import argparse
import copy
import json
import logging
import math
import os
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Tuple, Union

import torch
from datasets import Dataset, DatasetDict
from deepspeed.ops.adam import DeepSpeedCPUAdam
from mlsdk import (
    CacheOptions,
    Context,
    MNDevice,
    TensorProxy,
    set_tensor_name,
    set_tensor_name_in_module,
    storage,
    trace_event,
    trace_scope,
)
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2TokenizerFast,
    get_scheduler,
    pipeline,
)

SHOW_DETAILED_INFO_ITER = 20
VOCAB = 151936
CHAN = 1536
HEAD = 12


@dataclass
class TrainingConfig:
    dataset_json: Path
    model_name: str
    sequence_length: int
    n_steps: int
    batch_size: int
    n_hidden_layers: int
    max_steps: int
    learning_rate: float
    output_dir: Path
    enable_load_codegen_dir: bool
    save_model_dir: Optional[Path]
    device: str
    dtype: str
    run_name: Optional[str]
    generate: Literal["always", "skip", "first_only", "last_only"]
    distable_progress_bar: bool
    train_log_path: Optional[str]
    tloss_threshold: Optional[float]
    eloss_threshold: Optional[float]
    grad_accumulation_step: int = 1


def gen_logger(name: Optional[str] = None) -> logging.Logger:
    logging_env = "INFO"
    loglevel = logging.getLevelName(logging_env)

    fmt = (
        "%(levelname)s %(asctime)s %(thread)d %(threadName)s "
        "%(filename)s:%(lineno)d] %(message)s"
    )
    date_format = "%H:%M:%S"
    formatter = logging.Formatter(fmt, date_format)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(loglevel)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    logger.addHandler(stream_handler)
    return logger


_logger = gen_logger(__name__)


def save_train_log(  # noqa: CFQ002
    epoch: int,
    loss: float,
    mean_token_accuracy: float,
    learning_rate: float,
    grad_norm: float,
    num_tokens: int,
    step_count: int,
    max_steps: int,
    save_path: str,
) -> None:
    _logger.info(f"save_train_log {epoch} {step_count}/{max_steps}")

    timestamp = datetime.now(timezone.utc).isoformat()
    save_data = {
        "epoch": epoch,
        "loss": loss,
        "mean_token_accuracy": mean_token_accuracy,
        "timestamp": timestamp,
        "learning_rate": learning_rate,
        "grad_norm": grad_norm,
        "num_tokens": num_tokens,
        "step_count": step_count,
        "max_steps": max_steps,
    }

    with open(save_path, "a") as f:
        f.write(json.dumps(save_data) + "\n")
        f.flush()


def get_dataloaders(
    tokenizer: Qwen2TokenizerFast, conf: TrainingConfig
) -> dict[str, DataLoader]:
    return get_dataloaders_conversation_json(tokenizer, conf)


def get_dataloaders_conversation_json(  # noqa: CFQ001
    tokenizer: Qwen2TokenizerFast, conf: TrainingConfig
) -> dict[str, DataLoader]:
    sequence_length = conf.sequence_length
    batch_size = conf.batch_size
    num_valid_samples = 1

    if not hasattr(get_dataloaders, "_cache"):
        get_dataloaders._cache = {}
    if conf.dataset_json in get_dataloaders._cache:
        data = get_dataloaders._cache[conf.dataset_json]
        print(f"Using cached data for {conf.dataset_json}")
    else:
        with open(conf.dataset_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            get_dataloaders._cache[conf.dataset_json] = data

    instructions = []
    outputs = []
    for conversation in data:
        for i in range(0, len(conversation), 2):
            if (
                i + 1 < len(conversation)
                and conversation[i]["role"] == "user"
                and conversation[i + 1]["role"] == "assistant"
            ):
                instructions.append(conversation[i]["content"])
                outputs.append(conversation[i + 1]["content"])
    full_dataset = Dataset.from_dict({"instruction": instructions, "output": outputs})
    raw_datasets = DatasetDict({"full": full_dataset.shuffle(seed=42)})

    def tokenize(element: dict[str, Any]) -> dict[str, list[list[int]]]:
        all_input_ids = []
        all_labels = []
        assistant_marker_tokens = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        assert len(assistant_marker_tokens) == 3
        for instruction, output in zip(element["instruction"], element["output"]):
            conversation = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
            )
            assert isinstance(text, str)
            lines = text.splitlines()
            if "system" in lines[0]:
                lines = lines[2:]
                text = "\n".join(lines)
            assert "user" in lines[0], lines
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            all_marker_positions = []
            for i in range(len(input_ids) - len(assistant_marker_tokens) + 1):
                if (
                    input_ids[i : i + len(assistant_marker_tokens)]
                    == assistant_marker_tokens
                ):
                    all_marker_positions.append(i)
            assert len(all_marker_positions) == 1
            assistant_start_idx = all_marker_positions[0] + len(assistant_marker_tokens)

            # User part and markers are -100 (mask)
            labels = [
                -100 if i < assistant_start_idx else token_id
                for i, token_id in enumerate(input_ids)
            ]
            all_input_ids.extend(input_ids)
            all_labels.extend(labels)
        input_batch = []
        labels_batch = []
        for i in range(0, len(all_input_ids), sequence_length):
            input_chunk = all_input_ids[i : i + sequence_length]
            label_chunk = all_labels[i : i + sequence_length]
            if len(input_chunk) < sequence_length:
                padding_length = sequence_length - len(input_chunk)
                input_chunk += [tokenizer.pad_token_id] * padding_length
                label_chunk += [
                    -100
                ] * padding_length  # Padding tokens are masked with -100
            input_batch.append(input_chunk)
            labels_batch.append(label_chunk)

        return {
            "input_ids": input_batch,
            "labels": labels_batch,
        }

    tokenized_full_dataset_dict = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["full"].column_names
    )
    tokenized_full_dataset = tokenized_full_dataset_dict["full"]
    assert len(tokenized_full_dataset) >= num_valid_samples + 1

    valid_set = tokenized_full_dataset.select(range(num_valid_samples))
    train_set = tokenized_full_dataset.select(
        range(num_valid_samples, len(tokenized_full_dataset))
    )

    tokenized_datasets = DatasetDict({"train": train_set, "valid": valid_set})

    _logger.info(f"{len(tokenized_datasets['train'])=}")
    _logger.info(f"{len(tokenized_datasets['valid'])=}")
    train_sampler = None
    shuffle_train_loader = True
    g = torch.Generator()
    g.manual_seed(42)
    if len(tokenized_datasets["train"]) < batch_size:
        train_sampler = RandomSampler(
            tokenized_datasets["train"],
            replacement=True,
            num_samples=batch_size,
            generator=g,
        )
        shuffle_train_loader = False
        # the shuffle argument cannot be used with the sampler

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    # To retain label, use a custom collator instead of DataCollatorForLanguageModeling
    def custom_data_collator(examples):
        input_ids = torch.stack([torch.tensor(ex["input_ids"]) for ex in examples])
        labels = torch.stack([torch.tensor(ex["labels"]) for ex in examples])
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    dataloaders = {
        "train": DataLoader(
            tokenized_datasets["train"],
            shuffle=shuffle_train_loader,
            sampler=train_sampler,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=custom_data_collator,
            batch_size=batch_size,
            drop_last=True,
        ),
        "valid": DataLoader(
            tokenized_datasets["valid"],
            shuffle=False,
            collate_fn=custom_data_collator,
            drop_last=True,
        ),
    }
    _logger.info(f"{len(dataloaders['train'])=}")
    _logger.info(f"{len(dataloaders['valid'])=}")
    assert (
        len(dataloaders["train"]) > 0
    ), f"{len(dataloaders['train'])=} No training data"
    assert len(dataloaders["valid"]) == num_valid_samples
    return dataloaders


def split_batch_for_py_gradient_accumulation(
    batch: dict[str, torch.Tensor], grad_accumulation_step: int
) -> list[dict[str, torch.Tensor]]:
    batch_size = len(batch["input_ids"])
    assert (
        batch_size % grad_accumulation_step == 0
    ), f"{batch_size=} {grad_accumulation_step=}"

    batch_for_gradient_accumulation: list[dict[str, torch.Tensor]] = []
    for i in range(grad_accumulation_step):
        batch_for_gradient_accumulation.append(
            {
                "input_ids": batch["input_ids"][i::grad_accumulation_step],
                "labels": batch["labels"][i::grad_accumulation_step],
                "attention_mask": batch["attention_mask"][i::grad_accumulation_step],
            }
        )
    return batch_for_gradient_accumulation


def get_model(
    model_name: str,
    dtype: str,
    device: str,
    n_hidden_layers: int,
) -> torch.nn.Module:
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        .to(device)
        .to(getattr(torch, dtype))
    )
    if n_hidden_layers > 0:
        assert n_hidden_layers <= len(model.model.layers)
        model.model.layers = model.model.layers[:n_hidden_layers]
        _logger.info(
            f"Reducing the number of layers from {len(model.model.layers)} to {n_hidden_layers} because the num_hidden_layers parameter was modified via args."  # NOQA: B950
        )

    _logger.info(type(model))
    _logger.info(model.config)
    model_size = sum(t.numel() for t in model.parameters())
    model_dtype_size: dict[torch.dtype, int] = {}
    for t in model.parameters():
        model_dtype_size[t.dtype] = model_dtype_size.get(t.dtype, 0) + t.numel()
    _logger.info(f"{model_size / 1000 ** 2:.1f}M parameters: {model_dtype_size}")
    return model


def get_tokenizer(model_name: str) -> Qwen2TokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase), f"{type(tokenizer)=}"
    assert isinstance(tokenizer, Qwen2TokenizerFast), f"{type(tokenizer)=}"
    return tokenizer


def get_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    weight_decay = 0.0
    return DeepSpeedCPUAdam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        adamw_mode=True,
    )


def gen_layout_specs(vocab, chan, seqlen, head, bsize, out="/tmp"):  # NOQA: CFQ001
    embed_shape = [vocab, chan]
    indices_shape = [bsize, seqlen]
    indices_idx_add_shape = [bsize * seqlen]
    feats_shape = [bsize, seqlen, chan]
    feats_idx_add_shape = [bsize * seqlen, chan]

    embed_gemm_layout = "((25_Time:1, 3:1536, 8_L2B:1, 16_MAB:1, 4:1, 4:4), (96:16, 4_W:1, 4_PE:1); B@[L1B])"  # NOQA: B950
    embed_idx_layout = "((75_Time:4, 8_L2B:1, 16_MAB:1, 4:1, 4:4), (4_Time:1, 24:16, 4_W:1, 4_PE:1); B@[L1B])"  # NOQA: B950
    feats_gemm_layout = (
        "((), (8:3072, 8_L1B:1, 2_Time:1, 2:1536, 16:1), (96:16, 4_W:1, 4_PE:1))"
    )
    feats_gather_layout = (
        "((), (8_L2B:1, 8_L1B:1, 64:1), (4_Time:1, 24:64, 4_W:1, 4_PE:1))"
    )
    feats_idx_add_layout = feats_gather_layout.replace("(), ", "")
    logits_layout = "((), (8:96, 8_L1B:1, 2_Time:25, 2:48, 16:1), (25_Time:1, 3:16, 8_L2B:1, 16_MAB:1, 4_W:1, 4_PE:1))"  # NOQA: B950
    indices_gather_layout = "((), (8_L2B:1, 8_L1B:1, 32:1, 2_W:1))"
    indices_idx_add_layout = indices_gather_layout.replace("(), ", "")

    gemm_layout_spec = [
        {
            "key": "embed_tokens",
            "A": feats_gemm_layout,
            "BT": embed_gemm_layout,
            "C": logits_layout,
        }
    ]

    layout_spec = []
    for trigger, op_type in [
        ("SwitchInput", "Gather"),
        ("SwitchInput", "ChainerIndexAdd"),
        ("FixOutput", "ChainerIndexAdd"),
    ]:
        layout_spec.append(
            {
                "trigger": trigger,
                "op_type": op_type,
                "shape": embed_shape,
                "layout": embed_idx_layout,
            }
        )

    layout_spec.append(
        {
            "trigger": "SwitchInput",
            "op_type": "Gather",
            "shape": indices_shape,
            "layout": indices_gather_layout,
        }
    )
    layout_spec.append(
        {
            "trigger": "SwitchInput",
            "op_type": "ChainerIndexAdd",
            "shape": indices_idx_add_shape,
            "layout": indices_idx_add_layout,
        }
    )

    layout_spec.append(
        {
            "trigger": "FixOutput",
            "op_type": "Gather",
            "shape": feats_shape,
            "layout": feats_gather_layout,
        }
    )
    layout_spec.append(
        {
            "trigger": "SwitchInput",
            "op_type": "ChainerIndexAdd",
            "shape": feats_idx_add_shape,
            "layout": feats_idx_add_layout,
        }
    )

    # For gradient accumulation of weights with time slice.
    for trigger in ["SwitchInput", "FixOutput"]:
        layout_spec.append(
            {
                "trigger": trigger,
                "op_type": "Add",
                "shape": embed_shape,
                "layout": embed_idx_layout,
            }
        )
    for trigger in ["SwitchInput", "FixOutput"]:
        layout_spec.append(
            {
                "trigger": trigger,
                "op_type": "Add",
                "shape": [8960, chan],
                "layout": "((16_MAB:1, 5_Time:1, 7:16, 4:1, 4:4), (96:112, 4_W:1, 4_PE:1); B@[L1B,L2B])",  # NOQA: B950
            }
        )
    for trigger in ["SwitchInput", "FixOutput"]:
        layout_spec.append(
            {
                "trigger": trigger,
                "op_type": "Add",
                "shape": [chan, 8960],
                "layout": "((96:112, 4_W:1, 4_PE:1), (16_MAB:1, 5_Time:1, 7:16, 4:1, 4:4); B@[L1B,L2B])",  # NOQA: B950
            }
        )

    # All-gather at L2B and L1B before Expand in GQA.
    layout_spec.append(
        {
            "trigger": "SwitchInput",
            "op_type": "Expand",
            "shape": [1, 2, 1, seqlen, chan // head],
            "layout": "((), (2_Time:1), (), (16_MAB:1, 16:16, 4:1, 4:4), (8:256, 4_W:1, 4_PE:1); B@[L1B,L2B])",  # NOQA: B950
        }
    )
    layout_spec.append(
        {
            "trigger": "FixOutput",
            "op_type": "Expand",
            "shape": [1, 2, 6, seqlen, chan // head],
            "layout": "((), (2_Time:6), (6_Time:1), (16_MAB:1, 16:16, 4:1, 4:4), (8:256, 4_W:1, 4_PE:1); B@[L1B,L2B])",  # NOQA: B950
        }
    )

    with open(os.path.join(out, "gemm_layout_spec.json"), "w") as f:
        json.dump(gemm_layout_spec, f, indent=2)

    with open(os.path.join(out, "layout_spec.json"), "w") as f:
        json.dump(layout_spec, f, indent=2)

    return os.path.join(out, "gemm_layout_spec.json"), os.path.join(
        out, "layout_spec.json"
    )


def get_compile_options(conf: TrainingConfig) -> dict[str, Any]:
    gemm_layout_spec, layout_spec = gen_layout_specs(
        vocab=VOCAB, chan=CHAN, seqlen=conf.sequence_length, head=HEAD, bsize=1
    )  # NOQA: CFQ001
    compile_options = {
        "gemm_layout_spec": gemm_layout_spec,
        "layout_spec": layout_spec,
        "scheduler": "auto_recompute_sa",
    }
    return compile_options


def save_huggingface_format(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, save_directory: Path
):
    _logger.info(f"Saving model to {save_directory}")
    model.save_pretrained(str(save_directory))
    tokenizer.save_pretrained(str(save_directory))


def generate(
    model: torch.nn.Module,
    tokenizer: Qwen2TokenizerFast,
    device: str,
    num_return_sequences: int = 1,
) -> None:
    model.eval()

    def create_test_prompt(tokenizer: Qwen2TokenizerFast, user_instruction: str) -> str:
        conversation = [
            {"role": "user", "content": user_instruction},
        ]
        prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        lines = prompt.splitlines()
        if "system" in lines[0]:
            lines = lines[2:]
            prompt = "\n".join(lines)
        assert "user" in lines[0]
        return prompt + "\n"

    with torch.no_grad():
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=32,
        )
        txt = [
            create_test_prompt(tokenizer, "日本の首都は？"),
            create_test_prompt(tokenizer, "田中、野球しようぜ。"),
            create_test_prompt(tokenizer, "君はだれ？"),
            create_test_prompt(tokenizer, "Hello! How are you?"),
            create_test_prompt(tokenizer, "田中、一緒に海に行こうよ。"),
        ]
        results = pipe(txt, num_return_sequences=num_return_sequences)
        for x in results:
            for y in x:
                _logger.info(y["generated_text"])
                _logger.info("=" * 20)
    model.train()


def eval(
    model: torch.nn.Module,
    tokenizer: Qwen2TokenizerFast,
    dataloaders: dict[str, DataLoader],
    eval_step: Any,
    device: str,
) -> float:
    model.eval()
    with torch.no_grad():
        eloss = Average()
        for batch in tqdm(dataloaders["valid"], desc="eval"):
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["labels"] = batch["labels"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            with torch.no_grad():
                loss = eval_step(batch)["loss"]
                eloss.update(loss)
        _logger.info(f"eloss: {eloss.avg()}")
        model.train()
        return eloss.avg()


def compile_for_py_grad_host_optimizer(  # noqa: CFQ002, CFQ004
    *,
    conf: TrainingConfig,
    context: Optional[Context],
    model: torch.nn.Module,
    tokenizer: Qwen2TokenizerFast,
) -> Tuple[
    Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]],
    Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]],
]:  # noqa CFQ004
    def calc_grads_and_loss(
        inp: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        assert inp["input_ids"].size() == inp["attention_mask"].size()
        outputs = model(
            input_ids=inp["input_ids"],
            labels=inp["labels"],
            attention_mask=inp["attention_mask"],
        )
        loss = outputs.loss
        loss.backward()
        return {"loss": loss}

    def eval_step(inp: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        outputs = model(
            input_ids=inp["input_ids"],
            labels=inp["labels"],
            attention_mask=inp["attention_mask"],
        )
        loss = outputs.loss
        return {"loss": loss}

    set_tensor_name_in_module(model, "model")
    for n, p in model.named_parameters():
        context.register_param(p)

        # Register grad tensor
        p.grad = torch.nn.Parameter(torch.zeros_like(p))
        set_tensor_name(p.grad, f"{n}_grad".replace(".", "@"))
        context.register_param(p.grad)

    train_codegen_dir = storage.path(
        target=str(conf.output_dir / "codegen" / "train_step")
    )
    eval_codegen_dir = storage.path(
        target=str(conf.output_dir / "codegen" / "eval_step")
    )
    if conf.enable_load_codegen_dir:
        compiled_calc_grads_and_loss = context.load_codegen_dir(train_codegen_dir)
        compiled_eval_step = context.load_codegen_dir(eval_codegen_dir)
        return compiled_calc_grads_and_loss, compiled_eval_step

    dataloaders: dict[str, DataLoader] = get_dataloaders(tokenizer, conf)
    dat_train: DataLoader = dataloaders["train"]
    it = iter(dat_train)

    sample_input = next(it)

    splitted_sample_input = split_batch_for_py_gradient_accumulation(
        sample_input, conf.grad_accumulation_step
    )[0]

    _logger.info(f"{splitted_sample_input['input_ids'].size()=}")
    _logger.info(f"{splitted_sample_input['labels'].size()=}")
    _logger.info(f"{splitted_sample_input['attention_mask'].size()=}")

    compile_options = get_compile_options(conf)

    compiled_calc_grads_and_loss = context.compile(
        calc_grads_and_loss,
        splitted_sample_input,
        train_codegen_dir,
        options=compile_options,
        cache_options=CacheOptions(str(conf.output_dir / "cache" / "train")),
    )

    compiled_eval_step = context.compile(
        eval_step,
        next(iter(dataloaders["valid"])),
        eval_codegen_dir,
        options=compile_options,
        cache_options=CacheOptions(str(conf.output_dir / "cache" / "eval")),
    )
    return compiled_calc_grads_and_loss, compiled_eval_step


class Average:
    def __init__(self) -> None:
        self.v = 0.0
        self.n = 0

    def update(self, v: Union[float, TensorProxy]) -> None:
        if isinstance(v, TensorProxy):
            v = v.cpu()
        self.v += v
        self.n += 1

    def avg(self) -> float:
        return self.v / self.n if self.n > 0 else torch.nan


def run_with_py_gradient_accumulation(  # noqa: CFQ001, CFQ002, CFQ004
    conf: TrainingConfig,
) -> None:
    # To make the result reproducible.
    torch.manual_seed(0)
    # To show more digits in the log.
    torch.set_printoptions(precision=10)  # type: ignore

    _logger.info(f"run_with_py_gradient_accumulation {conf.output_dir=}")
    if conf.run_name is None:
        conf.run_name = str(uuid.uuid4())
    _logger.info(f"run name: {conf.run_name}")

    context = Context(MNDevice(conf.device))
    Context.switch_context(context)
    tokenizer = get_tokenizer(conf.model_name)
    model = get_model(
        model_name=conf.model_name,
        dtype=conf.dtype,
        device="cpu",  # Torch device is always CPU
        n_hidden_layers=conf.n_hidden_layers,
    )

    # This function will register model parameters and their grads to the context,
    # effectively moving the device from CPU to context's device (MN-Core).
    calc_grads_and_loss, eval_step = compile_for_py_grad_host_optimizer(
        conf=conf,
        context=context,
        model=model,
        tokenizer=tokenizer,
    )

    dataloaders = get_dataloaders(tokenizer, conf)
    # Create a copy of the model on CPU used for on-host optimization.
    model_copy_on_cpu = copy.deepcopy(model).to("cpu")
    optimizer = get_optimizer(model_copy_on_cpu, lr=conf.learning_rate)

    if conf.n_steps == -1:
        n_steps_per_epoch = len(dataloaders["train"])
        _logger.info(f"{n_steps_per_epoch=}")
    else:
        n_steps_per_epoch = conf.n_steps

    num_training_steps = conf.max_steps
    n_epochs = math.ceil(conf.max_steps / n_steps_per_epoch)

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    _logger.info(f"{num_training_steps=}")

    if conf.generate == "first_only":
        _logger.info("Generate is set to first_only. Generating before training...")
        generate(model=model, tokenizer=tokenizer, device="cpu", num_return_sequences=1)
        _logger.info("Generation before training done. Exit immediately.")
        return

    _logger.info(f"{n_steps_per_epoch=}")

    step_count = 0
    last_tloss = math.inf

    for e in range(n_epochs):
        _logger.info(f"EPOCH: {e} / {n_epochs}")
        model.train()
        bar = tqdm(range(n_steps_per_epoch), disable=conf.distable_progress_bar)
        tloss = Average()
        for i, batch in enumerate(dataloaders["train"]):
            if i >= n_steps_per_epoch:
                break

            # Split batch for gradient accumulation
            batches = split_batch_for_py_gradient_accumulation(
                batch, conf.grad_accumulation_step
            )
            assert len(batches) == conf.grad_accumulation_step, len(batches)

            batch_loss = torch.tensor(0.0, device="cpu")

            # Accumulate gradients on the cpu (host)
            for b in batches:
                out = calc_grads_and_loss(b)
                # This is not typo! The first `.cpu()` move the tensor in
                # the codegen world to the torch world. The second `.cpu()`
                # move the tensor from GPU or CPU to CPU.
                loss = out["loss"].cpu().cpu()
                batch_loss += loss
                _logger.info(f"mini batch {loss=}")
                del out, loss
            del batches, batch
            batch_loss /= conf.grad_accumulation_step
            tloss.update(batch_loss)

            if (e == 0 and i < SHOW_DETAILED_INFO_ITER) or i % 100 == 0:
                _logger.info(f"{e=}, {i=}, {batch_loss=} {lr_scheduler.get_last_lr()=}")

            # Fetch accumulated gradients from the device
            context.synchronize()

            # Copy grads to model on the host
            grad_dict: dict[str, torch.Tensor] = {}
            for k, v in model.named_parameters():
                assert v.grad is not None
                grad_dict[k] = v.grad.cpu()
            for k, v in model_copy_on_cpu.named_parameters():
                v.grad = grad_dict[k]

            # Run optimizer on the host
            with torch.no_grad():
                with trace_event("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()

            with trace_event("HtoD"):  # Host to Device
                with torch.no_grad():
                    # asynchronously copy model parameters to the device
                    for k, v in model.named_parameters():
                        context.get_registered_value_proxy(v).load_from(
                            model_copy_on_cpu.state_dict()[k], clone=False
                        )
                        assert v.grad is not None
                        context.get_registered_value_proxy(v.grad).load_from(
                            v.grad.zero_(), clone=False
                        )
                    # copy model_on_cpu parameters to the model
                    for k, v in model.named_parameters():
                        v.copy_(model_copy_on_cpu.state_dict()[k])
                        assert v.grad is not None
                        # Note: We cannot clear grads in this way. This clear grads
                        # forever.
                        # p.grad = torch.nn.Parameter(torch.zero_like(p))
                        if context is None:
                            v.grad.zero_()

            last_tloss = tloss.avg()
            lr_scheduler.step()  # do lr_scheduler.step after saving
            bar.set_description(f"epoch: {e}, tloss: {last_tloss}")
            bar.update(1)
            if conf.train_log_path is not None:
                save_train_log(
                    epoch=e,
                    loss=batch_loss.item(),
                    mean_token_accuracy=0,
                    learning_rate=lr_scheduler.get_last_lr(),
                    grad_norm=0,
                    num_tokens=0,
                    step_count=step_count,
                    max_steps=conf.max_steps,
                    save_path=conf.train_log_path,
                )

            step_count += 1
            _logger.info(f"{step_count=}")

            if step_count >= conf.max_steps:
                break

        eloss = eval(model, tokenizer, dataloaders, eval_step, device="cpu")
        if e == n_epochs - 1 and conf.eloss_threshold is not None:
            assert (
                eloss <= conf.eloss_threshold
            ), f"eloss {eloss} is greater than threshold {conf.eloss_threshold}"

        if (conf.generate == "always") or (
            (conf.generate == "last_only") and (e == n_epochs - 1)
        ):
            with torch.no_grad():
                model.eval()
                generate(model=model, tokenizer=tokenizer, device="cpu")
                model.train()

        context.synchronize()
        if step_count >= conf.max_steps:
            break
    if conf.tloss_threshold is not None:
        _logger.info(
            f"last tloss: {last_tloss}, tloss_threshold: {conf.tloss_threshold}"
        )
        assert last_tloss <= conf.tloss_threshold
    if conf.save_model_dir:
        save_huggingface_format(model, tokenizer, conf.save_model_dir)


def main() -> None:  # noqa: CFQ001
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--dataset_json",
        type=Path,
        help="Path to the conversation dataset in JSON format.",
    )
    argparser.add_argument(
        "--model",
        type=str,
        default="tiny-swallow-1.5b",
        help="Model name to use for training. Short names: qwen2.5-1.5b, tiny-swallow-1.5b",
    )
    argparser.add_argument("--sequence_length", type=int, default=4096)
    argparser.add_argument("--n_steps", type=int, default=-1)
    argparser.add_argument("--max_steps", type=int, default=40)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--learning_rate", type=float, default=3e-5)
    argparser.add_argument("--n_hidden_layers", type=int, default=-1)
    argparser.add_argument(
        "--codegen_output_dir", type=Path, default=Path("/tmp/slm_sft_gian_output")
    )
    argparser.add_argument("--enable_load_codegen_dir", action="store_true")
    argparser.add_argument("--save_model_dir", type=Path, default=None)
    argparser.add_argument("--device", type=str, default="mncore2:auto")
    argparser.add_argument(
        "--dtype", type=str, default="float", choices=["float", "bfloat16", "float16"]
    )
    argparser.add_argument("--run", type=str, default=None)
    argparser.add_argument(
        "--generate",
        type=str,
        default="last_only",
        choices=[
            "always",
            "skip",
            "first_only",
            "last_only",
        ],
    )
    argparser.add_argument("--tloss_threshold", type=float, default=None)
    argparser.add_argument("--eloss_threshold", type=float, default=None)

    argparser.add_argument("--disable_progress_bar", action="store_true")
    argparser.add_argument(
        "--perfetto_trace", type=str, default=None, help="perfetto trace file"
    )
    argparser.add_argument("--train_log_path", type=str, default=None)

    args = argparser.parse_args()

    if args.perfetto_trace:
        perfetto_trace = args.perfetto_trace
    else:
        perfetto_trace = os.path.join(args.codegen_output_dir, "perfetto_trace.pb")

    if args.model == "qwen2.5-1.5b":
        args.model = "Qwen/Qwen2.5-1.5B-Instruct"
    elif args.model == "tiny-swallow-1.5b":
        args.model = "SakanaAI/TinySwallow-1.5B"

    if not os.path.exists(args.dataset_json):
        raise FileNotFoundError(
            f"Dataset JSON file {args.dataset_json} does not exist."
        )

    if (args.save_model_dir is not None) and os.path.exists(args.save_model_dir):
        _logger.error(f"Model output directory {args.save_model_dir} already exists.")
        _logger.error("This may overwrite existing files.")
        exit(1)

    assert args.max_steps > 0, "max_steps must be greater than 0"
    assert args.device.startswith("pfvm") or args.device.startswith(
        "mncore"
    ), "Only pfvm/mncore devices are supported."

    conf = TrainingConfig(
        dataset_json=args.dataset_json,
        model_name=args.model,
        sequence_length=args.sequence_length,
        n_steps=args.n_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_hidden_layers=args.n_hidden_layers,
        output_dir=args.codegen_output_dir,
        enable_load_codegen_dir=args.enable_load_codegen_dir,
        save_model_dir=args.save_model_dir,
        device=args.device,
        dtype=args.dtype,
        run_name=args.run,
        generate=args.generate,
        distable_progress_bar=args.disable_progress_bar,
        tloss_threshold=args.tloss_threshold,
        eloss_threshold=args.eloss_threshold,
        grad_accumulation_step=args.batch_size,
        train_log_path=args.train_log_path,
    )

    # Create trace directory if it does not exist
    trace_dir = os.path.dirname(perfetto_trace)
    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)

    with trace_scope(perfetto_trace):
        run_with_py_gradient_accumulation(conf)


if __name__ == "__main__":
    main()
