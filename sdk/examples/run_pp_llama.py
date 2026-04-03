import argparse
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from mlsdk import Context, MNDevice, storage
from mlsdk.experimental.llm.attention_mask import (
    prepare_4d_causal_attention_mask_with_cache_position,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import LlamaConfig

logger = logging.getLogger(__name__)


def prepare_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        fallback_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
        logger.info(f"Falling back to TinyLlama tokenizer:\n{e}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # For static KV caching

    return tokenizer


def prepare_inputs(tokenizer, max_length, prompt, system_prompt):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    chat_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        chat_template,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
    )
    inputs["chat_template"] = chat_template
    inputs["position_ids"] = inputs["attention_mask"].cumsum(1)
    inputs["attention_mask"] = prepare_4d_causal_attention_mask_with_cache_position(
        inputs["attention_mask"],
        inputs["position_ids"],
        torch.float32,
    )

    return inputs


def prepare_sendrecv_buffers(
    rank, world_size, bsize, max_length, hidden_size, hidden_size_per_head
):
    sendrecv_buffers = {}
    if rank == 0:
        sendrecv_buffers["input_ids"] = torch.zeros(bsize, 1).long()
    if rank == (world_size - 1):
        sendrecv_buffers["prefill"] = torch.zeros(bsize, max_length, hidden_size)
        sendrecv_buffers["decode"] = torch.zeros(bsize, 1, hidden_size)
    else:
        sendrecv_buffers["prefill"] = torch.zeros(
            bsize, max_length, hidden_size + 2 * hidden_size_per_head
        )
        sendrecv_buffers["decode"] = torch.zeros(
            bsize, 1, hidden_size + 2 * hidden_size_per_head
        )
    return sendrecv_buffers


def load_codegen_dirs(context, rank, prefill_codegen_dirs, decode_codegen_dirs, outdir):
    def forward_dummy(_):
        raise RuntimeError("This function should not be called.")

    prefill_func = context.load_codegen_dir(storage.path(prefill_codegen_dirs[rank]))
    decode_func = context.load_codegen_dir(storage.path(decode_codegen_dirs[rank]))

    return prefill_func, decode_func


def send(dist, tensor, dst):
    logger.debug(f"Sending {tensor.shape} ({tensor.dtype}) to rank {dst}.")
    dist.send(tensor=tensor, dst=dst)
    logger.debug(f"Successfully sent {tensor.shape} ({tensor.dtype}) to rank {dst}.")


def recv(dist, tensor, src):
    logger.debug(f"Receiving {tensor.shape} ({tensor.dtype}) from rank {src}.")
    dist.recv(tensor=tensor, src=src)
    logger.debug(
        f"Successfully received {tensor.shape} ({tensor.dtype}) from rank {src}."
    )


def forward(  # NOQA
    rank,
    world_size,
    model_name,
    bsize,
    max_length,
    prefill_codegen_dirs,
    decode_codegen_dirs,
    system_prompt,
    prompt,
    outdir,
    verbose,
):
    logging.basicConfig(
        format=f"[%(asctime)s %(filename)s:%(lineno)d] [%(process)d] [Rank {rank}] %(levelname)s: %(message)s",  # NOQA
        level=logging.DEBUG if verbose else logging.INFO,
    )

    logger.info(
        f"Loading precompiled prefill and decode codegen directories for stage {rank}."
    )
    device = MNDevice(f"mncore2:{rank}")
    context = Context(device)
    Context.switch_context(context)
    prefill_func, decode_func = load_codegen_dirs(
        context,
        rank,
        prefill_codegen_dirs,
        decode_codegen_dirs,
        outdir,
    )

    logger.info("Preparing inputs.")
    tokenizer = prepare_tokenizer(model_name)
    inputs = prepare_inputs(tokenizer, max_length, prompt, system_prompt)
    if rank == 0:
        logger.info(f"Prompt:\n{inputs['chat_template']}")

    if model_name.endswith(".json"):
        config = LlamaConfig.from_json_file(model_name)
    else:
        config = AutoConfig.from_pretrained(model_name)
    hidden_size = config.hidden_size
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    sendrecv_buffers = prepare_sendrecv_buffers(
        rank,
        world_size,
        bsize,
        max_length,
        hidden_size,
        hidden_size_per_head,
    )

    torch.distributed.barrier()

    n_input_tokens = inputs["position_ids"].max().item()
    max_new_tokens = max_length - n_input_tokens

    if rank == 0:
        logger.debug(f"{n_input_tokens} input tokens, {max_new_tokens} max new tokens.")

        output_ids = []

    start = time.time()
    for step in range(max_new_tokens):
        if rank == 0:
            inputs_for_stage = {
                "input_ids": inputs["input_ids"],
                "position_ids": inputs["position_ids"],
            }
        elif rank < world_size - 1:
            buffer = sendrecv_buffers["prefill" if step == 0 else "decode"]
            recv(dist, buffer, rank - 1)
            inputs_for_stage = {
                "hidden_states": buffer[:, :, :hidden_size],
                "position_embeddings_cos": buffer[
                    :, :, hidden_size : hidden_size + hidden_size_per_head
                ],
                "position_embeddings_sin": buffer[
                    :, :, hidden_size + hidden_size_per_head :
                ],
                "attention_mask": inputs["attention_mask"],
                "position_ids": inputs["position_ids"],
            }
            if step > 0:
                assert "past_key_values" in inputs
                inputs_for_stage["past_key_values"] = inputs["past_key_values"]
        else:
            buffer = sendrecv_buffers["prefill" if step == 0 else "decode"]
            recv(dist, buffer, rank - 1)
            inputs_for_stage = {"hidden_states": buffer}

        logger.debug(f"Step {step}. Compiled function start.")
        if step == 0:
            func = prefill_func
        else:
            func = decode_func
        outputs = func(inputs_for_stage)
        for key in outputs:
            logger.debug(f"Step {step}. Copying '{key}' to host start.")
            outputs[key] = outputs[key].cpu()
            logger.debug("Successfully copied.")
        logger.debug(f"Step {step}. Compiled function end.")

        if rank == 0:
            hidden_states = outputs["hidden_states_out"]
            position_embeddings_cos = outputs["position_embeddings_cos_out"]
            position_embeddings_sin = outputs["position_embeddings_sin_out"]
            tensor = torch.cat(
                [hidden_states, position_embeddings_cos, position_embeddings_sin],
                dim=-1,
            )
            send(dist, tensor, 1)
        elif rank < world_size - 1:
            if rank == world_size - 2:
                tensor = outputs["hidden_states_out"]
            else:
                hidden_states = outputs["hidden_states_out"]
                position_embeddings_cos = outputs["position_embeddings_cos_out"]
                position_embeddings_sin = outputs["position_embeddings_sin_out"]
                tensor = torch.cat(
                    [hidden_states, position_embeddings_cos, position_embeddings_sin],
                    dim=-1,
                )
            send(dist, tensor, rank + 1)
            past_key_values = outputs["past_key_values_out"]

            inputs["past_key_values"] = past_key_values
        else:
            logits = outputs["logits"]
            input_ids = logits.argmax(dim=2)[:, -1:]  # Greedy decoding.
            send(dist, input_ids, 0)

        if 0 < rank < world_size - 1:
            next_attention_mask = inputs["attention_mask"][:, :, -1:, :]
            next_attention_mask = torch.roll(next_attention_mask, shifts=-1, dims=-1)
            next_attention_mask[:, :, :, -1] = 0
            inputs["attention_mask"] = next_attention_mask.detach()

        if rank < world_size - 1:
            next_position_ids = inputs["position_ids"][:, -1:] + 1
            inputs["position_ids"] = next_position_ids.detach()

        if rank == 0:
            recv(dist, sendrecv_buffers["input_ids"], world_size - 1)
            inputs["input_ids"] = sendrecv_buffers["input_ids"]
            output_id = inputs["input_ids"][0, 0].item()
            output_ids.append(output_id)
            output_id_decoded = tokenizer.decode(
                output_id, clean_up_tokenization_spaces=True
            )
            # @todo (hvy): Break the loop for all ranks if the output is EOS.
            logger.info(f"Step {step}. Decoded output: '{output_id_decoded}'.")

        logger.debug(f"Step {step}. Enter barrier.")
        torch.distributed.barrier()
        logger.debug(f"Step {step}. Exited barrier.")

    end = time.time()
    context.invalidate()

    if rank == 0:
        output_ids_decoded = tokenizer.decode(
            output_ids, clean_up_tokenization_spaces=True
        )
        logger.info(f"Final output:\n{output_ids_decoded}")
        inputs = tokenizer(
            inputs["chat_template"],
            return_tensors="pt",
        )
        logger.info(f"Generated {max_new_tokens / (end - start)} tokens/sec.")
        start = time.time()
        if not model_name.endswith(".json"):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True
                )
            except Exception as e:
                logger.info(
                    f"Failed to load model from Hugging Face, trying to load from config. {e}"
                )
                model = AutoModelForCausalLM.from_config(config)
            model.eval()
            assert bsize == 1
            output_ids = model.generate(
                inputs["input_ids"],
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )[
                0
            ]  # Batch size of 1.
            end = time.time()
            output_ids_decoded = tokenizer.decode(
                output_ids, clean_up_tokenization_spaces=True
            )
            logger.info(f"model.generate output:\n{output_ids_decoded}")


def main(args):
    model_name = args.model_name
    bsize = args.bsize
    max_length = args.max_length
    prefill_codegen_dirs = args.prefill_codegen_dirs
    decode_codegen_dirs = args.decode_codegen_dirs
    prompt = args.prompt
    system_prompt = args.system_prompt
    outdir = args.outdir
    verbose = args.verbose

    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    assert len(decode_codegen_dirs) == len(prefill_codegen_dirs)
    assert len(decode_codegen_dirs) == world_size

    backend = "gloo"
    torch.distributed.init_process_group(
        backend=backend, init_method="env://", timeout=datetime.timedelta(hours=2)
    )
    torch.distributed.barrier()

    with torch.set_grad_enabled(False):
        forward(
            rank,
            world_size,
            model_name,
            bsize,
            max_length,
            prefill_codegen_dirs,
            decode_codegen_dirs,
            system_prompt,
            prompt,
            outdir,
            verbose,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--bsize", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--prefill_codegen_dirs", nargs="+")
    parser.add_argument("--decode_codegen_dirs", nargs="+")
    parser.add_argument(
        "--prefill_or_decode",
        type=str,
        choices=["prefill", "decode"],
        default="prefill",
    )
    parser.add_argument("--prompt", type=str, default="Write a song about MN-Core")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a friendly chatbot who is an expert on MN-Core.",
    )
    parser.add_argument("--outdir", type=str, default="/tmp/mlsdk_llm_pp_infer_dummy")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
