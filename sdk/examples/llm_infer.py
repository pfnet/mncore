import argparse
import os
from typing import Mapping

import torch
from mlsdk import CacheOptions, Context, MNDevice, storage
from mlsdk.experimental.llm.attention_mask import (
    prepare_4d_causal_attention_mask_with_cache_position,
)
from mlsdk.experimental.llm.kv_cache import (
    kv_cache_to_legacy,
    kv_cache_to_plamo,
    kv_cache_to_tensor,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def prepare_prompt(tokenizer, prompt, system_prompt):
    if tokenizer.chat_template:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def infer_with_generate(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt")
    # Greedy decoding for simplicity for comparing the results with the compiled version.
    output_ids = model.generate(
        inputs["input_ids"], do_sample=False, max_new_tokens=max_new_tokens
    )
    assert isinstance(output_ids, torch.Tensor)
    return output_ids


def infer_with_compilation(  # NOQA: CFQ002, CFQ001
    *,
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_length: int,
    max_new_tokens: int,
    compile_prefill: bool,
    compile_decode: bool,
    device_name: str,
    outdir: str,
    check_intermediate_outputs: bool,
    prepare_attention_mask_on_cpu: bool,
    disable_cache: bool,
    num_compiler_threads: int,
) -> torch.Tensor:
    is_plamo_model = any("plamo" in a.lower() for a in model.config.architectures)

    def forward(inputs: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        assert all(isinstance(v, torch.Tensor) for v in inputs.values()), {
            k: type(v) for k, v in inputs.items()
        }
        if "past_key_values" in inputs:
            if is_plamo_model:
                kv_cache_func = kv_cache_to_plamo
            else:
                # @todo (hvy): Stop using the deprecated legacy KV cache format of tuples.
                kv_cache_func = kv_cache_to_legacy
            past_key_values = kv_cache_func(inputs["past_key_values"])
        else:
            past_key_values = None

        outputs = model.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            past_key_values=past_key_values,
            use_cache=True,
        )
        return {
            "logits": outputs.logits,
            "next_past_key_values": kv_cache_to_tensor(outputs.past_key_values)[
                :, :, :, :, 1:, :
            ],  # Do every operation, including the shifting, for the KV cache on device.
        }

    assert tokenizer.padding_side == "left"
    inputs = tokenizer(
        prompt, return_tensors="pt", padding="max_length", max_length=max_length
    )
    # @todo (hvy): Consider subtracting 1 from the position_ids to match modeling_llama.py.
    assert "position_ids" not in inputs
    inputs["position_ids"] = inputs["attention_mask"].cumsum(1)
    if prepare_attention_mask_on_cpu:
        inputs["attention_mask"] = prepare_4d_causal_attention_mask_with_cache_position(
            inputs["attention_mask"], inputs["position_ids"], model.dtype
        )
    output_ids = inputs["input_ids"]

    device = MNDevice(device_name)
    context = Context(device)
    Context.switch_context(context)
    context.registry.register("model", model)

    compiled_funcs = {}

    for step in range(max_new_tokens):
        if step == 0:
            if compile_prefill and "prefill" not in compiled_funcs:
                # Set codegen internal flags
                os.environ["CODEGEN_TIME_SLICE_SCATTERED_INDEXING_BCAST"] = "1"
                os.environ["CODEGEN_OP_DEF"] = "Gather=GatherBcast"

                compiled_funcs["prefill"] = context.compile(
                    forward,
                    inputs,
                    storage.path(outdir + "/prefill"),
                    cache_options=(
                        CacheOptions(outdir + "/prefill_cache")
                        if not disable_cache
                        else None
                    ),
                    num_compiler_threads=num_compiler_threads,
                )
            forward_for_step = compiled_funcs.get("prefill", forward)
        else:
            if compile_decode and "decode" not in compiled_funcs:
                compiled_funcs["decode"] = context.compile(
                    forward,
                    inputs,
                    storage.path(outdir + "/decode"),
                    cache_options=(
                        CacheOptions(outdir + "/decode_cache")
                        if not disable_cache
                        else None
                    ),
                    num_compiler_threads=num_compiler_threads,
                )
            forward_for_step = compiled_funcs.get("decode", forward)

        outputs = forward_for_step(inputs)

        if check_intermediate_outputs:
            # @todo (hvy): Consider using a more sophisticated check for the outputs.
            if "mncore" in device_name:
                atol = 1.0
            else:
                assert device_name == "pfvm:cpu"
                atol = 5e-3
            n_tokens = inputs["position_ids"].max()
            outputs_expected = forward(inputs)
            logits = outputs["logits"][:, -n_tokens:]
            logits_expected = outputs_expected["logits"][:, -n_tokens:]
            next_past_key_values = outputs["next_past_key_values"][
                :, :, :, :, -n_tokens:
            ]
            next_past_key_values_expected = outputs_expected["next_past_key_values"][
                :, :, :, :, -n_tokens:
            ]

            assert torch.allclose(logits, logits_expected, atol=atol), (
                step,
                (logits - logits_expected).abs().max(),
            )
            assert torch.allclose(
                next_past_key_values, next_past_key_values_expected, atol=atol
            ), (
                step,
                (next_past_key_values - next_past_key_values_expected).abs().max(),
            )

        next_input_ids = (
            outputs["logits"].cpu().argmax(dim=2)[:, -1:]
        )  # Greedy decoding.
        if prepare_attention_mask_on_cpu:
            next_attention_mask = inputs["attention_mask"][:, :, -1:, :]
            next_attention_mask = torch.roll(next_attention_mask, shifts=-1, dims=-1)
            next_attention_mask[:, :, :, -1] = 0
        else:
            next_attention_mask = inputs["attention_mask"]
            next_attention_mask = torch.roll(next_attention_mask, shifts=-1, dims=-1)
            next_attention_mask[:, -1] = 1
        next_position_ids = inputs["position_ids"][:, -1:] + 1
        next_past_key_values = outputs["next_past_key_values"].cpu()
        inputs = {
            "input_ids": next_input_ids.detach(),
            "attention_mask": next_attention_mask.detach(),
            "position_ids": next_position_ids.detach(),
            "past_key_values": next_past_key_values.detach(),
        }

        output_ids = torch.cat([output_ids, next_input_ids], dim=1)

        if next_input_ids.item() == tokenizer.eos_token_id:
            break

    return output_ids[:, max_new_tokens:]


def main(args):
    prompt = args.prompt
    system_prompt = args.system_prompt
    model_name = args.model_name
    max_length = args.max_length
    max_new_tokens = args.max_new_tokens
    compile_prefill = args.compile_prefill
    compile_decode = args.compile_decode
    device_name = args.device_name
    outdir = args.outdir
    check_intermediate_outputs = args.check_intermediate_outputs
    prepare_attention_mask_on_cpu = args.prepare_attention_mask_on_cpu
    disable_cache = args.disable_cache
    num_compiler_threads = args.num_compiler_threads

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()  # Some models do not return the KV cache in training mode.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # For static KV caching
    tokenizer.truncation_side = "left"

    prompt = prepare_prompt(tokenizer, prompt, system_prompt)

    outputs = infer_with_compilation(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        compile_prefill=compile_prefill,
        compile_decode=compile_decode,
        device_name=device_name,
        outdir=outdir,
        check_intermediate_outputs=check_intermediate_outputs,
        prepare_attention_mask_on_cpu=prepare_attention_mask_on_cpu,
        disable_cache=disable_cache,
        num_compiler_threads=num_compiler_threads,
    )
    print(
        "=========== Generated with compilation ==========\n",
        tokenizer.decode(outputs[0]),
    )

    outputs_expected = infer_with_generate(prompt, model, tokenizer, max_new_tokens)
    print(
        "========== Generated with model.generate ==========\n",
        tokenizer.decode(outputs_expected[0]),
    )

    # @todo (hvy): Do not rely on `max_new_tokens` tokens always being generated?
    assert torch.equal(
        outputs[:, -max_new_tokens:], outputs_expected[:, -max_new_tokens:]
    ), "Outputs differed. Check generated outputs above."
    print("Generated outputs matched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default='The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens. With some proper optimization, we can achieve this within a span of "just" 90 days using 16 A100-40G GPUs 🚀🚀. The training has started on 2023-09-01.',  # NOQA
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a friendly chatbot who is an expert on MN-Core.",
    )
    parser.add_argument(
        "--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_compiler_threads", type=int, default=-1)
    parser.add_argument("--compile_prefill", action="store_true")
    parser.add_argument("--compile_decode", action="store_true")
    parser.add_argument("--device_name", type=str, default="mncore2:auto")
    parser.add_argument("--outdir", type=str, default="/tmp/mlsdk_llm_infer")
    parser.add_argument("--check_intermediate_outputs", action="store_true")
    parser.add_argument("--prepare_attention_mask_on_cpu", action="store_true")
    parser.add_argument("--disable_cache", action="store_true")
    args = parser.parse_args()
    main(args)
