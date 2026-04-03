import argparse
import copy

import torch
from mlsdk import Context, MNDevice, storage
from mlsdk.experimental.llm.kv_cache import kv_cache_to_legacy, kv_cache_to_tensor
from mlsdk.experimental.llm.modeling_staged_llama import split_llama
from transformers import AutoModelForCausalLM
from transformers.models.llama import LlamaConfig


def main(args):  # NOQA
    n_stages = args.n_stages
    bsize = args.bsize
    max_length = args.max_length
    phase = args.phase
    model_name = args.model_name
    device_name = args.device_name
    outdir = args.outdir

    if model_name.endswith(".json"):
        config = LlamaConfig.from_json_file(model_name)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    config = model.config
    n_layers = config.num_hidden_layers

    if phase == "prefill":
        inputs = {
            "input_ids": torch.randint(0, config.vocab_size, (bsize, max_length)),
            "attention_mask": torch.ones(bsize, 1, max_length, max_length),
            "position_ids": torch.ones(bsize, max_length).cumsum(1),
        }
    else:
        inputs = {
            "input_ids": torch.randint(0, config.vocab_size, (bsize, 1)),
            "attention_mask": torch.ones(bsize, 1, 1, max_length),
            "position_ids": torch.full((bsize, 1), fill_value=10),
            "past_key_values": torch.rand(
                n_layers,
                2,
                bsize,
                config.num_key_value_heads,
                max_length - 1,  # Appended with one token during decoding.
                config.hidden_size // model.config.num_attention_heads,
            ),
        }

    expected_inputs = copy.deepcopy(inputs)
    expected_inputs["use_cache"] = True
    expected_outputs = model(**expected_inputs)

    models = split_llama(model, n_stages)
    for model in models:
        model.eval()

    n_decoder_stages = n_stages - 2
    n_layers_per_stage = (n_layers + n_decoder_stages - 1) // n_decoder_stages
    h = None

    collected_past_key_values = []
    for stage, model in enumerate(models):
        print("Compiling stage", stage)
        device = MNDevice(device_name)
        context = Context(device)
        Context.switch_context(context)
        context.registry.register("model", model)

        if stage == 0:

            def forward(inputs, model=model):
                return model(**inputs)

            inputs_for_stage = {
                "input_ids": inputs["input_ids"],
                "position_ids": inputs["position_ids"],
            }
        elif stage == len(models) - 1:

            def forward(inputs, model=model):
                return model(**inputs)

            inputs_for_stage = {
                "hidden_states": h["hidden_states_out"],
            }
        else:

            def forward(inputs, model=model):
                if "past_key_values" in inputs:
                    inputs["past_key_values"] = kv_cache_to_legacy(
                        inputs["past_key_values"]
                    )
                else:
                    inputs["past_key_values"] = None
                outputs = model(**inputs)
                outputs["past_key_values_out"] = kv_cache_to_tensor(
                    outputs["past_key_values_out"]
                )[:, :, :, :, 1:, :]
                return outputs

            inputs_for_stage = {
                "hidden_states": h["hidden_states_out"],
                "position_embeddings_cos": h["position_embeddings_cos_out"],
                "position_embeddings_sin": h["position_embeddings_sin_out"],
                "attention_mask": inputs["attention_mask"],
                "position_ids": inputs["position_ids"],
            }
            if "past_key_values" in inputs:
                assert phase == "decode"
                inputs_for_stage["past_key_values"] = inputs["past_key_values"][
                    n_layers_per_stage * (stage - 1) : n_layers_per_stage * stage
                ]

        compiled_forward = context.compile(
            forward,
            inputs_for_stage,
            storage.path(outdir + f"/{phase}" + f"/stage_{stage}"),
        )
        h = compiled_forward(inputs_for_stage)
        for k in h.keys():
            h[k] = h[k].cpu().detach()

        context.invalidate()

        if "past_key_values_out" in h:
            assert 1 <= stage <= (n_stages - 2)
            collected_past_key_values.append(h["past_key_values_out"])

    outputs = h
    outputs["past_key_values"] = torch.cat(collected_past_key_values)

    actual = outputs["logits"]
    expected = expected_outputs["logits"]
    diff = actual - expected
    print(f"logits {diff.abs().max()=}")
    for i in range(n_layers):
        for kv in range(2):
            actual = outputs["past_key_values"][i][kv]
            expected = expected_outputs["past_key_values"][i][kv][:, :, 1:, :]
            diff = actual - expected
            print(f"kv {i=} {kv=} {diff.abs().max()=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument("--n_stages", type=int, default=4)
    parser.add_argument("--bsize", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device_name", type=str, default="mncore2:auto")
    parser.add_argument("--outdir", type=str, default="/tmp/mlsdk_llm_pp_infer")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["prefill", "decode"],
        default="prefill",
    )
    args = parser.parse_args()
    main(args)
