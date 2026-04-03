import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

import tomllib
import torch
from mlsdk import (
    CompiledFunction,
    Context,
    get_tensor_name,
    set_tensor_name_in_module,
    storage,
)


def register_model(
    context: Context,
    name: str,
    model: torch.nn.Module,
) -> None:
    if (
        get_tensor_name(next(model.parameters())) is None
    ):  # in case the model obj isn't registered to the context
        set_tensor_name_in_module(model, name)
        for p in model.parameters():
            context.register_param(p)
        for b in model.buffers():
            context.register_buffer(b)


def compile_fn(  # noqa: CFQ002
    context: Context,
    target_fn: Callable[
        [
            dict[str, torch.Tensor],
        ],
        dict[str, torch.Tensor],
    ],  # compiled fn
    model: torch.nn.Module,
    sample_input: dict[str, torch.Tensor],
    outdir: str = "/tmp/example_output",
    model_name: str = "example",
    option_json: Path | None = None,
) -> CompiledFunction:

    if option_json is None:
        option_json = Path("/opt/pfn/pfcomp/codegen/preset_options/O1.json")

    compile_options = {"option_json": str(option_json)}

    compile_args = {
        "function": target_fn,
        "inputs": sample_input,
        "options": compile_options,
    }

    codegen_base_dir = storage.path(outdir)
    compile_args["codegen_dir"] = codegen_base_dir / model_name

    register_model(context, "model", model)

    return context.compile(**compile_args)


# for type hint of the configs from toml
class TomlValue:
    str | int | float | bool | list["TomlValue"] | dict[str, "TomlValue"]


class TomlDict:
    dict[str, TomlValue]


def read_configs_from_toml(
    toml_path: str,
) -> TomlDict:

    configs_dict = None
    with open(toml_path, mode="rb") as f:
        configs_dict = tomllib.load(f)

    return configs_dict


def apply_toml_defaults(
    configs: TomlDict | str | os.PathLike,
    parser: argparse.ArgumentParser,
) -> None:

    if isinstance(configs, dict):
        for k, v in configs.items():
            if isinstance(v, dict):  # in case v is (nested) dict
                apply_toml_defaults(v, parser)
            else:
                parser.add_argument(f"--{k}", default=v, type=type(v))
    elif isinstance(configs, str) or isinstance(configs, os.PathLike):
        configs_dict = read_configs_from_toml(configs)

        apply_toml_defaults(configs_dict, parser)
    else:
        sys.exit("")
