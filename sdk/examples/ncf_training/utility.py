import argparse
import os
import pathlib
import random
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import tomllib
import torch
from fx2onnx import set_tensor_name
from mlsdk import (
    CacheOptions,
    CompiledFunction,
    Context,
    MNCoreOptimizer,
    get_tensor_name,
    set_buffer_name_in_optimizer,
    set_tensor_name_in_module,
    storage,
)


def set_deterministic_mode(seed: int) -> None:
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set cudnn.benchmark mode and specify the use of deterministic algorithms
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


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
    model: torch.nn.Module | dict[str, torch.nn.Module],
    sample_input: dict[str, torch.Tensor],
    outdir: str = "/tmp/example_output",
    model_name: str = "example",
    is_train: bool = True,
    optimizers: (
        list[MNCoreOptimizer] | None
    ) = None,  # list[] is for multiple optimizers
    option_json: str = "/opt/pfn/pfcomp/codegen/preset_options/O1.json",
    preset_options_dir: str | None = None,
    enable_cache: bool = False,
    **kwargs: Any,  # used in `compile_args` in Context.compile()
) -> CompiledFunction:

    if preset_options_dir is None:
        preset_options_dir = pathlib.Path.cwd().parent.parent.parent / "preset_options"

    compile_options = {"option_json": option_json}

    compile_args = {
        "function": target_fn,
        "inputs": sample_input,
        "options": compile_options,
    }

    codegen_base_dir = storage.path(outdir)
    compile_args["codegen_dir"] = codegen_base_dir / model_name

    if enable_cache:
        compile_args["cache_options"] = CacheOptions(
            f"{outdir}/{model_name}/cache",
            enable_app_cache=True,
            enable_onnx_cache=True,
            enable_codegen_cache=True,
            enable_gpfn2obj_cache=True,
        )

    if isinstance(model, torch.nn.Module):
        register_model(context, model_name, model)
    else:  # if isinstance(models, dict[str, torch.nn.Module]):
        for name, actual_model in model.items():
            register_model(context, name, actual_model)

    if is_train:
        if optimizers is None:  # in case that optimizer.step() will be done at the host
            if isinstance(model, torch.nn.Module):
                for n, p in model.named_parameters():
                    p.grad = torch.nn.Parameter(
                        torch.zeros_like(p), requires_grad=p.requires_grad
                    )
                    set_tensor_name(p.grad, f"{model_name}@{n}@grad".replace(".", "_"))
                    context.register_param(p.grad)
            else:
                for name, actual_model in model.items():
                    for n, p in actual_model.named_parameters():
                        p.grad = torch.nn.Parameter(
                            torch.zeros_like(p), requires_grad=p.requires_grad
                        )
                        set_tensor_name(p.grad, f"{name}@{n}".replace(".", "_"))
                        context.register_param(p.grad)
        else:
            for idx, optimizer in enumerate(optimizers):
                optimizer_name = "optimizer" + str(idx)
                set_buffer_name_in_optimizer(optimizer, optimizer_name)
                context.register_optimizer_buffers(optimizer)

    compile_args.update(kwargs)

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


def str2bool(v: bool | str) -> bool:
    if v.lower() in ("yes", "true", "on", "enable", "y", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "off", "disable", "n", "f", "0"):
        return False
    elif isinstance(v, str | bool):
        return v
    else:
        raise argparse.ArgumentTypeError("Str or boolean value expected")


def apply_toml_defaults(
    configs: TomlDict | str | os.PathLike,
    parser: argparse.ArgumentParser,
) -> None:

    if isinstance(configs, dict):
        for k, v in configs.items():
            if isinstance(v, dict):  # in case v is (nested) dict
                apply_toml_defaults(v, parser)
            else:
                # just checking whether v is list is enough for array args
                # because array in toml is converted to the list by tomllib
                args_type = None
                if isinstance(v, list):
                    args_type = type(v[0])
                elif isinstance(v, bool):
                    args_type = str2bool
                else:
                    args_type = type(v)
                parser.add_argument(
                    f"--{k}",
                    default=v,
                    type=args_type,
                    nargs="*" if isinstance(v, list) else "?",
                )
    elif isinstance(configs, str | os.PathLike):
        configs_dict = read_configs_from_toml(configs)

        apply_toml_defaults(configs_dict, parser)
    else:
        sys.exit("")
