import argparse
import datetime
import gc
import math
import os
import random
import sys
import time
from collections.abc import Callable, Iterable
from typing import Any, Self

import numpy as np
import tomllib
import torch
from mlsdk import (
    CacheOptions,
    CompiledFunction,
    Context,
    MNCoreAdam,
    MNCoreAdamW,
    MNCoreOptimizer,
    MNCoreSGD,
    get_tensor_name,
    set_buffer_name_in_optimizer,
    set_tensor_name,
    set_tensor_name_in_module,
    storage,
)

PRESET_OPTIONS_DIR = os.environ.get(
    "PRESET_OPTIONS_DIR", "/opt/pfn/pfcomp/codegen/preset_options"
)


def decide_outdir(
    device: str,
    example_name: str = "example",
    basedir: str | os.PathLike = ".",
) -> str:
    outdir = f"{basedir}/{example_name}_"
    if "mncore2" in device:
        outdir += "mncore_fx2onnx_"
    elif device == "pfvm:cpu":
        outdir += "mncore_cpu_fx2onnx_"
    elif device == "pfvm:cuda":
        outdir += "mncore_gpu_fx2onnx_"
    elif device == "cuda":
        outdir += "gpu_torch_"
    else:
        outdir += "cpu_torch_"

    # suffix is the "_out_{datetime}"
    outdir += f"_out_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}"

    # Create output dirs in case of torch_cpu and torch_gpu to save the outputs
    if outdir in ["cpu_torch_out", "gpu_torch_out"]:
        os.makedirs(outdir, exist_ok=True)

    return outdir


def decide_model_name(
    device: str,
    learning_rate: float,
    momentum: float,
    epoch: int | None = None,
    basename: str = "",
) -> str:
    model_name = basename
    if "mncore2" in device:
        model_name += "_mncore2_"
    elif device == "pfvm:cuda":
        model_name += "_pfvm_cuda_"
    elif device == "pfvm:cpu":
        model_name += "_pfvm_cpu_"
    elif device == "cuda":
        model_name += "_cuda_torch_"
    else:
        model_name += "_cpu_torch_"

    if "mncore2" in device or device in ["pfvm:cuda", "pfvm:cpu"]:
        model_name += "fx2onnx_"

    model_name += "lr" + str(learning_rate) + "_M" + str(momentum)

    if epoch is not None:
        model_name += "_e" + str(epoch)

    return model_name


def create_optimizer(  # noqa: CFQ002
    name: str,
    params: Iterable[
        torch.nn.Parameter
    ],  # pass the iterator which is returned by model.parameters()
    use_mncore: bool,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    dampening: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-08,
) -> torch.optim.Optimizer | MNCoreOptimizer:
    if use_mncore and ("mncore_" not in name):
        name = "mncore_" + name
    optimizer = None
    if name == "adam":
        optimizer = torch.optim.Adam(
            params=params,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params=params,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
    elif name == "sgd":
        optimizer = torch.optim.SGD(
            params=params,
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
        )
    elif name == "mncore_adam":
        # The "try-except" statements below are for backward compatibility
        try:
            optimizer = MNCoreAdam(
                params,
                learning_rate=learning_rate,
                alpha=beta1,
                beta=beta2,
                norm_coefficient=0.0,
                epsilon=eps,
                chainer_use_torch=False,
            )
        except (NameError, TypeError):
            optimizer = MNCoreAdam(
                params,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
                chainer_use_torch=True,
            )
        except Exception as e:
            raise e
    elif name == "mncore_adamw":
        try:
            optimizer = MNCoreAdamW(
                params,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay,
            )
        except (NameError, TypeError):
            optimizer = MNCoreAdamW(
                params,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
            )
        except Exception as e:
            raise e
    elif name == "mncore_sgd":
        optimizer = MNCoreSGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer name: {name}")

    return optimizer


def set_deterministic_mode(seed: int) -> None:
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set cudnn.benchmark mode and specify the use of deterministic algorithms
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def output_result_times(  # noqa: CFQ002
    train_times: list[float] | None = None,
    eval_times: list[float] | None = None,
    train_iter: int | None = None,
    eval_iter: int | None = None,
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    optimizer_name: str | None = None,
    backend_name: str = "cpu",
    sample_name: str = "sample",
    optimize_option: str = "debug",
    time_scale: str = "s",
) -> None:
    scaling_factor = 1.0
    if time_scale == "ms":
        scaling_factor = 1000.0
    elif time_scale == "us":
        scaling_factor = 1000000.0
    elif time_scale == "ns":
        scaling_factor = 1000000000.0

    onnx_exporter = "None"
    optimize_flag = "None"
    if backend_name not in ["cpu", "cuda"]:
        onnx_exporter = "fx2onnx"

        optimize_flag = optimize_option

    print("\n########## Performances ##########")
    print("\n---------- configs ----------")
    print(f"sample name: {sample_name}")
    print(f"backend device: {backend_name}")
    print(f"onnx exporter: {onnx_exporter}")
    print(f"optimizer: {optimizer_name}")
    print(f"optimize_option: {optimize_flag}")
    print("------------------------------")

    if train_times is not None and len(train_times) != 0:
        print("\n---------- performance of training part ----------")
        total_time = math.fsum(train_times) * scaling_factor
        average_time = total_time / len(train_times)
        iter_per_sec = train_iter * len(train_times) / total_time
        print(f"train epochs: {len(train_times)}")
        print(f"batch size: {train_batch_size}")
        print(f"iterations per epoch: {train_iter}")
        print(f"total time [{time_scale}]: {total_time}")
        print(f"average per epoch [{time_scale}]: {average_time}")
        print(f"averaged [iter/{time_scale}]: {iter_per_sec}")
        print("--------------------------------------------------")

    if eval_times is not None and len(eval_times) != 0:
        print("\n---------- performance of evaluation/inference part ----------")
        total_time = math.fsum(eval_times) * scaling_factor
        average_time = total_time / len(eval_times)
        iter_per_sec = eval_iter * len(eval_times) / total_time
        print(f"eval epochs: {len(eval_times)}")
        print(f"batch size: {eval_batch_size}")
        print(f"iterations per epoch: {eval_iter}")
        print(f"total time [{time_scale}]: {total_time}")
        print(f"average per epoch [{time_scale}]: {average_time}")
        print(f"averaged [iter/{time_scale}]: {iter_per_sec}")
        print("--------------------------------------------------------------")
    print("\n##################################")


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
    optimizer_configs: (
        list[tuple[MNCoreOptimizer | torch.optim.Optimizer, float]] | None
    ) = None,  # list[] is for multiple optimizers
    optimize_option: str = "debug",
    preset_options_dir: str = PRESET_OPTIONS_DIR,
    enable_cache: bool = False,
    envs: dict[str, str] | None = None,
    **kwargs: Any,  # used in `compile_args` in Context.compile()
) -> CompiledFunction:

    compile_options = {
        "option_json": os.path.join(preset_options_dir, optimize_option + ".json"),
        "envs": envs if envs is not None else {},
    }

    compile_args = {
        "function": target_fn,
        "inputs": sample_input,
        "options": compile_options,
        "codegen_dir": storage.path(outdir) / model_name,
    }

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
        if (
            optimizer_configs is None
        ):  # in case that optimizer.step() will be done at the host
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
                        p.grad = torch.nn.Parameter(torch.zeros_like(p))
                        set_tensor_name(p.grad, f"{name}@{n}".replace(".", "_"))
                        context.register_param(p.grad)
        else:
            for idx, optimizer_config in enumerate(optimizer_configs):
                optimizer_name = "optimizer" + str(idx)
                set_buffer_name_in_optimizer(optimizer_config[0], optimizer_name)
                context.register_optimizer_buffers(optimizer_config[0])

    compile_args.update(kwargs)

    return context.compile(**compile_args)


class Timer:
    def __init__(self) -> None:
        self.time = None

    def __enter__(self) -> Self:
        gc.disable()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        end_time = time.perf_counter()
        gc.enable()
        self.time = end_time - self.start_time
        return None


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
    elif isinstance(v, str) | isinstance(v, bool):
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
    elif isinstance(configs, str) or isinstance(configs, os.PathLike):
        configs_dict = read_configs_from_toml(configs)

        apply_toml_defaults(configs_dict, parser)
    else:
        sys.exit("")
