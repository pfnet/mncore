import argparse
import os
from pathlib import Path
from typing import Any, Optional, Union

import timm
import torch
from mlsdk import (
    Context,
    MNCoreSGD,
    MNDevice,
    set_buffer_name_in_optimizer,
    set_tensor_name_in_module,
    storage,
)
from PIL import Image

SAMPLE_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "./datasets/mncore2_chip.png"
)


def escape_path(path: str) -> str:
    escaped = ""
    for c in path:
        if c.isalnum() or c in "_-":
            escaped += c
        else:
            escaped += "_"
    return escaped


def create_model_with_cache(
    model_name: str, model_cache_dir: Optional[str] = None, **kwargs: Any
) -> Any:
    if not model_cache_dir:
        return timm.create_model(model_name, **kwargs)
    else:
        timm_version = "timm_version" + timm.__version__
        torch_version = "torch_version" + torch.__version__
        cache_dir = os.path.join(
            model_cache_dir,
            escape_path(f"{torch_version}_{timm_version}_{model_name}"),
        )
        # Load the model always from the cache to return the same model object always.
        # This should also create the cache if it does not exist.
        return timm.create_model(model_name, **kwargs, cache_dir=cache_dir)


def imagenet_classes() -> list[str]:
    script_dir = os.path.dirname(__file__)
    imagenet_classes_path = os.path.join(script_dir, "imagenet_classes.txt")
    with open(imagenet_classes_path) as f:
        return [line.strip() for line in f]


def run_inference(
    args: argparse.Namespace,
) -> None:
    img = Image.open(SAMPLE_IMAGE_PATH)
    model = create_model_with_cache(
        args.model_name,
        pretrained=True,
        model_cache_dir=args.model_cache_dir,
    )
    model = model.eval()

    def infer(input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            x = input["images"]
            return {"out": model(x)}

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    images = transforms(img).unsqueeze(0).expand(args.batch_size, -1, -1, -1)
    sample = {"images": images}

    device = MNDevice(args.device)
    context = Context(device)
    Context.switch_context(context)
    context.registry.register("model", model)

    compile_options: dict[str, str] = {}
    if args.option_json is not None:
        compile_options = {"option_json": str(args.option_json)}

    compiled_infer = context.compile(
        infer,
        sample,
        storage.path(args.outdir) / "infer",
        options=compile_options,
    )

    if args.action == "compile":
        context.synchronize()
        return

    result_as_proxy = compiled_infer(sample)

    if args.action == "run":
        context.synchronize()
        return

    result_on_torch = infer(sample)

    # Tensors obtained via ".cpu()" from TensorProxy exist on GPU in CUDA environments,
    # so they need to be moved to CPU before the comparison.
    result = result_as_proxy["out"].cpu()
    if result.is_cuda:
        result = result.cpu()

    context.synchronize()
    torch.allclose(result, result_on_torch["out"], atol=1e-5)

    if "in1k" in args.model_name:
        classes = imagenet_classes()
        mncore_top5_classes = torch.topk(result[0], 5).indices.cpu()
        print("MNCore2 top-5 classes:")
        for i in mncore_top5_classes:
            print(f"- {classes[i]} ({i.item()})")
        torch_top5_classes = torch.topk(result_on_torch["out"][0], 5).indices
        print("Torch top-5 classes:")
        for i in torch_top5_classes:
            print(f"- {classes[i]} ({i.item()})")


# return mncore.runtime_core._context._function.CompiledFunction
# but this is not directly exposed in the public API, so we use Any here.
def compile_train_step_with_torch_onnx(
    model: Any,
    sample: dict[str, Any],
    context: Context,
    outdir: str,
    option_json: str | None = None,
) -> Any:
    model = model.train()
    context.registry.register("model0", model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    context.registry.register("optimizer0", optimizer)
    loss_fn = torch.nn.CrossEntropyLoss()

    def f(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"loss": loss_fn(model(inputs["images"]), inputs["labels"])}

    compile_options: dict[str, Union[str, bool]] = {"backprop": True}
    if option_json is not None:
        compile_options["option_json"] = str(option_json)

    compiled_train_step = context.compile(
        f,
        sample,
        storage.path(outdir) / "train_step_torch_onnx",
        optimizers=[optimizer],
        options=compile_options,
    )

    def wrapped(inputs: dict[str, Any]) -> Any:
        inputs["optimizer0@0@mncore_learning_rate"] = torch.tensor(0.1)
        inputs["optimizer0@0@mncore_global_step"] = torch.tensor(wrapped.global_step)  # type: ignore
        inputs["mncore_grad_scale_factor"] = torch.tensor(1)
        wrapped.global_step += 1  # type: ignore
        return compiled_train_step(inputs)

    wrapped.global_step = 0  # type: ignore

    return wrapped


# return mncore.runtime_core._context._function.CompiledFunction
# but this is not directly exposed in the public API, so we use Any here.
def compile_train_step_with_fx2onnx(
    model: Any,
    sample: dict[str, Any],
    context: Context,
    outdir: str,
    option_json: str | None = None,
) -> Any:
    model = model.train()
    set_tensor_name_in_module(model, "model0")
    for p in model.parameters():
        context.register_param(p)
    optimizer = MNCoreSGD(model.parameters(), 0.1, 0.9, 0.0)
    set_buffer_name_in_optimizer(optimizer, "optimizer0")
    context.register_optimizer_buffers(optimizer)
    loss_fn = torch.nn.CrossEntropyLoss()

    def train_step(input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = input["images"]
        t = input["labels"]
        optimizer.zero_grad()
        y = model(x)
        loss = loss_fn(y, t)
        loss.backward()
        optimizer.step()
        return {"loss": loss}

    compile_options: dict[str, Union[str, bool]] = {}
    if option_json is not None:
        compile_options["option_json"] = str(option_json)

    return context.compile(
        train_step,
        sample,
        storage.path(outdir) / "train_step_fx2onnx",
        options=compile_options,
        export_kwargs={"use_fx2onnx": True},
    )


def run_training(
    args: argparse.Namespace,
) -> None:
    device = MNDevice(args.device)
    context = Context(device)
    Context.switch_context(context)

    img = Image.open(SAMPLE_IMAGE_PATH)
    model = create_model_with_cache(
        args.model_name,
        pretrained=True,
        num_classes=1000,
        model_cache_dir=args.model_cache_dir,
    )
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    images = transforms(img).unsqueeze(0).expand(args.batch_size, -1, -1, -1)
    labels = torch.randint(0, 1000, (args.batch_size,))
    sample = {"images": images, "labels": labels}

    # TODO (akirakawata): Should we make this argument?
    use_fx2onnx = not bool(
        int(os.environ.get("MNCORE_USE_LEGACY_ONNX_EXPORTER", False))
    )
    if use_fx2onnx:
        # NOTE (puchupala): fx2onnx training needs the optimizer in the
        # exported graph and lr, step, and grad scale factor in the inputs,
        # so it follows a separate code path.
        compiled_train_step = compile_train_step_with_fx2onnx(
            model,
            sample,
            context,
            args.outdir,
            option_json=args.option_json,
        )
    else:
        compiled_train_step = compile_train_step_with_torch_onnx(
            model,
            sample,
            context,
            args.outdir,
            option_json=args.option_json,
        )

    if args.action == "compile":
        context.synchronize()
        return

    first_loss = compiled_train_step(sample)["loss"].cpu()

    if args.action == "run":
        context.synchronize()
        return

    for _ in range(args.num_iters - 2):
        compiled_train_step(sample)
    last_loss = compiled_train_step(sample)["loss"].cpu()
    context.synchronize()

    assert last_loss < first_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="/tmp/mlsdk_timm")
    parser.add_argument("--option_json", type=Path, default=None)
    parser.add_argument("--mode", type=str, default="infer", choices=["infer", "train"])
    parser.add_argument(
        "--device",
        type=str,
        default="mncore2:auto",
        choices=["mncore2:auto", "pfvm:cpu", "pfvm:cuda"],
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Directory to cache the model weights. "
        "If not set, weights are always downloaded from the hub. default: None",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="validate",
        choices=["compile", "run", "validate"],
        help="Whether to only compile, run without validation, "
        "or run with validation (default: validate)",
    )

    train_group = parser.add_argument_group(
        "Training options", "Options for training mode (ignored in inference mode)"
    )
    train_group.add_argument(
        "--num_iters",
        type=int,
        default=12,
        help="Number of training iterations to run (default: 12)",
    )

    args = parser.parse_args()

    # Simple args validation
    assert args.batch_size > 0, "Batch size must be positive"
    assert (
        args.num_iters >= 2
    ), "Number of iterations must be at least 2 to observe loss decrease"
    if args.option_json is not None:
        assert (
            args.option_json.is_file()
        ), f"Option JSON file not found: {args.option_json}"

    if args.mode == "train":
        run_training(args)
    elif args.mode == "infer":
        run_inference(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
