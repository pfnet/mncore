import argparse
import os
from pathlib import Path
from typing import Any, Optional, Union

import mncore  # noqa: F401
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
    model_name: str,
    batch_size: int,
    outdir: str,
    option_json_path: Optional[Path],
    device_str: str,
    model_cache_dir: Optional[str],
) -> None:
    img = Image.open(SAMPLE_IMAGE_PATH)
    model = create_model_with_cache(
        model_name,
        pretrained=True,
        model_cache_dir=model_cache_dir,
    )
    model = model.eval()

    def infer(input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            x = input["images"]
            return {"out": model(x)}

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    images = transforms(img).unsqueeze(0).expand(batch_size, -1, -1, -1)
    sample = {"images": images}

    device = MNDevice(device_str)
    context = Context(device)
    Context.switch_context(context)
    context.registry.register("model", model)

    compile_options: dict[str, str] = {}
    if option_json_path is not None:
        compile_options = {"option_json": str(option_json_path)}

    compiled_infer = context.compile(
        infer,
        sample,
        storage.path(outdir) / "infer",
        options=compile_options,
    )
    result_as_proxy = compiled_infer(sample)
    result_on_torch = infer(sample)

    # Tensors obtained via ".cpu()" from TensorProxy exist on GPU in CUDA environments,
    # so they need to be moved to CPU before the comparison.
    result = result_as_proxy["out"].cpu()
    if result.is_cuda:
        result = result.cpu()

    torch.allclose(result, result_on_torch["out"], atol=1e-5)

    if "in1k" in model_name:
        classes = imagenet_classes()
        mncore_top5_classes = torch.topk(result[0], 5).indices.cpu()
        print("MNCore2 top-5 classes:")
        for i in mncore_top5_classes:
            print(f"- {classes[i]} ({i.item()})")
        torch_top5_classes = torch.topk(result_on_torch["out"][0], 5).indices
        print("Torch top-5 classes:")
        for i in torch_top5_classes:
            print(f"- {classes[i]} ({i.item()})")


def run_training_torch_onnx(
    model_name: str,
    batch_size: int,
    outdir: str,
    option_json_path: Optional[Path],
    device: str,
    model_cache_dir: Optional[str],
) -> None:
    device = MNDevice(device)
    context = Context(device)
    Context.switch_context(context)

    img = Image.open(SAMPLE_IMAGE_PATH)

    model = create_model_with_cache(
        model_name,
        pretrained=True,
        num_classes=1000,
        model_cache_dir=model_cache_dir,
    )
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    images = transforms(img).unsqueeze(0).expand(batch_size, -1, -1, -1)
    labels = torch.randint(0, 1000, (batch_size,))
    sample = {"images": images, "labels": labels}

    model = model.train()
    context.registry.register("model", model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    context.registry.register("optimizer", optimizer)
    loss_fn = torch.nn.CrossEntropyLoss()

    def f(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"loss": loss_fn(model(inputs["images"]), inputs["labels"])}

    compile_options: dict[str, Union[str, bool]] = {}
    if option_json_path is not None:
        compile_options = {"option_json": str(option_json_path)}
    compile_options["backprop"] = True
    compiled_f = context.compile(
        f,
        sample,
        storage.path(outdir) / "train_step_torch_onnx",
        optimizers=[optimizer],
        options=compile_options,
    )

    first_loss = compiled_f(sample)["loss"].cpu()
    for _ in range(10):
        compiled_f(sample)
    context.synchronize()
    last_loss = compiled_f(sample)["loss"].cpu()

    assert last_loss < first_loss


def run_training_fx2onnx(
    model_name: str,
    batch_size: int,
    outdir: str,
    option_json_path: Optional[Path],
    device_str: str,
    model_cache_dir: Optional[str],
) -> None:
    device = MNDevice(device_str)
    context = Context(device)
    Context.switch_context(context)

    img = Image.open(SAMPLE_IMAGE_PATH)

    model = create_model_with_cache(
        model_name,
        pretrained=True,
        num_classes=1000,
        model_cache_dir=model_cache_dir,
    )
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

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    images = transforms(img).unsqueeze(0).expand(batch_size, -1, -1, -1)
    labels = torch.randint(0, 1000, (batch_size,))
    sample = {"images": images, "labels": labels}

    compile_options: dict[str, str] = {}
    if option_json_path is not None:
        compile_options = {"option_json": str(option_json_path)}

    compiled_train_step = context.compile(
        train_step,
        sample,
        storage.path(outdir) / "train_step_fx2onnx",
        options=compile_options,
        export_kwargs={"use_fx2onnx": True},
    )

    first_loss = compiled_train_step(sample)["loss"].cpu()
    for _ in range(10):
        compiled_train_step(sample)
    context.synchronize()
    last_loss = compiled_train_step(sample)["loss"].cpu()

    assert last_loss < first_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, required=True)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--outdir", type=str, default="/tmp/mlsdk_timm")
    parser.add_argument("--option_json", type=Path, default=None)
    parser.add_argument("--is_training", action="store_true")
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
    args = parser.parse_args()

    outdir = args.outdir
    if outdir is None:
        outdir = f"/tmp/MLSDK_codegen_dir_{args.model_name}"
        if args.is_training:
            outdir += "_training"
        else:
            outdir += "_inference"

    # TODO (akirakawata): Should we make this argument?
    use_fx2onnx = not bool(
        int(os.environ.get("MNCORE_USE_LEGACY_ONNX_EXPORTER", False))
    )
    if args.is_training:
        if use_fx2onnx:
            run_training_fx2onnx(
                args.model_name,
                args.batch_size,
                outdir,
                args.option_json,
                args.device,
                args.model_cache_dir,
            )
        else:
            run_training_torch_onnx(
                args.model_name,
                args.batch_size,
                args.outdir,
                args.option_json,
                args.device,
                args.model_cache_dir,
            )
    else:
        run_inference(
            args.model_name,
            args.batch_size,
            args.outdir,
            args.option_json,
            args.device,
            args.model_cache_dir,
        )
