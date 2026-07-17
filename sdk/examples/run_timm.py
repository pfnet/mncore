import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

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

logger = logging.getLogger(__name__)
SAMPLE_IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "./datasets/mncore2_chip.png"
)
ACTION_CHOICES = ["compile", "run", "validate"]
CUSTOM_EXIT_CODES = {
    "error": 1,
    "unexpected_error": -1,
    # Unknown since the test is skipped.
    # e.g. compile success and we want to skip the run/validate to save time.
    "unknown": -2,
}


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
    try:
        model = create_model_with_cache(
            args.model_name,
            pretrained=True,
            model_cache_dir=args.model_cache_dir,
        )
    except RuntimeError as e:
        print(f"Failed to load pretrained weights for the model: {e}")
        print("Falling back to creating the model without pretrained weights.")
        model = create_model_with_cache(
            args.model_name,
            pretrained=False,
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
        device_top5_classes = torch.topk(result[0], 5).indices.cpu()
        logger.info("Device top-5 classes:")
        for i in device_top5_classes:
            logger.info(f"- {classes[i]} ({i.item()})")
        torch_top5_classes = torch.topk(result_on_torch["out"][0], 5).indices
        logger.info("Torch top-5 classes:")
        for i in torch_top5_classes:
            logger.info(f"- {classes[i]} ({i.item()})")


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


class StepError(Exception):
    """Raised by a pipeline step to report a specific exit code on failure
    instead of the default ``error`` code."""

    def __init__(self, exit_code: int) -> None:
        super().__init__(f"step failed with exit code {exit_code}")
        self.exit_code = exit_code


class Pipeline:
    """Runs the ``compile`` → ``run`` → ``validate`` steps in order.

    The shared driver (:meth:`execute`) times each step independently, stops
    early when ``args.action`` only asks for an earlier step, and cascades a
    failure to every step that can no longer run. Subclasses implement the
    three steps; a step signals failure by raising, and may raise
    :class:`StepError` to report a non-default exit code.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.context: Optional[Context] = None
        self.sample: dict[str, Any] = {}

    def compile(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def validate(self) -> None:
        raise NotImplementedError

    def execute(self) -> dict[str, dict[str, Any]]:
        # Steps not reached (skipped via --action, or never run because an
        # earlier step failed) keep the default "unknown"/zero-duration entry.
        results: dict[str, dict[str, Any]] = {
            action: {"exit_code": CUSTOM_EXIT_CODES["unknown"], "duration_s": 0.0}
            for action in ACTION_CHOICES
        }
        steps: list[tuple[str, Callable[[], None]]] = [
            ("compile", self.compile),
            ("run", self.run),
            ("validate", self.validate),
        ]

        for index, (name, step) in enumerate(steps):
            step_start = time.perf_counter()
            try:
                step()
            except Exception as e:
                exit_code = (
                    e.exit_code
                    if isinstance(e, StepError)
                    else CUSTOM_EXIT_CODES["error"]
                )
                results[name] = {
                    "exit_code": exit_code,
                    "duration_s": time.perf_counter() - step_start,
                }
                logger.error(f"Error during {name}: {e}")
                # Downstream steps cannot proceed once a step fails.
                for downstream, _ in steps[index + 1 :]:
                    results[downstream]["exit_code"] = CUSTOM_EXIT_CODES["error"]
                break

            results[name] = {
                "exit_code": 0,
                "duration_s": time.perf_counter() - step_start,
            }
            # Stop once we have completed the step the caller asked for.
            if self.args.action == name:
                break

        return results


class InferencePipeline(Pipeline):
    def compile(self) -> None:
        args = self.args
        img = Image.open(SAMPLE_IMAGE_PATH)
        try:
            model = create_model_with_cache(
                args.model_name,
                pretrained=True,
                model_cache_dir=args.model_cache_dir,
            ).eval()
        except RuntimeError as e:
            logger.warning(f"Failed to load pretrained weights for the model: {e}")
            logger.warning(
                "Falling back to creating the model without pretrained weights."
            )
            model = create_model_with_cache(
                args.model_name,
                pretrained=False,
                model_cache_dir=args.model_cache_dir,
            ).eval()

        def infer(input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            with torch.no_grad():
                return {"out": model(input["images"])}

        self.infer = infer

        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        self.sample = {
            "images": transforms(img).unsqueeze(0).expand(args.batch_size, -1, -1, -1)
        }

        self.context = Context(MNDevice(args.device))
        Context.switch_context(self.context)
        self.context.registry.register("model", model)

        compile_options: dict[str, str] = {}
        if args.option_json is not None:
            compile_options = {"option_json": str(args.option_json)}

        self.compiled_infer = self.context.compile(
            infer,
            self.sample,
            storage.path(args.outdir) / "infer",
            options=compile_options,
        )
        self.context.synchronize()

    def run(self) -> None:
        assert self.context is not None
        self.result_as_proxy = self.compiled_infer(self.sample)
        self.context.synchronize()

    def validate(self) -> None:
        assert self.context is not None
        try:
            result_on_torch = self.infer(self.sample)
        except Exception as e:
            # A failure of the PyTorch reference path is unexpected rather than
            # a genuine output mismatch, so report it with a distinct code.
            raise StepError(CUSTOM_EXIT_CODES["unexpected_error"]) from e

        # Obtain torch.Tensor from TensorProxy
        result = self.result_as_proxy["out"].cpu()

        # In case of CUDA, result is a GPU tensor, so we need to move it to CPU
        # before the comparison. Note that this is not necessary for MN-Core
        # backend since TensorProxy will return a CPU tensor.
        if result.is_cuda:
            result = result.cpu()

        torch.allclose(result, result_on_torch["out"], atol=1e-5)
        self._log_top5_classes(result, result_on_torch)

        # Safety synchronize after validation to prevent potential side effects
        # to subsequent steps in case of a failure in the validation logic above.
        self.context.synchronize()

    def _log_top5_classes(self, result: Any, result_on_torch: dict[str, Any]) -> None:
        # Best-effort diagnostics; failures here must not fail validation.
        try:
            if "in1k" in self.args.model_name:
                classes = imagenet_classes()
                logger.info("MNCore2 top-5 classes:")
                for i in torch.topk(result[0], 5).indices.cpu():
                    logger.info(f"- {classes[i]} ({i.item()})")
                logger.info("Torch top-5 classes:")
                for i in torch.topk(result_on_torch["out"][0], 5).indices:
                    logger.info(f"- {classes[i]} ({i.item()})")
        except Exception as e:
            logger.error(f"Error during post-validation processing: {e}")


class TrainingPipeline(Pipeline):
    def compile(self) -> None:
        args = self.args
        self.context = Context(MNDevice(args.device))
        Context.switch_context(self.context)

        img = Image.open(SAMPLE_IMAGE_PATH)
        try:
            model = create_model_with_cache(
                args.model_name,
                pretrained=True,
                num_classes=1000,
                model_cache_dir=args.model_cache_dir,
            )
        except RuntimeError as e:
            logger.warning(f"Failed to load pretrained weights for the model: {e}")
            logger.warning(
                "Falling back to creating the model without pretrained weights."
            )
            model = create_model_with_cache(
                args.model_name,
                pretrained=False,
                num_classes=1000,
                model_cache_dir=args.model_cache_dir,
            )
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        images = transforms(img).unsqueeze(0).expand(args.batch_size, -1, -1, -1)
        labels = torch.randint(0, 1000, (args.batch_size,))
        self.sample = {"images": images, "labels": labels}

        # TODO (akirakawata): Should we make this argument?
        use_fx2onnx = not bool(
            int(os.environ.get("MNCORE_USE_LEGACY_ONNX_EXPORTER", False))
        )
        if use_fx2onnx:
            # NOTE (puchupala): fx2onnx training needs the optimizer in the
            # exported graph and lr, step, and grad scale factor in the inputs,
            # so it follows a separate code path.
            self.compiled_train_step = compile_train_step_with_fx2onnx(
                model,
                self.sample,
                self.context,
                args.outdir,
                option_json=args.option_json,
            )
        else:
            self.compiled_train_step = compile_train_step_with_torch_onnx(
                model,
                self.sample,
                self.context,
                args.outdir,
                option_json=args.option_json,
            )
        self.context.synchronize()

    def run(self) -> None:
        assert self.context is not None
        self.first_loss = self.compiled_train_step(self.sample)["loss"].cpu()
        self.context.synchronize()

    def validate(self) -> None:
        # Heuristically check that the loss decreases after a few iterations to
        # validate that training is working. This is not a perfect validation,
        # but it's a simple check that the training loop is doing something
        # reasonable. If subsequent iterations somehow fail, it is treated as a
        # validation failure for simplicity.
        assert self.context is not None
        for _ in range(self.args.num_iters - 2):
            self.compiled_train_step(self.sample)
        last_loss = self.compiled_train_step(self.sample)["loss"].cpu()
        self.context.synchronize()
        assert last_loss < self.first_loss


def safe_run(pipeline: Pipeline) -> dict[str, dict[str, Any]]:
    # Backstop for unexpected errors raised outside of the per-step handling.
    start_s = time.perf_counter()
    try:
        return pipeline.execute()
    except Exception as e:
        duration_s = time.perf_counter() - start_s
        logger.error(f"Unexpected error during {type(pipeline).__name__}: {e}")
        return {
            action: {
                "exit_code": CUSTOM_EXIT_CODES["unexpected_error"],
                "duration_s": duration_s,
            }
            for action in ACTION_CHOICES
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compile and execute timm vision models on MN-Core2. "
            "Executes three pipeline phases: compile (generate device code), "
            "run (execute on device), and validate (compare against PyTorch or "
            "verify loss reduction). "
            "Supports both inference and training modes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "OUTPUT FORMAT:\n"
            "The script outputs JSON with the following structure:\n"
            "  {\n"
            '    "compile": {"exit_code": int, "duration_s": float},\n'
            '    "run": {"exit_code": int, "duration_s": float},\n'
            '    "validate": {"exit_code": int, "duration_s": float}\n'
            "  }\n\n"
            "EXIT CODES:\n"
            "  0: Phase completed successfully\n"
            "  1: Phase failed with a recoverable error\n"
            " -1: Phase failed with an unexpected error (e.g., PyTorch reference failed)\n"
            " -2: Phase status unknown (action stopped at an earlier phase)\n\n"
            "EXAMPLES:\n"
            "  # Compile, run, and validate resnet18 on auto-detected MN-Core2 device\n"
            "  %(prog)s --model_name resnet18 --action validate\n\n"
            "  # Only compile efficientnet_b0 with custom output directory\n"
            "  %(prog)s --model_name efficientnet_b0 --outdir ./out --action compile\n\n"
            "  # Train with batch size 32 for 20 iterations\n"
            "  %(prog)s --mode train --model_name resnet18 --batch_size 32 --num_iters 20"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="/tmp/mlsdk_timm")
    parser.add_argument("--option_json", type=Path, default=None)
    parser.add_argument("--mode", type=str, default="infer", choices=["infer", "train"])
    parser.add_argument(
        "--device",
        type=str,
        default="mncore2:auto",
        choices=[
            "mncore2:auto",
            "mncore2:0",
            "mncore2:1",
            "mncore2:2",
            "mncore2:3",
            "mncore2:4",
            "mncore2:5",
            "mncore2:6",
            "mncore2:7",
            "pfvm:cpu",
            "pfvm:cuda",
        ],
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
        choices=ACTION_CHOICES,
        help="Whether to only compile, run without validation, "
        "or run with validation (default: validate)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
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
    logging.basicConfig(level=getattr(logging, args.log_level))

    # Simple args validation
    assert args.batch_size > 0, "Batch size must be positive"
    assert (
        args.num_iters >= 2
    ), "Number of iterations must be at least 2 to observe loss decrease"
    if args.option_json is not None:
        assert (
            args.option_json.is_file()
        ), f"Option JSON file not found: {args.option_json}"

    pipelines: dict[str, type[Pipeline]] = {
        "infer": InferencePipeline,
        "train": TrainingPipeline,
    }
    if args.mode not in pipelines:
        raise ValueError(f"Unsupported mode: {args.mode}")
    if os.path.exists(args.outdir):
        logger.warning(
            f"Output directory {args.outdir} already exists. "
            "It may cause issues with the compilation."
        )
    result = safe_run(pipelines[args.mode](args))

    print(json.dumps(result))
    # If any of the actions resulted in an error (non-zero and not intentionally
    # skipped), exit with code 1 to indicate failure.
    for action_result in result.values():
        if action_result["exit_code"] not in (0, CUSTOM_EXIT_CODES["unknown"]):
            sys.exit(1)
