import argparse
import random
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from mlsdk import (
    Context,
    MNCoreSGD,
    MNDevice,
    set_buffer_name_in_optimizer,
    set_tensor_name_in_module,
    storage,
)
from mnist_common import MNCoreClassifier, mnist_loaders

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def main(outdir: str, option_json_path: Optional[Path], device_str: str) -> None:
    batch_size = 64
    eval_batch_size = 125

    device = MNDevice(device_str)
    context = Context(device)
    Context.switch_context(context)

    train_loader, eval_loader = mnist_loaders(batch_size, eval_batch_size)

    model_with_loss_fn = MNCoreClassifier()
    model_with_loss_fn.train()
    set_tensor_name_in_module(model_with_loss_fn, "model_with_loss_fn")
    for p in model_with_loss_fn.parameters():
        context.register_param(p)
    for b in model_with_loss_fn.buffers():
        context.register_buffer(b)

    optimizer = MNCoreSGD(model_with_loss_fn.parameters(), 0.1, 0.9, 0.0)
    set_buffer_name_in_optimizer(optimizer, "optimizer")
    context.register_optimizer_buffers(optimizer)

    def train_step(inp: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        x = inp["x"]
        t = inp["t"]
        optimizer.zero_grad()
        output = model_with_loss_fn(x, t)
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        return {"loss": loss}

    compile_options = {}
    if option_json_path is not None:
        compile_options["option_json"] = str(option_json_path)

    sample = next(iter(train_loader))
    compiled_train_step = context.compile(
        train_step,
        sample,
        storage.path(outdir) / "train_step",
        options=compile_options,
    )

    for epoch in range(10):
        loss = 0.0
        for i, sample in enumerate(train_loader):
            curr_loss = compiled_train_step(sample)["loss"].item()
            loss += (curr_loss - loss) / (i + 1)
            if i % 100 == 0:
                print(f"epoch {epoch}, iter {i:4}, loss {loss}")
        print(f"epoch {epoch}, loss {loss}")

    context.synchronize()

    torch.save(
        {
            "model_state_dict": model_with_loss_fn.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        },
        storage.path(outdir) / "checkpoint.pt",
    )

    model_with_loss_fn.eval()

    def eval_step(inp: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        x = inp["x"]
        t = inp["t"]
        output = model_with_loss_fn(x, t)
        y = output["y"]
        _, predicted = torch.max(y, 1)
        correct = (predicted == t).sum()
        return {"correct": correct}

    sample = next(iter(eval_loader))
    compiled_eval_step = context.compile(
        eval_step,
        sample,
        storage.path(outdir) / "eval_step",
        options=compile_options,
    )
    correct = 0
    for sample in eval_loader:
        correct += compiled_eval_step(sample)["correct"].item()
    print(
        f"Correct: {correct} / {len(eval_loader.dataset)}. "
        f"Accuracy: {correct / len(eval_loader.dataset)}"
    )
    assert 0.94 < correct / len(eval_loader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A standalone MNIST training and inference script."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/tmp/mlsdk_mnist",
        help="Path to store compiled and trained results",
    )
    parser.add_argument(
        "--option_json",
        type=Path,
        default=None,
        help="""
        Path to a JSON file specifying compilation configs,
        e.g. /opt/pfn/pfcomp/codegen/preset_options/O1.json
        """,
    )
    parser.add_argument(
        "--device", type=str, default="mncore2:auto", help="device_name for MNDevice"
    )
    args = parser.parse_args()
    main(args.outdir, args.option_json, args.device)
