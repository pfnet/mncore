import argparse
import random
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from mnist_common import MNCoreClassifier, mnist_loaders

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def main(
    checkpoint_path: str, outdir: str, option_json_path: Optional[Path], device_str: str
) -> None:
    batch_size = 64
    eval_batch_size = 125

    _, eval_loader = mnist_loaders(batch_size, eval_batch_size)

    checkpoint = torch.load(checkpoint_path)

    model_with_loss_fn = MNCoreClassifier()
    model_with_loss_fn.load_state_dict(checkpoint["model_state_dict"])
    model_with_loss_fn.eval()

    def eval_step(inp: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        x = inp["x"]
        t = inp["t"]
        output = model_with_loss_fn(x, t)
        y = output["y"]
        _, predicted = torch.max(y, 1)
        correct = (predicted == t).sum()
        return {"correct": correct}

    correct = 0
    for sample in eval_loader:
        correct += eval_step(sample)["correct"]
    print(
        f"Correct: {correct} / {len(eval_loader.dataset)}. "
        f"Accuracy: {correct / len(eval_loader.dataset)}"
    )
    assert 0.94 < correct / len(eval_loader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        A script designed to be used with mnist_train.py,
        specifically for running MNIST inference operations.
        """
    )
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--outdir", type=str, default="/tmp/mlsdk_mnist_infer")
    parser.add_argument("--option_json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="mncore2:auto")
    args = parser.parse_args()
    main(args.checkpoint_path, args.outdir, args.option_json, args.device)
