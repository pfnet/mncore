import argparse
import os
import random
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from mlsdk import storage
from mnist_common import MNCoreClassifier, mnist_loaders

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def main(outdir: str, option_json_path: Optional[Path], device_str: str) -> None:
    batch_size = 64
    eval_batch_size = 125

    train_loader, _ = mnist_loaders(batch_size, eval_batch_size)

    model_with_loss_fn = MNCoreClassifier()
    model_with_loss_fn.train()

    optimizer = torch.optim.SGD(model_with_loss_fn.parameters(), 0.1, 0.9, 0.0)

    def train_step(inp: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        x = inp["x"]
        t = inp["t"]
        optimizer.zero_grad()
        output = model_with_loss_fn(x, t)
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        return {"loss": loss}

    for epoch in range(10):
        loss = 0.0
        for i, sample in enumerate(train_loader):
            curr_loss = train_step(sample)["loss"]
            loss += (curr_loss - loss) / (i + 1)
            if i % 100 == 0:
                print(f"epoch {epoch}, iter {i:4}, loss {loss}")
        print(f"epoch {epoch}, loss {loss}")

    os.makedirs(outdir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model_with_loss_fn.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        },
        storage.path(outdir) / "checkpoint.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        A script designed to be used with mnist_infer.py,
        specifically for running MNIST training operations.
        """
    )
    parser.add_argument("--outdir", type=str, default="/tmp/mlsdk_mnist_train")
    parser.add_argument("--option_json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="mncore2:auto")
    args = parser.parse_args()
    main(args.outdir, args.option_json, args.device)
