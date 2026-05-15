import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from alias_generator import AliasSample
from mlsdk import Context, MNCoreAdam, MNCoreOptimizer, MNDevice
from ncf_eval import run_eval
from ncf_utils import (
    generate_neg_dataset,
    generate_padding,
    load_eval_data,
    load_model,
    load_train_pos_data,
    save_model,
)

# import from externals/mlcommons-ncf/recommendation/pytorch
from neumf import NeuMF
from torch.utils.data import ConcatDataset, DataLoader
from utility import apply_toml_defaults, compile_fn, set_deterministic_mode


def run_train(  # noqa: CFQ002
    args: argparse.Namespace,
    model: nn.Module,
    device: str,
    context: Context,
    optimizer: MNCoreOptimizer,
    loss_fn: nn.BCEWithLogitsLoss,
    outdir: str,
    pos_dataset: torch.utils.data.Dataset,
    neg_sampler: AliasSample,
    num_items: np.int64,
) -> None:
    # Define training functions
    def train_fx2onnx(sample_d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        optimizer.zero_grad()

        outputs = model(sample_d["user"], sample_d["item"])
        loss = loss_fn(outputs, sample_d["label"]).float()
        loss = torch.mean(loss.view(-1), 0)

        loss.backward()
        optimizer.step()
        return {"result": loss}

    # Dummy input for compilation
    samples = {
        "user": torch.randint(1, (args.train_batch_size,), dtype=torch.int64),
        "item": torch.randint(1, (args.train_batch_size,), dtype=torch.int64),
        "label": torch.rand(args.train_batch_size).view(-1, 1),
    }

    train_fn = compile_fn(
        context,
        train_fx2onnx,
        model,
        samples,
        outdir=outdir,
        model_name="ncf_train",
        is_train=True,
        optimizers=[optimizer],
        option_json=str(args.option_json),
    )

    model.train()

    total_iter = (
        len(pos_dataset) * (1 + args.train_neg_ratio)
    ) // args.train_batch_size

    for epoch in range(args.epoch):
        pos_users, _, pos_labels = pos_dataset.tensors
        neg_dataset = generate_neg_dataset(
            pos_users,
            pos_labels.size(),
            args.train_neg_ratio,
            num_items,
            args.allow_collision_with_pos,
            neg_sampler,
        )
        dataloader = DataLoader(
            ConcatDataset([pos_dataset, neg_dataset]),
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers,
            drop_last=True,
        )

        for num_batch, (users, items, labels) in enumerate(dataloader):
            samples["user"] = users
            samples["item"] = items
            samples["label"] = labels.view(-1, 1)

            output = train_fn(samples)
            epoch_str = f"Epoch {epoch + 1}/{args.epoch}"
            iteration_str = f"Iteration {num_batch + 1}/{total_iter}"
            loss_str = f"Loss: {output["result"].item():.4}"
            print(f"{epoch_str}, {iteration_str}, {loss_str}")

    # Synchronize tensors on MN-Core 2"s DRAM and PyTorch tensors
    context.synchronize()


def main(args: argparse.Namespace) -> None:  # noqa: CFQ001

    if args.save_path != "":
        dir_path = os.path.dirname(args.save_path) or "."
        if not os.access(dir_path, os.W_OK):
            raise ValueError("Parent directory of save_path is not writable.")

    # Fix seed values for reproducibility
    set_deterministic_mode(args.seed)

    # Decide device and outdir from given options
    device_name = args.device
    outdir = args.outdir

    # Load positive data for training and create dataset
    data_dir = f"/tmp/ncf_training/{args.dataset}"
    scaled_data_dir = (
        f"{data_dir}/{args.dataset}x{args.user_scaling}x{args.item_scaling}"
    )
    train_pos_dataset, num_users, num_items, neg_sampler = load_train_pos_data(
        scaled_data_dir, args.user_scaling, args.item_scaling
    )

    # Define model
    model = NeuMF(
        num_users,
        num_items,
        mf_dim=args.factors,
        mf_reg=0.0,
        mlp_layer_sizes=args.mlp_layers,
        mlp_layer_regs=([0.0] * len(args.mlp_layers)),
    )

    # Create loss function object
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # Create optimizer object
    optimizer = MNCoreAdam(
        model.parameters(),
        lr=args.learning_rate,
        chainer_use_torch=True,
    )

    # Pass the device information to context or move the model and optimizer to specified device
    device = MNDevice(device_name)
    train_context = Context(device)
    eval_context = Context(device)
    Context.switch_context(train_context)

    # Load pre-trained model
    if args.load_path != "":
        load_model(model, optimizer, model_path=args.load_path)

    # Run training
    run_train(
        args,
        model,
        device,
        train_context,
        optimizer,
        loss_fn,
        outdir,
        train_pos_dataset,
        neg_sampler,
        num_items,
    )

    # Save trained model
    if args.save_path != "":
        save_model(model, optimizer, outdir, model_path=args.save_path)

    # Load positive and negative data for evaluation and create dataloader
    eval_dataset, samples_per_user = load_eval_data(
        scaled_data_dir,
        num_users,
        args.user_scaling,
        args.item_scaling,
        args.eval_neg_ratio,
    )
    users_per_eval_batch = max(args.eval_batch_size // samples_per_user, 1)
    eval_dataset = ConcatDataset(
        [
            eval_dataset,
            generate_padding(len(eval_dataset), users_per_eval_batch, samples_per_user),
        ]
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=users_per_eval_batch,
        shuffle=False,
        num_workers=args.loader_num_workers,
    )

    # Switch to evaluation context
    Context.switch_context(eval_context)

    # Run evaluation
    run_eval(
        args,
        model,
        device,
        eval_context,
        outdir,
        eval_dataloader,
        samples_per_user,
        num_users,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # mlsdk options
    parser.add_argument("--device", type=str, default="mncore2:auto")
    parser.add_argument("--outdir", type=str, default="/tmp/mlsdk_ncf_training/out")
    parser.add_argument(
        "--option_json",
        type=Path,
        default="/opt/pfn/pfcomp/codegen/preset_options/O1.json",
    )

    apply_toml_defaults(str(Path(__file__).parent / "configs.toml"), parser)

    # Parse command line args and opts
    args = parser.parse_args()

    main(args)
