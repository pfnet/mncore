import os
import pickle

import numpy as np
import torch

# import from externals/mlcommons-ncf/recommendation/pytorch
from alias_generator import AliasSample
from convert import CACHE_FN, generate_negatives
from mlsdk import MNCoreOptimizer
from torch.utils.data import TensorDataset


def load_sampler(
    data_dir: str, user_scaling: int, item_scaling: int
) -> tuple[AliasSample, np.ndarray, np.ndarray, np.int64]:
    fn_prefix = data_dir + "/" + CACHE_FN.format(user_scaling, item_scaling)
    sampler_cache = fn_prefix + "cached_sampler.pkl"

    if os.path.exists(data_dir):
        print(f"Using alias file: {sampler_cache}")
        with open(sampler_cache, "rb") as f:
            sampler, pos_users, pos_items, num_items, _ = pickle.load(f)
    else:
        raise ValueError(f"sampler directory does not exist: {data_dir}")

    return (sampler, pos_users, pos_items, num_items)


def generate_neg_dataset(
    pos_users: torch.Tensor,
    label_size: torch.Size,
    neg_ratio: int,
    num_items: np.int64,
    allow_collision: bool,
    sampler: AliasSample,
) -> TensorDataset:
    if allow_collision:
        neg_users = pos_users.repeat(neg_ratio)
        neg_items = torch.empty_like(neg_users, dtype=torch.int64).random_(
            0, int(num_items)
        )
    else:
        # Use sampler which had been generated in convert.py
        # The sampler avoids collision of item id between positive and negative data
        negatives = generate_negatives(sampler, neg_ratio, pos_users.numpy())
        negatives = torch.from_numpy(negatives)
        neg_users = negatives[:, 0]
        neg_items = negatives[:, 1]
    neg_labels = torch.zeros(label_size, dtype=torch.float32).repeat(neg_ratio)

    return TensorDataset(neg_users, neg_items, neg_labels)


def load_train_pos_data(
    data_dir: str, user_scaling: int, item_scaling: int
) -> tuple[TensorDataset, int, np.int64, AliasSample]:
    sampler, pos_users, pos_items, num_items = load_sampler(
        data_dir, user_scaling, item_scaling
    )

    num_users = len(sampler.num_regions)
    pos_users = torch.from_numpy(pos_users).type(torch.LongTensor)
    pos_items = torch.from_numpy(pos_items).type(torch.LongTensor)
    pos_labels = torch.ones_like(pos_users, dtype=torch.float32)
    dataset = TensorDataset(pos_users, pos_items, pos_labels)

    return dataset, num_users, num_items, sampler


def load_eval_data(
    data_dir: str, num_users: int, user_scaling: int, item_scaling: int, neg_ratio: int
) -> tuple[TensorDataset, int]:
    # Load positive items
    pos_item_chunks = []
    for chunk_id in range(user_scaling):
        pos_ratings = torch.from_numpy(
            np.load(
                f"{data_dir}/testx{user_scaling}x{item_scaling}_{chunk_id}.npz",
                encoding="bytes",
            )["arr_0"]
        )
        pos_item_chunks.append(pos_ratings[:, 1].reshape(-1, 1))

    # Load negative items
    neg_item_chunks = []
    for chunk_id in range(user_scaling):
        neg_ratings = torch.from_numpy(
            np.load(
                f"{data_dir}/test_negx{user_scaling}x{item_scaling}_{chunk_id}.npz",
                encoding="bytes",
            )["arr_0"]
        )
        neg_item_chunks.append(neg_ratings[:, 1].reshape(-1, neg_ratio))

    # Concat positive and negative items
    item_chunks = [
        torch.cat((negs, poses), dim=1)
        for negs, poses in zip(neg_item_chunks, pos_item_chunks)
    ]

    # Get indices of positive items in concatenated items
    pos_item_index_chunks = []
    for items, pos_items in zip(item_chunks, pos_item_chunks):
        is_positive_mask = items == pos_items
        pos_item_index_chunks.append(torch.argmax(is_positive_mask.long(), dim=1))

    # Create a mask to identify duplicate items to avoid them during evaluation
    dup_mask_chunks = []
    for items in item_chunks:
        stable_indices = torch.argsort(items, dim=1, stable=True)
        sorted_items = torch.gather(items, 1, stable_indices)

        is_duplicate_sorted = sorted_items[:, 1:] == sorted_items[:, :-1]
        dup_mask_sorted = torch.cat(
            [
                torch.zeros(is_duplicate_sorted.shape[0], 1, dtype=torch.bool),
                is_duplicate_sorted,
            ],
            dim=1,
        )

        # Unsort the mask back to the original item order
        inverse_indices = torch.argsort(stable_indices, dim=1)
        dup_mask = torch.gather(dup_mask_sorted, 1, inverse_indices)
        dup_mask_chunks.append(dup_mask)

    # Concatenate all chunks into final Tensors
    items = torch.cat(item_chunks, dim=0).long()
    dup_mask = torch.cat(dup_mask_chunks, dim=0)
    pos_item_indices = torch.cat(pos_item_index_chunks, dim=0)

    # Replicate each user ID for the number of item samples they have
    users = torch.arange(num_users, dtype=torch.long).unsqueeze(1)
    users = users.repeat(1, items.shape[1])

    dataset = TensorDataset(users, items, dup_mask, pos_item_indices)
    samples_per_user = items.size(1)

    return dataset, samples_per_user


def generate_padding(
    data_len: int, users_per_batch: int, samples_per_user: int
) -> TensorDataset:
    remainder_users = data_len % users_per_batch
    padding_users = users_per_batch - remainder_users if remainder_users > 0 else 0

    dummy_users = torch.zeros(padding_users, samples_per_user, dtype=torch.long)
    dummy_items = torch.zeros(padding_users, samples_per_user, dtype=torch.long)
    dummy_dup_mask = torch.zeros(padding_users, samples_per_user, dtype=torch.bool)
    dummy_pos_item_indices = torch.full((padding_users,), -1, dtype=torch.long)

    return TensorDataset(
        dummy_users, dummy_items, dummy_dup_mask, dummy_pos_item_indices
    )


def load_model(
    model: torch.nn.Module,
    optimizer: MNCoreOptimizer | None = None,
    model_path: str | os.PathLike = "./ncf_model.pth",
) -> None:
    print(f"loading model from {model_path}")
    weights = torch.load(model_path, weights_only=True)

    model.load_state_dict(weights["model"])
    if optimizer is not None:
        optimizer.load_state_dict(weights["optimizer"])


def save_model(
    model: torch.nn.Module,
    optimizer: MNCoreOptimizer,
    outdir: str,
    model_path: str | os.PathLike = "ncf_model.pth",
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(outdir, model_path),
    )

    print(f"model saved to {model_path}")
