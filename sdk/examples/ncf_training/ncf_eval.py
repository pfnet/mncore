import argparse
import math

import torch
from mlsdk import CompiledFunction, Context
from utility import compile_fn


# Measure inference accuracy as hit ratio (HR) and normalized documented cumulative gain (NDCG)
def measure_acc(  # noqa: CFQ002
    args: argparse.Namespace,
    device: str,
    dataloader: torch.utils.data.DataLoader,
    samples_per_user: int,
    infer_fn: CompiledFunction,
    K: int,
    num_user: int,
) -> None:
    log_2 = math.log(2)

    hits = torch.tensor(0.0)
    ndcg = torch.tensor(0.0)

    with torch.no_grad():
        for user, item, dup_mask, pos_item_indices in dataloader:
            samples = {
                "user": user.view(-1),
                "item": item.view(-1),
            }

            scores = infer_fn(samples)["result"].detach().view(-1, samples_per_user)

            # Set scores of duplicate items to -1 to exclude them from top-k
            scores[dup_mask.bool()] = -1
            _, top_k_indices = torch.topk(scores, K)

            # Check if the positive item is among the top-k recommendations (a "hit")
            hit_mask = top_k_indices == pos_item_indices.unsqueeze(1)
            hits += hit_mask.sum().item()

            # Find normalized documented cumulative gain (NDCG)
            hit_ranks = torch.nonzero(hit_mask)[:, 1].view(-1).to(torch.float)
            ndcg += (log_2 / (hit_ranks + 2).log_()).sum()

    hit_rate = hits.item() / num_user
    ndcg = ndcg.item() / num_user
    print(f"HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f}")


def run_eval(  # noqa: CFQ002
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: str,
    context: Context,
    outdir: str,
    dataloader: torch.utils.data.DataLoader,
    samples_per_user: int,
    num_user: int,
) -> None:
    # Define inference functions
    def infer_fx2onnx(sample_d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = None
        with torch.no_grad():
            output = model(sample_d["user"], sample_d["item"], sigmoid=True)

        return {"result": output}

    # Sample input for compilation
    user, item = next(iter(dataloader))[:2]
    sample = {
        "user": user.view(-1),
        "item": item.view(-1),
    }

    infer_fn = compile_fn(
        context,
        infer_fx2onnx,
        model,
        sample,
        outdir=outdir,
        model_name="ncf_eval",
        is_train=False,
        option_json=str(args.option_json),
    )

    model.eval()

    measure_acc(
        args, device, dataloader, samples_per_user, infer_fn, args.topk, num_user
    )
