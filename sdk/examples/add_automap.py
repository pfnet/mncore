from dataclasses import dataclass
from typing import List, NamedTuple, Tuple

import torch
from mlsdk import (
    Context,
    MNDevice,
    register_pytree_dataclass,
    register_pytree_namedtuple,
    storage,
)


@register_pytree_namedtuple
class NT(NamedTuple):
    x: torch.Tensor


@register_pytree_dataclass
@dataclass
class DC:
    x: torch.Tensor


def run_add():
    device = MNDevice("mncore2:auto")
    context = Context(device)
    Context.switch_context(context)

    def add(
        arg1: Tuple[torch.Tensor],
        arg2: NT,
        *,
        kwarg_ls: List[torch.Tensor],
        kwarg_dc: DC,
    ) -> torch.Tensor:
        return arg1[0] + arg2.x + kwarg_ls[0] + kwarg_dc.x

    arg1 = (torch.randn(3, 4),)
    arg2 = NT(x=torch.randn(3, 4))
    kwarg_ls = [torch.randn(3, 4)]
    kwarg_dc = DC(x=torch.randn(3, 4))

    compiled_add = context.compile_automap(
        add,
        (arg1, arg2),
        {"kwarg_ls": kwarg_ls, "kwarg_dc": kwarg_dc},
        storage.path("/tmp/add_many_tensors"),
        options={"float_dtype": "float"},
    )
    result = compiled_add(
        (torch.ones(3, 4),),
        NT(x=torch.ones(3, 4)),
        kwarg_ls=[torch.ones(3, 4)],
        kwarg_dc=DC(x=torch.ones(3, 4)),
    )
    result_on_cpu = result.cpu()
    print(f"{result_on_cpu=}")
    assert torch.allclose(result_on_cpu, torch.ones(3, 4) * 4)


if __name__ == "__main__":
    run_add()
