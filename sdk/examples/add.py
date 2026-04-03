import torch
from mlsdk import CacheOptions, Context, MNDevice, storage


def run_add():
    device = MNDevice("mncore2:auto")
    context = Context(device)
    Context.switch_context(context)

    def add(input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = input["x"]
        y = input["y"]
        return {"out": x + y}

    sample = {"x": torch.randn(3, 4), "y": torch.randn(3, 4)}

    compiled_add = context.compile(
        add,
        sample,
        storage.path("/tmp/add_two_tensors"),
        options={"float_dtype": "float"},
        cache_options=CacheOptions("/tmp/add_two_tensors_cache"),
    )
    result = compiled_add({"x": torch.ones(3, 4), "y": torch.ones(3, 4)})
    result_on_cpu = result["out"].cpu()
    print(f"{result_on_cpu=}")
    assert torch.allclose(result_on_cpu, torch.ones(3, 4) * 2)


if __name__ == "__main__":
    run_add()
