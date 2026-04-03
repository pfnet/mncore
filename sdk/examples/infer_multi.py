import torch
from mlsdk import Context, MNDevice, set_tensor_name_in_module, storage


def run_infer():
    device = MNDevice("mncore2:auto")
    context = Context(device)
    Context.switch_context(context)

    model0 = torch.nn.Linear(4, 4, bias=False)
    model0.weight = torch.nn.Parameter(torch.ones(4, 4))
    model0.eval()
    model1 = torch.nn.Linear(4, 4, bias=False)
    model1.weight = torch.nn.Parameter(torch.ones(4, 4) * 2)
    model1.eval()

    # To differentiate each model, Context uses the name specified by
    # set_tensor_name_in_module, so these names must be set appropriately.
    # Similarly, during training, set the name in set_buffer_name_in_optimizer
    # as well.
    set_tensor_name_in_module(model0, "model0")
    set_tensor_name_in_module(model1, "model1")
    for p in model0.parameters():
        context.register_param(p)
    for b in model0.buffers():
        context.register_buffer(b)
    for p in model1.parameters():
        context.register_param(p)
    for b in model1.buffers():
        context.register_buffer(b)

    def infer0(input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = input["x"]
        y = model0(x)
        return {"out": y}

    def infer1(input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = input["x"]
        y = model1(x)
        return {"out": y}

    sample = {"x": torch.ones(4, 4)}

    compiled_infer0 = context.compile(
        infer0,
        sample,
        storage.path("/tmp/infer0"),
    )
    compiled_infer1 = context.compile(
        infer1,
        sample,
        storage.path("/tmp/infer1"),
    )
    result0 = compiled_infer0({"x": torch.ones(4, 4)})
    result_on_cpu0 = result0["out"].cpu()
    assert torch.allclose(result_on_cpu0, torch.ones(4, 4) @ torch.ones(4, 4))
    result1 = compiled_infer1({"x": torch.ones(4, 4)})
    result_on_cpu1 = result1["out"].cpu()
    assert torch.allclose(result_on_cpu1, torch.ones(4, 4) @ (torch.ones(4, 4) * 2))


if __name__ == "__main__":
    run_infer()
