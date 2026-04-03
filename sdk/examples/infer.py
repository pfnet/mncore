import torch
from mlsdk import Context, MNDevice, set_tensor_name_in_module, storage

torch.manual_seed(0)


def run_infer():
    device = MNDevice("mncore2:auto")
    context = Context(device)
    Context.switch_context(context)

    model = torch.nn.Linear(4, 4)
    model.eval()
    set_tensor_name_in_module(model, "model")
    for p in model.parameters():
        context.register_param(p)
    for b in model.buffers():
        context.register_buffer(b)

    def infer(input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = input["x"]
        y = model(x)
        return {"out": y}

    sample = {"x": torch.randn(4, 4)}

    compiled_infer = context.compile(
        infer,
        sample,
        storage.path("/tmp/infer"),
        options={"float_dtype": "mixed"},
    )
    result = compiled_infer({"x": torch.ones(4, 4)})
    result_on_cpu = result["out"].cpu()
    print(result_on_cpu)


if __name__ == "__main__":
    run_infer()
