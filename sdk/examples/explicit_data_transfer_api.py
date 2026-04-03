import torch
from mlsdk import Context, MNDevice, set_tensor_name_in_module, storage

torch.manual_seed(0)


def run_explicit_data_transfer():
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
        storage.path("/tmp/proxy_transfer"),
    )

    input_proxies_allocated = compiled_infer.allocate_input_proxy()
    input_proxies_allocated["x"].load_from(torch.ones(4, 4), clone=False)

    for model_param in model.parameters():
        model_param_proxy = context.get_registered_value_proxy(model_param)
        model_param_proxy.load_from(model_param)

    result = compiled_infer(input_proxies_allocated)
    result_on_cpu = result["out"].cpu()
    print(result_on_cpu)


if __name__ == "__main__":
    run_explicit_data_transfer()
