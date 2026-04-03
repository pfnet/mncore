import torch
from mlsdk import (
    Context,
    MNCoreSGD,
    MNDevice,
    set_buffer_name_in_optimizer,
    set_tensor_name_in_module,
    storage,
)


def run_train():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)

        def forward(self, *, x):
            return {"y": self.linear(x)}

    device = MNDevice("mncore2:auto")
    context = Context(device)
    Context.switch_context(context)

    sample = {"x": torch.ones(1, 10)}
    model = Model()
    model.linear.weight.data.fill_(1.0)
    model.linear.bias.data.fill_(1.0)
    model.train()

    set_tensor_name_in_module(model, "model")
    for p in model.parameters():
        context.register_param(p)
    for b in model.buffers():
        context.register_buffer(b)

    optimizer = MNCoreSGD(model.parameters(), 0.1, 0.9, 0.0)
    set_buffer_name_in_optimizer(optimizer, "optim0")
    context.register_optimizer_buffers(optimizer)

    def f(inp):
        optimizer.zero_grad()
        loss = torch.relu(model(**inp)["y"]).sum()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        return {"loss": loss}

    compiled_f = context.compile(
        f,
        sample,
        storage.path("/tmp/train"),
    )

    print(f"Original {model.linear.bias=}")
    for _ in range(10):
        compiled_f(sample)
    context.synchronize()
    print(f"Optimized {model.linear.bias=}")


if __name__ == "__main__":
    run_train()
