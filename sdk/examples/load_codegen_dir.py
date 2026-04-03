import torch
from mlsdk import Context, MNDevice, storage

device = MNDevice("mncore2:auto")
context = Context(device)
Context.switch_context(context)

# Load the previously compiled function
loaded_add = context.load_codegen_dir(storage.path("/tmp/add_two_tensors"))

# Now you can use it directly
result = loaded_add({"x": torch.ones(3, 4), "y": torch.ones(3, 4)})
result_on_cpu = result["out"].cpu()
print(f"Result from loaded function: {result_on_cpu=}")
assert torch.allclose(result_on_cpu, torch.ones(3, 4) * 2)
