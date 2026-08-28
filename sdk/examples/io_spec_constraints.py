"""Connect compiled functions without a device-to-device layout conversion.

The consumer is compiled first. The producer then uses the consumer input IOSpec
as a compile-time constraint for its output. At execution time, relocation lets
the consumer read the producer output buffer directly; no predefined device
buffer or host copy is needed.
"""

import argparse

import torch
from mlsdk import Context, MNDevice, storage

TensorDict = dict[str, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mncore2:auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = Context(MNDevice(args.device))

    def producer(inputs: TensorDict) -> TensorDict:
        return {"shared": inputs["x"] * 2}

    def consumer(inputs: TensorDict) -> TensorDict:
        return {"result": inputs["shared"] + 1}

    sample = torch.randn(32, 64)
    compile_options = {"float_dtype": "float"}

    # Compile the consumer first, because it is the side whose preferred input
    # layout the producer output must satisfy.
    compiled_consumer = context.compile(
        consumer,
        {"shared": sample},
        storage.path("/tmp/io_spec_constraints_consumer"),
        options=compile_options,
        training=False,
    )

    # The mapping key is a producer input/output name. The value can come from a
    # differently named function IO; using the same name is the common case.
    shared_names = {"shared"}
    compiled_producer = context.compile(
        producer,
        {"x": sample},
        storage.path("/tmp/io_spec_constraints_producer"),
        options=compile_options,
        training=False,
        io_spec_constraints={
            name: compiled_consumer.input_specs[name] for name in shared_names
        },
    )

    producer_outputs = compiled_producer({"x": torch.ones_like(sample)})

    # producer_outputs["shared"] is a TensorProxy. Since the two IOSpecs are
    # compatible, the runtime relocates the consumer input to that device buffer
    # instead of compiling or executing a DRAM-to-DRAM layout conversion.
    outputs = compiled_consumer(producer_outputs)
    result = outputs["result"].cpu()

    assert compiled_producer.output_specs["shared"].is_memcopyable_to(
        compiled_consumer.input_specs["shared"]
    )
    assert torch.allclose(result, torch.full_like(result, 3))
    print("producer -> consumer direct-use succeeded")


if __name__ == "__main__":
    main()
