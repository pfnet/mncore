# MN-Core Software Samples & Build Resources

This repository provides the following resources for working with MN-Core:

* Examples for the MN-Core SDK
* Resources for setting up the MN-Core SDK environment
* Resources for managing bare-metal machines equipped with MN-Core

## Try the MN-Core SDK

The MN-Core SDK enables developers to build and optimize programs for the MN-Core architecture. The SDK consists of two primary components:

* MLSDK: Provides PyTorch-compatible interface for developing machine learning models.
* HPCSDK: Provides a general-purpose programming environment in C/C++ with OpenCL-compatible and directive-based programming models. Currently, only MNCL, an OpenCL-compatible environment, is provided.

Programs developed with the MN-Core SDK are portable. They can run on MN-Core hardware and emulators, and MLSDK programs can also run on GPUs. This flexibility allows you to verify the compatibility of your programs quickly even without direct access to MN-Core hardware.

We recommend installing the MN-Core SDK environment using Docker for a consistent setup. Please refer to the following documentation for detailed installation instructions:

* [`sdk/0.6/README.md`](sdk/0.6/README.md)

When using MLSDK, the following documentation will be helpful:

* MLSDK Documentation: [en](https://dev.mn-core.com/sdk/latest/MLSDK/docs/en/), [ja](https://dev.mn-core.com/sdk/latest/MLSDK/docs/ja/)
* [MLSDK Examples](sdk/examples/README.md)

When testing HPCSDK, please refer to the header files included with the SDK. Note that HPCSDK is currently in an alpha stage.

## Run Programs with MN-Core

Programs developed with the MN-Core SDK can run in the following three environments:

### Run on the emulator

The MN-Core SDK includes an MN-Core emulator that runs programs without access to actual hardware. This allows you to easily evaluate portability to MN-Core and estimate expected performance.

The usage of the emulator differs between MLSDK and HPCSDK. Please refer to their respective documentation for details.

### Run on bare metal (MN-Core 2 Devkit / MN-Server 2)

If you have access to an MN-Core 2 Devkit or MN-Server 2, you can run programs on actual MN-Core hardware instead of the emulator. These systems allow you to execute programs within a Docker container built using the instructions in [`sdk/0.6/README.md`](sdk/0.6/README.md).

### Run on the cloud (Preferred Computing Platform; PFCP)

The Preferred Computing Platform (PFCP) also provides environments equipped with MN-Core processors.
You can use pre-built Docker images on PFCP that are equivalent to images built with [`sdk/0.6/README.md`](sdk/0.6/README.md).
This makes it straightforward to deploy MN-Core applications on PFCP after verifying them locally with the emulator or on bare-metal systems.
For instructions on using MN-Core on PFCP, please refer to the [PFCP documentation](https://docs.pfcomputing.com/).

## Repository Contents

* `apt/` -- Resources to install MN-Core packages via `apt`
  * `add_mncore_packages.sh` -- Script to add the APT repository in your Ubuntu system
* `sdk/` -- MN-Core SDK
  * `0.4/`, `0.5/`, `0.6/` -- MN-Core SDK versions. Each contains:
    * `README.md` -- Explaining how to prepare MN-Core SDK environment
    * `mncore-sdk-minimal.Dockerfile` — Dockerfile for minimal image with dependencies
    * `mncore-sdk-full.Dockerfile` — Dockerfile for "full" image with extra packages
    * `create_dev_ctr.sh` — Script to start a development container
  * `examples/` — Examples for the latest MLSDK
