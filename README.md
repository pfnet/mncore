# MN-Core Software Samples & Build Resources

This repository provides resources for working with MN-Core.

## Try the MN-Core SDK

The MN-Core SDK enables developers to build and optimize programs for the MN-Core architecture. The SDK consists of two primary components:

* MLSDK: Provides PyTorch-compatible interface for developing machine learning models.
* HPCSDK: Provides OpenCL-compatible environment for developing C++ applications.

Programs developed with the MN-Core SDK are portable. They can be executed on MN-Core emulators and GPUs in addition to MN-Core hardware. This flexibility allows you to verify the compatibility of your programs quickly even without direct access to MN-Core hardware.

We recommend installing the MN-Core SDK environment using Docker for consistent setup. Please refer to the following documentation for detailed installation instructions:

* [`sdk/0.4/README.md`](sdk/0.4/README.md)

When using MLSDK, the following documentation will be helpful:

* [MLSDK Documentation](https://dev.mn-core.com/sdk/0.4/MLSDK/docs/en/)
* [MLSDK Examples](sdk/examples/README.md)

When testing HPCSDK, please refer to the header files included with the SDK. Note that HPCSDK is currently in alpha stage.

## Run Programs on MN-Core

There are two primary methods for running programs on actual MN-Core hardware:

### Run on the cloud (Preferred Computing Platform; PFCP)

The Preferred Computing Platform (PFCP) provides environments equipped with MN-Core processors.
You can use pre-built Docker images on PFCP that are equivalent to images built with [`sdk/0.4/README.md`](sdk/0.4/README.md).
These pre-built images available on PFCP allow you to seamlessly deploy programs developed and verified in your local environment directly onto MN-Core hardware.
For instructions on utilizing MN-Core on PFCP, please refer to the [PFCP documentation](https://docs.pfcomputing.com/).

### Run on bare metal (MN-Core 2 Devkit / MN-Server 2)

MN-Core is also available on bare-metal machine products.
These machines allow you to execute programs on actual MN-Core hardware within a Docker container built according to [`sdk/0.4/README.md`](sdk/0.4/README.md).

## Repository Contents

* `apt/`  -- Resources to install MN-Core packages via `apt`
  * `add_mncore_packages.sh`  -- Script to add the APT repository in your Ubuntu system
* `sdk/`  -- MN-Core SDK
  * `0.4/`  -- MN-Core SDK 0.4
    * `mncore-sdk-minimal.Dockerfile`  -- Dockerfile to build the MN-Core SDK 0.4 minimal Docker image with dependencies
    * `mncore-sdk-full.Dockerfile`  -- Dockerfile to build the MN-Core SDK 0.4 "full" Docker image with extra packages
    * `create_dev_ctr.sh`  -- Script to start an MN-Core SDK 0.4 development container
  * `examples/`  -- Examples for the latest MLSDK
