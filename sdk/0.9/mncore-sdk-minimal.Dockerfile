# syntax=docker/dockerfile:1

# Dockerfile to build the MN-Core SDK 0.9 minimal Docker image (mncore-sdk-minimal:0.9).

# Example command line to build this Docker image to run on this directory:
#
# docker build -t mncore-sdk-minimal:0.9 -f mncore-sdk-minimal.Dockerfile .


FROM docker.io/library/ubuntu:24.04

ARG torch_index_url=https://download.pytorch.org/whl/cpu/
ARG torch_version=2.9.0
ARG torchvision_version=0.24.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://asia-northeast1-apt.pkg.dev/doc/repo-signing-key.gpg /usr/share/keyrings/mncore-packages-archive-keyring.gpg.asc

RUN chmod 644 /usr/share/keyrings/mncore-packages-archive-keyring.gpg.asc

RUN echo "deb [signed-by=/usr/share/keyrings/mncore-packages-archive-keyring.gpg.asc] https://asia-northeast1-apt.pkg.dev/projects/mncore-packages noble main" > /etc/apt/sources.list.d/mncore-packages.list

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libnuma1 \
    libopenmpi3 \
    openmpi-bin \
    libjpeg8 \
    libpng16-16 \
    libjson-c5 \
    libunwind8 \
    libgomp1 \
    g++ \
    git \
    pkgconf \
    time \
    zstd \
    libtbb-dev \
    libpython3.12 \
    python3.12-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    python3.12-venv \
    libgpfn3-0 \
    libgpfn3-dev \
    gpfn3-smi \
    mncore-sdk=0.9 \
    && \
    apt-mark hold mncore-sdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Users can pin "uv" to a specific version when needed.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN rm /usr/lib/python3.12/EXTERNALLY-MANAGED && \
    uv pip install \
        --system \
        --python /usr/bin/python3.12 \
        --no-python-downloads \
        --compile-bytecode \
        --link-mode copy \
        --preview-features pylock \
        -r /opt/pfn/pfcomp/pylock.toml

LABEL "org.opencontainers.image.description"="This is a minimal image of the MN-Core SDK (MN-Core development environment). It is used as a base image when using the SDK via CLI or when deploying workloads. Before using this MN-Core SDK image, please review the MN-Core SDK End-User License Agreement (EULA) at https://dev.mn-core.com/sdk/eula/MN-Core_SDK_End-User_License_Agreement_20261002.pdf. The EULA is located also at /opt/pfn/pfcomp/licenses/ in the image. By using the SDK, you acknowledge and agree to the terms and conditions set forth therein."
