# syntax=docker/dockerfile:1

# Dockerfile to build the MN-Core SDK 0.5 minimal Docker image (mncore-sdk-minimal:0.5).

# Example command line to build this Docker image to run on this directory:
#
# docker build -t mncore-sdk-minimal:0.5 -f mncore-sdk-minimal.Dockerfile .


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
    mncore-sdk=0.5 \
    && \
    apt-mark hold mncore-sdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# MN-Core SDK 0.5 has been confirmed to work only in the Python environment constructed as shown below.
# We are considering support for more general Python environments (ex. user's venv) in future releases.
#
# Note that the SDK won't work with PyTorch's official Python packages. It uses our own custom build.
RUN rm /usr/lib/python3.12/EXTERNALLY-MANAGED && \
    pip3 install --break-system-packages \
    annotated-doc==0.0.4 \
    anyio==4.13.0 \
    boto3==1.42.96 \
    botocore==1.42.96 \
    certifi==2026.4.22 \
    click==8.3.3 \
    deprecation==2.1.0 \
    filelock==3.29.0 \
    fsspec==2026.3.0 \
    h11==0.16.0 \
    hf-xet==1.4.3 \
    httpcore==1.0.9 \
    httpx==0.28.1 \
    huggingface_hub==1.12.0 \
    idna==3.13 \
    Jinja2==3.1.6 \
    jmespath==1.1.0 \
    markdown-it-py==4.0.0 \
    MarkupSafe==3.0.3 \
    mdurl==0.1.2 \
    ml_dtypes==0.5.4 \
    mpmath==1.3.0 \
    networkx==3.6.1 \
    numpy==2.4.4 \
    onnx==1.21.0 \
    onnx-ir==0.2.1 \
    onnxscript==0.6.2 \
    packaging==26.2 \
    pfio==2.7.1 \
    pillow==12.2.0 \
    pip==24.0 \
    protobuf==7.34.1 \
    Pygments==2.20.0 \
    python-dateutil==2.9.0.post0 \
    PyYAML==6.0.3 \
    retrying==1.3.3 \
    rich==15.0.0 \
    s3transfer==0.16.1 \
    safetensors==0.7.0 \
    setuptools==68.1.2 \
    shellingham==1.5.4 \
    six==1.17.0 \
    sympy==1.14.0 \
    timm==0.9.2 \
    tqdm==4.67.3 \
    typer==0.25.0 \
    typing_extensions==4.15.0 \
    urllib3==2.6.3 \
    wheel==0.42.0 \
    pytorch-pfn-extras==0.9.0 \
    "torch==${torch_version}" \
    "torchvision==${torchvision_version}" \
    --extra-index-url "${torch_index_url}"

LABEL "org.opencontainers.image.description"="This is a minimal image of the MN-Core SDK (MN-Core development environment). It is used as a base image when using the SDK via CLI or when deploying workloads. Before using this MN-Core SDK image, please review the MN-Core_SDK_End-User_License_Agreement.pdf located in either /opt/pfn/licenses/ or /opt/pfn/pfcomp/licenses/ within the image. By using the SDK, you acknowledge and agree to the terms and conditions set forth therein."
