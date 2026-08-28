# syntax=docker/dockerfile:1

# Dockerfile to build the MN-Core SDK 0.8 minimal Docker image (mncore-sdk-minimal:0.8).

# Example command line to build this Docker image to run on this directory:
#
# docker build -t mncore-sdk-minimal:0.8 -f mncore-sdk-minimal.Dockerfile .


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
    mncore-sdk=0.8 \
    && \
    apt-mark hold mncore-sdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# MN-Core SDK 0.8 has been confirmed to work only in the Python environment constructed as shown below.
# We are considering support for more general Python environments (ex. user's venv) in future releases.
#
# Note that the SDK won't work with PyTorch's official Python packages. It uses our own custom build.
RUN rm /usr/lib/python3.12/EXTERNALLY-MANAGED && \
    pip3 install --break-system-packages \

    accelerate==1.4.0 \
    anyio==4.14.2 \
    boto3==1.43.77 \
    botocore==1.43.77 \
    certifi==2026.7.22 \
    click==8.4.2 \
    deprecation==2.1.0 \
    filelock==3.32.3 \
    fsspec==2026.7.0 \
    h11==0.16.0 \
    hf-xet==1.6.0 \
    httpcore==1.0.9 \
    httpx==0.28.1 \
    huggingface_hub==1.28.0 \
    idna==3.19 \
    Jinja2==3.1.6 \
    jmespath==1.1.0 \
    MarkupSafe==3.0.3 \
    ml_dtypes==0.6.0 \
    mpmath==1.3.0 \
    networkx==3.6.1 \
    numpy==2.5.2 \
    onnx==1.22.0 \
    onnx-ir==1.0.0 \
    onnxscript==0.6.2 \
    packaging==26.3 \
    pfio==2.7.1 \
    pillow==12.3.0 \
    pip==24.0 \
    protobuf==7.36.0 \
    psutil==7.2.2 \
    python-dateutil==2.9.0.post0 \
    PyYAML==6.0.3 \
    retrying==1.3.3 \
    s3transfer==0.19.2 \
    safetensors==0.8.0 \
    setuptools==68.1.2 \
    six==1.17.0 \
    sympy==1.14.0 \
    timm==0.9.2 \
    tqdm==4.70.0 \
    typing_extensions==4.16.0 \
    urllib3==2.7.0 \
    wheel==0.42.0 \
    pytorch-pfn-extras==0.9.0 \
    "torch==${torch_version}" \
    "torchvision==${torchvision_version}" \
    --extra-index-url "${torch_index_url}"

LABEL "org.opencontainers.image.description"="This is a minimal image of the MN-Core SDK (MN-Core development environment). It is used as a base image when using the SDK via CLI or when deploying workloads. Before using this MN-Core SDK image, please review the MN-Core_SDK_End-User_License_Agreement.pdf located in either /opt/pfn/licenses/ or /opt/pfn/pfcomp/licenses/ within the image. By using the SDK, you acknowledge and agree to the terms and conditions set forth therein."
