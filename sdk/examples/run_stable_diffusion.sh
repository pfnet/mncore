#!/bin/bash
set -ex -o pipefail

CURRENT_DIR=$(realpath $(dirname $0))
CODEGEN_DIR=$(realpath ${CURRENT_DIR}/../../)
BUILD_DIR=${BUILD_DIR:-${CODEGEN_DIR}/build}

venv_dir=/tmp/run_stable_diffusion_venv
if [[ ! -d $venv_dir ]]; then
    python3 -m venv /tmp/run_stable_diffusion_venv --system-site-packages
    source /tmp/run_stable_diffusion_venv/bin/activate
    # Fix versions to avoid breaking changes
    pip install diffusers==0.8.0 transformers==4.44.0 huggingface-hub==0.24.7
fi
source /tmp/run_stable_diffusion_venv/bin/activate

source "${BUILD_DIR}/codegen_pythonpath.sh"

export MNCORE_USE_EXTERNAL_DATA_FORMAT=1


python3 ${CURRENT_DIR}/stable_diffusion.py "$@"
