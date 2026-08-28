#! /bin/bash

set -eux -o pipefail

CURRENT_DIR=$(realpath $(dirname $0))

EXAMPLE_NAME="whisper_inference"
VENV_DIR=/tmp/${EXAMPLE_NAME}_venv

# Whisper requires ffmpeg for audio processing.
if ! command -v ffmpeg > /dev/null 2>&1; then
    apt update && apt install -y ffmpeg
fi

if [[ ! -d ${VENV_DIR} ]]; then
    python3 -m venv --system-site-packages ${VENV_DIR}
    source ${VENV_DIR}/bin/activate
    pip install -r ${CURRENT_DIR}/requirements.txt
    pip install -r ${CURRENT_DIR}/requirements_cpu.txt --index-url https://download.pytorch.org/whl/cpu
else
    source ${VENV_DIR}/bin/activate
fi

CODEGEN_DIR=${CODEGEN_DIR:-${CURRENT_DIR}/../../../}
BUILD_DIR=${BUILD_DIR:-${CODEGEN_DIR}/build}
source "${BUILD_DIR}/codegen_pythonpath.sh"

# To enable deterministic behavior, set env. var. as follows
export CUBLAS_WORKSPACE_CONFIG=:16:8

# MLSDK config
export CODEGEN_GEMM_FORCE_WEIGHT_ON_DRAM=1

exec python3 ${CURRENT_DIR}/${EXAMPLE_NAME}.py "$@" \
    --preset_options_dir ${CODEGEN_DIR}/preset_options
