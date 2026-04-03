#! /bin/bash

set -eux -o pipefail

EXAMPLE_NAME=run_llm_infer
VENVDIR=/tmp/${EXAMPLE_NAME}_venv

CURRENT_DIR=$(realpath $(dirname $0))
CODEGEN_DIR=$(realpath ${CURRENT_DIR}/../../)
BUILD_DIR=${BUILD_DIR:-${CODEGEN_DIR}/build}

if [[ ! -d ${VENVDIR} ]]; then
    python3 -m venv --system-site-packages ${VENVDIR}
fi

source ${VENVDIR}/bin/activate
# See https://discuss.huggingface.co/t/cas-bridge-xethub-hf-co-broke/158626/8 for hf_xet.
pip3 install transformers==4.44.0 huggingface-hub==0.34.4 hf_xet==v1.1.5

source "${BUILD_DIR}/codegen_pythonpath.sh"

export MNCORE_USE_LEGACY_ONNX_EXPORTER=1
export MNCORE_USE_EXTERNAL_DATA_FORMAT=1

exec python3 ${CURRENT_DIR}/llm_infer.py "$@"
