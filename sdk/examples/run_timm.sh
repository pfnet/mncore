#! /bin/bash

set -eux -o pipefail

EXAMPLE_NAME=run_timm
VENVDIR=/tmp/${EXAMPLE_NAME}_venv

CURRENT_DIR=$(realpath $(dirname $0))
CODEGEN_DIR=$(realpath ${CURRENT_DIR}/../../)
BUILD_DIR=${BUILD_DIR:-${CODEGEN_DIR}/build}

if [[ ! -d ${VENVDIR} ]]; then
    python3 -m venv --system-site-packages ${VENVDIR}
    source ${VENVDIR}/bin/activate
    pip3 install timm==1.0.14 huggingface-hub==0.28.1
else
    source ${VENVDIR}/bin/activate
fi

source "${BUILD_DIR}/codegen_pythonpath.sh"

exec python3 ${CURRENT_DIR}/${EXAMPLE_NAME}.py "$@"
