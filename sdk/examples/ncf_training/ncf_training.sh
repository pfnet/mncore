#! /bin/bash

set -eux -o pipefail

EXAMPLE_NAME=ncf_training

CURRENT_DIR=$(realpath $(dirname $0))
CODEGEN_DIR=$(realpath ${CURRENT_DIR}/../../../)

VENV_DIR="/tmp/${EXAMPLE_NAME}/training_venv"

# Use virtual environment
if [[ ! -d ${VENV_DIR} ]]; then
    python3 -m venv --system-site-packages ${VENV_DIR}
    source ${VENV_DIR}/bin/activate
    pip install -r ${CURRENT_DIR}/requirements_training.txt
else
    source ${VENV_DIR}/bin/activate
fi

GIT_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH="${GIT_ROOT}:/tmp/${EXAMPLE_NAME}/externals/mlcommons-ncf/recommendation/pytorch${PYTHONPATH:+:${PYTHONPATH}}"

BUILD_DIR=${BUILD_DIR:-${CODEGEN_DIR}/build}

source "${BUILD_DIR}/codegen_preloads.sh"
source "${BUILD_DIR}/codegen_pythonpath.sh"

python3 ${CURRENT_DIR}/${EXAMPLE_NAME}.py "$@"
