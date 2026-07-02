#!/bin/bash
set -eux -o pipefail

#########################################################
# Set up paths and environment variables
#########################################################

# Must use the same env as training to load the trained model correctly,
# especially when the diffusers version differs from the one used in training.
VENV_DIR=/tmp/stable_diffusion_venv
CURRENT_DIR=$(realpath $(dirname $0))
CODEGEN_DIR=$(realpath ${CURRENT_DIR}/../../../)

# Set Huggingface cache directory to avoid filling up the home directory
export HF_HOME=${HF_HOME:-"/tmp/huggingface"}
export MNCORE_USE_EXTERNAL_DATA_FORMAT=1

#########################################################
# Set up python environment
#########################################################

if [[ ! -d ${VENV_DIR} ]]; then
    python3 -m venv --system-site-packages ${VENV_DIR}
    source ${VENV_DIR}/bin/activate
    pip3 install -r ${CURRENT_DIR}/requirements.txt
else
    source ${VENV_DIR}/bin/activate
fi

source "${CODEGEN_DIR}/build/codegen_pythonpath.sh"

#########################################################
# Run Stable Diffusion evaluation
#########################################################

PRESET_OPTIONS_DIR=${CODEGEN_DIR}/preset_options \
    python3 ${CURRENT_DIR}/stable_diffusion_eval.py "$@"
