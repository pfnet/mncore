#! /bin/bash

set -eux -o pipefail

EXAMPLE_NAME=run_llm_infer_pp
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

# @todo (hvy): Fix args.
model_name="${2}"
n_stages="${4}"
outdir="${6}"
max_length="${8}"

export MNCORE_USE_LEGACY_ONNX_EXPORTER=1
export MNCORE_USE_EXTERNAL_DATA_FORMAT=1
export CODEGEN_LAYOUT_PLANNER_Z=1
export CODEGEN_USE_ADDR_FIRST_Z=1

CODEGEN_TIME_SLICE_SCATTERED_INDEXING_BCAST=1 \
CODEGEN_OP_DEF=Gather=GatherBcast \
python3 ${CURRENT_DIR}/compile_pp_llama.py \
    --model_name ${model_name} \
    --phase prefill \
    --n_stages ${n_stages} \
    --outdir ${outdir} \
    --max_length ${max_length}

python3 ${CURRENT_DIR}/compile_pp_llama.py \
    --model_name ${model_name} \
    --phase decode \
    --n_stages ${n_stages} \
    --outdir ${outdir} \
    --max_length ${max_length}

prefill_codegen_dirs=(${outdir}/prefill/*)
decode_codegen_dirs=(${outdir}/decode/*)
prefill_codegen_dirs_str="${prefill_codegen_dirs[*]}"
decode_codegen_dirs_str="${decode_codegen_dirs[*]}"

mpiexec --allow-run-as-root -n ${n_stages} --bind-to none -x OMP_NUM_THREADS=1 python3 ${CURRENT_DIR}/run_pp_llama.py \
   --model_name ${model_name} \
   --prefill_codegen_dirs ${prefill_codegen_dirs_str} \
   --decode_codegen_dirs ${decode_codegen_dirs_str} \
   --verbose \
   --max_length ${max_length}
