#!/bin/bash
set -ex -o pipefail

preset=$1
dataset_json=$2
codegen_output_dir=${3:-"/tmp/run_slm_sft"}

#########################################################
# Set up paths and environment variables
#########################################################

CURRENT_DIR=$(realpath $(dirname $0))
VENVDIR=/tmp/run_slm_sft_gian_venv
# Set Huggingface cache directory to avoid filling up the home directory
export HF_HOME=${HF_HOME:-"/tmp/huggingface"}

#########################################################
# Set training parameters based on preset
#########################################################
#
# NOTE: eloss_threshold in the presets are set according to the experiment result.
# eloss after 40 steps for qwen and swal according to the experiment on 2026-02-11
# were as follows.
#
# |      | TANUKI.json        | BUSHI.json         |
# |------|--------------------|--------------------|
# | qwen | 4.276381492614746  | 3.3284881114959717 |
# | swal | 4.3392558097839355 | 3.6840715408325195 |

if [[ "$preset" == "qwen" ]]; then
    # Same as sft-gian-qwen-pyac-mncore2_nightly CI
    device="mncore2:auto"
    model="qwen2.5-1.5b"
    tloss_threshold="0.018174"
    eloss_threshold="4.500000"
    max_steps="40"  # Set to -1 to use default max_steps
    batch_size=32
    n_hidden_layers=-1
elif [[ "$preset" == "swal" ]]; then
    # Same as sft-gian-swal-pyac-mncore2_nightly CI
    device="mncore2:auto"
    model="tiny-swallow-1.5b"
    tloss_threshold="0.018174"
    eloss_threshold="4.500000"
    max_steps="40"  # Set to -1 to use default max_steps
    batch_size=32
    n_hidden_layers=-1
elif [[ "$preset" == "swal-small" ]]; then
    # Same as sft-gian-swal-pyac-small-mncore2_pr CI
    device="mncore2:auto"
    model="tiny-swallow-1.5b"
    model="tiny-swallow-1.5b"
    tloss_threshold="12.758696"
    eloss_threshold="10.933026"
    max_steps="3"  # Set to -1 to use default max_steps
    batch_size=32
    n_hidden_layers=2
else
    echo "invalid preset: $preset"
    exit 1
fi

#########################################################
# MLSDK configuration via environment variables based on preset
#########################################################

if [[ "$preset" == "swal-small" ]]; then
    export CODEGEN_SA_STEPS=100
    export CODEGEN_NUM_SA_THREADS=22
    export CODEGEN_N_TRANSPOSE_THREADS=8
    export CODEGEN_N_DEV_COPY_STREAMS_THREADS=8
else
    export CODEGEN_SA_STEPS=10000
    export CODEGEN_NUM_SA_THREADS=10
    export CODEGEN_N_TRANSPOSE_THREADS=27
    export CODEGEN_N_DEV_COPY_STREAMS_THREADS=27
fi

export CODEGEN_N_TRANSPOSE_THREADS=${CODEGEN_N_TRANSPOSE_THREADS:-27}
export CODEGEN_N_DEV_COPY_STREAMS_THREADS=${CODEGEN_N_DEV_COPY_STREAMS_THREADS:-27}
export CODEGEN_SA_STEPS=${CODEGEN_SA_STEPS:-10000}
export CODEGEN_NUM_SA_THREADS=${CODEGEN_NUM_SA_THREADS:-22}
export CODEGEN_ENABLE_SET_PARTIAL_LOCATION=1
export CODEGEN_GEMM_FORCE_CHANNEL_SPLIT=1
export CODEGEN_OP_DEF=ChainerIndexAdd=IndexAddBcast
export CODEGEN_SKIP_RESOLVE_NEGATIVE_INDICES=1
export CODEGEN_TIME_SLICE_SCATTERED_INDEXING_BCAST=1
export CODEGEN_LAYOUT_PLANNER_Z_HONOR_LAYOUT_SPEC=1
export CODEGEN_MAX_TIME_SLICE=400
export CODEGEN_IGNORE_LAYOUT_CHECK=1
export CODEGEN_ALLOW_UNUSED_LAYOUT_SPEC=1
export CODEGEN_USE_ADDR_FIRST_Z=1
export CODEGEN_LAYOUT_PLANNER_Z=1
export CODEGEN_ALARM=7200
# qwen's embedding and lm_head share weight and they use equivalent but
# different layout. The layout plan will be confused if eval sets the
# same layout to both.
# TODO(hamaji): Come up with a way to handle reused shared weights.
export CODEGEN_IGNORE_REUSED_VALUE_LAYOUT=1
export CODEGEN_DEFER_SIMPLIFY=ReplaceAttention,ReplaceAttentionGrad
export CODEGEN_NODE_SIM_ALLOW_UNEXPECTED_FAIL=1
export CODEGEN_FORCE_ATTENTION_GRAD_AFTER_FORWARD=1
export CODEGEN_AUTO_RECOMPUTE_HACK_FOR_QWEN=1
export CODEGEN_STOP_USING_GENERIC_INDEXING_GATHER_INDEX_ADD=1
export CODEGEN_GEMM_FORCE_WEIGHT_ON_DRAM=1
export CODEGEN_LPZ_SKIP_PROPAGATE_TIME=1
export CODEGEN_OPS_ON_HOST=ChainerAdamW
export MNCORE_USE_EXTERNAL_DATA_FORMAT=1
export PFVM_DISABLE_CONSTANT_REUSE=1

#########################################################
# Set up python environment
#########################################################

if [[ ! -d ${VENVDIR} ]]; then
    python3 -m venv --system-site-packages ${VENVDIR}
    source ${VENVDIR}/bin/activate
    pip3 install -r ${CURRENT_DIR}/requirements.txt
else
    source ${VENVDIR}/bin/activate
fi
CODEGEN_DIR=$(realpath ${CURRENT_DIR}/../../../)
source "${CODEGEN_DIR}/build/codegen_pythonpath.sh"

#########################################################
# Run SLM SFT
#########################################################

mkdir -p ${codegen_output_dir}
python3 $(realpath $(dirname $0))/slm_sft.py \
    --model ${model} \
    --device ${device} \
    --batch_size ${batch_size} \
    --n_hidden_layer ${n_hidden_layers} \
    --max_steps ${max_steps} \
    --tloss_threshold ${tloss_threshold} \
    --eloss_threshold ${eloss_threshold} \
    --run mlsdk_examples_slm_sft \
    --codegen_output_dir ${codegen_output_dir} \
    --dataset_json ${dataset_json}
