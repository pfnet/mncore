#! /bin/bash

set -eux -o pipefail

EXAMPLE_NAME="mlsdk_ssd_inference"
VENV_DIR=${VENV_DIR:-"/tmp/${EXAMPLE_NAME}/venv"}
EXTERNAL_DIR=${EXTERNAL_DIR:-"/tmp/${EXAMPLE_NAME}/external"}
COCO_DIR=${COCO_DIR:-"/tmp/${EXAMPLE_NAME}/coco"}
OUT_DIR=${OUT_DIR:-"/tmp/${EXAMPLE_NAME}/out"}

CURRENT_DIR=$(realpath $(dirname $0))
CODEGEN_DIR=$(realpath ${CURRENT_DIR}/../../../)
BUILD_DIR="${CODEGEN_DIR}/build"

### Prepare and source venv/

if [[ ! -d ${VENV_DIR} ]]; then
    python3 -m venv --system-site-packages ${VENV_DIR}
    source ${VENV_DIR}/bin/activate
    pip3 install -r ${CURRENT_DIR}/requirements.txt
else
    source ${VENV_DIR}/bin/activate
fi

### Prepare external/ items

mkdir -p ${EXTERNAL_DIR}
pushd ${EXTERNAL_DIR}

if [[ ! -d ai-reference-models ]]; then
    git clone 'https://github.com/intel/ai-reference-models.git' ai-reference-models --depth 1 --branch v3.3
fi
if [[ ! -f resnet34-ssd1200.pth ]]; then
    # For downloading the trained model, please refer to this documentation.
    # Ref: https://github.com/intel/ai-reference-models/blob/main/models_v2/pytorch/ssd-resnet34/inference/cpu/CONTAINER.md
    wget --no-check-certificate \
        'https://docs.google.com/uc?export=download&id=13kWgEItsoxbVKUlkQz4ntjl1IZGk6_5Z' \
        -O resnet34-ssd1200.pth
fi

popd

### Prepare coco/ items

mkdir -p ${COCO_DIR}
pushd ${COCO_DIR}

# download the COCO datasets and annotations
if [[ ! -d val2017 ]]; then
    curl -O 'http://images.cocodataset.org/zips/val2017.zip'
    unzip val2017.zip
    rm -f val2017.zip
fi
if [[ ! -d annotations ]]; then
    curl -O 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    unzip annotations_trainval2017.zip
    rm -f annotations_trainval2017.zip
fi
# fetch images and annotations for inference
if [[ ! -f annotations/fetched_annotations_eval.json ]]; then
    python3 ${CURRENT_DIR}/coco_preparation.py \
        -a annotations/instances_val2017.json \
        -i val2017 \
        -f annotations/fetched_annotations_eval.json
fi

popd

### Run ssd_inference.py

source "${BUILD_DIR}/codegen_pythonpath.sh"

# Do not split a small h/w image among L1/L2 blocks.
export CODEGEN_SPATIAL_SPLIT_THRESHOLD_L2B_L1B=32

PYTHONPATH=${PYTHONPATH}:"${EXTERNAL_DIR}/ai-reference-models/models_v2/pytorch/ssd-resnet34/inference/cpu"
python3 ${CURRENT_DIR}/ssd_inference.py \
    --model_path "${EXTERNAL_DIR}/resnet34-ssd1200.pth" \
    --coco_data_path "${COCO_DIR}/val2017" \
    --coco_annotation_path "${COCO_DIR}/annotations/fetched_annotations_eval.json" \
    --out_img_dir "${CURRENT_DIR}" \
    --outdir "${OUT_DIR}" \
    --option_json "${CODEGEN_DIR}/preset_options/O1.json" \
    ${@}
