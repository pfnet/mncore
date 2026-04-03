#! /bin/bash

set -eux -o pipefail

EXAMPLE_NAME="mlsdk_detr_inference"
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
if [[ ! -d detr ]]; then
    git clone https://github.com/facebookresearch/detr.git --depth 1
fi
popd

TARGET_FILES=(
    "models/detr.py"
    "models/matcher.py"
)

for REL_PATH in "${TARGET_FILES[@]}"; do
    BASE_NAME=$(basename "$REL_PATH" .py)
    PATCH_TARGET="${EXTERNAL_DIR}/detr/${REL_PATH}"
    PATCH_FILE="${CURRENT_DIR}/patches/${BASE_NAME}.patch"
    patch --forward --backup -i "$PATCH_FILE" "$PATCH_TARGET" || [ $? -eq 1 ]
done

cp ${CURRENT_DIR}/lsa.py ${EXTERNAL_DIR}/detr/models/

### Run detr_inference.py

export PYTHONPATH="${EXTERNAL_DIR}/detr${PYTHONPATH:+:${PYTHONPATH}}"
echo PYTHONPATH

source "${BUILD_DIR}/codegen_pythonpath.sh"

export MNCORE_USE_EXTERNAL_DATA_FORMAT=1

python3 ${CURRENT_DIR}/detr_inference.py ${@}
