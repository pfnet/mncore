#!/bin/bash
# Note: this script is not for testing MN-Core devices, but for maintaining code samples in the MLSDK document.

if [[ -z "${OMP_NUM_THREADS}" ]]; then
    # Prevent from creating too many threads in the OpenMP runtime.
    # Four threads are enouth to train and infer a simple MLP model.
    export OMP_NUM_THREADS=4
fi

set -eux -o pipefail

MLSDK_DIR="$(realpath "$(dirname "$0")")"
pushd "${MLSDK_DIR}"

OUTDIR="/tmp/mlsdk_mnist_train"
./exec_with_env.sh python3 mnist_train.py --outdir "${OUTDIR}"
./exec_with_env.sh python3 mnist_infer.py "${OUTDIR}/checkpoint.pt"

