#! /bin/bash
# Runs some of the MLSDK examples under codegen/MLSDK/examples, that runs under limited dependencies.
#
# In other words, it only needs mncore-sdk-minimal.
set -u -o pipefail
echo "starting at $(date "+%Y-%m-%d-%H-%M")" >&2

if [[ -e /opt/pfn/pfcomp/image_date.txt ]]; then
    echo "base image is $(cat /opt/pfn/pfcomp/image_date.txt)" >&2
elif [[ -e /opt/pfn/pfcomp/VERSION && -e /opt/pfn/pfcomp/GIT_COMMIT ]]; then
    echo "base image is $(cat /opt/pfn/pfcomp/VERSION) ($(cat /opt/pfn/pfcomp/GIT_COMMIT))" >&2
else
    echo "Warning: no base image was found. please ensure that you are running this script in the MLSDK environment." >&2
fi

DEVICE="mncore2:auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        "--device")
            shift
            if [[ -n "$1" ]]; then
                DEVICE="$1"
            fi
            ;;
        *)
            :  # ignore
            ;;
    esac
    shift
done

if [[ "${DEVICE}" == "mncore2"* ]]; then
    EXAMPLE_FILES=(
        "add.py"
        "load_codegen_dir.py"
        "add_trace.py"
        "explicit_data_transfer_api.py"
        "infer.py"
        "infer_multi.py"
        "train.py"
        "mnist.py"
    )
    echo "listing available MN-Core 2 boards:" >&2
    gpfn3-smi list
else
    EXAMPLE_FILES=(
        "mnist.py"
    )
fi

# running examples; one by one
set -ex
MLSDK_DIR="$(realpath "$(dirname "$0")")"
pushd "${MLSDK_DIR}"

for filename in "${EXAMPLE_FILES[@]}"; do
    ./exec_with_env.sh python3 "${filename}" --device "${DEVICE}"
done

./run_timm.sh --model_name resnet50.a1h_in1k --batch_size 16 --device "${DEVICE}"

