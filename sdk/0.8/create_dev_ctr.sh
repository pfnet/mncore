#! /bin/bash
set -eu -o pipefail

MNCORE_REGEX='/dev/mnc.*'

DOCKER="docker"
IMAGE="mncore-sdk-full:0.8"
SEMAPHORE_MOUNT="/opt/mncore_shared_semaphore:/var/tmp/mncore"
DOCKER_OPTS=()

function usage() {
    echo "$0 -- Launch docker container for MN-Core devenv" >&2
    echo "" >&2
    echo "$0 [-A] [-i IMAGE] [-s MOUNT] [-R RUNTIME_CLI] [-O DOCKER_OPTS] devices..." >&2
    echo "" >&2
    echo "Options:" >&2
    echo "  -A              Mount all devices on the node to the container" >&2
    echo "  -i IMAGE[:TAG]  Use specified IMAGE (${IMAGE})" >&2
    echo "  -s MOUNT        Use specified MOUNT for sharing semaphore (${SEMAPHORE_MOUNT})" >&2
    echo "  -R RUNTIME_CLI  Container runtime CLI (${DOCKER})" >&2
    echo "  -O DOCKER_OPTS  Additional docker-run options" >&2
    echo "" >&2
    exit 1
}

USE_ALL=0
while getopts "Ai:s:R:O:h" opt; do
    case "${opt}" in
    A) USE_ALL=1;;
    i) IMAGE=${OPTARG};;  # localhost/ prefix is required when run with -R podman, that is, the options would be -R podman -i localhost/mncore-sdk-full:0.8
    s) SEMAPHORE_MOUNT=${OPTARG};;
    R) DOCKER=${OPTARG};;
    O) DOCKER_OPTS+=(${OPTARG});;
    h) usage
    esac
done
shift "$((OPTIND - 1))"

DEVICES=()
if [[ $# -ne 0 ]]; then
    DEVICES=($@)
elif [[ ${USE_ALL} -ne 0 ]]; then
    echo "Enumerating devices" >&2
    while IFS= read -r dev; do
        DEVICES+=("${dev}")
    done < <(find /dev -regex "${MNCORE_REGEX}")
    if [[ ${#DEVICES[@]} -eq 0 ]]; then
        echo "[WARN] No MN-Core device found. You can use emulator backend only" >&2
    fi
else
    echo "E: No device list nor use-all flag (-A) specified." >&2
    usage $0
fi

DEVICE_OPTS=()
for dev in "${DEVICES[@]+"${DEVICES[@]}"}"; do
    DEVICE_OPTS+=("--device=${dev}")
done

MOUNT_OPTS=()
if [[ ${#DEVICES[@]} -gt 0 ]]; then
    MOUNT_OPTS+=("-v" "${SEMAPHORE_MOUNT}")
else
    echo "[INFO] No semaphore mount (${SEMAPHORE_MOUNT}) as no MN-Core device found." >&2
fi

# check if the image is already loaded to the system
if ! ${DOCKER} image ls --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE}$"; then
    echo "E: Image ${IMAGE} is not found on the system."
    exit 1
fi

echo "Starting container" >&2
CTR_ID=$(${DOCKER} run \
    --privileged \
    -d \
    "${MOUNT_OPTS[@]+"${MOUNT_OPTS[@]}"}" \
    "${DEVICE_OPTS[@]+"${DEVICE_OPTS[@]}"}" \
    "${DOCKER_OPTS[@]+"${DOCKER_OPTS[@]}"}" \
    ${IMAGE} \
    sleep inf)
echo ${CTR_ID}

echo "Setting up devices" >&2
for dev in "${DEVICES[@]+"${DEVICES[@]}"}"; do
    echo "  ${dev}" >&2

    if ! ${DOCKER} exec --privileged -it ${CTR_ID} gpfn3-smi config ${dev#/dev/} clock --core=750 --gddr6=15000; then
        echo "E: Failed to setup device ${dev}. Aborting" >&2
        ${DOCKER} stop ${CTR_ID} >/dev/null
        exit 1
    fi

    if ! ${DOCKER} exec --privileged -it ${CTR_ID} gpfn3-smi config ${dev#/dev/} mab; then
        echo "E: Failed to setup device ${dev}. Aborting" >&2
        ${DOCKER} stop ${CTR_ID} >/dev/null
        exit 1
    fi
done

echo -e "Setup succeeded. To launch a shell in the container:\n\n\$ \033[1m${DOCKER} exec --privileged -it ${CTR_ID:0:12} bash\033[0m\n\nTo shutdown the container:\n\n\$ \033[1m${DOCKER} stop ${CTR_ID:0:12}\033[0m\n" >&2
