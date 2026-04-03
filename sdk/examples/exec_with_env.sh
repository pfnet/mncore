#! /bin/bash

set -eux -o pipefail

CURRENT_DIR=$(realpath $(dirname $0))
CODEGEN_DIR=$(realpath ${CURRENT_DIR}/../../)
BUILD_DIR=${CODEGEN_DIR}/build

# Different from README.md, we don't use /opt/pfn/pfcomp/codegen/build here.
# This is because we are using this script also in our development environment.
source ${BUILD_DIR}/codegen_pythonpath.sh

exec "$@"
