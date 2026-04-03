#! /bin/bash
# Runs all the MLSDK examples under codegen/MLSDK/examples, including examples that need more dependencies than the minimal ones.
#
# In other words, it needs mncore-sdk-full. It does not work with mncore-sdk-minimal.
set -eux -o pipefail
"$(realpath "$(dirname "$0")")/run_minimal_examples.sh" "$@"
"$(realpath "$(dirname "$0")")/run_mnist_on_torch.sh"

