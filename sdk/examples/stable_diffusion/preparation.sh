#!/bin/bash

set -eux -o pipefail
dataset_dir=${dataset_dir:-"."}

# install necessary commands
if !(type tar > /dev/null 2>&1 ); then
    apt update
    apt install tar || exit 1
fi

# prepare the Caltech 256 dataset
if [ ! -d "${dataset_dir}/data" ]; then
    mkdir -p "${dataset_dir}/data" || exit 1
fi

if [ $(ls -UF ${dataset_dir}/data/train/ | grep / | wc -l) -lt 1 ]; then
    # download the dataset if there doesn't exist
    if [ ! -e "${dataset_dir}/256_ObjectCategories.tar" ]; then
        curl -L https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1 -o "${dataset_dir}/256_ObjectCategories.tar"
    fi
    # unpack tar file if it has not been unpacked yet
    if [ $(ls -UF ${dataset_dir}/data/train | grep / | wc -l) -lt 1 ]; then
        cat "${dataset_dir}/256_ObjectCategories.tar" | tar -xf - -C "${dataset_dir}/data" --no-same-owner
        mv "${dataset_dir}/data/256_ObjectCategories" "${dataset_dir}/data/train" || exit 1
    fi
fi

# rename image folders
if [ $(find  ${dataset_dir}/data/train/ -name 'metadata.csv' | wc -l) -ne $(ls -UF ${dataset_dir}/data/train | grep / | wc -l) ]; then
    python3 preparation.py --dataset_dir "${dataset_dir}/data/train" "$@"
fi
