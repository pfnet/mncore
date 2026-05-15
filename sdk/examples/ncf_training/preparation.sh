#!/bin/bash

set -eux -o pipefail

EXAMPLE_NAME=ncf_training

CURRENT_DIR=$(realpath $(dirname $0))
CODEGEN_DIR=$(realpath ${CURRENT_DIR}/../../../)
pushd ${CODEGEN_DIR}/MLSDK/examples/${EXAMPLE_NAME}

source "./parse_args.sh"
parse_arguments "$@"
if [ $? -ne 0 ]; then
    echo "Argument parser failed."
    exit 1
fi

# install necessary commands
if !(type unzip > /dev/null 2>&1 ); then
    apt update || exit 1
    apt install -y unzip || exit 1
fi
if !(type curl > /dev/null 2>&1 ); then
    apt update || exit 1
    apt install -y curl || exit 1
fi

# clone mlcommons repository for python modules and data preprocess.
GIT_ROOT=$(git rev-parse --show-toplevel)
if [ ! -d "/tmp/${EXAMPLE_NAME}/externals/mlcommons-ncf" ]; then
    git clone https://github.com/mlcommons/training.git /tmp/${EXAMPLE_NAME}/externals/mlcommons-ncf || exit 1
    git -c safe.directory=$(realpath /tmp/${EXAMPLE_NAME}/externals/mlcommons-ncf) -C /tmp/${EXAMPLE_NAME}/externals/mlcommons-ncf checkout 3683b7b || exit 1
fi
DATA_EXPANSION_DIR=/tmp/${EXAMPLE_NAME}/externals/mlcommons-ncf/data_generation/fractal_graph_expansions
RECOMMENDATION_DIR=/tmp/${EXAMPLE_NAME}/externals/mlcommons-ncf/recommendation

# prepare dataset directory

function download_20m {
	echo "Download ml-20m"
	curl -O https://files.grouplens.org/datasets/movielens/ml-20m.zip
}

function download_1m {
	echo "Downloading ml-1m"
	curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
}

if [ ! -z "${EXTERNAL_DATA_DIR}" ]; then # in case of mounting the specified path
    if [ ! -d "${EXTERNAL_DATA_DIR}" ]; then
        mkdir "${EXTERNAL_DATA_DIR}"
    fi
    ln -sfn "${EXTERNAL_DATA_DIR}" /tmp/${EXAMPLE_NAME}/${DATASET}
fi
DATA_DIR=/tmp/${EXAMPLE_NAME}/${DATASET}
if [ ! -d ${DATA_DIR} ]; then
    mkdir ${DATA_DIR}
fi

# download dataset if necessary
cd ${DATA_DIR}
if [ ! -d "./${DATASET}" ]; then
    if [ ! -e "./${DATASET}.zip" ]; then
        # bash ${RECOMMENDATION_DIR}/download_dataset.sh ${DATASET} || exit 1
        if [[ ${DATASET} == "ml-1m" ]]
        then
            download_1m
        else
            download_20m
        fi
        bash ${RECOMMENDATION_DIR}/verify_dataset.sh ${DATASET} | grep -q "FAILED" && { echo "dataset verification failed."; exit 1;}
    fi
    unzip ./${DATASET}.zip
    if [ "${DATASET}" = "ml-1m" ]; then
        # create ratings.csv from ratings.dat
        echo "userId,movieId,rating,timestamp" > ./${DATASET}/ratings.csv
        sed 's/::/,/g' ./${DATASET}/ratings.dat >> ./${DATASET}/ratings.csv
    fi
fi
cd -

# install libraries for building python and Pillow package
PKGS=(libbz2-dev libsqlite3-dev libncurses-dev libreadline-dev liblzma-dev libjpeg-dev zlib1g-dev)
if ! dpkg -s ${PKGS[@]} &> /dev/null; then
    apt update || exit 1
    apt install -y ${PKGS[@]}
fi

# install pyenv and python 3.6.15
export PYENV_ROOT="/tmp/${EXAMPLE_NAME}/dataset_pyenv"
export PATH="${PYENV_ROOT}/bin:${PATH}"
if ! command -v pyenv &> /dev/null; then
    curl https://pyenv.run | bash
fi
if ! pyenv versions --bare | grep -q "^3.6.15"; then
    pyenv install 3.6.15
fi
eval "$(pyenv init - bash)"
pyenv shell 3.6.15

# create venv and install dependencies
VENV_DIR="/tmp/${EXAMPLE_NAME}/dataset_venv"
if [[ ! -d ${VENV_DIR} ]]; then
    python3 -m venv ${VENV_DIR}
    source ${VENV_DIR}/bin/activate
    pip3 install -r requirements_datagen.txt
else
    source ${VENV_DIR}/bin/activate
fi

# run dataset expansion
export DATASET
export USER_MUL=${USER_SCALING}
export ITEM_MUL=${ITEM_SCALING}
export DATA_DIR=$(realpath ${DATA_DIR})
SCALED_DATASET_DIR="${DATA_DIR}/${DATASET}x${USER_SCALING}x${ITEM_SCALING}"
if [ ! -d ${SCALED_DATASET_DIR} ]; then
    cd ${DATA_EXPANSION_DIR}
    bash data_gen.sh
    cd -
fi

# switch back to system python
set +u; deactivate; set -u
pyenv shell --unset

# create another venv for generating negative dataset and training
VENV_DIR="/tmp/${EXAMPLE_NAME}/training_venv"
if [[ ! -d ${VENV_DIR} ]]; then
    python3 -m venv --system-site-packages ${VENV_DIR}
    source ${VENV_DIR}/bin/activate
    pip3 install -r requirements_training.txt
else
    source ${VENV_DIR}/bin/activate
fi

# run negative dataset generation
if ! ls "${SCALED_DATASET_DIR}"/test_neg* >/dev/null 2>&1; then
    python3 ${RECOMMENDATION_DIR}/pytorch/convert.py ${SCALED_DATASET_DIR} --valid-negative ${EVAL_NEG_RATIO} --user_scaling ${USER_SCALING} --item_scaling ${ITEM_SCALING} ${REMAINING_ARGS[@]}
fi

popd
