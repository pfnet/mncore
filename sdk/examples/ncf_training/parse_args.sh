#!/bin/bash

parse_arguments() {
    local long_opts="dataset:,external_data_dir:,user_scaling:,item_scaling:,eval_neg_ratio:"
    local parsed_args
    parsed_args=$(getopt -o "" -l "$long_opts" -- "$@")

    # check if getopt failed
    if [ $? -ne 0 ]; then
        return 1
    fi

    eval set -- "$parsed_args"

    # initialize
    DATASET="ml-1m"
    EXTERNAL_DATA_DIR=""
    USER_SCALING="1"
    ITEM_SCALING="1"
    EVAL_NEG_RATIO="999"
    REMAINING_ARGS=()

    while true; do
        case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --external_data_dir)
            EXTERNAL_DATA_DIR="$2"
            shift 2
            ;;
        --user_scaling)
            USER_SCALING="$2"
            shift 2
            ;;
        --item_scaling)
            ITEM_SCALING="$2"
            shift 2
            ;;
        --eval_neg_ratio)
            EVAL_NEG_RATIO="$2"
            shift 2
            ;;
        --)
            shift
            REMAINING_ARGS=("$@")
            break
            ;;
        *)
            echo "Internal error while parsing arguments." >&2
            return 1
            ;;
        esac
    done

    return 0
}
