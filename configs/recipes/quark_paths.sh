#!/usr/bin/env bash
# Resolve public Quark recipe paths before changing the working directory.
QUARK_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

quark_absolute_path() {
    case "$1" in
        ''|/*|*://*) printf '%s' "$1" ;;
        *) printf '%s/%s' "$QUARK_REPO_ROOT" "$1" ;;
    esac
}

quark_export_path() {
    local quark_env_name
    case "$1" in
        --dataset_configs) quark_env_name=DATASET_CONFIGS ;;
        --tokenizer_weight) quark_env_name=TOKENIZER_WEIGHT ;;
        --output_dir) quark_env_name=OUTPUT_DIR ;;
        --log_dir) quark_env_name=LOG_DIR ;;
        --resume) quark_env_name=RESUME ;;
    esac
    printf -v "$quark_env_name" '%s' "$2"
    export "$quark_env_name"
}

quark_prepare_paths() {
    local quark_key quark_value
    for quark_key in DATASET_CONFIGS TOKENIZER_WEIGHT OUTPUT_DIR LOG_DIR RESUME; do
        quark_value="${!quark_key:-}"
        if [[ -n "$quark_value" ]]; then
            printf -v "$quark_key" '%s' "$(quark_absolute_path "$quark_value")"
            export "$quark_key"
        fi
    done
    QUARK_ARGS=()
    while (( $# )); do
        case "$1" in
            --dataset_configs|--tokenizer_weight|--output_dir|--log_dir|--resume)
                if (( $# < 2 )) || [[ "$2" == --* ]]; then
                    printf 'Missing path after %s\n' "$1" >&2
                    return 2
                fi
                quark_value="$(quark_absolute_path "$2")"
                QUARK_ARGS+=("$1" "$quark_value")
                quark_export_path "$1" "$quark_value"
                shift 2
                ;;
            --dataset_configs=*|--tokenizer_weight=*|--output_dir=*|--log_dir=*|--resume=*)
                quark_value="$(quark_absolute_path "${1#*=}")"
                QUARK_ARGS+=("${1%%=*}=$quark_value")
                quark_export_path "${1%%=*}" "$quark_value"
                shift
                ;;
            *) QUARK_ARGS+=("$1"); shift ;;
        esac
    done
}
