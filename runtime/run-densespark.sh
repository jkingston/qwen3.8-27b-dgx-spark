#!/usr/bin/env bash
set -euo pipefail

profile="${1:-agents}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export DENSESPARK_IMAGE="${DENSESPARK_IMAGE:-densespark:latest}"
export DENSESPARK_CONTAINER="${DENSESPARK_CONTAINER:-qwen38-27b-densespark}"
export DENSESPARK_SERVED_NAME="${DENSESPARK_SERVED_NAME:-qwen3.8-27b}"
export DENSESPARK_PORT="${DENSESPARK_PORT:-18083}"

case "$profile" in
    agents)
        export DENSESPARK_CONCURRENCY="${DENSESPARK_CONCURRENCY:-16}"
        export DENSESPARK_MAX_LEN="${DENSESPARK_MAX_LEN:-65536}"
        export DENSESPARK_MAX_NUM_SEQS="${DENSESPARK_MAX_NUM_SEQS:-16}"
        ;;
    interactive)
        export DENSESPARK_CONCURRENCY="${DENSESPARK_CONCURRENCY:-1}"
        export DENSESPARK_MAX_LEN="${DENSESPARK_MAX_LEN:-131072}"
        ;;
    *)
        echo "usage: $0 [agents|interactive]" >&2
        exit 2
        ;;
esac

cd -- "$script_dir/densespark"
exec ./configs/launch-densespark.sh
