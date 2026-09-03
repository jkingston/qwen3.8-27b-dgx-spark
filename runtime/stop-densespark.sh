#!/usr/bin/env bash
set -euo pipefail

container="${DENSESPARK_CONTAINER:-qwen38-27b-densespark}"
if docker container inspect "$container" >/dev/null 2>&1; then
    docker stop "$container"
else
    printf '%s is not running\n' "$container"
fi
