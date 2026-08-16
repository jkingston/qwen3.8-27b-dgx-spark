#!/usr/bin/env bash
set -euo pipefail

container_name=${QWEN38_CONTAINER_NAME:-qwen38-sglang}

if ! docker inspect "$container_name" >/dev/null 2>&1; then
  exit 0
fi

docker stop --timeout 30 "$container_name"

for _ in {1..20}; do
  if ! docker inspect "$container_name" >/dev/null 2>&1; then
    exit 0
  fi
  sleep 0.5
done

echo "container did not auto-remove: $container_name" >&2
exit 1
