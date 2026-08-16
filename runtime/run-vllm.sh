#!/usr/bin/env bash
set -euo pipefail

variant=${1:-nvfp4}
speculation=${2:-none}

container_image=${QWEN38_IMAGE:-ghcr.io/aeon-7/aeon-vllm-ultimate@sha256:1aa47363e4c9cfa0a85411c669d39b7f9fa3adb3e735ef1ca5760be3044dacd7}
models_root=${QWEN38_MODELS_ROOT:-${HOME}/models/qwen3.8-27b}
cache_root=${XDG_CACHE_HOME:-${HOME}/.cache}
cache_dir=${QWEN38_CACHE:-${cache_root}/qwen3.8-vllm}
port=${QWEN38_PORT:-18083}
context=${QWEN38_CONTEXT:-131072}
gpu_util=${QWEN38_GPU_MEMORY_UTILIZATION:-0.70}
max_seqs=${QWEN38_MAX_SEQS:-8}
max_batched_tokens=${QWEN38_MAX_BATCHED_TOKENS:-16384}
kv_cache_dtype=${QWEN38_KV_CACHE_DTYPE:-auto}
mtp_tokens=${QWEN38_MTP_TOKENS:-3}
container_name=${QWEN38_CONTAINER_NAME:-qwen38-vllm}

usage() {
  echo "usage: $0 {bf16|fp8|nvfp4} {none|mtp}" >&2
}

case "$variant" in
  bf16|fp8|nvfp4) model_dir=$models_root/$variant ;;
  *) usage; exit 2 ;;
esac

case "$speculation" in
  none) speculative_args=() ;;
  mtp)
    speculative_args=(
      --speculative-config
      "{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp_tokens}"
    )
    ;;
  *) usage; exit 2 ;;
esac

if [[ ! -s "$model_dir/config.json" ]]; then
  echo "Qwen3.8 $variant checkpoint is incomplete: $model_dir" >&2
  exit 1
fi

mkdir -p "$cache_dir/$variant"

compilation_args=()
if [[ "${QWEN38_CUDAGRAPHS:-1}" != 1 ]]; then
  compilation_args=(--compilation-config '{"cudagraph_mode":"NONE"}')
fi

exec docker run --rm \
  --name "$container_name" \
  --gpus all \
  --ipc host \
  --network host \
  -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_ENFORCE_STRICT_TOOL_CALLING=0 \
  -v "$model_dir:/model:ro" \
  -v "$cache_dir/$variant:/root/.cache" \
  --entrypoint vllm \
  "$container_image" \
  serve /model \
  --served-model-name qwen3.8 \
  --host 127.0.0.1 \
  --port "$port" \
  --mamba-cache-dtype float32 \
  --max-model-len "$context" \
  --gpu-memory-utilization "$gpu_util" \
  --max-num-seqs "$max_seqs" \
  --max-num-batched-tokens "$max_batched_tokens" \
  --kv-cache-dtype "$kv_cache_dtype" \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"preserve_thinking":false}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --trust-remote-code \
  "${compilation_args[@]}" \
  "${speculative_args[@]}"
