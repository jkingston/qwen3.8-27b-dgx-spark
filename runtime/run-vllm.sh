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
auto_download=${QWEN38_AUTO_DOWNLOAD:-1}

usage() {
  echo "usage: $0 {bf16|fp8|nvfp4} {none|mtp}" >&2
}

case "$variant" in
  bf16)
    model_repo=Qwen/Qwen3.8-27B
    model_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
    ;;
  fp8)
    model_repo=Qwen/Qwen3.8-27B-FP8
    model_revision=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
    ;;
  nvfp4)
    model_repo=unsloth/Qwen3.8-27B-NVFP4
    model_revision=16b6615af3548b88e2d8e382457bc705b00479cf
    ;;
  *) usage; exit 2 ;;
esac
model_dir=$models_root/$variant
download_marker=$model_dir/.qwen38-download-in-progress

case "$speculation" in
  none|mtp) ;;
  *) usage; exit 2 ;;
esac

if [[ ! -s "$model_dir/config.json" || -e "$download_marker" ]]; then
  if [[ "$auto_download" != 1 ]]; then
    echo "Qwen3.8 $variant checkpoint is missing or incomplete: $model_dir" >&2
    echo "Automatic downloads are disabled by QWEN38_AUTO_DOWNLOAD=0." >&2
    exit 1
  fi

  mkdir -p "$model_dir"
  touch "$download_marker"
  echo "Downloading $model_repo at $model_revision to $model_dir..."
  if command -v hf >/dev/null 2>&1; then
    hf download "$model_repo" --revision "$model_revision" --local-dir "$model_dir"
  elif command -v uvx >/dev/null 2>&1; then
    uvx --from huggingface-hub hf download \
      "$model_repo" --revision "$model_revision" --local-dir "$model_dir"
  else
    echo "Install the Hugging Face CLI or uv before downloading checkpoints:" >&2
    echo "  https://huggingface.co/docs/huggingface_hub/guides/cli" >&2
    exit 1
  fi

  if [[ ! -s "$model_dir/config.json" ]]; then
    echo "checkpoint download did not produce $model_dir/config.json" >&2
    exit 1
  fi
  rm -f "$download_marker"
fi

mkdir -p "$cache_dir/$variant"

vllm_args=(
  serve /model
  --served-model-name qwen3.8
  --host 127.0.0.1
  --port "$port"
  --mamba-cache-dtype float32
  --max-model-len "$context"
  --gpu-memory-utilization "$gpu_util"
  --max-num-seqs "$max_seqs"
  --max-num-batched-tokens "$max_batched_tokens"
  --kv-cache-dtype "$kv_cache_dtype"
  --reasoning-parser qwen3
  --default-chat-template-kwargs '{"preserve_thinking":false}'
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --enable-chunked-prefill
  --enable-prefix-caching
  --trust-remote-code
)

if [[ "${QWEN38_CUDAGRAPHS:-1}" != 1 ]]; then
  vllm_args+=(--compilation-config '{"cudagraph_mode":"NONE"}')
fi
if [[ "$speculation" == mtp ]]; then
  vllm_args+=(
    --speculative-config
    "{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp_tokens}"
  )
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
  "${vllm_args[@]}"
