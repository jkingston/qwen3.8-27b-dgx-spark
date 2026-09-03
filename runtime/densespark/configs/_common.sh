#!/usr/bin/env bash
# Shared launch settings. Sourced by every configs/launch-*.sh so the profiles
# differ only in the one thing each is meant to isolate.
set -euo pipefail

IMAGE="${DENSESPARK_IMAGE:-densespark:latest}"
MODEL="${DENSESPARK_MODEL:-Frozenlock/Qwen3.8-27B-int4-AutoRound}"
SERVED_NAME="${DENSESPARK_SERVED_NAME:-densespark-qwen3.8-27b}"
CONTAINER="${DENSESPARK_CONTAINER:-densespark}"
PORT="${DENSESPARK_PORT:-8000}"

# 64K by default. The model supports 262144, but the KV cache for the 16 full
# attention layers grows with it; raise this when you need the context more than
# the concurrency.
MAX_LEN="${DENSESPARK_MAX_LEN:-65536}"
GPU_UTIL="${DENSESPARK_GPU_UTIL:-0.90}"
HF_HOME_HOST="${HF_HOME:-$HOME/.cache/huggingface}"

# Text-only serving by default. Qwen3.8-27B carries a vision tower; refusing
# image and video inputs keeps the multimodal preprocessing paths out of the
# way of a decode-speed measurement. Drop this to serve images.
LIMIT_MM="${DENSESPARK_LIMIT_MM:-{\"image\":0,\"video\":0\}}"

# Backend selection. Both are left to vLLM unless set, so an unset environment
# reproduces stock behaviour exactly. ATTN covers the 16 full-attention layers
# (FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION); MAMBA covers the 48
# linear-attention layers (TRITON, FLASHINFER), which carry a fifth of the
# per-token weight traffic and so are worth sweeping separately.
ATTN_BACKEND="${DENSESPARK_ATTN_BACKEND:-}"
MAMBA_BACKEND="${DENSESPARK_MAMBA_BACKEND:-}"
KV_DTYPE="${DENSESPARK_KV_DTYPE:-}"
# Qwen3.8 is a hybrid GDN model. The default pair below is the configuration
# that passes the raw-token oracle at prompt lengths {1, k, k+1, k+2}. Prefix
# caching plus ``align`` remains an explicit experiment: on pinned vLLM 0.27.1
# a repeated one-token prompt is not deterministic even with the k+1 dispatch
# guard. Setting only DENSESPARK_PREFIX_CACHING=1 selects ``align``; explicit
# mode/cache combinations must remain coherent.
PREFIX_CACHING="${DENSESPARK_PREFIX_CACHING:-0}"
if [ -n "${DENSESPARK_MAMBA_CACHE_MODE:-}" ]; then
    MAMBA_CACHE_MODE="$DENSESPARK_MAMBA_CACHE_MODE"
elif [ "$PREFIX_CACHING" = "1" ]; then
    MAMBA_CACHE_MODE="align"
else
    MAMBA_CACHE_MODE="none"
fi

# PQ draft head. The structure is a 33.3 MiB file built once by
# patches/04-pq-draft-head/build_pq_artifact.py; it is mounted rather than baked
# in so the image stays independent of the checkpoint. The runtime refuses to
# use a structure that was not trained on the head it finds loaded.
PQ_ARTIFACT="${DENSESPARK_PQ_ARTIFACT_HOST:-$HOME/.cache/densespark/pq_head_m128.pt}"

# The API server has no authentication of its own, so it is published on the
# loopback interface and reaching it from another machine is a decision the
# operator makes deliberately. Set DENSESPARK_BIND_HOST to an interface address,
# or 0.0.0.0 for all of them, when that is what you want.
BIND_HOST="${DENSESPARK_BIND_HOST:-127.0.0.1}"

# Tool calling. These two are capability gates, not a mode: vLLM only reaches
# the tool-parsing branch when a request actually carries a "tools" field, so a
# plain chat request is unaffected and nothing about generation changes. Without
# them an agent client asking for tool_choice "auto" gets
#   400 "auto" tool choice requires --enable-auto-tool-choice and
#   --tool-call-parser to be set
# and cannot use the server at all.
#
# qwen3_xml matches this checkpoint's own chat template, which emits
# <tool_call><function=NAME><parameter=NAME> rather than JSON inside the tag.
TOOL_CALL_PARSER="${DENSESPARK_TOOL_CALL_PARSER:-qwen3_xml}"

# Escape hatch for sweeping a flag that does not have a knob of its own yet.
# Word-split on purpose, e.g. DENSESPARK_EXTRA_ARGS="--max-num-batched-tokens 8192".
EXTRA_ARGS="${DENSESPARK_EXTRA_ARGS:-}"

# Marlin column-block dispatch. GB10's L2 is 24 MiB and Marlin re-reads its
# packed weight once per row tile, so the merged 34816-wide gate_up projection
# (85 MiB) runs at 43% of the bf16 tensor ceiling while L2-resident blocks reach
# 67-70%. The limit is the maximum output columns per Marlin call; 0 is off and
# is the default so an unset environment reproduces stock behaviour. The split
# only engages at or above MIN_M rows, because below the measured crossover it
# costs more launches than it saves traffic.
MARLIN_NSPLIT="${DENSESPARK_MARLIN_NSPLIT:-0}"
MARLIN_NSPLIT_MIN_M="${DENSESPARK_MARLIN_NSPLIT_MIN_M:-256}"

launch() {
    local profile="$1"; shift
    local tuning=()
    [ -n "$ATTN_BACKEND" ]  && tuning+=(--attention-backend "$ATTN_BACKEND")
    [ -n "$MAMBA_BACKEND" ] && tuning+=(--mamba-backend "$MAMBA_BACKEND")
    [ -n "$KV_DTYPE" ]      && tuning+=(--kv-cache-dtype "$KV_DTYPE")
    case "${PREFIX_CACHING}:${MAMBA_CACHE_MODE}" in
        1:align) tuning+=(--enable-prefix-caching --mamba-cache-mode align) ;;
        0:none) tuning+=(--no-enable-prefix-caching --mamba-cache-mode none) ;;
        *)
            echo "DENSESPARK_PREFIX_CACHING and DENSESPARK_MAMBA_CACHE_MODE must be 0:none or 1:align" >&2
            return 2
            ;;
    esac
    # shellcheck disable=SC2206  # word splitting is the point here
    [ -n "$EXTRA_ARGS" ]    && tuning+=($EXTRA_ARGS)
    printf 'DenseSpark: %s\n  image %s\n  model %s\n  context %s\n  tuning %s\n' \
        "$profile" "$IMAGE" "$MODEL" "$MAX_LEN" "${tuning[*]:-(vLLM defaults)}"
    local mounts=()
    if [ "${DENSESPARK_PQ_DRAFT:-0}" = "1" ]; then
        if [ ! -f "$PQ_ARTIFACT" ]; then
            echo "DENSESPARK: DENSESPARK_PQ_DRAFT=1 but no artifact at $PQ_ARTIFACT" >&2
            echo "  run ./install.sh to build it, or set DENSESPARK_PQ_DRAFT=0" >&2
            return 1
        fi
        mounts+=(-v "${PQ_ARTIFACT}:/opt/densespark/pq_head_m128.pt:ro")
    fi
    # A --rm container otherwise starts with an empty torch.compile cache.
    # Mounting this cache was the tested differentiator: five consecutive
    # restarts produced byte-identical greedy output and E to four decimal
    # places, while two empty-cache restarts agreed on only 2 of 6 prompts and
    # E ranged 4.05 to 4.44. The experiment did not isolate the underlying
    # compiler or fusion mechanism. Set DENSESPARK_VLLM_CACHE= (empty) to opt out.
    if [ -n "${DENSESPARK_VLLM_CACHE-$HOME/.cache/densespark-vllm}" ]; then
        DENSESPARK_VLLM_CACHE="${DENSESPARK_VLLM_CACHE-$HOME/.cache/densespark-vllm}"
        # The column-block dispatch changes the traced Python inside the
        # compiled region, but vLLM's AOT cache key covers the engine config and
        # the vLLM source, not this environment variable. Sharing one cache
        # directory between the two settings makes the second launch silently
        # replay the first one's graph, which turns any A/B into a measurement
        # of the same code twice. Give each setting its own directory.
        # Backend and scheduler flags can also change the compiled graph. The
        # old suffix covered local Python dispatch knobs only, so a Humming run
        # could replay an AOT graph created by Marlin. Bind every tuning flag to
        # the cache path; DENSESPARK_CACHE_VARIANT remains an explicit escape
        # hatch for image-local changes not represented in argv/environment.
        local tuning_cache_material tuning_cache_sha lab89_cache_abi
        lab89_cache_abi="off"
        if [ "${DENSESPARK_LAB89_HYBRID_LINEAR:-0}" = "1" ]; then
            # cond-v2 is intentionally part of the path. The original Lab 89
            # Python branch was constant-folded by AOT, and replaying that
            # cache would make a corrected image look hybrid while still
            # executing Humming for every M.
            lab89_cache_abi="cond-v2"
        fi
        tuning_cache_material="${ATTN_BACKEND}|${MAMBA_BACKEND}|${KV_DTYPE}|${PREFIX_CACHING}|${MAMBA_CACHE_MODE}|${EXTRA_ARGS}|${DENSESPARK_CACHE_VARIANT:-default}|${DENSESPARK_LAB89_HYBRID_LINEAR:-0}|${DENSESPARK_LAB89_HUMMING_MIN_M:-256}|${lab89_cache_abi}"
        if [ "${DENSESPARK_LAB90_EXACT_SAMPLER:-0}" = "1" ]; then
            tuning_cache_material="${tuning_cache_material}|lab90-exact-v1"
        fi
        tuning_cache_sha="$(printf '%s' "$tuning_cache_material" | sha256sum)"
        tuning_cache_sha="${tuning_cache_sha%% *}"
        tuning_cache_sha="${tuning_cache_sha:0:12}"
        DENSESPARK_VLLM_CACHE="${DENSESPARK_VLLM_CACHE}/tuning-${tuning_cache_sha}-nsplit-${MARLIN_NSPLIT}"\
"-head-interleaved-${DENSESPARK_HEAD_INTERLEAVED:-1}"\
"-head-autotune-${DENSESPARK_HEAD_AUTOTUNE:-1}"\
"-draft-match-${DENSESPARK_DRAFT_MATCH_FILTERS:-0}"
        mkdir -p "$DENSESPARK_VLLM_CACHE"
        printf '  compile cache %s\n' "$DENSESPARK_VLLM_CACHE"
        mounts+=(-v "${DENSESPARK_VLLM_CACHE}:/root/.cache/vllm")
    fi
    # vLLM reads a number of knobs straight from the environment rather than
    # from a server flag - VLLM_MARLIN_INPUT_DTYPE and VLLM_DISABLED_KERNELS
    # among them. Forward whatever the caller exported so a sweep does not need
    # an edit here for every new one.
    local vllm_env=()
    local name
    for name in $(compgen -e | grep '^VLLM_' | sort || true); do
        vllm_env+=(-e "${name}=${!name}")
    done
    [ ${#vllm_env[@]} -gt 0 ] && printf '  vllm env %s\n' "${vllm_env[*]}"

    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    exec docker run --rm --name "$CONTAINER" \
        --gpus all \
        "${mounts[@]}" \
        --ipc=host \
        -p "${BIND_HOST}:${PORT}:8000" \
        -v "${HF_HOME_HOST}:/root/.cache/huggingface" \
        -e "DENSESPARK_INT8_LMHEAD=${DENSESPARK_INT8_LMHEAD:-1}" \
        -e "DENSESPARK_HEAD_BATCH_DOT=${DENSESPARK_HEAD_BATCH_DOT:-1}" \
        -e "DENSESPARK_HEAD_AUTOTUNE=${DENSESPARK_HEAD_AUTOTUNE:-1}" \
        -e "DENSESPARK_HEAD_CHUNK16=${DENSESPARK_HEAD_CHUNK16:-1}" \
        -e "DENSESPARK_HEAD_INTERLEAVED=${DENSESPARK_HEAD_INTERLEAVED:-1}" \
        -e "DENSESPARK_PQ_DRAFT=${DENSESPARK_PQ_DRAFT:-0}" \
        -e "DENSESPARK_PQ_CANDIDATES=${DENSESPARK_PQ_CANDIDATES:-2048}" \
        -e "DENSESPARK_PQ_BATCH_SCAN=${DENSESPARK_PQ_BATCH_SCAN:-1}" \
        -e "DENSESPARK_LAB86_SPARSE_PQ=${DENSESPARK_LAB86_SPARSE_PQ:-0}" \
        -e "DENSESPARK_LAB89_HYBRID_LINEAR=${DENSESPARK_LAB89_HYBRID_LINEAR:-0}" \
        -e "DENSESPARK_LAB89_HUMMING_MIN_M=${DENSESPARK_LAB89_HUMMING_MIN_M:-256}" \
        -e "DENSESPARK_LAB113_ENABLE=${DENSESPARK_LAB113_ENABLE:-0}" \
        -e "DENSESPARK_LAB118_ENABLE=${DENSESPARK_LAB118_ENABLE:-0}" \
        -e "DENSESPARK_LAB118_RUNTIME_AUDIT=${DENSESPARK_LAB118_RUNTIME_AUDIT:-}" \
        -e "DENSESPARK_LAB133_M8000=${DENSESPARK_LAB133_M8000:-0}" \
        -e "DENSESPARK_LAB90_EXACT_SAMPLER=${DENSESPARK_LAB90_EXACT_SAMPLER:-0}" \
        -e "DENSESPARK_DRAFT_MATCH_FILTERS=${DENSESPARK_DRAFT_MATCH_FILTERS:-0}" \
        -e "DENSESPARK_PROFILE=${profile}" \
        -e "DENSESPARK_MARLIN_NSPLIT=${MARLIN_NSPLIT}" \
        -e "DENSESPARK_MARLIN_NSPLIT_MIN_M=${MARLIN_NSPLIT_MIN_M}" \
        -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}" \
        "${vllm_env[@]}" \
        "$IMAGE" serve "$MODEL" \
            --served-model-name "$SERVED_NAME" \
            --max-model-len "$MAX_LEN" \
            --tool-call-parser "$TOOL_CALL_PARSER" \
            --enable-auto-tool-choice \
            --gpu-memory-utilization "$GPU_UTIL" \
            --limit-mm-per-prompt "$LIMIT_MM" \
            "${tuning[@]}" \
            "$@"
}
