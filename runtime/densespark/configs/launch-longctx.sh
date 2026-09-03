#!/usr/bin/env bash
# Full 256K context with an FP8 KV cache, everything else as in launch-mtp.sh.
# Only the 16 full-attention layers hold a KV cache; the 48 linear-attention
# layers keep recurrent state instead, which is what makes 256K affordable here.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./_common.sh
DENSESPARK_MAX_LEN="${DENSESPARK_MAX_LEN:-262144}"
MAX_LEN="$DENSESPARK_MAX_LEN"
SPEC_TOKENS="${DENSESPARK_SPEC_TOKENS:-3}"
launch "256K context, fp8 KV, mtp-${SPEC_TOKENS}" \
    --kv-cache-dtype fp8 \
    --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${SPEC_TOKENS}}"
