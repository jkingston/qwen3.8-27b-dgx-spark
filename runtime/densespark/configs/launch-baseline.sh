#!/usr/bin/env bash
# Baseline: stock vLLM behaviour. No speculative decoding, stock BF16 LM head.
# Every other profile is measured against this one.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./_common.sh
DENSESPARK_INT8_LMHEAD=0 launch "baseline (stock head, no speculation)"
