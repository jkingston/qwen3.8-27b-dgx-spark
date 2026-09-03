#!/usr/bin/env bash
# Speculative decoding with the model's own MTP head, plus the INT8 LM head.
#
# The checkpoint declares mtp_num_hidden_layers=1, so vLLM reuses that single
# module to propose DENSESPARK_SPEC_TOKENS tokens per step. On a dense model the
# draft is cheap relative to the target read, which is the opposite of the
# mixture-of-experts case where a large BF16 draft head ate the win.
#
# Six draft tokens is the measured production setting. It amortizes the target
# pass without extending the chain past the point where acceptance saturates.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./_common.sh
SPEC_TOKENS="${DENSESPARK_SPEC_TOKENS:-6}"
DRAFT_SAMPLE_METHOD="${DENSESPARK_DRAFT_SAMPLE_METHOD:-}"
case "$DRAFT_SAMPLE_METHOD" in
    "") SAMPLE_JSON="" ;;
    greedy|probabilistic)
        SAMPLE_JSON=", \"draft_sample_method\": \"${DRAFT_SAMPLE_METHOD}\""
        ;;
    *)
        echo "DENSESPARK_DRAFT_SAMPLE_METHOD must be greedy or probabilistic" >&2
        exit 2
        ;;
esac
launch "mtp-${SPEC_TOKENS} + int8-lm-head" \
    --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${SPEC_TOKENS}${SAMPLE_JSON}}"
