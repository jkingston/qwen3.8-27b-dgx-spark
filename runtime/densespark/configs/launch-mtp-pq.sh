#!/usr/bin/env bash
# MTP speculation with the product-quantized draft head, plus the INT8 LM head.
#
# The MTP proposer's argmax is the second-largest weight read in the step: one
# full pass over the 1.271 GB (1.184 GiB) INT8 head per proposed token, which is about 70%
# of the marginal cost of a chain position. A proposal is not an emission - the
# target verifies it and the rejection sampler emits the target's argmax - so
# the draft path is the one place in the step where an approximate head is
# admissible.
#
# The PQ head scans 33.3 MiB of structure and applies the deployed gathered-INT8
# reduction to the top 2048 candidates. In the production replay it matches the
# exact proposal on 2400 of 2409 captured calls, and one call takes about
# 0.385 ms instead of 5.50 ms for the exact head.
#
# Needs the structure at ~/.cache/densespark/pq_head_m128.pt; build it with
# patches/04-pq-draft-head/build_pq_artifact.py.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Every knob that _common.sh snapshots at source time has to be set and exported
# BEFORE the source, or a normal launch silently runs without it. This profile
# used to omit the three the concurrent-throughput profile sets, which made any
# comparison against launch-c16.sh a comparison of speculation minus every
# prefill optimization against a fully optimized baseline.
DENSESPARK_MARLIN_NSPLIT="${DENSESPARK_MARLIN_NSPLIT:-8192}"
export DENSESPARK_MARLIN_NSPLIT

extra="--max-num-batched-tokens 8192"
if [ "${DENSESPARK_MTP_SSM_BF16:-1}" != "0" ]; then
    extra="$extra --mamba-ssm-cache-dtype bfloat16"
fi
DENSESPARK_EXTRA_ARGS="${DENSESPARK_EXTRA_ARGS:-$extra}"
export DENSESPARK_EXTRA_ARGS

source ./_common.sh
# Depth 6 was tuned at C=1. The measured optimum falls as concurrency rises -
# at C=16 depth 1 is the stable winner and depth 6 is a large regression - so
# set DENSESPARK_SPEC_TOKENS explicitly for any concurrent profile.
SPEC_TOKENS="${DENSESPARK_SPEC_TOKENS:-6}"
export DENSESPARK_PQ_DRAFT="${DENSESPARK_PQ_DRAFT:-1}"
launch "mtp-${SPEC_TOKENS} + pq-draft-head + int8-lm-head" \
    --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${SPEC_TOKENS}}"
