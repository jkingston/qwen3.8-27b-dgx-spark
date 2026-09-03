#!/usr/bin/env bash
# The DenseSpark profile: one launcher carrying every measured decision.
#
# The other launchers in this directory each isolate one intervention, which is
# what made them useful while the interventions were being decided. This one is
# the composition, and it is the only launcher whose numbers the study reports
# as a profile rather than as a contrast.
#
# Every component here is quality-neutral against baseline AutoRound INT4 with
# MTP. That is a measured property, not an intention: over 23,997 tokens of
# 8,000-token prompts, the composed stack sits 0.00051 nat/token from exact
# dequantization, the FlashInfer GDN prefill backend contributes -0.000487 and
# the M=8000 route -0.000700. Structured NVFP4 is deliberately absent - it buys
# prefill latency for 0.0159 nat/token, thirty times the whole stack's distance
# from exact, and the fused SwiGLU is absent with it because it fuses into the
# NVFP4 quantizer and cannot be taken separately.
#
# This profile is not one configuration. Three settings are selected by the
# concurrency the deployment is tuned for, and the block that defines those
# tables below is the single place their policy is stated - which values are
# measured, which are still provisional, and what each one was measured against.
# Override any of them per deployment with the matching DENSESPARK_ variable.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# _common.sh snapshots these at source time, so they must be exported first.
DENSESPARK_MARLIN_NSPLIT="${DENSESPARK_MARLIN_NSPLIT:-0}"
DENSESPARK_MARLIN_NSPLIT_MIN_M="${DENSESPARK_MARLIN_NSPLIT_MIN_M:-256}"
DENSESPARK_LAB89_HYBRID_LINEAR="${DENSESPARK_LAB89_HYBRID_LINEAR:-1}"
DENSESPARK_LAB89_HUMMING_MIN_M="${DENSESPARK_LAB89_HUMMING_MIN_M:-256}"
DENSESPARK_LAB113_ENABLE="${DENSESPARK_LAB113_ENABLE:-1}"
DENSESPARK_LAB118_ENABLE="${DENSESPARK_LAB118_ENABLE:-1}"
DENSESPARK_LAB133_M8000="${DENSESPARK_LAB133_M8000:-1}"
DENSESPARK_INT8_LMHEAD="${DENSESPARK_INT8_LMHEAD:-1}"
DENSESPARK_HEAD_BATCH_DOT="${DENSESPARK_HEAD_BATCH_DOT:-1}"
DENSESPARK_HEAD_AUTOTUNE="${DENSESPARK_HEAD_AUTOTUNE:-0}"
DENSESPARK_HEAD_CHUNK16="${DENSESPARK_HEAD_CHUNK16:-1}"
DENSESPARK_HEAD_INTERLEAVED="${DENSESPARK_HEAD_INTERLEAVED:-1}"
DENSESPARK_PQ_DRAFT="${DENSESPARK_PQ_DRAFT:-1}"
DENSESPARK_PQ_CANDIDATES="${DENSESPARK_PQ_CANDIDATES:-2048}"
DENSESPARK_PQ_BATCH_SCAN="${DENSESPARK_PQ_BATCH_SCAN:-1}"
DENSESPARK_DRAFT_MATCH_FILTERS="${DENSESPARK_DRAFT_MATCH_FILTERS:-0}"
export DENSESPARK_MARLIN_NSPLIT DENSESPARK_MARLIN_NSPLIT_MIN_M \
       DENSESPARK_LAB89_HYBRID_LINEAR \
       DENSESPARK_LAB89_HUMMING_MIN_M DENSESPARK_LAB113_ENABLE \
       DENSESPARK_LAB118_ENABLE DENSESPARK_LAB133_M8000 \
       DENSESPARK_INT8_LMHEAD DENSESPARK_HEAD_BATCH_DOT \
       DENSESPARK_HEAD_AUTOTUNE DENSESPARK_HEAD_CHUNK16 \
       DENSESPARK_HEAD_INTERLEAVED DENSESPARK_PQ_DRAFT \
       DENSESPARK_PQ_CANDIDATES DENSESPARK_PQ_BATCH_SCAN \
       DENSESPARK_DRAFT_MATCH_FILTERS

# _common.sh already emits --mamba-cache-mode and --no-enable-prefix-caching
# from DENSESPARK_MAMBA_CACHE_MODE and DENSESPARK_PREFIX_CACHING, whose defaults
# are the ones this profile wants. Repeating them here only produced a duplicate
# -key warning from vLLM.
extra="--max-num-batched-tokens 8192 --mamba-ssm-cache-dtype bfloat16"
extra="$extra --linear-backend humming --gdn-prefill-backend flashinfer"
# This model thinks before it answers, and its chat template opens the <think>
# block itself, so the model only ever emits the closing tag. Without a parser
# every OpenAI-compatible client shows the reasoning as part of the reply, with
# a stray </think> in the middle of it. The parser moves it to
# reasoning_content, which clients collapse. It formats the chat response and
# does not touch generation, but it does change what a streaming client sees:
# the trace is buffered until its closing tag, so a request whose budget runs
# out first receives nothing at all. bench_densespark.py measures against
# /v1/chat/completions and therefore turns the reasoning mode off per request.
extra="$extra --reasoning-parser qwen3"
export VLLM_HUMMING_INPUT_QUANT_CONFIG="${VLLM_HUMMING_INPUT_QUANT_CONFIG:-{\"dtype\":\"int8\"}}"

# Speculation depth by expected concurrency. Depth costs a forward pass per
# proposed token and pays only while the target accepts, so the profitable depth
# falls as concurrency rises. DENSESPARK_CONCURRENCY selects the row;
# DENSESPARK_SPEC_TOKENS overrides it outright.
# install.sh asks once for the maximum concurrency this deployment serves and
# writes it here. An explicit DENSESPARK_CONCURRENCY still wins.
_saved_concurrency="${XDG_CONFIG_HOME:-$HOME/.config}/densespark/concurrency"
if [ -z "${DENSESPARK_CONCURRENCY:-}" ] && [ -r "${_saved_concurrency}" ]; then
    IFS= read -r _saved < "${_saved_concurrency}" || _saved=""
    case "${_saved}" in
        ''|*[!0-9]*) ;;
        *) DENSESPARK_CONCURRENCY="${_saved}" ;;
    esac
fi

# Three knobs are selected per concurrency, each as a 16-wide table read with
# the same row.
#
# Depth, measured, and the confidence differs by row.
#
#   C=1 -> 8. Fifteen measurements per arm on the five-prompt single-stream
#   benchmark: 57.76 cross-prompt mean at depth 8 against 55.21 at 6 and 55.88
#   at 10. The maximum is interior, so it is not an artefact of where the grid
#   stopped, and it clears the 2026-08-25 record of 53.9.
#
#   C=2 -> 6. Weaker. The eighty-cell sweep puts depth 6 at 81.94 and depth 8 at
#   77.07 against depth 3's 71.81, so both deep chains beat the shallow one and
#   the mechanism says they should at low concurrency - a long proposal pays off
#   exactly when the machine would otherwise wait on one request. But that cell's
#   own run-to-run spread is 59%, so the margin is not established, only the
#   direction.
#
#   C>=3 -> 3. At three requests the deep chains lose outright, 66.67 at depth 6
#   against 77.29 at depth 3, and depth 3 wins clearly again from 13 to 16 by 4.0
#   to 16.2%. Between those it trails the best cell by 1 to 2.8%, inside spreads
#   of up to 17%, so a per-C argmax there would be fitted to noise.
#
# Sparse PQ: OFF everywhere, and not as a tuning choice. The sparse draft head
# samples from a sparse q, so its rejection step needs real draft_probs, and it
# raises rather than approximate when a batch is greedy. A greedy request is
# the normal case for deterministic output, and the raise kills the engine
# process, not just the request. It was also slower than the plain head at C=1
# in every paired cell, so nothing is being given up by removing it.
#
# Draft sampling: probabilistic, which with the plain head is safe on greedy
# batches and measured slightly ahead at C=1 (58.25/55.46 against 57.99/51.47).
DENSESPARK_SPEC_TOKENS_BY_C="${DENSESPARK_SPEC_TOKENS_BY_C:-8,6,3,3,3,3,3,3,3,3,3,3,3,3,3,3}"
DENSESPARK_DRAFT_SAMPLE_BY_C="${DENSESPARK_DRAFT_SAMPLE_BY_C:-probabilistic,probabilistic,probabilistic,probabilistic,probabilistic,probabilistic,probabilistic,probabilistic,probabilistic,probabilistic,probabilistic,probabilistic,probabilistic,probabilistic,probabilistic,probabilistic}"
DENSESPARK_SPARSE_PQ_BY_C="${DENSESPARK_SPARSE_PQ_BY_C:-0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}"
concurrency="${DENSESPARK_CONCURRENCY:-16}"
case "${concurrency}" in
    ''|*[!0-9]*) echo "DENSESPARK_CONCURRENCY must be a positive integer" >&2; exit 2 ;;
esac
if [ "${concurrency}" -lt 1 ]; then
    echo "DENSESPARK_CONCURRENCY must be at least 1" >&2
    exit 2
fi
if [ "${concurrency}" -gt 16 ]; then
    _row=15
else
    _row=$((concurrency - 1))
fi
# Read in the current shell, never in a command substitution: a width check
# that runs in a subshell can only exit the subshell, so the launcher would
# print the error and then serve with an empty setting.
_table_entry() {
    local name="$1" value="$2"
    local -a cells
    IFS=',' read -r -a cells <<< "${value}"
    if [ "${#cells[@]}" -ne 16 ]; then
        echo "${name} must list exactly 16 entries, one per concurrency" >&2
        exit 2
    fi
    _entry="${cells[${_row}]}"
    if [ -z "${_entry}" ]; then
        echo "${name} has an empty entry for concurrency ${concurrency}" >&2
        exit 2
    fi
}
_table_entry DENSESPARK_SPEC_TOKENS_BY_C "${DENSESPARK_SPEC_TOKENS_BY_C}"
_default_depth="${_entry}"
_table_entry DENSESPARK_DRAFT_SAMPLE_BY_C "${DENSESPARK_DRAFT_SAMPLE_BY_C}"
_default_draft="${_entry}"
_table_entry DENSESPARK_SPARSE_PQ_BY_C "${DENSESPARK_SPARSE_PQ_BY_C}"
_default_sparse="${_entry}"
DENSESPARK_LAB86_SPARSE_PQ="${DENSESPARK_LAB86_SPARSE_PQ:-${_default_sparse}}"
export DENSESPARK_LAB86_SPARSE_PQ
DRAFT_SAMPLE_METHOD="${DENSESPARK_DRAFT_SAMPLE_METHOD:-${_default_draft}}"
case "${DRAFT_SAMPLE_METHOD}" in
    probabilistic|default) ;;
    *) echo "DENSESPARK_DRAFT_SAMPLE_METHOD must be probabilistic or default" >&2; exit 2 ;;
esac
# The concurrency you tune for is not a limit. Capping the running batch at it
# would make traffic above it queue, which sounds better than it measures: at
# one request, depth 8, the cap gives 56.93 cross-prompt mean and 54.53
# token-weighted tok/s against 58.39 and 59.14 without it - it costs 2.6% and
# 8.5% by forbidding the scheduler exactly the batch it wants. So nothing is
# emitted by default and vLLM's own admission limit applies.
#
# Set DENSESPARK_MAX_NUM_SEQS to cap it deliberately - for a shared deployment
# where predictable latency for the requests already running matters more than
# the throughput of the ones waiting.
MAX_NUM_SEQS="${DENSESPARK_MAX_NUM_SEQS:-}"
if [ -n "${MAX_NUM_SEQS}" ]; then
    case "${MAX_NUM_SEQS}" in
        *[!0-9]*) echo "DENSESPARK_MAX_NUM_SEQS must be a positive integer" >&2; exit 2 ;;
    esac
    [ "${MAX_NUM_SEQS}" -ge 1 ] || { echo "DENSESPARK_MAX_NUM_SEQS must be at least 1" >&2; exit 2; }
    extra="$extra --max-num-seqs ${MAX_NUM_SEQS}"
fi
DENSESPARK_EXTRA_ARGS="${DENSESPARK_EXTRA_ARGS:-$extra}"
export DENSESPARK_EXTRA_ARGS

SPEC_TOKENS="${DENSESPARK_SPEC_TOKENS:-${_default_depth}}"
case "${SPEC_TOKENS}" in
    ''|*[!0-9]*) echo "DENSESPARK_SPEC_TOKENS must be a positive integer" >&2; exit 2 ;;
esac
# This profile has no no-speculation mode, and the reason is structural rather
# than a policy choice. The three-way linear dispatch closes its load audit at
# exactly 260 linears: 256 backbone plus the four of one full-attention MTP
# layer. Without a speculative configuration that layer never loads, the count
# stops at 256, the closure never fires and the first forward raises. Refusing
# here is better than starting a server that dies on its first request.
if [ "${SPEC_TOKENS}" -lt 1 ]; then
    echo "this profile requires speculation: the three-way linear dispatch closes" >&2
    echo "its load audit on 260 linears, four of which belong to the MTP layer." >&2
    echo "Set DENSESPARK_SPEC_TOKENS to 1 or more, or use a different profile." >&2
    exit 2
fi

source ./_common.sh

# "default" means: emit no draft_sample_method at all and let vLLM choose. The
# field is not a no-op with a neutral value, so the arm without it has to be the
# arm that omits it.
if [ "${DRAFT_SAMPLE_METHOD}" = "default" ]; then
    _spec_config="{\"method\": \"mtp\", \"num_speculative_tokens\": ${SPEC_TOKENS}}"
else
    _spec_config="{\"method\": \"mtp\", \"num_speculative_tokens\": ${SPEC_TOKENS}, \"draft_sample_method\": \"${DRAFT_SAMPLE_METHOD}\"}"
fi

_pq_label="sparse-pq"
[ "${DENSESPARK_LAB86_SPARSE_PQ}" = "1" ] || _pq_label="pq"
launch "densespark C=${concurrency}, mtp-${SPEC_TOKENS} + ${_pq_label} + ${DRAFT_SAMPLE_METHOD}-draft + int8-head + three-way" \
    --speculative-config "${_spec_config}"
