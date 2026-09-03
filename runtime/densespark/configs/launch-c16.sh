#!/usr/bin/env bash
# Concurrent-throughput profile. This is the C=16 measured winner: INT8 LM head,
# no speculation, Marlin column-block dispatch, and a bfloat16 GDN recurrent
# state.
#
# Measured on the forum protocol at C=16, best retained run against one
# published NVFP4 observation each:
#
#   Decode-heavy 1000/8000   155.19 tok/s   against 132.07   won on both runs
#   Balanced     1000/1000   161.47 tok/s   against 134.41   won once warm
#   Prompt-heavy 8000/1000    71.63 tok/s   against  87.91   lost
#
# The bfloat16 recurrent state is the reason this profile is separate rather
# than the default. The checkpoint asks for float32 and vLLM warns when it is
# overridden; the change is worth +10.9% throughput and -11.3% TPOT, and its
# quality evidence is six prompts that showed content-preserving divergence,
# which is a sanity check and not a task-accuracy benchmark.
# Drop DENSESPARK_C16_SSM_BF16=0 to serve the same profile with the state the
# checkpoint asks for.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

DENSESPARK_MARLIN_NSPLIT="${DENSESPARK_MARLIN_NSPLIT:-8192}"
export DENSESPARK_MARLIN_NSPLIT

extra="--max-num-batched-tokens 8192"
if [ "${DENSESPARK_C16_SSM_BF16:-1}" != "0" ]; then
    extra="$extra --mamba-ssm-cache-dtype bfloat16"
fi
DENSESPARK_EXTRA_ARGS="${DENSESPARK_EXTRA_ARGS:-$extra}"
export DENSESPARK_EXTRA_ARGS

source ./_common.sh
DENSESPARK_INT8_LMHEAD=1 launch "c16-throughput"
