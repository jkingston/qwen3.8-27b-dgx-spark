#!/usr/bin/env bash
# INT8 LM head plus Marlin column-block dispatch. Isolates the prefill
# optimization: every projection wider than 8192 output columns is issued as
# equal blocks that fit GB10's 24 MiB L2, but only for calls with at least 256
# rows, so decode keeps the single-call path bit-for-bit.
#
# Measured on the merged 34816-wide gate_up projection, L2 evicted before every
# timed call: 13.881 ms unsplit against 9.935 ms split at M=2048, and identical
# output at M<=128.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
DENSESPARK_MARLIN_NSPLIT="${DENSESPARK_MARLIN_NSPLIT:-8192}"
export DENSESPARK_MARLIN_NSPLIT
source ./_common.sh
DENSESPARK_INT8_LMHEAD=1 launch "int8-head-marlin-nsplit"
