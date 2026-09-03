#!/usr/bin/env bash
# INT8 LM head only. Isolates optimization 1: the 248320 x 5120 head drops from
# 2.54 GB to 1.27 GB per decoded token.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./_common.sh
DENSESPARK_INT8_LMHEAD=1 launch "int8-lm-head"
