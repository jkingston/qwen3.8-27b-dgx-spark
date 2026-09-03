"""Split a wide Marlin projection into column blocks that fit the L2 cache.

WHY
---
On GB10 the L2 is 24 MiB. Marlin re-reads its packed weight once per row tile,
so a projection whose weight exceeds L2 streams that weight from DRAM on every
row tile, while a narrower one is served from L2 after the first pass. Measured
on this part at K=5120, group 128, with L2 evicted before every timed call, as a
percentage of the 125.24 TFLOP/s bf16 tensor ceiling:

    N        weight     M=2048      M=16000
    2048      5.0 MiB    77.6%       70.9%
    4096     10.0 MiB    73.1%       72.0%
    6144     15.0 MiB    70.9%       70.1%
    8192     20.0 MiB    67.7%       67.8%
    10240    25.0 MiB    57.7%       59.6%     <- crosses the 24 MiB L2
    12288    30.0 MiB    45.2%       45.3%
    17408    42.5 MiB    43.0%       43.8%
    34816    85.0 MiB    42.5%       42.8%

The cliff sits exactly on the cache size, and it is a within-call effect: the
numbers above are unchanged whether or not L2 is flushed between calls.

Qwen3.8-27B lands on the wrong side of it. vLLM merges gate_proj and up_proj
into one 34,816-wide linear in all 64 layers, and that single projection is the
largest share of prefill GPU time. Splitting it back into blocks the cache can
hold is worth 1.38-1.42x on the projection at M >= 256, measured through the
real ``apply_gptq_marlin_linear`` path with the concatenation cost included.

The split is not free at decode. Same measurement, same shape, full versus an
eight-way split:

    M        full      split      decision
    1        0.4286    0.4549     full
    16       0.4373    0.4820     full
    64       0.4557    0.5621     full
    112      0.9063    0.9037     tie
    256      1.7849    1.2910     split, 1.383x
    2048    13.8638    9.7733     split, 1.419x

so the choice has to be made per call, from the row count. Both weight layouts
are therefore kept resident. That costs one extra copy of every split
projection - about 7.2 GiB for this model - which the 128 GiB unified memory
absorbs. Set ``DENSESPARK_MARLIN_NSPLIT=0`` to opt out entirely.

WHAT IT DOES NOT DO
-------------------
The split is bit-exact in its inputs but not in its output: a narrower GEMM
picks a different reduction geometry, and the measured difference against the
unsplit call is 1.88e-05 relative, 1.56e-02 absolute at bf16. That is ULP-level
for this dtype but it is not zero, so enabling the split changes generated text
exactly the way any other kernel change does. Treat it as an A/B, not a
transparent speedup.

Only the 16-bit activation path is eligible. With ``VLLM_MARLIN_INPUT_DTYPE``
set to int8 or fp8 the repacked weight uses a different permutation that does
not slice along N - verified, not assumed - so the split declines and the layer
keeps its single call. Layers with act-order, zero points, tile padding, a
bias, or a non-4-bit weight type are also left alone.

ONE HAZARD WORTH KNOWING
------------------------
The split happens in Python inside the region vLLM compiles, and vLLM's AOT
compilation cache is keyed on the engine configuration and the vLLM source, not
on this module's environment variables. A server that finds a cached graph will
replay it - "Directly load AOT compilation from path ..." in the log - and the
toggle will have no effect at all, in either direction. Observed: an A/B run
with a shared cache produced 146.25 against 146.10 output tok/s, a 0.10%
difference, because both arms executed the same unsplit graph. Give each setting
its own cache directory, which is what configs/_common.sh now does, or clear the
cache between settings. When the split is live the log carries one
"nsplit: KxN -> P blocks of B columns" line per eligible shape AND the startup
must show "Compiling model again", not "Directly load AOT compilation".

TOGGLES
-------
    DENSESPARK_MARLIN_NSPLIT        maximum output columns per Marlin call.
                                    0 (default) disables the patch entirely.
    DENSESPARK_MARLIN_NSPLIT_MIN_M  minimum row count for the split path.
                                    Default 256, the measured crossover.
"""

import os

import torch

MAX_PARTS = 16

_logged = set()


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def column_limit():
    """Maximum output columns a single Marlin call may cover. 0 disables."""
    return _env_int("DENSESPARK_MARLIN_NSPLIT", 0)


def min_rows():
    """Row count at or above which the split path is used."""
    return _env_int("DENSESPARK_MARLIN_NSPLIT_MIN_M", 256)


def choose_parts(size_n, limit, max_parts=MAX_PARTS):
    """Fewest equal column blocks of at most ``limit`` columns.

    Every block must divide ``size_n`` exactly and stay a multiple of the 64
    column tile Marlin repacks with, so that slicing the repacked tensor lands
    on a tile boundary. Returns 1 when no such split exists, which is the
    signal to leave the layer alone.
    """
    if limit <= 0 or size_n <= limit:
        return 1
    for parts in range(2, max_parts + 1):
        if size_n % parts:
            continue
        block = size_n // parts
        if block <= limit and block % 64 == 0:
            return parts
    return 1


def eligible(config, has_bias):
    """Whether a layer's static configuration permits an N split."""
    if has_bias:
        # The permuted bias would have to be sliced too. No projection in the
        # target model carries one, so this stays unimplemented rather than
        # unverified.
        return False, "bias"
    act = getattr(config, "act_type", None)
    if act is not None and getattr(act, "itemsize", 2) != 2:
        return False, "8-bit activations use a different weight permutation"
    if getattr(config, "has_g_idx", False):
        return False, "act-order"
    if getattr(config, "zero_points", False):
        return False, "zero points"
    if getattr(config.weight_type, "size_bits", 0) != 4:
        return False, "weight type is not 4-bit"
    return True, None


def attach(kernel, layer, repacked_nk, logger=None):
    """Pre-slice a wide projection after the stock weight preparation.

    Leaves ``kernel`` untouched and returns False whenever anything about the
    layer is not exactly what the slicing was verified against.
    """
    limit = column_limit()
    if limit <= 0:
        return False

    config = kernel.config
    bias = getattr(layer, "bias", None)
    ok, reason = eligible(config, bias is not None)
    if not ok:
        _log_once(logger, f"nsplit: skipping a layer ({reason})")
        return False

    size_k, size_n = config.partition_weight_shape
    parts = choose_parts(size_n, limit)
    if parts < 2:
        return False

    q = getattr(layer, kernel.w_q_name).data
    s = getattr(layer, kernel.w_s_name).data

    # The repacked tensor has to describe exactly this (n, k). Tile padding
    # would put the real columns somewhere other than where the slice looks.
    packed_n, packed_k = repacked_nk(q, 4)
    if (packed_n, packed_k) != (size_n, size_k):
        _log_once(logger, "nsplit: skipping a padded layer")
        return False
    if q.dim() != 2 or s.dim() != 2:
        return False
    if q.shape[1] % parts or s.shape[1] != size_n:
        return False

    block_n = size_n // parts
    block_q = q.shape[1] // parts
    weights = [q[:, i * block_q:(i + 1) * block_q].contiguous() for i in range(parts)]
    scales = [s[:, i * block_n:(i + 1) * block_n].contiguous() for i in range(parts)]

    kernel._ds_nsplit = (block_n, weights, scales)
    _log_once(
        logger,
        f"nsplit: {size_k}x{size_n} -> {parts} blocks of {block_n} columns",
    )
    return True


def apply_split(kernel, layer, x, apply_linear):
    """Run the pre-sliced blocks and rejoin them, or return None to fall back."""
    plan = getattr(kernel, "_ds_nsplit", None)
    if plan is None:
        return None
    rows = x.numel() // x.shape[-1]
    if rows < min_rows():
        return None

    block_n, weights, scales = plan
    config = kernel.config
    empty = layer.g_idx_sort_indices
    zp = getattr(layer, kernel.w_zp_name)
    gidx = getattr(layer, kernel.w_gidx_name)
    outputs = [
        apply_linear(
            input=x,
            weight=weights[i],
            weight_scale=scales[i],
            weight_zp=zp,
            g_idx=gidx,
            g_idx_sort_indices=empty,
            workspace=kernel.workspace,
            wtype=config.weight_type,
            input_size_per_partition=config.partition_weight_shape[0],
            output_size_per_partition=block_n,
            is_k_full=kernel.is_k_full,
            input_global_scale=None,
            bias=None,
            input_dtype=config.act_type,
        )
        for i in range(len(weights))
    ]
    return torch.cat(outputs, dim=-1)


def _log_once(logger, message):
    if logger is None or message in _logged:
        return
    _logged.add(message)
    try:
        logger.info(message)
    except Exception:
        pass
