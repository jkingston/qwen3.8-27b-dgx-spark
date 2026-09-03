"""Lab 113: AOT-opaque Humming/CUTLASS prefill dispatcher.

This module wraps the
pinned vLLM ``HummingLinearKernel`` at import time and is inert unless
``DENSESPARK_LAB113_ENABLE=1``.

For the three projection families confirmed by Lab 109, the load hook expands
the still-canonical symmetric INT4/G128 payload to a second FP8 E4M3,
128x128-block-scaled layout before Humming destroys the canonical layout.  The
stock Humming layout remains resident.  At runtime an opaque ``torch.library``
operator chooses CUTLASS only for the two measured row counts M={8192,16000};
all other M use the unchanged Humming call.

The route intentionally lives *inside* the CUDA implementation of the custom
operator.  Dynamo/AOT therefore sees one symbolic-shape operator and cannot
specialize a Python branch to its example M.  The operator's fake
implementation preserves symbolic M in its output shape.  Direct execution
uses the exact Lab 105 mutable-output ABI and includes activation quantization,
every chunk launch, and the final output write.

Component timing and
numeric confirmation come from immutable Lab 109.  Model-load/AOT integration,
task quality, and Prompt-heavy serving performance are separate gates.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch


ENABLE_ENV = "DENSESPARK_LAB113_ENABLE"
AUDIT_PATH_ENV = "DENSESPARK_LAB113_RUNTIME_AUDIT"
EXPECTED_VLLM_VERSION = "0.27.1"
EXPECTED_HUMMING_VERSION = "0.1.13"
GROUP_SIZE = 128
CHUNK_ROWS = 2048
LAYOUT_CHUNK_N = 1024
DIRECT_M_VALUES = (8192, 16000)
FP8_MAX = 448.0

HUMMING = "humming_w4a8_int8"
DIRECT = "cutlass_direct_output_m2048"
DIRECT_FAMILY_SHAPES: dict[tuple[int, int], str] = {
    (5120, 34816): "mlp.gate_up",
    (5120, 14336): "attn.qkv",
    (5120, 16384): "gdn.qkvz",
}
EXPECTED_FAMILY_COUNTS = {
    "mlp.gate_up": 64,
    "attn.qkv": 16,
    "gdn.qkvz": 48,
}

WEIGHT_BUFFER = "_densespark_lab113_cutlass_weight"
SCALE_BUFFER = "_densespark_lab113_cutlass_scale"
FAMILY_ATTR = "_densespark_lab113_family"

_INSTALLED = False
_LOAD_RECORDS: list[dict[str, Any]] = []
_LOAD_VALIDATED = False
_LIBRARY: torch.library.Library | str | None = None


def _strict_toggle() -> bool:
    value = os.environ.get(ENABLE_ENV, "0")
    if value not in ("0", "1"):
        raise RuntimeError(f"{ENABLE_ENV} must be exactly 0 or 1, observed {value!r}")
    return value == "1"


def flattened_rows(x: torch.Tensor) -> int:
    if x.ndim < 2 or x.shape[-1] <= 0:
        raise RuntimeError(f"Lab113 expected [...,K], observed {tuple(x.shape)}")
    return x.numel() // x.shape[-1]


def route_for_rows(rows: int) -> str:
    if rows <= 0:
        raise RuntimeError(f"Lab113 expected positive M, observed {rows}")
    return DIRECT if rows in DIRECT_M_VALUES else HUMMING


def family_from_config(config: Any) -> str | None:
    shape = tuple(int(value) for value in config.partition_weight_shape)
    return DIRECT_FAMILY_SHAPES.get(shape)


def fp8_layout_bytes(k: int, n: int, count: int = 1) -> dict[str, int]:
    if min(k, n, count) <= 0 or k % GROUP_SIZE or n % GROUP_SIZE:
        raise ValueError("K/N must be positive multiples of 128 and count positive")
    weight = k * n * count
    scale = (k // GROUP_SIZE) * (n // GROUP_SIZE) * 4 * count
    return {"weight": weight, "scale": scale, "total": weight + scale}


def expected_duplicate_layout_memory() -> dict[str, Any]:
    rows = []
    total = 0
    for (k, n), family in DIRECT_FAMILY_SHAPES.items():
        count = EXPECTED_FAMILY_COUNTS[family]
        memory = fp8_layout_bytes(k, n, count)
        rows.append({"family": family, "K": k, "N": n, "count": count, **memory})
        total += memory["total"]
    return {
        "rows": rows,
        "total_bytes": total,
        "total_decimal_GB": total / 1.0e9,
        "total_GiB": total / 2**30,
        "scope": "resident FP8 weight plus FP32 128x128 scale for three selected families",
    }


def unpack_uint4b8(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.int32 or packed.ndim != 2:
        raise ValueError("packed qweight must be rank-2 int32 [K/8,N]")
    shifts = torch.arange(8, dtype=torch.int32, device=packed.device) * 4
    stored = (packed[:, None, :] >> shifts[None, :, None]) & 0xF
    return stored.reshape(packed.shape[0] * 8, packed.shape[1]) - 8


def quantize_weight_128x128(
    canonical: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if canonical.ndim != 2:
        raise ValueError("canonical weight must be [N,K]")
    n, k = canonical.shape
    if n % GROUP_SIZE or k % GROUP_SIZE:
        raise ValueError("canonical N/K must be multiples of 128")
    view = canonical.float().reshape(
        n // GROUP_SIZE, GROUP_SIZE, k // GROUP_SIZE, GROUP_SIZE
    )
    scale = view.abs().amax(dim=(1, 3)).clamp_min(1.0e-10) / FP8_MAX
    quantized = (view / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    return quantized.to(torch.float8_e4m3fn).reshape(n, k), scale


def build_direct_layout(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    k: int,
    n: int,
    chunk_n: int = LAYOUT_CHUNK_N,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Expand one already-loaded canonical vLLM parameter pair.

    The source scale dtype is recorded rather than silently described as the
    FP16 checkpoint value.  In the pinned BF16 vLLM load path it is BF16,
    whereas Lab109 independently loaded the original FP16 safetensor.  This
    distinction forces a separate task-quality gate for the integrated image.
    """

    if chunk_n < GROUP_SIZE or chunk_n % GROUP_SIZE:
        raise ValueError("chunk_n must be a positive multiple of 128")
    if (k % GROUP_SIZE) or (n % GROUP_SIZE) or (n % chunk_n):
        raise RuntimeError(f"unsupported layout K={k}, N={n}, chunk_n={chunk_n}")
    if tuple(qweight.shape) != (k // 8, n) or qweight.dtype != torch.int32:
        raise RuntimeError(
            f"canonical qweight mismatch: {tuple(qweight.shape)} {qweight.dtype}"
        )
    if tuple(scales.shape) != (k // GROUP_SIZE, n):
        raise RuntimeError(f"canonical scale shape mismatch: {tuple(scales.shape)}")
    if scales.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"canonical scale dtype is unverified: {scales.dtype}")

    weight = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=qweight.device)
    weight_scale = torch.empty(
        (n // GROUP_SIZE, k // GROUP_SIZE),
        dtype=torch.float32,
        device=qweight.device,
    )
    for start in range(0, n, chunk_n):
        stop = start + chunk_n
        logical = unpack_uint4b8(qweight[:, start:stop]).transpose(0, 1).contiguous()
        row_scale = scales[:, start:stop].transpose(0, 1).float().contiguous()
        canonical = logical.float() * row_scale.repeat_interleave(GROUP_SIZE, dim=1)
        quantized, block_scale = quantize_weight_128x128(canonical)
        weight[start:stop].copy_(quantized)
        weight_scale[start // GROUP_SIZE : stop // GROUP_SIZE].copy_(block_scale)
        del logical, row_scale, canonical, quantized, block_scale
    memory = fp8_layout_bytes(k, n)
    observed = (
        weight.numel() * weight.element_size()
        + weight_scale.numel() * weight_scale.element_size()
    )
    if observed != memory["total"]:
        raise RuntimeError(f"layout byte mismatch: observed {observed}, modeled {memory}")
    return weight, weight_scale, {
        "K": k,
        "N": n,
        "source_scale_dtype": str(scales.dtype),
        "weight_shape": list(weight.shape),
        "weight_stride": list(weight.stride()),
        "scale_shape": list(weight_scale.shape),
        "scale_stride": list(weight_scale.stride()),
        "bytes": memory,
    }


def _validate_kernel_contract(kernel: Any, layer: torch.nn.Module, family: str) -> None:
    config = kernel.config
    versions = {
        "vllm": importlib.metadata.version("vllm"),
        "humming-kernels": importlib.metadata.version("humming-kernels"),
    }
    if versions != {
        "vllm": EXPECTED_VLLM_VERSION,
        "humming-kernels": EXPECTED_HUMMING_VERSION,
    }:
        raise RuntimeError(f"Lab113 runtime versions changed: {versions}")
    if config.has_g_idx or config.zero_points:
        raise RuntimeError("Lab113 requires symmetric GPTQ without act-order")
    if getattr(config.weight_type, "size_bits", None) != 4 or config.group_size != 128:
        raise RuntimeError("Lab113 requires canonical INT4/G128")
    if config.act_type is not torch.bfloat16:
        raise RuntimeError(f"Lab113 requires BF16 model activations: {config.act_type}")
    if getattr(layer, "bias", None) is not None:
        raise RuntimeError("Lab113 has no bias-preserving direct route")
    partition = tuple(int(value) for value in config.partition_weight_shape)
    full = tuple(int(value) for value in config.full_weight_shape)
    if partition != full or DIRECT_FAMILY_SHAPES.get(partition) != family:
        raise RuntimeError(
            f"Lab113 requires TP1 pinned family shape, partition={partition}, full={full}"
        )
    for name in (kernel.w_q_name, kernel.w_s_name):
        if not hasattr(layer, name):
            raise RuntimeError(f"Lab113 canonical parameter {name!r} is missing")


def _audit_payload() -> dict[str, Any]:
    counts = Counter(row["family"] for row in _LOAD_RECORDS)
    observed_bytes = sum(row["bytes"]["total"] for row in _LOAD_RECORDS)
    expected = expected_duplicate_layout_memory()
    return {
        "schema": "densespark.lab113.runtime-load-audit.v1",
        "enabled": _strict_toggle(),
        "load_records": list(_LOAD_RECORDS),
        "observed_family_counts": dict(sorted(counts.items())),
        "expected_family_counts": EXPECTED_FAMILY_COUNTS,
        "observed_duplicate_bytes": observed_bytes,
        "expected_duplicate_memory": expected,
        # Runtime routing is deliberately side-effect free.  The model AOT
        # graph and separate request-level audit establish the exercised M;
        # never mutate Python state or write files from the traced call path.
        "dispatch_records": [],
        "load_validated": _LOAD_VALIDATED,
    }


def _write_runtime_audit() -> None:
    target_value = os.environ.get(AUDIT_PATH_ENV)
    if not target_value:
        return
    target = Path(target_value)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_audit_payload(), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=target.name + ".", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    os.replace(temporary, target)


def validate_complete_load() -> dict[str, Any]:
    global _LOAD_VALIDATED
    payload = _audit_payload()
    failures = []
    if payload["observed_family_counts"] != EXPECTED_FAMILY_COUNTS:
        failures.append("selected family counts differ from 64/16/48")
    if payload["observed_duplicate_bytes"] != payload["expected_duplicate_memory"][
        "total_bytes"
    ]:
        failures.append("duplicate resident bytes differ from exact model")
    if any(row["source_scale_dtype"] != "torch.bfloat16" for row in _LOAD_RECORDS):
        failures.append("integrated canonical scale is not pinned BF16")
    if failures:
        raise RuntimeError("Lab113 incomplete model load: " + "; ".join(failures))
    _LOAD_VALIDATED = True
    _write_runtime_audit()
    return _audit_payload()


def record_loaded_layout(record: dict[str, Any]) -> None:
    """Record one selected layout and close the load gate outside forward.

    ``process_weights_after_loading`` is not part of Dynamo/AOT capture.  The
    exact 128th selected layout therefore provides the only safe place to
    validate counts/bytes and atomically write the immutable load manifest.
    Any later selected layout is an immediate pinned-geometry failure.
    """

    expected_total = sum(EXPECTED_FAMILY_COUNTS.values())
    if _LOAD_VALIDATED or len(_LOAD_RECORDS) >= expected_total:
        raise RuntimeError("Lab113 observed a selected layout after exact load closure")
    _LOAD_RECORDS.append(record)
    if len(_LOAD_RECORDS) == expected_total:
        validate_complete_load()


def _hybrid_cuda(
    inputs: torch.Tensor,
    humming_weight: torch.Tensor,
    humming_scale: torch.Tensor,
    humming_locks: torch.Tensor,
    cutlass_weight: torch.Tensor,
    cutlass_scale: torch.Tensor,
    layer_config: str,
    compute_config: str,
    chunk_rows: int,
) -> torch.Tensor:
    m, k = inputs.shape
    n = cutlass_weight.shape[0]
    route = route_for_rows(m)
    if route == HUMMING:
        return torch.ops.humming.humming_gemm(
            layer_config=layer_config,
            compute_config=compute_config,
            tuning_config=None,
            inputs=inputs,
            weight=humming_weight,
            weight_scale=humming_scale,
            locks=humming_locks,
        )

    if chunk_rows != CHUNK_ROWS or k % GROUP_SIZE or n % GROUP_SIZE:
        raise RuntimeError("Lab113 direct runtime ABI changed")
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    output = torch.empty((m, n), dtype=torch.bfloat16, device=inputs.device)
    cutlass_b = cutlass_weight.transpose(0, 1)
    cutlass_scale_b = cutlass_scale.transpose(0, 1)
    for start in range(0, m, chunk_rows):
        stop = min(start + chunk_rows, m)
        a_q, a_scale = per_token_group_quant_fp8(
            inputs[start:stop],
            GROUP_SIZE,
            use_ue8m0=False,
            column_major_scales=True,
        )
        expected_scale_shape = (stop - start, k // GROUP_SIZE)
        if (
            a_q.dtype != torch.float8_e4m3fn
            or tuple(a_q.shape) != (stop - start, k)
            or tuple(a_scale.shape) != expected_scale_shape
            or a_scale.stride(0) != 1
            or a_scale.stride(1) != stop - start
            or not output[start:stop].is_contiguous()
        ):
            raise RuntimeError("Lab113 activation/output CUTLASS ABI mismatch")
        torch.ops._C.cutlass_scaled_mm(
            output[start:stop],
            a_q,
            cutlass_b,
            a_scale,
            cutlass_scale_b,
            None,
        )
    return output


def _hybrid_fake(
    inputs: torch.Tensor,
    humming_weight: torch.Tensor,
    humming_scale: torch.Tensor,
    humming_locks: torch.Tensor,
    cutlass_weight: torch.Tensor,
    cutlass_scale: torch.Tensor,
    layer_config: str,
    compute_config: str,
    chunk_rows: int,
) -> torch.Tensor:
    del humming_weight, humming_scale, humming_locks, cutlass_scale
    del layer_config, compute_config, chunk_rows
    return inputs.new_empty(
        (inputs.shape[0], cutlass_weight.shape[0]), dtype=torch.bfloat16
    )


def register_hybrid_op() -> None:
    global _LIBRARY
    if _LIBRARY is not None:
        return
    if getattr(torch.ops.densespark_lab113, "hybrid_linear", None) is not None:
        # A test/auditor may load this immutable file through a second module
        # name in the same interpreter.  The first module still owns the
        # Library object and its registrations; never redefine the schema.
        _LIBRARY = "registered-by-earlier-module"
        return
    library = torch.library.Library("densespark_lab113", "FRAGMENT")
    library.define(
        "hybrid_linear(Tensor inputs, Tensor humming_weight, Tensor humming_scale, "
        "Tensor humming_locks, Tensor cutlass_weight, Tensor cutlass_scale, "
        "str layer_config, str compute_config, int chunk_rows) -> Tensor"
    )
    library.impl("hybrid_linear", _hybrid_cuda, dispatch_key="CUDA")
    library._register_fake("hybrid_linear", _hybrid_fake)
    _LIBRARY = library


def install(humming_kernel_cls: type) -> None:
    """Wrap the exact pinned Humming kernel once."""

    global _INSTALLED
    if _INSTALLED:
        return
    register_hybrid_op()
    original_process = humming_kernel_cls.process_weights_after_loading
    original_apply = humming_kernel_cls.apply_weights

    def process_weights_after_loading(self: Any, layer: torch.nn.Module) -> None:
        if not _strict_toggle():
            original_process(self, layer)
            return
        family = family_from_config(self.config)
        if family is None:
            original_process(self, layer)
            return
        if hasattr(layer, WEIGHT_BUFFER) or hasattr(layer, SCALE_BUFFER):
            raise RuntimeError("Lab113 direct buffers already exist before load hook")
        _validate_kernel_contract(self, layer, family)
        k, n = (int(value) for value in self.config.partition_weight_shape)
        qweight = getattr(layer, self.w_q_name)
        scales = getattr(layer, self.w_s_name)
        weight, weight_scale, record = build_direct_layout(
            qweight, scales, k=k, n=n
        )
        original_process(self, layer)
        layer.register_buffer(WEIGHT_BUFFER, weight, persistent=False)
        layer.register_buffer(SCALE_BUFFER, weight_scale, persistent=False)
        setattr(layer, FAMILY_ATTR, family)
        record["family"] = family
        record_loaded_layout(record)

    def apply_weights(
        self: Any,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not _strict_toggle() or family_from_config(self.config) is None:
            return original_apply(self, layer, x, bias)
        if bias is not None:
            raise RuntimeError("Lab113 received an unverified runtime bias")
        if not _LOAD_VALIDATED:
            raise RuntimeError("Lab113 forward reached before exact load closure")
        family = family_from_config(self.config)
        if getattr(layer, FAMILY_ATTR, None) != family:
            raise RuntimeError("Lab113 family tag disappeared after model load")
        weight = getattr(layer, WEIGHT_BUFFER, None)
        weight_scale = getattr(layer, SCALE_BUFFER, None)
        if weight is None or weight_scale is None:
            raise RuntimeError("Lab113 direct layout disappeared after model load")

        from vllm.utils.humming import HummingMethod

        meta = HummingMethod._get_meta(layer)
        flattened = x.view(-1, x.size(-1))
        output = torch.ops.densespark_lab113.hybrid_linear(
            flattened,
            layer.weight,
            layer.weight_scale,
            layer.locks,
            weight,
            weight_scale,
            meta.to_str(),
            layer.compute_config,
            CHUNK_ROWS,
        )
        return output.view(*x.shape[:-1], output.size(-1))

    humming_kernel_cls.process_weights_after_loading = process_weights_after_loading
    humming_kernel_cls.apply_weights = apply_weights
    _INSTALLED = True


register_hybrid_op()
