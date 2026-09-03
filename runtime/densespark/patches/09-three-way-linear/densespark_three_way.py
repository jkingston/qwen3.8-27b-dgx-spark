"""Lab118: compose the immutable Lab89 and Lab113 linear runtimes.

The composition order is the experiment:

1. Lab113 wraps clean Humming with an AOT-opaque direct-CUTLASS route for the
   three Lab109-selected families at exactly M={8192,16000}.
2. Lab89 then wraps that result in ``torch.cond(M >= 256)`` and preserves a
   stock Marlin shadow for the false branch.

Consequently selected families route Marlin below 256, Humming from 256 except
at the two exact direct points, and CUTLASS at those points.  Unselected
families route Marlin/Humming at the same threshold and never use CUTLASS.
No kernel implementation or component policy is changed here.

This module is inert unless all three exact toggles are enabled.
Load accounting is performed only after each weight-finalization call; forward
and custom-op paths contain no file I/O, print, or mutable audit state.
"""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch


ENABLE_ENV = "DENSESPARK_LAB118_ENABLE"
AUDIT_PATH_ENV = "DENSESPARK_LAB118_RUNTIME_AUDIT"
LAB89_ENABLE_ENV = "DENSESPARK_LAB89_HYBRID_LINEAR"
LAB89_THRESHOLD_ENV = "DENSESPARK_LAB89_HUMMING_MIN_M"
LAB113_ENABLE_ENV = "DENSESPARK_LAB113_ENABLE"

THRESHOLD = 256
DIRECT_M_VALUES = (8192, 16000)
MARLIN = "marlin_w4a16"
HUMMING = "humming_w4a8_int8"
DIRECT = "cutlass_direct_output_m2048"

LAB89_MODULE = "vllm._densespark_lab118_marlin_component"
LAB113_MODULE = "vllm._densespark_lab118_cutlass_component"

# The two output projections have identical physical shape and identical
# policy.  The low-level MPLinear hook has no stable module pathname, so they
# are intentionally audited as one shape family rather than guessed apart.
FAMILY_SHAPES: dict[tuple[int, int], str] = {
    (5120, 34816): "mlp.gate_up",
    (17408, 5120): "mlp.down",
    (5120, 14336): "attn.qkv",
    (6144, 5120): "attn_or_gdn.out",
    (5120, 16384): "gdn.qkvz",
}
EXPECTED_FAMILY_COUNTS = {
    # 256 backbone linears plus one full-attention MTP layer with four
    # linears.  The MTP layer adds gate_up/down/qkv/o exactly once.
    "mlp.gate_up": 65,
    "mlp.down": 65,
    "attn.qkv": 17,
    "attn_or_gdn.out": 65,
    "gdn.qkvz": 48,
}
SELECTED_FAMILIES = frozenset(("mlp.gate_up", "attn.qkv", "gdn.qkvz"))
LAB113_BACKBONE_COUNTS = {
    "mlp.gate_up": 64,
    "attn.qkv": 16,
    "gdn.qkvz": 48,
}
LAB113_COMPOSED_COUNTS = {
    "mlp.gate_up": 65,
    "attn.qkv": 17,
    "gdn.qkvz": 48,
}

_INSTALLED = False
_LOAD_VALIDATED = False
_LOAD_RECORDS: list[dict[str, Any]] = []


def _strict_toggle(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise RuntimeError(f"{name} must be exactly 0 or 1, observed {value!r}")
    return value == "1"


def validate_environment() -> None:
    if not _strict_toggle(ENABLE_ENV):
        raise RuntimeError(f"{ENABLE_ENV}=1 is required for the Lab118 route")
    required = {
        LAB89_ENABLE_ENV: "1",
        LAB113_ENABLE_ENV: "1",
        LAB89_THRESHOLD_ENV: str(THRESHOLD),
    }
    observed = {name: os.environ.get(name) for name in required}
    if observed != required:
        raise RuntimeError(
            f"Lab118 component environment must be exact: required={required}, "
            f"observed={observed}"
        )


def route_for_rows(rows: int, *, selected: bool) -> str:
    if rows <= 0:
        raise RuntimeError(f"Lab118 expected positive M, observed {rows}")
    if rows < THRESHOLD:
        return MARLIN
    if selected and rows in DIRECT_M_VALUES:
        return DIRECT
    return HUMMING


def family_from_config(config: Any) -> str | None:
    return FAMILY_SHAPES.get(tuple(int(value) for value in config.partition_weight_shape))


def marlin_payload_bytes(k: int, n: int, count: int = 1) -> dict[str, int]:
    if min(k, n, count) <= 0 or k % 128 or n % 256:
        raise ValueError("Marlin payload requires aligned positive K/N/count")
    qweight = k * n * 4 // 8 * count
    scales = (k // 128) * n * 2 * count
    return {"qweight": qweight, "scales_bf16": scales, "total": qweight + scales}


def direct_payload_bytes(k: int, n: int, count: int = 1) -> dict[str, int]:
    if min(k, n, count) <= 0 or k % 128 or n % 128:
        raise ValueError("direct payload requires aligned positive K/N/count")
    weight = k * n * count
    scales = (k // 128) * (n // 128) * 4 * count
    return {"weight_fp8": weight, "scales_fp32": scales, "total": weight + scales}


def expected_extra_memory() -> dict[str, Any]:
    rows = []
    marlin_total = 0
    direct_total = 0
    for (k, n), family in FAMILY_SHAPES.items():
        count = EXPECTED_FAMILY_COUNTS[family]
        marlin = marlin_payload_bytes(k, n, count)
        direct = direct_payload_bytes(k, n, count) if family in SELECTED_FAMILIES else None
        marlin_total += marlin["total"]
        direct_total += 0 if direct is None else direct["total"]
        rows.append(
            {
                "family": family,
                "K": k,
                "N": n,
                "count": count,
                "marlin": marlin,
                "direct": direct,
            }
        )
    total = marlin_total + direct_total
    return {
        "rows": rows,
        "marlin_qweight_scale_bytes": marlin_total,
        "direct_fp8_scale_bytes": direct_total,
        "total_modeled_payload_bytes": total,
        "total_modeled_payload_GiB": total / 2**30,
        "scope": (
            "extra transformed qweight/scale payload only; excludes allocator overhead, "
            "per-layer Marlin workspace, empty g_idx tensors, and compile cache"
        ),
    }


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def inspect_loaded_layer(
    kernel: Any,
    layer: torch.nn.Module,
    marlin_component: Any,
    cutlass_component: Any,
) -> dict[str, Any]:
    family = family_from_config(kernel.config)
    if family is None:
        raise RuntimeError(
            "Lab118 encountered an unknown quantized linear shape: "
            f"{tuple(kernel.config.partition_weight_shape)}"
        )
    k, n = (int(value) for value in kernel.config.partition_weight_shape)
    if tuple(int(value) for value in kernel.config.full_weight_shape) != (k, n):
        raise RuntimeError("Lab118 is pinned to TP1 full partitions")

    shadow = getattr(layer, marlin_component.SHADOW_NAME, None)
    marlin_kernel = getattr(kernel, "_densespark_lab89_marlin_kernel", None)
    if shadow is None or marlin_kernel is None:
        raise RuntimeError("Lab118 Marlin shadow/kernel is missing after load")
    qweight = getattr(shadow, marlin_kernel.w_q_name)
    scales = getattr(shadow, marlin_kernel.w_s_name)
    marlin_observed = _tensor_bytes(qweight) + _tensor_bytes(scales)
    marlin_expected = marlin_payload_bytes(k, n)["total"]
    if marlin_observed != marlin_expected:
        raise RuntimeError(
            f"Lab118 Marlin payload mismatch for {family}: "
            f"observed={marlin_observed}, expected={marlin_expected}"
        )

    for name in ("weight", "weight_scale", "locks", "compute_config"):
        if not hasattr(layer, name):
            raise RuntimeError(f"Lab118 Humming layout lost {name!r} for {family}")

    selected = family in SELECTED_FAMILIES
    direct_weight = getattr(layer, cutlass_component.WEIGHT_BUFFER, None)
    direct_scale = getattr(layer, cutlass_component.SCALE_BUFFER, None)
    if selected:
        if direct_weight is None or direct_scale is None:
            raise RuntimeError(f"Lab118 selected direct layout is missing for {family}")
        direct_observed = _tensor_bytes(direct_weight) + _tensor_bytes(direct_scale)
        direct_expected = direct_payload_bytes(k, n)["total"]
        if direct_observed != direct_expected:
            raise RuntimeError(
                f"Lab118 direct payload mismatch for {family}: "
                f"observed={direct_observed}, expected={direct_expected}"
            )
    else:
        if direct_weight is not None or direct_scale is not None:
            raise RuntimeError(f"Lab118 unselected family gained direct buffers: {family}")
        direct_observed = 0

    return {
        "family": family,
        "K": k,
        "N": n,
        "selected_for_direct": selected,
        "marlin_payload_bytes": marlin_observed,
        "direct_payload_bytes": direct_observed,
    }


def _audit_payload() -> dict[str, Any]:
    counts = Counter(row["family"] for row in _LOAD_RECORDS)
    marlin_bytes = sum(row["marlin_payload_bytes"] for row in _LOAD_RECORDS)
    direct_bytes = sum(row["direct_payload_bytes"] for row in _LOAD_RECORDS)
    expected = expected_extra_memory()
    return {
        "schema": "densespark.lab118.runtime-load-audit.v1",
        "load_validated": _LOAD_VALIDATED,
        "load_records": list(_LOAD_RECORDS),
        "observed_family_counts": dict(sorted(counts.items())),
        "expected_family_counts": EXPECTED_FAMILY_COUNTS,
        "observed_marlin_payload_bytes": marlin_bytes,
        "observed_direct_payload_bytes": direct_bytes,
        "expected_extra_memory": expected,
        "routing_contract": {
            "selected": "Marlin M<256; CUTLASS M in {8192,16000}; Humming otherwise",
            "unselected": "Marlin M<256; Humming M>=256",
        },
        "dispatch_records": [],
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
    expected = payload["expected_extra_memory"]
    failures = []
    if payload["observed_family_counts"] != EXPECTED_FAMILY_COUNTS:
        failures.append("quantized family counts differ from pinned 256-linears geometry")
    if payload["observed_marlin_payload_bytes"] != expected["marlin_qweight_scale_bytes"]:
        failures.append("Marlin payload bytes differ from exact model")
    if payload["observed_direct_payload_bytes"] != expected["direct_fp8_scale_bytes"]:
        failures.append("direct payload bytes differ from exact model")
    if failures:
        raise RuntimeError("Lab118 incomplete model load: " + "; ".join(failures))
    _LOAD_VALIDATED = True
    _write_runtime_audit()
    return _audit_payload()


def install(humming_kernel_cls: type) -> None:
    """Install the two immutable components in the only valid order."""

    global _INSTALLED
    if _INSTALLED:
        return
    if not _strict_toggle(ENABLE_ENV):
        return
    validate_environment()
    cutlass_component = importlib.import_module(LAB113_MODULE)
    marlin_component = importlib.import_module(LAB89_MODULE)

    # Immutable Lab113 closed after the 128 selected backbone layouts because
    # its original serving integration had no drafter.  Lab122 loads a second
    # Qwen3.5 MTP model containing one additional gate_up and qkv selected
    # layout.  Extend the exact geometry *before* either component is installed;
    # the component still closes and rejects every selected layout after 130.
    observed_cutlass_counts = dict(cutlass_component.EXPECTED_FAMILY_COUNTS)
    if observed_cutlass_counts != LAB113_BACKBONE_COUNTS:
        raise RuntimeError(
            "Lab118 immutable Lab113 selected geometry changed: "
            f"{observed_cutlass_counts}"
        )
    cutlass_component.EXPECTED_FAMILY_COUNTS = dict(LAB113_COMPOSED_COUNTS)

    # Order is semantic: Lab89's true branch must capture the already wrapped
    # Lab113 apply method, while its false branch remains stock Marlin.
    cutlass_component.install(humming_kernel_cls)
    marlin_component.install(humming_kernel_cls)
    composed_process = humming_kernel_cls.process_weights_after_loading
    composed_apply = humming_kernel_cls.apply_weights

    def process_weights_after_loading(self: Any, layer: torch.nn.Module) -> None:
        if not _strict_toggle(ENABLE_ENV):
            raise RuntimeError("Lab118 was disabled after its wrapper was installed")
        composed_process(self, layer)
        expected_total = sum(EXPECTED_FAMILY_COUNTS.values())
        if _LOAD_VALIDATED or len(_LOAD_RECORDS) >= expected_total:
            raise RuntimeError("Lab118 observed a linear after exact load closure")
        _LOAD_RECORDS.append(
            inspect_loaded_layer(self, layer, marlin_component, cutlass_component)
        )
        if len(_LOAD_RECORDS) == expected_total:
            validate_complete_load()

    def apply_weights(
        self: Any,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not _strict_toggle(ENABLE_ENV):
            raise RuntimeError("Lab118 was disabled after its wrapper was installed")
        if not _LOAD_VALIDATED:
            raise RuntimeError("Lab118 forward reached before exact load closure")
        return composed_apply(self, layer, x, bias)

    humming_kernel_cls.process_weights_after_loading = process_weights_after_loading
    humming_kernel_cls.apply_weights = apply_weights
    _INSTALLED = True
