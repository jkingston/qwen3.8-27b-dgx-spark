"""Lab 89: keep Humming and Marlin layouts and dispatch on the real row count.

The target checkpoint is canonical symmetric GPTQ INT4/G128.  vLLM normally
chooses one ``MPLinearKernel`` while creating a layer, then lets that kernel
destructively transform the checkpoint tensors.  Consequently
``--linear-backend humming`` and ``--linear-backend marlin`` cannot be selected
per invocation: by the time ``apply_weights`` runs, only one physical layout
still exists.

This runtime wraps the selected Humming kernel.  During weight
finalization it deep-copies the still-canonical vLLM parameters, transforms the
copy with the *stock* Marlin preparation path, and transforms the original with
Humming.  Calls below the configured M threshold use the Marlin copy; larger
calls use Humming.  Dispatch is expressed as ``torch.cond`` rather than a
Python branch: vLLM traces a symbolic row count, and a Python ``if`` specializes
that symbol to the example input and silently leaves only one backend in the
AOT graph.  The extra weight layout is intentional on the 128 GiB GB10.

The runtime is inert unless ``DENSESPARK_LAB89_HYBRID_LINEAR=1``.  An enabled
run fails closed for every format outside the exact Qwen3.8 experiment
contract.  It never silently drops either layout or changes backend.
"""

from __future__ import annotations

import importlib.metadata
import os
from typing import Any, Callable

import torch


ENABLE_ENV = "DENSESPARK_LAB89_HYBRID_LINEAR"
THRESHOLD_ENV = "DENSESPARK_LAB89_HUMMING_MIN_M"
EXPECTED_HUMMING_VERSION = "0.1.13"
DEFAULT_HUMMING_MIN_M = 256
SHADOW_NAME = "_densespark_lab89_marlin_layout"

_INSTALLED = False


def _strict_toggle() -> bool:
    value = os.environ.get(ENABLE_ENV, "0")
    if value not in ("0", "1"):
        raise RuntimeError(f"{ENABLE_ENV} must be exactly 0 or 1, observed {value!r}")
    return value == "1"


def humming_min_m() -> int:
    raw = os.environ.get(THRESHOLD_ENV, str(DEFAULT_HUMMING_MIN_M))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{THRESHOLD_ENV} must be an integer, observed {raw!r}") from exc
    if not 17 <= value <= 16384:
        raise RuntimeError(
            f"{THRESHOLD_ENV} must be in [17, 16384], observed {value}"
        )
    return value


def flattened_rows(x: torch.Tensor) -> int:
    if x.ndim < 2 or x.shape[-1] <= 0:
        raise RuntimeError(f"Lab 89 expected [...,K] input, observed {tuple(x.shape)}")
    return x.numel() // x.shape[-1]


def choose_backend(rows: int, threshold: int) -> str:
    if rows <= 0:
        raise RuntimeError(f"Lab 89 expected a positive M, observed {rows}")
    return "humming" if rows >= threshold else "marlin"


def dispatch_dual_layout(
    x: torch.Tensor,
    threshold: int,
    humming_branch: Callable[[torch.Tensor], torch.Tensor],
    marlin_branch: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Keep both backend branches when ``M`` is a ``torch.SymInt``.

    ``torch.cond`` executes exactly one branch in eager mode and represents
    both branches explicitly under Dynamo/export/AOT.  Do not replace this
    with a Python conditional: vLLM compiles one graph for M=1..8192 with guard
    evaluation disabled, so specializing the first trace makes the route
    permanently Humming or permanently Marlin.
    """

    rows = flattened_rows(x)
    return torch.cond(
        rows >= threshold,
        humming_branch,
        marlin_branch,
        (x,),
    )


def _copy_parameter(parameter: torch.nn.Parameter) -> torch.nn.Parameter:
    """Copy both storage and vLLM parameter-subclass metadata.

    Marlin's preparation asserts ``BasevLLMParameter`` rather than accepting a
    plain ``torch.nn.Parameter``.  ``detach().clone()`` would therefore lose a
    required part of the loader ABI.  Construct the same tensor subclass over
    cloned storage and copy vLLM's dimension/loader attributes explicitly.
    """

    # BasevLLMParameter.__new__ accepts only its data tensor, while PyTorch's
    # Parameter.__deepcopy__ calls subclasses with ``(data, requires_grad)``;
    # the seemingly natural deepcopy therefore raises for the pinned loader.
    # _make_subclass is the same zero-conversion mechanism used by tensor
    # subclasses: construct the exact concrete type around independent storage,
    # then restore vLLM's Python-side metadata.
    cloned = torch.Tensor._make_subclass(
        type(parameter),
        parameter.detach().clone(),
        parameter.requires_grad,
    )
    for name, value in parameter.__dict__.items():
        # Values are dimension integers and loader callables in the pinned
        # BasevLLMParameter classes.  Sharing a loader callable is harmless
        # after loading; deepcopying a bound callable could recursively clone
        # its owning module.
        setattr(cloned, name, value)
    if cloned is parameter or cloned.data_ptr() == parameter.data_ptr():
        raise RuntimeError("Lab 89 parameter copy unexpectedly aliases its source")
    return cloned


def _make_marlin_shadow(layer: torch.nn.Module) -> torch.nn.Module:
    shadow = torch.nn.Module()
    parameters = list(layer.named_parameters(recurse=False))
    if not parameters:
        raise RuntimeError("Lab 89 found no canonical parameters to duplicate")
    for name, parameter in parameters:
        shadow.register_parameter(name, _copy_parameter(parameter))
    for name, buffer in layer.named_buffers(recurse=False):
        shadow.register_buffer(name, buffer.detach().clone())
    if not hasattr(shadow, "bias"):
        shadow.bias = None
    return shadow


def _validate_contract(kernel: Any, layer: torch.nn.Module) -> None:
    config = kernel.config
    if importlib.metadata.version("humming-kernels") != EXPECTED_HUMMING_VERSION:
        raise RuntimeError(
            "Lab 89 requires humming-kernels "
            f"{EXPECTED_HUMMING_VERSION}"
        )
    if config.has_g_idx:
        raise RuntimeError("Lab 89 does not support GPTQ act-order")
    if config.zero_points:
        raise RuntimeError("Lab 89 requires symmetric weights without zero points")
    if getattr(config.weight_type, "size_bits", None) != 4:
        raise RuntimeError("Lab 89 requires a 4-bit MPLinear weight type")
    if config.group_size != 128:
        raise RuntimeError(
            f"Lab 89 requires group_size=128, observed {config.group_size}"
        )
    if config.act_type is not torch.bfloat16:
        raise RuntimeError(
            f"Lab 89 requires BF16 activations, observed {config.act_type}"
        )
    if getattr(layer, "bias", None) is not None:
        raise RuntimeError("Lab 89 has no verified bias-preserving dual-layout path")
    size_k, size_n = config.partition_weight_shape
    if size_k % 128 or size_n % 256:
        raise RuntimeError(
            "Lab 89 requires Humming-aligned K/N, observed "
            f"K={size_k}, N={size_n}"
        )
    for name in (kernel.w_q_name, kernel.w_s_name, kernel.w_gidx_name, kernel.w_zp_name):
        if name is not None and not hasattr(layer, name):
            raise RuntimeError(f"Lab 89 canonical parameter {name!r} is missing")


def install(humming_kernel_cls: type) -> None:
    """Wrap vLLM's pinned ``HummingLinearKernel`` exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    original_process = humming_kernel_cls.process_weights_after_loading
    original_apply = humming_kernel_cls.apply_weights

    def process_weights_after_loading(self: Any, layer: torch.nn.Module) -> None:
        if not _strict_toggle():
            original_process(self, layer)
            return

        _validate_contract(self, layer)
        threshold = humming_min_m()
        shadow = _make_marlin_shadow(layer)

        # Import after the environment gate so an inert image remains exactly
        # the stock Humming route.  The production image wraps Marlin with its
        # optional N-split dispatcher; this hybrid deliberately calls the saved
        # stock methods because every Marlin invocation is below the crossover
        # and a third split layout would waste about 7.2 GiB.
        from vllm.model_executor.kernels.linear.mixed_precision import marlin as marlin_module

        marlin_cls = marlin_module.MarlinLinearKernel
        marlin_kernel = marlin_cls(
            self.config,
            w_q_param_name=self.w_q_name,
            w_s_param_name=self.w_s_name,
            w_zp_param_name=self.w_zp_name,
            w_gidx_param_name=self.w_gidx_name,
        )
        stock_process: Callable[..., Any] = getattr(
            marlin_module,
            "_ds_nsplit_orig_process",
            marlin_cls.process_weights_after_loading,
        )
        stock_apply: Callable[..., Any] = getattr(
            marlin_module,
            "_ds_nsplit_orig_apply",
            marlin_cls.apply_weights,
        )
        stock_process(marlin_kernel, shadow)

        # Do not register ``shadow`` before this call.  Humming converts
        # ``dict(layer.named_parameters())`` recursively, so an early child
        # registration would mix the second layout into its checkpoint input.
        original_process(self, layer)
        layer.add_module(SHADOW_NAME, shadow)
        self._densespark_lab89_marlin_kernel = marlin_kernel
        self._densespark_lab89_marlin_apply = stock_apply
        self._densespark_lab89_threshold = threshold

    def apply_weights(
        self: Any,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not _strict_toggle():
            return original_apply(self, layer, x, bias)
        if bias is not None:
            raise RuntimeError("Lab 89 received an unverified runtime bias")
        if not hasattr(self, "_densespark_lab89_marlin_kernel"):
            raise RuntimeError("Lab 89 apply reached a layer without both weight layouts")
        threshold = self._densespark_lab89_threshold
        shadow = getattr(layer, SHADOW_NAME, None)
        if shadow is None:
            raise RuntimeError("Lab 89 Marlin shadow disappeared after weight loading")

        def humming_branch(operand: torch.Tensor) -> torch.Tensor:
            return original_apply(self, layer, operand, bias)

        def marlin_branch(operand: torch.Tensor) -> torch.Tensor:
            return self._densespark_lab89_marlin_apply(
                self._densespark_lab89_marlin_kernel,
                shadow,
                operand,
                bias,
            )

        return dispatch_dual_layout(
            x,
            threshold,
            humming_branch,
            marlin_branch,
        )

    humming_kernel_cls.process_weights_after_loading = process_weights_after_loading
    humming_kernel_cls.apply_weights = apply_weights
    _INSTALLED = True
