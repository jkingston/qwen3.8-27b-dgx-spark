#!/usr/bin/env python3
"""Let a wide Marlin projection be issued as L2-resident column blocks.

GB10 has a 24 MiB L2 and Marlin re-reads its packed weight once per row tile, so
a projection whose weight exceeds the cache streams that weight from DRAM on
every tile. Measured on this part, the achieved fraction of the bf16 tensor
ceiling falls from about 70% at 20 MiB to about 43% at 30 MiB, and the cliff sits
exactly on the cache size. Qwen3.8-27B merges gate_proj and up_proj into one
34,816-wide linear in all 64 layers, which lands on the wrong side of it.

This installs a runtime module and wraps two ``MarlinLinearKernel`` methods:
weight preparation additionally pre-slices an eligible wide projection into
column blocks, and the apply path uses those blocks only when the call has
enough rows to profit. Everything is inert unless ``DENSESPARK_MARLIN_NSPLIT``
is set, so an unset environment reproduces stock behaviour exactly.

The patch is version-specific and fails closed when the apply site differs from
the vLLM 0.27.x source it was validated against.

Usage:
    python3 patch_marlin_nsplit.py
    python3 patch_marlin_nsplit.py --vllm-root /path/to/vllm
"""

import argparse
import os
import shutil
import sys


MARKER = "DENSESPARK_MARLIN_NSPLIT"

FALLBACK_ROOTS = (
    "/usr/local/lib/python3.12/dist-packages/vllm",
    "/usr/lib/python3/dist-packages/vllm",
    "/opt/venv/lib/python3.12/site-packages/vllm",
)

RELATIVE_TARGET = os.path.join(
    "model_executor", "kernels", "linear", "mixed_precision", "marlin.py"
)
RUNTIME_NAME = "_densespark_nsplit.py"
RUNTIME_SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "densespark_nsplit.py"
)

# The tail of MarlinLinearKernel.apply_weights. The hook is appended after it,
# so this anchor exists only to pin the file to the version the slicing was
# verified against.
ANCHOR = '''            input_global_scale=getattr(layer, "input_global_scale", None),
            bias=bias,
            input_dtype=c.act_type,
        )
'''

HOOK = '''

# ── DENSESPARK_MARLIN_NSPLIT ──────────────────────────────────────────────────
# Issue a wide projection as several L2-resident column blocks when the call has
# enough rows to profit. See vllm/_densespark_nsplit.py for the measurements and
# the toggles. Inert unless DENSESPARK_MARLIN_NSPLIT is set.
from vllm import _densespark_nsplit as _ds_nsplit  # noqa: E402
from vllm.logger import init_logger as _ds_init_logger  # noqa: E402
from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: E402
    marlin_repacked_nk as _ds_marlin_repacked_nk,
)

_ds_nsplit_logger = _ds_init_logger(__name__)
_ds_nsplit_orig_process = MarlinLinearKernel.process_weights_after_loading
_ds_nsplit_orig_apply = MarlinLinearKernel.apply_weights


def _ds_nsplit_process_weights_after_loading(self, layer) -> None:
    _ds_nsplit_orig_process(self, layer)
    try:
        _ds_nsplit.attach(self, layer, _ds_marlin_repacked_nk, _ds_nsplit_logger)
    except Exception as exc:  # never let an optimization break weight loading
        self._ds_nsplit = None
        _ds_nsplit_logger.warning(
            "DENSESPARK_MARLIN_NSPLIT left a layer unsplit: %s", exc
        )


def _ds_nsplit_apply_weights(self, layer, x, bias=None):
    if bias is None:
        output = _ds_nsplit.apply_split(self, layer, x, apply_gptq_marlin_linear)
        if output is not None:
            return output
    return _ds_nsplit_orig_apply(self, layer, x, bias)


MarlinLinearKernel.process_weights_after_loading = (
    _ds_nsplit_process_weights_after_loading
)
MarlinLinearKernel.apply_weights = _ds_nsplit_apply_weights
'''


def find_root(vllm_root=None):
    """Return the installed vllm package directory, or ``None``."""
    if vllm_root:
        return vllm_root
    try:
        import vllm

        return os.path.dirname(vllm.__file__)
    except Exception:
        pass
    for root in FALLBACK_ROOTS:
        if os.path.exists(os.path.join(root, RELATIVE_TARGET)):
            return root
    return None


def install_runtime(root):
    """Copy the runtime module next to the vllm package."""
    if not os.path.exists(RUNTIME_SOURCE):
        print(f"ABORT: runtime module missing at {RUNTIME_SOURCE}", file=sys.stderr)
        return None
    destination = os.path.join(root, RUNTIME_NAME)
    shutil.copyfile(RUNTIME_SOURCE, destination)
    with open(destination, "r", encoding="utf-8") as handle:
        compile(handle.read(), destination, "exec")
    return destination


def apply(target):
    """Append the dispatch hook exactly once."""
    with open(target, "r", encoding="utf-8") as handle:
        source = handle.read()

    if MARKER in source:
        print(f"OK: already patched {target}")
        return 0
    if source.count(ANCHOR) != 1:
        print(f"ABORT: expected one apply_weights anchor in {target}", file=sys.stderr)
        return 1
    if not source.rstrip().endswith(ANCHOR.rstrip()):
        print(
            f"ABORT: {target} does not end with MarlinLinearKernel.apply_weights",
            file=sys.stderr,
        )
        return 1

    patched = source.rstrip("\n") + "\n" + HOOK
    compile(patched, target, "exec")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(patched)
    print(f"OK: Marlin column-block dispatch applied to {target}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm-root", default=None)
    args = parser.parse_args()
    root = find_root(args.vllm_root)
    if not root:
        print("ABORT: could not locate the installed vllm package", file=sys.stderr)
        return 1
    target = os.path.join(root, RELATIVE_TARGET)
    if not os.path.exists(target):
        print(f"ABORT: could not locate {RELATIVE_TARGET}", file=sys.stderr)
        return 1
    if install_runtime(root) is None:
        return 1
    return apply(target)


if __name__ == "__main__":
    raise SystemExit(main())
