#!/usr/bin/env python3
"""Prevent hybrid-model prefills from entering a decode CUDA graph.

vLLM 0.27.x classifies a uniform decode batch from tensor shape alone. A
prefill whose query length equals ``1 + num_speculative_tokens`` has the same
shape as a speculative verification step, so it can replay the FULL decode
graph before the GDN recurrent state has been initialized. The resulting state
is missing and subsequent tokens are incorrect.

This backports the narrow upstream fix: for hybrid models only, ask the
scheduler whether the batch still contains a context request. Such a batch is
forced away from uniform-decode dispatch. Genuine decode batches keep the
shape heuristic and therefore retain their FULL CUDA graphs and performance.

The patch is deliberately version-specific and fails closed when either call
site differs from the vLLM 0.27.x source it was validated against.

Usage:
    python3 patch_gdn_prefill_dispatch.py
    python3 patch_gdn_prefill_dispatch.py --vllm-root /path/to/vllm
"""

import argparse
import os
import sys


MARKER = "DENSESPARK_GDN_PREFILL_DISPATCH"

FALLBACK_ROOTS = (
    "/usr/local/lib/python3.12/dist-packages/vllm",
    "/usr/lib/python3/dist-packages/vllm",
    "/opt/venv/lib/python3.12/site-packages/vllm",
)

RELATIVE_TARGET = os.path.join("v1", "worker", "gpu_model_runner.py")

DISPATCH_ANCHOR = '''            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
'''

DISPATCH_REPLACEMENT = '''            # DENSESPARK_GDN_PREFILL_DISPATCH: tensor shape alone cannot
            # distinguish a short hybrid prefill from speculative verification.
            # Keep real decode batches on FULL graphs, but never dispatch a batch
            # containing context work through a decode graph.
            force_uniform_decode = None
            if self.model_config.is_hybrid:
                from vllm.v1.utils import compute_iteration_details as _ds_iter_details
                if _ds_iter_details(scheduler_output).num_ctx_requests > 0:
                    force_uniform_decode = False

            (
                cudagraph_mode,
                batch_desc,
                should_ubatch,
                num_tokens_across_dp,
                cudagraph_stats,
            ) = self._determine_batch_execution_and_padding(
'''

ARGUMENT_ANCHOR = '''                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
            )
'''

ARGUMENT_REPLACEMENT = '''                use_cascade_attn=cascade_attn_prefix_lens is not None,
                num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
                force_uniform_decode=force_uniform_decode,
            )
'''


def find_target(vllm_root=None):
    """Return the gpu model runner to patch, or ``None`` when unavailable."""
    if vllm_root:
        return os.path.join(vllm_root, RELATIVE_TARGET)
    try:
        import vllm

        return os.path.join(os.path.dirname(vllm.__file__), RELATIVE_TARGET)
    except Exception:
        pass
    for root in FALLBACK_ROOTS:
        candidate = os.path.join(root, RELATIVE_TARGET)
        if os.path.exists(candidate):
            return candidate
    return None


def apply(target):
    """Apply the guarded dispatch change exactly once."""
    with open(target, "r", encoding="utf-8") as handle:
        source = handle.read()

    if MARKER in source:
        print(f"OK: already patched {target}")
        return 0
    if source.count(DISPATCH_ANCHOR) != 1:
        print(
            f"ABORT: expected one runtime dispatch anchor in {target}",
            file=sys.stderr,
        )
        return 1
    if source.count(ARGUMENT_ANCHOR) != 1:
        print(
            f"ABORT: expected one runtime argument anchor in {target}",
            file=sys.stderr,
        )
        return 1

    patched = source.replace(DISPATCH_ANCHOR, DISPATCH_REPLACEMENT, 1)
    patched = patched.replace(ARGUMENT_ANCHOR, ARGUMENT_REPLACEMENT, 1)
    compile(patched, target, "exec")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(patched)
    print(f"OK: hybrid prefill dispatch guard applied to {target}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm-root", default=None)
    args = parser.parse_args()
    target = find_target(args.vllm_root)
    if not target or not os.path.exists(target):
        print("ABORT: could not locate vllm/v1/worker/gpu_model_runner.py", file=sys.stderr)
        return 1
    return apply(target)


if __name__ == "__main__":
    raise SystemExit(main())
