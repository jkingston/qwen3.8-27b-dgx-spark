#!/usr/bin/env python3
"""Product-quantized draft head for the MTP proposer.

The draft path spends one full pass over the 1.271 GB INT8 head per proposed
token, which is about 70% of the marginal cost of a chain position. A proposal
is not an emission: the target model verifies it, and vLLM's rejection sampler
emits the target argmax whichever token was drafted. An approximate head is
therefore admissible here and nowhere else in the step.

This patch routes the drafter's greedy argmax through a product quantization of
the same head - a 31.8 MB code scan and a gathered INT8 rerank of the top C
candidates - instead of the full read. The structure is a build-time artifact;
the standalone artifact builder trains it from the selected checkpoint.

Applies to the vLLM 0.27.x form of LLMBaseProposer._greedy_sample. An
unrecognised body aborts without editing the file.

The patched runtime is inert unless DENSESPARK_PQ_DRAFT=1 is set in the server
environment, so one image serves both sides of an A/B. It also disables itself,
loudly, when the artifact is absent or was not trained on the head that is
loaded.

Usage:
    python3 patch_pq_draft_head.py                     # patch the importable vLLM
    python3 patch_pq_draft_head.py --vllm-root /path   # patch a specific install
"""

import argparse
import os
import shutil
import sys

MARKER = "DENSESPARK_PQ_DRAFT"

FALLBACK_ROOTS = (
    "/usr/local/lib/python3.12/dist-packages/vllm",
    "/usr/lib/python3/dist-packages/vllm",
    "/opt/venv/lib/python3.12/site-packages/vllm",
)

RELATIVE_TARGET = os.path.join("v1", "spec_decode", "llm_base_proposer.py")
RUNTIME_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "densespark_pq.py")
RUNTIME_NAME = "_densespark_pq.py"

ANCHOR = '''    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        if self.use_local_argmax_reduction:'''

REPLACEMENT = '''    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy-sample draft tokens from hidden states."""
        # DENSESPARK_PQ_DRAFT: approximate the drafter's argmax with a product
        # quantization of the head and rerank against its deployed INT8 rows.
        # Only the proposal changes; the target still verifies every token.
        if not hasattr(self, '_ds_pq_head') and hidden_states.shape[0] > 0:
            self._ds_pq_head = None
            try:
                import os as _ds_os
                if _ds_os.environ.get('DENSESPARK_PQ_DRAFT', '0') == '1':
                    from vllm import _densespark_pq as _ds_pq
                    _ds_lm_head = getattr(self.model, 'lm_head', None)
                    if (_ds_lm_head is not None
                            and getattr(_ds_lm_head, '_ds_int8', None) is None):
                        # Patch 01 quantizes lazily inside the first _get_logits.
                        self.model.compute_logits(hidden_states[:1])
                    if _ds_lm_head is not None:
                        self._ds_pq_head = _ds_pq.build(
                            _ds_lm_head, hidden_states.shape[-1])
            except Exception as _ds_exc:
                import sys as _ds_sys
                print(f'DENSESPARK PQ: disabled: {_ds_exc!r}',
                      file=_ds_sys.stderr, flush=True)
                self._ds_pq_head = None
        if (getattr(self, '_ds_pq_head', None) is not None
                and not self.use_local_argmax_reduction
                and not self.use_heterogeneous_vocab):
            return self._ds_pq_head(hidden_states)
        if self.use_local_argmax_reduction:'''


def find_root(vllm_root=None):
    """Return the vllm package directory to patch, or None."""
    roots = []
    if vllm_root:
        roots.append(vllm_root)
    else:
        try:
            import vllm
            roots.append(os.path.dirname(vllm.__file__))
        except Exception:
            # A broken or CPU-only install must not surface a traceback here.
            pass
        roots.extend(FALLBACK_ROOTS)
    for root in roots:
        if os.path.exists(os.path.join(root, RELATIVE_TARGET)):
            return root
    return None


def apply(root):
    target = os.path.join(root, RELATIVE_TARGET)
    with open(target, encoding="utf-8") as handle:
        content = handle.read()

    if MARKER in content:
        print(f"SKIP: already applied to {target}")
        return 0

    if ANCHOR not in content:
        print(f"FAIL: _greedy_sample in {target} is not the vLLM 0.27.x form")
        print("      Patches are version specific; re-check against your vLLM.")
        return 1

    if not os.path.exists(RUNTIME_SOURCE):
        print(f"FAIL: runtime module missing at {RUNTIME_SOURCE}")
        return 1

    # The runtime lands inside the package so the injected hook can import it
    # by name without depending on sys.path.
    shutil.copyfile(RUNTIME_SOURCE, os.path.join(root, RUNTIME_NAME))
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(content.replace(ANCHOR, REPLACEMENT))
    print(f"OK: PQ draft head applied to {target}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm-root", help="path to the vllm package to patch")
    args = parser.parse_args()

    root = find_root(args.vllm_root)
    if root is None:
        print("FAIL: no vLLM installation found")
        print("      Pass --vllm-root /path/to/site-packages/vllm")
        return 1
    return apply(root)


if __name__ == "__main__":
    sys.exit(main())
