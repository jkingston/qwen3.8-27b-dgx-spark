#!/usr/bin/env python3
"""Fail-closed Lab 86 patcher for pinned vLLM 0.27.1.

``LLMBaseProposer._sample_draft_tokens`` receives the sparse early-return.  The
runner remains byte-for-byte unchanged; rejection receives only a fail-closed
shape/dtype/presence guard because its dense FP32 draft-probability ABI is the
correctness boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import py_compile
import tempfile
from pathlib import Path
from typing import Any


PINNED_VLLM = "0.27.1"
ENV_NAME = "DENSESPARK_LAB86_SPARSE_PQ"
RUNTIME_NAME = "_densespark_lab86_sparse_pq.py"
MANIFEST_NAME = "_densespark_lab86_manifest.json"
PACKAGE_FILES = {
    "proposer": Path("v1/spec_decode/llm_base_proposer.py"),
    "runner": Path("v1/worker/gpu_model_runner.py"),
    "rejection": Path("v1/sample/rejection_sampler.py"),
    "sampling_ops": Path("v1/sample/ops/topk_topp_sampler.py"),
    "pq_runtime": Path("_densespark_pq.py"),
}
EXPECTED_CLEAN_SHA256 = {
    "proposer": "62b3f21cb35b9c6f374cca89f037d1045bca19ca19d95f63cb555a1d416a7c89",
    "runner": "d32ac540eafeef13ead6bb65ed3818b7200c2f83a1e6c35fbe2d0c71ea3266ab",
    "rejection": "4bb87d7984d967be9c7e5b6a6b489c02571ef2f2913211cb3f1eea74b2e1f32f",
    "sampling_ops": "bc8c1d91ad624368d2bd87a53c0361e550d28b060bcd460462d2f86565c26426",
    "pq_runtime": "7870c525327571c43b39d1bfe8717e8ed2b67bb356013386a7d6ff8f1d27804a",
}

ANCHOR = '''    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            return self._greedy_sample(hidden_states), None
        logits = self.model.compute_logits(hidden_states)
'''

REPLACEMENT = f'''    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # DENSESPARK_LAB86_SPARSE_PQ: the lab runtime returns both the sampled
        # token and the exact same q scattered into the dense rejection ABI.
        import os as _ds_lab86_os
        _ds_lab86_toggle = _ds_lab86_os.environ.get("{ENV_NAME}", "0")
        if _ds_lab86_toggle not in ("0", "1"):
            raise RuntimeError("{ENV_NAME} must be exactly 0 or 1")
        if _ds_lab86_toggle == "1":
            if (not self._enable_probabilistic_draft_probs
                    or sampling_metadata.all_greedy):
                raise RuntimeError(
                    "Lab 86 requires draft_sample_method=probabilistic and "
                    "a non-greedy target batch"
                )
            from vllm import _densespark_lab86_sparse_pq as _ds_lab86
            return _ds_lab86.sample_sparse_probabilistic(
                self, hidden_states, sampling_metadata
            )
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            return self._greedy_sample(hidden_states), None
        logits = self.model.compute_logits(hidden_states)
'''

MARKER = "DENSESPARK_LAB86_SPARSE_PQ: the lab runtime returns"
REJECTION_MARKER = "DENSESPARK_LAB86_SPARSE_PQ: stochastic q must reach rejection"
REJECTION_ANCHOR = '''    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    # Create output buffer.
'''
REJECTION_REPLACEMENT = f'''    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)

    # DENSESPARK_LAB86_SPARSE_PQ: stochastic q must reach rejection exactly.
    # The stock runner may otherwise fall back to draft_probs=None when its
    # request remap misses, which is not valid after sampling from sparse q.
    import os as _ds_lab86_os
    _ds_lab86_toggle = _ds_lab86_os.environ.get("{ENV_NAME}", "0")
    if _ds_lab86_toggle not in ("0", "1"):
        raise RuntimeError("{ENV_NAME} must be exactly 0 or 1")
    if _ds_lab86_toggle == "1":
        if (draft_probs is None
                or draft_probs.dtype != torch.float32
                or not draft_probs.is_contiguous()
                or draft_probs.shape != target_logits.shape):
            raise RuntimeError(
                "Lab 86 requires contiguous FP32 draft_probs with exactly "
                "the target [draft_rows,vocab] shape"
            )

    # Create output buffer.
'''
RUNNER_CONTRACT = (
    "self._draft_probs = draft_probs",
    "self.drafter.take_last_draft_probs()",
    "return torch.cat(draft_probs_rows, dim=0).contiguous()",
)
REJECTION_CONTRACT = (
    "target_prob / draft_prob >= uniform_prob",
    "tl.maximum(target_prob - draft_prob, 0.0)",
    "draft_probs.is_contiguous()",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def installed_version() -> str | None:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None


def default_source_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("cannot locate the installed vLLM package")
    roots = list(spec.submodule_search_locations)
    if len(roots) != 1:
        raise RuntimeError(f"ambiguous vLLM package roots: {roots}")
    return Path(roots[0])


def patch_text(source: str) -> str:
    if source.count(ANCHOR) != 1:
        raise RuntimeError(
            f"Lab 86 expected one proposer anchor, observed {source.count(ANCHOR)}"
        )
    transformed = source.replace(ANCHOR, REPLACEMENT, 1)
    compile(transformed, "llm_base_proposer.py", "exec")
    return transformed


def patch_rejection_text(source: str) -> str:
    if source.count(REJECTION_ANCHOR) != 1:
        raise RuntimeError(
            "Lab 86 expected one rejection ABI anchor, observed "
            f"{source.count(REJECTION_ANCHOR)}"
        )
    transformed = source.replace(
        REJECTION_ANCHOR, REJECTION_REPLACEMENT, 1
    )
    compile(transformed, "rejection_sampler.py", "exec")
    return transformed


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _paths(source_root: Path) -> dict[str, Path]:
    return {name: source_root / relative for name, relative in PACKAGE_FILES.items()}


def _clean_audit(source_root: Path, version: str | None) -> dict[str, Any]:
    paths = _paths(source_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Lab 86 source audit is missing {missing}")
    if version != PINNED_VLLM:
        raise RuntimeError(f"Lab 86 requires vLLM {PINNED_VLLM}, observed {version}")
    observed = {name: sha256_file(path) for name, path in paths.items()}
    mismatched = {
        name: {"observed": observed[name], "expected": EXPECTED_CLEAN_SHA256[name]}
        for name in paths
        if observed[name] != EXPECTED_CLEAN_SHA256[name]
    }
    if mismatched:
        raise RuntimeError(f"Lab 86 clean source hash mismatch: {mismatched}")
    sources = {
        name: path.read_text(encoding="utf-8")
        for name, path in paths.items()
        if name in ("proposer", "runner", "rejection")
    }
    if sources["proposer"].count(ANCHOR) != 1:
        raise RuntimeError("Lab 86 proposer anchor is absent or ambiguous")
    if sources["rejection"].count(REJECTION_ANCHOR) != 1:
        raise RuntimeError("Lab 86 rejection anchor is absent or ambiguous")
    for needle in RUNNER_CONTRACT:
        if needle not in sources["runner"]:
            raise RuntimeError(f"Lab 86 runner ABI anchor absent: {needle}")
    for needle in REJECTION_CONTRACT:
        if needle not in sources["rejection"]:
            raise RuntimeError(f"Lab 86 rejection ABI anchor absent: {needle}")
    return {"source_sha256": observed, "paths": paths, "sources": sources}


def apply(
    source_root: Path,
    runtime_source: Path,
    *,
    version: str | None = None,
) -> dict[str, Any]:
    version = installed_version() if version is None else version
    proposer_path = source_root / PACKAGE_FILES["proposer"]
    rejection_path = source_root / PACKAGE_FILES["rejection"]
    if (
        proposer_path.is_file()
        and MARKER in proposer_path.read_text(encoding="utf-8")
    ):
        result = audit(source_root, version=version)
        if result["state"] != "patched":
            raise RuntimeError("Lab 86 existing patch failed audit")
        return result
    clean = _clean_audit(source_root, version)
    if not runtime_source.is_file():
        raise RuntimeError(f"Lab 86 runtime is missing: {runtime_source}")
    runtime_bytes = runtime_source.read_bytes()
    compile(runtime_bytes, str(runtime_source), "exec")
    transformed_proposer = patch_text(clean["sources"]["proposer"])
    transformed_rejection = patch_rejection_text(clean["sources"]["rejection"])

    runtime_target = source_root / RUNTIME_NAME
    manifest_path = source_root / MANIFEST_NAME
    manifest = {
        "schema": 1,
        "vllm_version": PINNED_VLLM,
        "clean_source_sha256": clean["source_sha256"],
        "patched_source_sha256": {
            "proposer": sha256_text(transformed_proposer),
            "rejection": sha256_text(transformed_rejection),
        },
        "lab_runtime_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
        "sampled_and_reported_q_contract": (
            "one sparse_probabilities tensor -> exponential-race sample + dense scatter"
        ),
    }
    # Install the runtime first.  A process can never import the patched hook
    # without its module; the final manifest makes partial installation fail audit.
    _atomic_write(runtime_target, runtime_bytes)
    _atomic_write(proposer_path, transformed_proposer.encode("utf-8"))
    _atomic_write(rejection_path, transformed_rejection.encode("utf-8"))
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return audit(source_root, version=version)


def audit(source_root: Path, *, version: str | None = None) -> dict[str, Any]:
    version = installed_version() if version is None else version
    if version != PINNED_VLLM:
        raise RuntimeError(f"Lab 86 requires vLLM {PINNED_VLLM}, observed {version}")
    proposer_path = source_root / PACKAGE_FILES["proposer"]
    rejection_path = source_root / PACKAGE_FILES["rejection"]
    runtime_path = source_root / RUNTIME_NAME
    manifest_path = source_root / MANIFEST_NAME
    if not proposer_path.is_file():
        raise RuntimeError("Lab 86 proposer source is missing")
    proposer = proposer_path.read_text(encoding="utf-8")
    rejection = rejection_path.read_text(encoding="utf-8")
    if MARKER not in proposer:
        clean = _clean_audit(source_root, version)
        return {
            "state": "clean",
            "vllm_version": version,
            "source_sha256": clean["source_sha256"],
        }
    if not runtime_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("Lab 86 patched source lacks runtime or manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patched_hashes = {
        "proposer": sha256_file(proposer_path),
        "rejection": sha256_file(rejection_path),
    }
    required = {
        "schema": 1,
        "vllm_version": PINNED_VLLM,
        "patched_source_sha256": patched_hashes,
        "lab_runtime_sha256": sha256_file(runtime_path),
    }
    malformed = {
        key: {"observed": manifest.get(key), "expected": expected}
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if malformed:
        raise RuntimeError(f"Lab 86 manifest mismatch: {malformed}")
    clean_hashes = manifest.get("clean_source_sha256")
    if clean_hashes != EXPECTED_CLEAN_SHA256:
        raise RuntimeError("Lab 86 manifest does not bind the pinned clean sources")
    for name in ("runner", "sampling_ops", "pq_runtime"):
        path = source_root / PACKAGE_FILES[name]
        if not path.is_file() or sha256_file(path) != EXPECTED_CLEAN_SHA256[name]:
            raise RuntimeError(f"Lab 86 immutable dependency changed: {name}")
    gate = f'_ds_lab86_os.environ.get("{ENV_NAME}", "0")'
    if proposer.count(MARKER) != 1 or proposer.count(gate) != 1:
        raise RuntimeError("Lab 86 proposer hook is absent or duplicated")
    if rejection.count(REJECTION_MARKER) != 1 or rejection.count(gate) != 1:
        raise RuntimeError("Lab 86 rejection guard is absent or duplicated")
    py_compile.compile(str(proposer_path), doraise=True)
    py_compile.compile(str(rejection_path), doraise=True)
    py_compile.compile(str(runtime_path), doraise=True)
    return {
        "state": "patched",
        "vllm_version": version,
        "patched_source_sha256": patched_hashes,
        "runtime_sha256": sha256_file(runtime_path),
        "immutable_dependency_sha256": {
            name: sha256_file(source_root / PACKAGE_FILES[name])
            for name in ("runner", "sampling_ops", "pq_runtime")
        },
        "manifest": manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.source_root or default_source_root()
    if args.apply:
        if args.runtime is None:
            parser.error("--apply requires --runtime")
        result = apply(root, args.runtime)
    else:
        result = audit(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
