#!/usr/bin/env python3
"""Version-pinned Lab132 patch enabling FlashInfer GDN on SM12.x.

vLLM 0.27.1 only admits family 10.x even though the bundled FlashInfer
0.6.16.post3 implementation contains explicit SM120 and SM90/SM120 CP paths.
This patch changes only backend eligibility; explicit ``--gdn-prefill-backend
flashinfer`` remains required by the experiment launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from pathlib import Path


VLLM_VERSION = "0.27.1"
FLASHINFER_VERSION = "0.6.16.post3"
SOURCE_SHA256 = "1227d6f385a52296e9f08223544b1c5fdc7e8d9aa09a848e7a8e522a8dc51214"
PATCHED_SHA256 = "d42cdc95d8d221b49693a46119c714fee3f290282bdfefa63f92f9725f1b20ea"
FLASHINFER_SHA256 = "a486f6d4a6a59681e754a19b8d52efd14b3bbf6c8835272fb2d846bb6b06debb"

OLD_DOC = '''    FlashInfer's GDN prefill kernel is chosen when:
    * ``requested in ["flashinfer", "auto"]``;
    * ``platform == cuda``;
    * one of the following:
      - Hopper (SM90) — no further constraints;
      - Blackwell (SM10.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``.

    In-tree CuteDSL GDN prefill kernel is chosen when:
    * "cutedsl" is requested; (opt-in only)
    * Blackwell (SM10.x) with ``head_k_dim == 128``;
'''
NEW_DOC = '''    FlashInfer's GDN prefill kernel is chosen when:
    * ``requested in ["flashinfer", "auto"]``;
    * ``platform == cuda``;
    * one of the following:
      - Hopper (SM90) — no further constraints;
      - Blackwell (SM10.x or SM12.x) with ``head_k_dim == 128`` and
        ``cuda_runtime >= 13``.  The installed FlashInfer package carries a
        distinct SM120 implementation.

    In-tree CuteDSL GDN prefill kernel is chosen when:
    * "cutedsl" is requested; (opt-in only)
    * Blackwell (SM10.x) with ``head_k_dim == 128``;
'''
OLD_CODE = '''    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True
'''
NEW_CODE = '''    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True
    elif (
        current_platform.is_device_capability_family(120)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        # FlashInfer ships a distinct SM120 implementation.  The in-tree
        # CuteDSL backend remains limited to the family it was written for.
        supports_flashinfer = True
'''


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def patched_text(source: str) -> str:
    if source.count(OLD_DOC) != 1 or source.count(OLD_CODE) != 1:
        raise RuntimeError("Lab132 pinned vLLM anchors are absent or duplicated")
    return source.replace(OLD_DOC, NEW_DOC).replace(OLD_CODE, NEW_CODE)


def validate_flashinfer(path: Path) -> None:
    payload = path.read_bytes()
    if sha256(payload) != FLASHINFER_SHA256:
        raise RuntimeError("Lab132 bundled FlashInfer source hash changed")
    source = payload.decode("utf-8")
    required = (
        "chunk_gated_delta_rule_sm120",
        "cp_delta_rule_dsl_sm120",
        "elif _arch_major == 12:",
        'raise NotImplementedError("SM120 GDN prefill DSL kernel is unavailable")',
    )
    missing = [anchor for anchor in required if anchor not in source]
    if missing:
        raise RuntimeError(f"Lab132 SM120 FlashInfer contracts are missing: {missing}")


def apply_or_check(vllm_path: Path, flashinfer_path: Path, *, check: bool) -> None:
    if importlib.metadata.version("vllm") != VLLM_VERSION:
        raise RuntimeError("Lab132 requires vLLM 0.27.1")
    if importlib.metadata.version("flashinfer-python") != FLASHINFER_VERSION:
        raise RuntimeError("Lab132 requires flashinfer-python 0.6.16.post3")
    validate_flashinfer(flashinfer_path)
    payload = vllm_path.read_bytes()
    observed = sha256(payload)
    if observed == PATCHED_SHA256:
        return
    if check:
        raise RuntimeError(f"Lab132 patch is absent; observed SHA256 {observed}")
    if observed != SOURCE_SHA256:
        raise RuntimeError(f"Lab132 vLLM source hash changed: {observed}")
    result = patched_text(payload.decode("utf-8")).encode("utf-8")
    if sha256(result) != PATCHED_SHA256:
        raise RuntimeError("Lab132 patched source does not match the frozen digest")
    vllm_path.write_bytes(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--vllm-path",
        type=Path,
        default=Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
            "layers/mamba/gdn/qwen_gdn_linear_attn.py"
        ),
    )
    parser.add_argument(
        "--flashinfer-path",
        type=Path,
        default=Path("/usr/local/lib/python3.12/dist-packages/flashinfer/gdn_prefill.py"),
    )
    args = parser.parse_args()
    apply_or_check(args.vllm_path, args.flashinfer_path, check=args.check)
    print(f"Lab132 patch verified: {PATCHED_SHA256}")


if __name__ == "__main__":
    main()
