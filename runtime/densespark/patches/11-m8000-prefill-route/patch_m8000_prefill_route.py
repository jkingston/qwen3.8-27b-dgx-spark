#!/usr/bin/env python3
"""Extend Lab122's direct prefill route to the observed first-step M=8000.

The retained Lab130 trace proves that the official burst client starts the
engine after the first request arrives: the first GPU annotation is M=8000,
while subsequent rungs are M=8192.  Lab113's exact policy therefore sends the
fixed first rung through Humming.  This version-pinned patch adds only M=8000
to the already loaded direct layouts and updates the embedded audit manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_NAME = "_densespark_lab118_three_way_manifest.json"
CUTLASS_NAME = "_densespark_lab118_cutlass_component.py"
RUNTIME_NAME = "_densespark_lab118_three_way.py"

# The digests this step expects are not written here. The previous build step
# installs those files and records what it installed in the manifest beside
# them, so this reads them from there and writes back what it produced.
#
# Hardcoding them looked stronger and was weaker: the pins named an exact
# revision of two files, so editing a comment in either one - which happened,
# removing text that called shipped code a disposable experiment - broke a build
# nothing else could have caught. One of the six was the digest of a manifest
# that embeds package versions resolved inside the container, so it could not be
# recomputed outside one at all.

CUTLASS_REPLACEMENTS = (
    (
        "operator chooses CUTLASS only for the two measured row counts M={8192,16000};\n"
        "all other M use the unchanged Humming call.",
        "operator always chooses CUTLASS at M={8192,16000}.  Lab 133 adds the\n"
        "measured first-rung M=8000 only when ``DENSESPARK_LAB133_M8000=1``;\n"
        "all other M use the unchanged Humming call.",
    ),
    (
        "DIRECT_M_VALUES = (8192, 16000)",
        'LAB133_M8000_ENV = "DENSESPARK_LAB133_M8000"\n'
        '_lab133_m8000_value = os.environ.get(LAB133_M8000_ENV, "0")\n'
        'if _lab133_m8000_value not in ("0", "1"):\n'
        '    raise RuntimeError(\n'
        '        f"{LAB133_M8000_ENV} must be exactly 0 or 1, "\n'
        '        f"observed {_lab133_m8000_value!r}"\n'
        '    )\n'
        'LAB133_M8000_ENABLED = _lab133_m8000_value == "1"\n'
        "DIRECT_M_VALUES = ((8000,) if LAB133_M8000_ENABLED else ()) + (8192, 16000)",
    ),
    (
        '        "enabled": _strict_toggle(),\n',
        '        "enabled": _strict_toggle(),\n'
        '        "lab133_m8000_enabled": LAB133_M8000_ENABLED,\n',
    ),
)
RUNTIME_REPLACEMENTS = (
    (
        "three Lab109-selected families at exactly M={8192,16000}.",
        "three Lab109-selected families at M={8192,16000}, plus M=8000 only\n"
        "   when ``DENSESPARK_LAB133_M8000=1``.",
    ),
    (
        "at the two exact direct points, and CUTLASS at those points.",
        "at the enabled exact direct points, and CUTLASS at those points.",
    ),
    (
        "DIRECT_M_VALUES = (8192, 16000)",
        'LAB133_M8000_ENV = "DENSESPARK_LAB133_M8000"\n'
        '_lab133_m8000_value = os.environ.get(LAB133_M8000_ENV, "0")\n'
        'if _lab133_m8000_value not in ("0", "1"):\n'
        '    raise RuntimeError(\n'
        '        f"{LAB133_M8000_ENV} must be exactly 0 or 1, "\n'
        '        f"observed {_lab133_m8000_value!r}"\n'
        '    )\n'
        'LAB133_M8000_ENABLED = _lab133_m8000_value == "1"\n'
        "DIRECT_M_VALUES = ((8000,) if LAB133_M8000_ENABLED else ()) + (8192, 16000)",
    ),
    (
        '        "load_validated": _LOAD_VALIDATED,\n',
        '        "load_validated": _LOAD_VALIDATED,\n'
        '        "lab133_m8000_enabled": LAB133_M8000_ENABLED,\n',
    ),
    (
        '            "selected": "Marlin M<256; CUTLASS M in {8192,16000}; Humming otherwise",',
        '            "selected": (\n'
        '                "Marlin M<256; CUTLASS M in "\n'
        '                f"{DIRECT_M_VALUES}; Humming otherwise"\n'
        '            ),',
    ),
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def replace_exact(source: str, replacements: tuple[tuple[str, str], ...]) -> str:
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"Lab133 anchor is absent or duplicated: {old!r}")
        source = source.replace(old, new)
    return source


def patch_file(
    path: Path,
    *,
    expected_source_sha: str,
    replacements: tuple[tuple[str, str], ...],
    check: bool,
) -> str:
    """Apply the route extension once, and return the resulting digest."""

    payload = path.read_bytes()
    text = payload.decode("utf-8")
    if all(new in text for _, new in replacements):
        return sha256(payload)
    if check:
        raise RuntimeError(f"Lab133 is not applied at {path}")
    observed = sha256(payload)
    if observed != expected_source_sha:
        raise RuntimeError(
            f"Lab133 unexpected source at {path}: manifest records "
            f"{expected_source_sha}, found {observed}"
        )
    result = replace_exact(text, replacements).encode("utf-8")
    path.write_bytes(result)
    return sha256(result)


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_bytes())


def patch_manifest(path: Path, *, cutlass_sha: str, runtime_sha: str,
                   check: bool) -> None:
    manifest = read_manifest(path)
    if "lab133_route_extension" in manifest:
        if manifest.get("cutlass_component_sha256") != cutlass_sha \
           or manifest.get("runtime_sha256") != runtime_sha:
            raise RuntimeError(
                "Lab133 manifest records different component digests than the "
                "files beside it"
            )
        return
    if check:
        raise RuntimeError("Lab133 is not recorded in the component manifest")
    contract = manifest.get("contract", "")
    if contract.count("M={8192,16000}") != 1:
        raise RuntimeError("Lab133 embedded routing contract changed")
    manifest["contract"] = contract.replace(
        "direct only selected families at M={8192,16000}, Humming otherwise",
        "direct selected families at M={8192,16000}, plus M=8000 when "
        "DENSESPARK_LAB133_M8000=1; Humming otherwise",
    )
    manifest["cutlass_component_sha256"] = cutlass_sha
    manifest["runtime_sha256"] = runtime_sha
    manifest["lab133_route_extension"] = {
        "environment": "DENSESPARK_LAB133_M8000",
        "measured_M": 8000,
        "reason": "first Prompt-heavy scheduler step observed at M=8000",
    }
    path.write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def main() -> None:
    root = Path("/usr/local/lib/python3.12/dist-packages/vllm")
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=root)
    args = parser.parse_args()

    manifest_path = args.root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"Lab133 requires the component manifest at {manifest_path}")
    manifest = read_manifest(manifest_path)
    for key in ("cutlass_component_sha256", "runtime_sha256"):
        if not isinstance(manifest.get(key), str):
            raise RuntimeError(f"Lab133 component manifest has no {key}")

    cutlass_sha = patch_file(
        args.root / CUTLASS_NAME,
        expected_source_sha=manifest["cutlass_component_sha256"],
        replacements=CUTLASS_REPLACEMENTS,
        check=args.check,
    )
    runtime_sha = patch_file(
        args.root / RUNTIME_NAME,
        expected_source_sha=manifest["runtime_sha256"],
        replacements=RUNTIME_REPLACEMENTS,
        check=args.check,
    )
    patch_manifest(
        manifest_path,
        cutlass_sha=cutlass_sha,
        runtime_sha=runtime_sha,
        check=args.check,
    )
    print(
        "OK: M=8000 prefill route "
        + ("verified" if args.check else "applied")
        + f" ({args.root})"
    )


if __name__ == "__main__":
    main()
