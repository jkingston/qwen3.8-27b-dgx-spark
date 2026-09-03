#!/usr/bin/env python3
"""Patch/audit the three-way Marlin/Humming/direct-CUTLASS linear composition.

``--apply`` rewrites the installed Humming linear kernel so that each backbone
projection dispatches on its row count.  ``--check`` re-audits an already
patched tree.  ``--audit-source`` is GPU-free and verifies that the shipped
runtime module implements the intended routing policy and memory model.
``--audit-compile-cache`` is a GPU/model-load gate and fails unless every
backbone linear retains the correct symbolic outer branch and all selected
large-M routes remain AOT-opaque.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PINNED_VLLM = "0.27.1"
PINNED_HUMMING = "0.1.13"
TARGET = Path("model_executor/kernels/linear/mixed_precision/humming.py")
MARLIN_TARGET = Path("model_executor/kernels/linear/mixed_precision/marlin.py")
AUTO_GPTQ_TARGET = Path("model_executor/layers/quantization/auto_gptq.py")
HUMMING_UTILS_TARGET = Path("model_executor/layers/quantization/utils/humming_utils.py")
CUSTOM_OPS_TARGET = Path("_custom_ops.py")
RUNTIME_NAME = "_densespark_lab118_three_way.py"
MARLIN_COMPONENT_NAME = "_densespark_lab118_marlin_component.py"
CUTLASS_COMPONENT_NAME = "_densespark_lab118_cutlass_component.py"
MANIFEST_NAME = "_densespark_lab118_three_way_manifest.json"
MARKER = "DENSESPARK_LAB118_THREE_WAY"
MANIFEST_SCHEMA = 1

EXPECTED_CLEAN_FILES = {
    "humming": (TARGET, "bf869f0a18282f9d74704a3bd487bed8895819048d6d32d4ef1ea05039a97893"),
    "marlin": (MARLIN_TARGET, "9c2b728f7a040a7e7bc52dba334c4af3d1285d529cdd65d9020dfdc712278dff"),
    "auto_gptq": (AUTO_GPTQ_TARGET, "0f9303581215064b45de51111f36d122d3c7d269631be85b7e996359927a02b2"),
    "humming_utils": (HUMMING_UTILS_TARGET, "71927176957f6c54a4d2969d92520a5b6bda2b0d70b98ab5dc693f35e9ec9fa2"),
    "custom_ops": (CUSTOM_OPS_TARGET, "9fb29a33a63cd625b42a0bf000d821bb371d5eefa24a2376525fc371155afd02"),
}

# The shipped component sources, pinned so the transform refuses to run against
# anything but the files it was verified with. Update these together with the
# files; the test suite fails first, before a container build does.
COMPONENT_SHA256 = {
    "three_way": "fc3a567f396c65e16cd35d868ba184028ca249644c7412cc6940d4aac1d9856a",
    "marlin_component": "6562d8c8c028f87d40e3853297de4fc19a76c0ace60694697c7bea77cd0f2ae3",
    "cutlass_component": "4ff7464406ec2e7806bb2491717a32a1df5a3c7801fe23a5ce5403df9b4332f4",
}

ANCHOR = """        return output.view(*x.shape[:-1], output.size(-1))
"""
HOOK = """

# DENSESPARK_LAB118_THREE_WAY: three-layout linear dispatch.
from vllm import _densespark_lab118_three_way as _ds_lab118  # noqa: E402

_ds_lab118.install(HummingLinearKernel)
"""

COND_OP = "torch.ops.higher_order.cond"
HUMMING_OP = "torch.ops.humming.humming_gemm"
MARLIN_OP = "torch.ops._C.marlin_gemm"
HYBRID_OP = "torch.ops.densespark_lab113.hybrid_linear"
CUTLASS_OP = "torch.ops._C.cutlass_scaled_mm"
EXPECTED_GRAPHS = {
    "backbone": {
        "conditionals": 256,
        "selected_true_hybrid": 128,
        "unselected_true_humming": 128,
        "false_marlin": 256,
    },
    # The MTP graph reuses loaded body layouts but has four additional linear
    # call sites: two selected shapes and two unselected shapes.
    "eagle_head": {
        "conditionals": 4,
        "selected_true_hybrid": 2,
        "unselected_true_humming": 2,
        "false_marlin": 4,
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_source_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("Lab118 cannot locate installed vLLM")
    roots = list(spec.submodule_search_locations)
    if len(roots) != 1:
        raise RuntimeError(f"Lab118 found ambiguous vLLM roots: {roots}")
    return Path(roots[0])


def _versions() -> dict[str, str]:
    versions = {
        "vllm": importlib.metadata.version("vllm"),
        "humming-kernels": importlib.metadata.version("humming-kernels"),
    }
    if versions != {"vllm": PINNED_VLLM, "humming-kernels": PINNED_HUMMING}:
        raise RuntimeError(f"Lab118 package versions changed: {versions}")
    return versions


def patch_text(source: str) -> str:
    if source.count(ANCHOR) != 1 or not source.rstrip().endswith(ANCHOR.rstrip()):
        raise RuntimeError("Lab118 clean Humming anchor is absent or ambiguous")
    transformed = source.rstrip("\n") + HOOK
    compile(transformed, str(TARGET), "exec")
    return transformed


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _clean_dependency_audit(source_root: Path) -> dict[str, Any]:
    rows = {}
    for name, (relative, expected) in EXPECTED_CLEAN_FILES.items():
        path = source_root / relative
        actual = sha256_file(path) if path.is_file() else None
        rows[name] = {"path": str(path), "sha256": actual, "expected_sha256": expected, "passed": actual == expected}
    if not all(row["passed"] for row in rows.values()):
        raise RuntimeError(f"Lab118 clean vLLM source mismatch: {rows}")
    return rows


def apply(
    source_root: Path,
    runtime_source: Path,
    marlin_component_source: Path,
    cutlass_component_source: Path,
) -> dict[str, Any]:
    versions = _versions()
    target = source_root / TARGET
    source = target.read_text(encoding="utf-8")
    if MARKER in source:
        return audit_patched_tree(source_root)
    dependencies = _clean_dependency_audit(source_root)
    components = {
        "runtime": (runtime_source, COMPONENT_SHA256["three_way"]),
        "marlin_component": (marlin_component_source, COMPONENT_SHA256["marlin_component"]),
        "cutlass_component": (cutlass_component_source, COMPONENT_SHA256["cutlass_component"]),
    }
    payloads = {}
    for name, (path, expected) in components.items():
        payload = path.read_bytes()
        compile(payload, str(path), "exec")
        observed = sha256_bytes(payload)
        if expected is not None and observed != expected:
            raise RuntimeError(f"Lab118 {name} source changed: expected {expected}, observed {observed}")
        payloads[name] = payload
    transformed = patch_text(source).encode()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "versions": versions,
        "contract": (
            "outer torch.cond M>=256: false stock Marlin; true opaque "
            "direct only selected families at M={8192,16000}, Humming otherwise"
        ),
        "clean_dependencies": dependencies,
        "patched_humming_sha256": sha256_bytes(transformed),
        "runtime_sha256": sha256_bytes(payloads["runtime"]),
        "marlin_component_sha256": sha256_bytes(payloads["marlin_component"]),
        "cutlass_component_sha256": sha256_bytes(payloads["cutlass_component"]),
        "install_order": ["cutlass_component", "marlin_component", "lab118_load_audit"],
    }
    _atomic_write(source_root / RUNTIME_NAME, payloads["runtime"])
    _atomic_write(source_root / MARLIN_COMPONENT_NAME, payloads["marlin_component"])
    _atomic_write(source_root / CUTLASS_COMPONENT_NAME, payloads["cutlass_component"])
    _atomic_write(target, transformed)
    _atomic_write(source_root / MANIFEST_NAME, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return audit_patched_tree(source_root)


def audit_patched_tree(source_root: Path) -> dict[str, Any]:
    versions = _versions()
    target = source_root / TARGET
    manifest_path = source_root / MANIFEST_NAME
    source = target.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {
        "runtime_sha256": source_root / RUNTIME_NAME,
        "marlin_component_sha256": source_root / MARLIN_COMPONENT_NAME,
        "cutlass_component_sha256": source_root / CUTLASS_COMPONENT_NAME,
    }
    checks = {
        "schema": manifest.get("schema") == MANIFEST_SCHEMA,
        "versions": manifest.get("versions") == versions,
        "single_marker": source.count(MARKER) == 1,
        "patched_humming": manifest.get("patched_humming_sha256") == sha256_file(target),
        "install_order": manifest.get("install_order") == ["cutlass_component", "marlin_component", "lab118_load_audit"],
        "components": all(path.is_file() and manifest.get(key) == sha256_file(path) for key, path in paths.items()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Lab118 patched-tree audit failed: {checks}")
    # The Humming target is intentionally patched, so validate its clean hash
    # through the immutable manifest and validate only the untouched siblings
    # against disk here.
    clean_manifest = manifest.get("clean_dependencies", {})
    dependency_checks = {}
    for name, (relative, expected) in EXPECTED_CLEAN_FILES.items():
        if clean_manifest.get(name, {}).get("expected_sha256") != expected:
            raise RuntimeError(f"Lab118 manifest lost clean pin for {name}")
        if name == "humming":
            continue
        observed = sha256_file(source_root / relative)
        dependency_checks[name] = observed == expected
    if not all(dependency_checks.values()):
        raise RuntimeError(f"Lab118 untouched dependency changed: {dependency_checks}")
    compile(source, str(target), "exec")
    return {"state": "patched", "passed": True, "checks": {**checks, **{f'unmodified_{name}': passed for name, passed in dependency_checks.items()}}, "versions": versions, **{key: sha256_file(path) for key, path in paths.items()}}


def audit_source(runtime_source: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("densespark_three_way_source_audit", runtime_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the three-way runtime {runtime_source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    route_checks = {
        "selected_m1_marlin": module.route_for_rows(1, selected=True) == module.MARLIN,
        "selected_m255_marlin": module.route_for_rows(255, selected=True) == module.MARLIN,
        "selected_m256_humming": module.route_for_rows(256, selected=True) == module.HUMMING,
        "selected_m2048_humming": module.route_for_rows(2048, selected=True) == module.HUMMING,
        "selected_m8192_direct": module.route_for_rows(8192, selected=True) == module.DIRECT,
        "selected_m16000_direct": module.route_for_rows(16000, selected=True) == module.DIRECT,
        "selected_m8193_humming": module.route_for_rows(8193, selected=True) == module.HUMMING,
        "unselected_m8192_humming": module.route_for_rows(8192, selected=False) == module.HUMMING,
        "unselected_m16000_humming": module.route_for_rows(16000, selected=False) == module.HUMMING,
    }
    memory = module.expected_extra_memory()
    memory_checks = {
        "marlin_payload": memory["marlin_qweight_scale_bytes"] == 12_735_528_960,
        "direct_payload": memory["direct_fp8_scale_bytes"] == 16_865_218_560,
        "combined_payload": memory["total_modeled_payload_bytes"] == 29_600_747_520,
    }
    source_text = runtime_source.read_text(encoding="utf-8")
    order_checks = {
        "cutlass_install_precedes_marlin": source_text.index("cutlass_component.install(humming_kernel_cls)") < source_text.index("marlin_component.install(humming_kernel_cls)"),
        "no_apply_io": all(token not in source_text[source_text.index("    def apply_weights("):source_text.index("    humming_kernel_cls.process_weights_after_loading = process_weights_after_loading")] for token in ("Path(", "open(", "print(", "_write_runtime_audit(")),
    }
    checks = {**route_checks, **memory_checks, **order_checks}
    if not all(checks.values()):
        raise RuntimeError(f"three-way source composition failed: {checks}")
    return {
        "schema": "densespark.three-way.source-readiness.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "verdict": "THREE_WAY_SOURCE_COMPOSITION_READY_GPU_NOT_RUN",
        "checks": checks,
        "runtime_sha256": sha256_file(runtime_source),
        "extra_memory": memory,
        "policy": {
            "selected_families": sorted(module.SELECTED_FAMILIES),
            "selected": "Marlin M<256; direct M={8192,16000}; Humming otherwise",
            "unselected": "Marlin M<256; Humming M>=256",
        },
    }


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return None if parent is None else f"{parent}.{node.attr}"
    return None


def _is_call(node: ast.AST, name: str) -> bool:
    observed = _dotted_name(node.func) if isinstance(node, ast.Call) else None
    return observed == name or (
        observed is not None and observed.startswith(name + ".")
    )


def _call_count(node: ast.AST, name: str) -> int:
    return sum(_is_call(candidate, name) for candidate in ast.walk(node))


def _assignments(function: ast.FunctionDef) -> dict[str, ast.AST]:
    result = {}
    for statement in function.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            result[statement.targets[0].id] = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
        ):
            result[statement.target.id] = statement.value
    return result


def _assignment_records(
    function: ast.FunctionDef,
) -> dict[str, list[tuple[ast.AST, ast.AST]]]:
    result: dict[str, list[tuple[ast.AST, ast.AST]]] = {}
    for statement in function.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    result.setdefault(target.id, []).append((target, statement.value))
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
        ):
            result.setdefault(statement.target.id, []).append(
                (statement.target, statement.value)
            )
    return result


def _position(node: ast.AST) -> tuple[int, int] | None:
    line = getattr(node, "lineno", None)
    column = getattr(node, "col_offset", None)
    if not isinstance(line, int) or not isinstance(column, int):
        return None
    return line, column


def _unique_non_none_value_before_cleanup(
    node: ast.Name, function: ast.FunctionDef
) -> ast.AST | None:
    """Resolve Dynamo SSA values without treating later liveness cleanup as data."""

    use_position = _position(node)
    if use_position is None:
        return None
    records = _assignment_records(function).get(node.id, [])
    producers = [
        (target, value)
        for target, value in records
        if not (isinstance(value, ast.Constant) and value.value is None)
    ]
    cleanups = [
        target
        for target, value in records
        if isinstance(value, ast.Constant) and value.value is None
    ]
    if len(producers) != 1 or len(producers) + len(cleanups) != len(records):
        return None
    producer_target, producer_value = producers[0]
    producer_position = _position(producer_target)
    cleanup_positions = [_position(target) for target in cleanups]
    if producer_position is None or producer_position >= use_position:
        return None
    if any(position is None or position <= use_position for position in cleanup_positions):
        return None
    return producer_value


def _certified_parameter_is_live(
    node: ast.Name,
    function: ast.FunctionDef,
    certified_parameters: frozenset[str],
) -> bool:
    if node.id not in certified_parameters:
        return False
    if node.id not in {argument.arg for argument in function.args.args[1:]}:
        return False
    use_position = _position(node)
    if use_position is None:
        return False
    for target, value in _assignment_records(function).get(node.id, []):
        target_position = _position(target)
        if (
            not isinstance(value, ast.Constant)
            or value.value is not None
            or target_position is None
            or target_position <= use_position
        ):
            return False
    return True


def _resolve(node: ast.AST, assignments: dict[str, ast.AST]) -> ast.AST:
    seen = set()
    while isinstance(node, ast.Name) and node.id in assignments:
        if node.id in seen:
            raise RuntimeError("Lab118 cyclic graph assignment")
        seen.add(node.id)
        node = assignments[node.id]
    return node


def _branch(node: ast.AST, assignments: dict[str, ast.AST], children: dict[str, ast.ClassDef]) -> ast.ClassDef | None:
    node = _resolve(node, assignments)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return children.get(node.attr)
    return None


def _predicate_is_256(
    node: ast.AST,
    assignments: dict[str, ast.AST],
) -> bool:
    node = _resolve(node, assignments)
    return isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.GtE) and len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value == 256


def _forward(node: ast.ClassDef) -> ast.FunctionDef | None:
    return next(
        (
            child
            for child in node.body
            if isinstance(child, ast.FunctionDef) and child.name == "forward"
        ),
        None,
    )


def _self_child_call(
    node: ast.AST, children: dict[str, ast.ClassDef]
) -> tuple[ast.Call, ast.ClassDef] | None:
    if not isinstance(node, ast.Call):
        return None
    function = node.func
    if (
        not isinstance(function, ast.Attribute)
        or not isinstance(function.value, ast.Name)
        or function.value.id != "self"
        or function.attr not in children
    ):
        return None
    return node, children[function.attr]


def _structural_predicate_is_256(
    node: ast.AST,
    function: ast.FunctionDef,
    children: dict[str, ast.ClassDef],
    certified_parameters: frozenset[str],
    trail: frozenset[tuple[int, str]] = frozenset(),
) -> bool:
    """Prove a predicate through FX assignments and child return slots.

    String annotations are deliberately ignored.  A lifted predicate is
    accepted only when the AST call/return dataflow reaches the literal
    ``symbol >= 256`` comparison, or a formal already certified from every
    callsite of this exact child module.
    """

    marker = (id(function), ast.dump(node, include_attributes=False))
    if marker in trail:
        return False
    trail = trail | {marker}
    if isinstance(node, ast.Name):
        if _certified_parameter_is_live(node, function, certified_parameters):
            return True
        value = _unique_non_none_value_before_cleanup(node, function)
        if value is not None:
            return _structural_predicate_is_256(
                value, function, children, certified_parameters, trail
            )
        if _assignment_records(function).get(node.id):
            return False
        return False
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.GtE)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == 256
    ):
        return True
    if not isinstance(node, ast.Subscript):
        return False
    index = node.slice
    if not isinstance(index, ast.Constant) or type(index.value) is not int:
        return False
    producer = node.value
    if isinstance(producer, ast.Name):
        value = _unique_non_none_value_before_cleanup(producer, function)
        if value is None:
            return False
        producer = value
    matched = _self_child_call(producer, children)
    if matched is None:
        return False
    _call, child = matched
    child_forward = _forward(child)
    if child_forward is None:
        return False
    returns = [
        statement
        for statement in child_forward.body
        if isinstance(statement, ast.Return) and statement.value is not None
    ]
    if len(returns) != 1:
        return False
    returned = returns[0].value
    if isinstance(returned, ast.Name):
        value = _unique_non_none_value_before_cleanup(returned, child_forward)
        if value is None:
            return False
        returned = value
    if not isinstance(returned, (ast.Tuple, ast.List)):
        return False
    if index.value < 0 or index.value >= len(returned.elts):
        return False
    child_children = {
        nested.name: nested
        for nested in child.body
        if isinstance(nested, ast.ClassDef)
    }
    return _structural_predicate_is_256(
        returned.elts[index.value],
        child_forward,
        child_children,
        frozenset(),
        trail,
    )


def _certified_child_parameters(
    child: ast.ClassDef,
    parent_forward: ast.FunctionDef,
    parent_children: dict[str, ast.ClassDef],
    parent_certified: frozenset[str],
) -> frozenset[str]:
    child_forward = _forward(child)
    if child_forward is None:
        return frozenset()
    calls = [
        candidate
        for candidate in ast.walk(parent_forward)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Attribute)
        and isinstance(candidate.func.value, ast.Name)
        and candidate.func.value.id == "self"
        and candidate.func.attr == child.name
    ]
    if not calls:
        return frozenset()
    certified = set()
    for position, argument in enumerate(child_forward.args.args[1:]):
        if all(
            len(call.args) > position
            and _structural_predicate_is_256(
                call.args[position],
                parent_forward,
                parent_children,
                parent_certified,
            )
            for call in calls
        ):
            certified.add(argument.arg)
    return frozenset(certified)


def _audit_graph_class(
    node: ast.ClassDef, certified_parameters: frozenset[str] = frozenset()
) -> Counter[str]:
    counts: Counter[str] = Counter()
    children = {child.name: child for child in node.body if isinstance(child, ast.ClassDef)}
    forward = _forward(node)
    if forward is not None:
        assignments = _assignments(forward)
        for call in (candidate for candidate in ast.walk(forward) if _is_call(candidate, COND_OP)):
            assert isinstance(call, ast.Call)
            if len(call.args) < 3 or not (
                _predicate_is_256(call.args[0], assignments)
                or _structural_predicate_is_256(
                    call.args[0],
                    forward,
                    children,
                    certified_parameters,
                )
            ):
                raise RuntimeError("Lab118 found a non-M>=256 outer conditional")
            true_branch = _branch(call.args[1], assignments, children)
            false_branch = _branch(call.args[2], assignments, children)
            if true_branch is None or false_branch is None:
                raise RuntimeError("Lab118 cannot resolve an AOT branch module")
            signature = (
                _call_count(true_branch, HYBRID_OP),
                _call_count(true_branch, HUMMING_OP),
                _call_count(true_branch, MARLIN_OP),
                _call_count(false_branch, MARLIN_OP),
                _call_count(false_branch, HUMMING_OP),
                _call_count(false_branch, HYBRID_OP),
            )
            if signature == (1, 0, 0, 1, 0, 0):
                counts["selected_true_hybrid"] += 1
            elif signature == (0, 1, 0, 1, 0, 0):
                counts["unselected_true_humming"] += 1
            else:
                raise RuntimeError(f"Lab118 AOT branch signature changed: {signature}")
            counts["conditionals"] += 1
            counts["false_marlin"] += 1
    for child in children.values():
        child_certified = (
            _certified_child_parameters(
                child, forward, children, certified_parameters
            )
            if forward is not None
            else frozenset()
        )
        counts.update(_audit_graph_class(child, child_certified))
    return counts


def audit_compile_cache(cache_root: Path, runtime_audit: Path) -> dict[str, Any]:
    graphs = sorted(cache_root.rglob("computation_graph.py"))
    graph_records = []
    for role, expected in EXPECTED_GRAPHS.items():
        matches = [path for path in graphs if role in path.parts]
        if len(matches) != 1:
            raise RuntimeError(
                f"Lab118 expected exactly one {role} graph, observed {matches}"
            )
        path = matches[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        top = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        if len(top) != 1:
            raise RuntimeError(f"Lab118 expected one top-level {role} GraphModule")
        counts = dict(_audit_graph_class(top[0]))
        if counts != expected:
            raise RuntimeError(
                f"Lab118 incomplete three-way {role} graph: {counts}"
            )
        if _call_count(tree, CUTLASS_OP) != 0:
            raise RuntimeError(
                f"Lab118 raw CUTLASS call escaped the opaque op in {role}"
            )
        graph_records.append(
            {"role": role, "path": str(path), "sha256": sha256_file(path), "counts": counts}
        )
    load = json.loads(runtime_audit.read_text(encoding="utf-8"))
    load_checks = {
        "schema": load.get("schema") == "densespark.lab118.runtime-load-audit.v1",
        "load_validated": load.get("load_validated") is True,
        "counts": load.get("observed_family_counts") == load.get("expected_family_counts"),
        "marlin_bytes": load.get("observed_marlin_payload_bytes") == load.get("expected_extra_memory", {}).get("marlin_qweight_scale_bytes"),
        "direct_bytes": load.get("observed_direct_payload_bytes") == load.get("expected_extra_memory", {}).get("direct_fp8_scale_bytes"),
    }
    if not all(load_checks.values()):
        raise RuntimeError(f"Lab118 runtime load audit failed: {load_checks}")
    return {
        "schema": "densespark.lab118.model-load-aot.v1",
        "passed": True,
        "verdict": "THREE_WAY_MODEL_LOAD_SYMBOLIC_AOT_CONFIRMED",
        "graphs": graph_records,
        "runtime_load": {"path": str(runtime_audit), "sha256": sha256_file(runtime_audit), "checks": load_checks},
        "task_quality": "NOT_RUN",
        "serving_performance": "NOT_RUN",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--runtime", type=Path, default=Path(__file__).with_name("densespark_three_way.py"))
    parser.add_argument("--marlin-component", type=Path, default=Path(__file__).with_name("densespark_marlin_component.py"))
    parser.add_argument("--cutlass-component", type=Path, default=Path(__file__).with_name("densespark_cutlass_component.py"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--audit-source", action="store_true")
    parser.add_argument("--audit-compile-cache", type=Path)
    parser.add_argument("--runtime-audit", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    modes = sum((args.apply, args.audit_source, args.audit_compile_cache is not None))
    if modes > 1:
        parser.error("choose only one of --apply, --audit-source, --audit-compile-cache")
    if args.audit_source:
        result = audit_source(args.runtime.resolve())
    elif args.audit_compile_cache is not None:
        if args.runtime_audit is None:
            parser.error("--audit-compile-cache requires --runtime-audit")
        result = audit_compile_cache(args.audit_compile_cache, args.runtime_audit)
    elif args.apply:
        result = apply(args.source_root or default_source_root(), args.runtime, args.marlin_component, args.cutlass_component)
    else:
        result = audit_patched_tree(args.source_root or default_source_root())
    result["command_argv"] = list(sys.argv)
    result["exact_command"] = shlex.join(sys.argv)
    result["script_sha256"] = sha256_file(Path(__file__).resolve())
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
