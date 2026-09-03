#!/usr/bin/env python3
"""Build the product-quantization structure the PQ draft head loads.

The structure is three tensors derived from the deployed INT8 head:

    codes      [248320, 128] uint8   31,784,960 B
    centroids  [128, 256, 40] fp16    2,621,440 B
    row_scale  [248320] fp16            496,640 B

The production builder owns the fixed deterministic training schedule:

  --train           train from the checkpoint's lm_head. Needs a GPU and about
                    8 GB of peak scratch.

  --validate-artifact PATH
                    check schema, payload SHA256, selected model id, and the
                    mounted snapshot revision without loading the head.

Usage:
    python3 build_pq_artifact.py --train --snapshot /checkpoint/snapshots/REV \
        --out /out/pq_head_m128.pt

The runtime separately hashes every deployed INT8 coefficient and FP16 scale;
the metadata-only validation mode is an installer fast path, not a substitute
for that startup gate.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

M = 128
KSUB = 256
DSUB = 40
VOCAB = 248320
HIDDEN = 5120
ARTIFACT_SCHEMA = 2
ARTIFACT_VARIANT = "norm_match_m128"
ARTIFACT_DIGEST_ALGORITHM = "sha256-densespark-pq-runtime-v2"
HEAD_DIGEST_ALGORITHM = "sha256-densespark-int8-row-chunks-v1"
CANONICAL_TRAINING_ITERS = 8
CANONICAL_TRAINING_SEED = 1
ASSIGN_CHUNK = 2048
HEAD_DIGEST_ROWS = 4096
DEFAULT_MODEL_REPO = "Frozenlock/Qwen3.8-27B-int4-AutoRound"
KEEP = ("codes", "centroids", "row_scale")
ARTIFACT_METADATA_KEYS = (
    "schema", "variant", "m", "ksub", "dsub", "vocab", "hidden",
    "row_scale_mode", "training_iters", "training_seed",
    "checkpoint_model", "checkpoint_revision", "head_digest_algorithm",
    "head_sha256", "artifact_digest_algorithm",
)


def update_field(digest, name, value):
    name_bytes = name.encode("utf-8")
    value_bytes = str(value).encode("utf-8")
    digest.update(len(name_bytes).to_bytes(4, "little"))
    digest.update(name_bytes)
    digest.update(len(value_bytes).to_bytes(8, "little"))
    digest.update(value_bytes)


def update_tensor(digest, name, tensor):
    value = tensor.detach().contiguous().cpu()
    update_field(digest, f"{name}.dtype", str(value.dtype))
    update_field(digest, f"{name}.shape", ",".join(map(str, value.shape)))
    digest.update(memoryview(value.view(torch.uint8).reshape(-1).numpy()))


def artifact_digest(artifact):
    """Digest exactly the immutable payload consumed by the runtime."""
    digest = hashlib.sha256()
    update_field(digest, "algorithm", ARTIFACT_DIGEST_ALGORITHM)
    for key in ARTIFACT_METADATA_KEYS:
        update_field(digest, key, artifact[key])
    for key in KEEP:
        update_tensor(digest, key, artifact[key])
    return digest.hexdigest()


def safe_load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def artifact_valid(artifact, model_repo=None, revision=None):
    valid = (
        isinstance(artifact, dict)
        and artifact.get("schema") == ARTIFACT_SCHEMA
        and artifact.get("variant") == ARTIFACT_VARIANT
        and artifact.get("m") == M
        and artifact.get("ksub") == KSUB
        and artifact.get("dsub") == DSUB
        and artifact.get("vocab") == VOCAB
        and artifact.get("hidden") == HIDDEN
        and artifact.get("row_scale_mode") == "norm-matching"
        and artifact.get("training_iters") == CANONICAL_TRAINING_ITERS
        and artifact.get("training_seed") == CANONICAL_TRAINING_SEED
        and isinstance(artifact.get("checkpoint_model"), str)
        and isinstance(artifact.get("checkpoint_revision"), str)
        and artifact.get("head_digest_algorithm") == HEAD_DIGEST_ALGORITHM
        and isinstance(artifact.get("head_sha256"), str)
        and len(artifact["head_sha256"]) == 64
        and artifact.get("artifact_digest_algorithm")
        == ARTIFACT_DIGEST_ALGORITHM
        and isinstance(artifact.get("codes"), torch.Tensor)
        and tuple(artifact["codes"].shape) == (VOCAB, M)
        and artifact["codes"].dtype == torch.uint8
        and isinstance(artifact.get("centroids"), torch.Tensor)
        and tuple(artifact["centroids"].shape) == (M, KSUB, DSUB)
        and artifact["centroids"].dtype == torch.float16
        and isinstance(artifact.get("row_scale"), torch.Tensor)
        and tuple(artifact["row_scale"].shape) == (VOCAB,)
        and artifact["row_scale"].dtype == torch.float16
    )
    if not valid:
        return False
    if model_repo is not None and artifact["checkpoint_model"] != model_repo:
        return False
    if revision is not None and artifact["checkpoint_revision"] != revision:
        return False
    try:
        expected = artifact_digest(artifact)
    except (KeyError, TypeError, ValueError):
        return False
    return artifact.get("artifact_sha256") == expected


def checkpoint_path(model_repo, snapshot=None):
    if snapshot is not None:
        path = Path(snapshot)
        return path if (path / "model.safetensors.index.json").is_file() else None
    marker = model_repo.replace("/", "--")
    model_root = Path(f"/hf/hub/models--{marker}")
    main_ref = model_root / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        candidate = model_root / "snapshots" / revision
        if (candidate / "model.safetensors.index.json").is_file():
            return candidate
    candidates = [
        Path(path)
        for path in glob.glob(str(model_root / "snapshots" / "*"))
        if (Path(path) / "model.safetensors.index.json").is_file()
    ]
    return candidates[0] if len(candidates) == 1 else None


def load_head(snapshot):
    """Load the checkpoint's lm_head onto the active CUDA device."""
    index_path = snapshot / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    key = next(name for name in weight_map if name.endswith("lm_head.weight"))
    shard = snapshot / weight_map[key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(key).cuda()
    if tuple(weight.shape) != (VOCAB, HIDDEN):
        raise RuntimeError(f"unexpected lm_head shape {tuple(weight.shape)}")
    return weight


def deployed_int8_dequant(weight):
    """Reproduce the deployed head and fingerprint every stored coefficient."""
    output = torch.empty(weight.shape, device=weight.device, dtype=torch.float32)
    digest = hashlib.sha256()
    update_field(digest, "algorithm", HEAD_DIGEST_ALGORITHM)
    update_field(digest, "vocab", VOCAB)
    update_field(digest, "hidden", HIDDEN)
    update_field(digest, "weight_dtype", str(torch.int8))
    update_field(digest, "scale_dtype", str(torch.float16))
    for start in range(0, VOCAB, HEAD_DIGEST_ROWS):
        end = min(start + HEAD_DIGEST_ROWS, VOCAB)
        block = weight[start:end].float()
        scale = (block.abs().amax(1) / 127.0).clamp_min(1e-12)
        quantized = (
            (block / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        )
        stored_scale = scale.to(torch.float16)
        output[start:end] = quantized.float() * stored_scale.float()[:, None]
        weight_cpu = quantized.contiguous().cpu()
        scale_cpu = stored_scale.contiguous().cpu()
        digest.update(
            memoryview(weight_cpu.view(torch.uint8).reshape(-1).numpy())
        )
        digest.update(
            memoryview(scale_cpu.view(torch.uint8).reshape(-1).numpy())
        )
    return output, digest.hexdigest()


def normalize_head_in_place(weight):
    """Normalize strict FP32 rows and return their pre-normalization norms."""
    norms = torch.empty(VOCAB, device=weight.device, dtype=torch.float32)
    for start in range(0, VOCAB, HEAD_DIGEST_ROWS):
        end = min(start + HEAD_DIGEST_ROWS, VOCAB)
        block_norms = weight[start:end].norm(dim=1)
        norms[start:end] = block_norms
        weight[start:end].div_(block_norms[:, None].clamp_min(1e-12))
    return norms


def assign_codes(weight, centroids):
    """Assign every row subspace to its closest centroid."""
    codes = torch.empty(
        (VOCAB, M), device=weight.device, dtype=torch.uint8
    )
    centroid_norm = centroids.square().sum(-1)
    for start in range(0, VOCAB, ASSIGN_CHUNK):
        end = min(start + ASSIGN_CHUNK, VOCAB)
        subspaces = (
            weight[start:end]
            .float()
            .reshape(end - start, M, DSUB)
            .permute(1, 0, 2)
            .contiguous()
        )
        distance = torch.bmm(subspaces, centroids.transpose(1, 2))
        distance.mul_(-2.0).add_(centroid_norm[:, None, :])
        codes[start:end] = distance.argmin(-1).T.to(torch.uint8)
    return codes


def update_centroids(weight, codes, previous):
    """Apply one Lloyd centroid update, retaining empty centroids."""
    device = weight.device
    sums = torch.zeros((M * KSUB, DSUB), device=device, dtype=torch.float32)
    counts = torch.zeros(M * KSUB, device=device, dtype=torch.int64)
    offsets = torch.arange(M, device=device, dtype=torch.int64)[:, None] * KSUB
    for start in range(0, VOCAB, ASSIGN_CHUNK):
        end = min(start + ASSIGN_CHUNK, VOCAB)
        subspaces = (
            weight[start:end]
            .float()
            .reshape(end - start, M, DSUB)
            .permute(1, 0, 2)
            .contiguous()
        )
        ids = (codes[start:end].T.long() + offsets).reshape(-1)
        sums.index_add_(0, ids, subspaces.reshape(-1, DSUB))
        counts += torch.bincount(ids, minlength=M * KSUB)
    updated = (sums / counts.clamp_min(1)[:, None]).reshape(M, KSUB, DSUB)
    empty = counts.reshape(M, KSUB) == 0
    updated[empty] = previous[empty]
    return updated


def train_pq(unit_weight, iterations):
    """Run the canonical fixed-seed assignment/update schedule."""
    if iterations != CANONICAL_TRAINING_ITERS:
        raise ValueError(
            "this artifact requires exactly "
            f"{CANONICAL_TRAINING_ITERS} iterations"
        )
    # Repeated-index FP32 index_add_ must use PyTorch's deterministic path;
    # a fixed seed alone does not define a reproducible production artifact.
    torch.use_deterministic_algorithms(True)
    device = unit_weight.device
    generator = torch.Generator(device=device).manual_seed(
        CANONICAL_TRAINING_SEED
    )
    initialization = torch.randperm(
        VOCAB, generator=generator, device=device
    )[:KSUB]
    centroids = (
        unit_weight[initialization]
        .float()
        .reshape(KSUB, M, DSUB)
        .permute(1, 0, 2)
        .contiguous()
    )
    started = time.time()
    codes = torch.empty(0, device=device, dtype=torch.uint8)
    for iteration in range(iterations):
        codes = assign_codes(unit_weight, centroids)
        used = torch.stack(
            [
                torch.bincount(codes[:, subspace].long(), minlength=KSUB)
                .gt(0)
                .sum()
                for subspace in range(M)
            ]
        )
        print(
            f"  Lloyd {iteration + 1}/{iterations}: "
            f"centroid use mean={used.float().mean():.2f}, "
            f"min={int(used.min())}",
            flush=True,
        )
        if iteration + 1 < iterations:
            centroids = update_centroids(unit_weight, codes, centroids)

    stored_centroids = centroids.to(torch.float16).contiguous()
    codes = assign_codes(unit_weight, stored_centroids.float()).contiguous()
    return codes, stored_centroids, time.time() - started


def fit_norm_match_scale(unit_weight, exact_norms, codes, centroids):
    """Compute FP16 beta = exact norm divided by PQ reconstruction norm."""
    device = unit_weight.device
    output = torch.empty(VOCAB, device=device, dtype=torch.float16)
    subspaces = torch.arange(M, device=device)[None, :]
    for start in range(0, VOCAB, ASSIGN_CHUNK):
        end = min(start + ASSIGN_CHUNK, VOCAB)
        reconstruction = centroids[
            subspaces, codes[start:end].long()
        ].reshape(end - start, HIDDEN)
        reconstruction_norm = reconstruction.float().norm(dim=1).clamp_min(1e-12)
        output[start:end] = (
            exact_norms[start:end] / reconstruction_norm
        ).to(torch.float16)

        if start == 0:
            unit_norm = unit_weight[start:end].norm(dim=1)
            if float((unit_norm - 1.0).abs().max()) > 2e-5:
                raise RuntimeError("normalized training rows are not unit length")
    return output


def train(model_repo, snapshot_arg=None):
    snapshot = checkpoint_path(
        model_repo, Path(snapshot_arg) if snapshot_arg is not None else None
    )
    if snapshot is None:
        raise SystemExit(f"FAIL: no checkpoint snapshot found for {model_repo}")
    print(f"checkpoint {snapshot}", flush=True)

    head = load_head(snapshot)
    dequantized, head_sha256 = deployed_int8_dequant(head)
    del head
    exact_norms = normalize_head_in_place(dequantized)
    codes, centroids, seconds = train_pq(
        dequantized, CANONICAL_TRAINING_ITERS
    )
    row_scale = fit_norm_match_scale(
        dequantized, exact_norms, codes, centroids
    )
    print(f"trained in {seconds:.2f} s", flush=True)
    return {
        "schema": ARTIFACT_SCHEMA,
        "codes": codes.cpu().contiguous(),
        "centroids": centroids.cpu().contiguous(),
        "row_scale": row_scale.cpu().contiguous(),
        "variant": ARTIFACT_VARIANT,
        "m": M,
        "ksub": KSUB,
        "dsub": DSUB,
        "vocab": VOCAB,
        "hidden": HIDDEN,
        "training_iters": CANONICAL_TRAINING_ITERS,
        "training_seed": CANONICAL_TRAINING_SEED,
        "row_scale_mode": "norm-matching",
        "row_scale_semantics": (
            "FP16 round(exact-row-norm / PQ-reconstruction-norm)"
        ),
        "training_schedule": (
            "eight assignments/seven updates, then "
            "FP16-centroid reassignment"
        ),
        "checkpoint_model": model_repo,
        "checkpoint_revision": snapshot.name,
        "head_digest_algorithm": HEAD_DIGEST_ALGORITHM,
        "head_sha256": head_sha256,
        "artifact_digest_algorithm": ARTIFACT_DIGEST_ALGORITHM,
        "head_space": (
            "strict FP32 q*runtime-FP16-scale coefficients; PQ trained "
            "on exact unit rows; no mean centering"
        ),
        "source": "build_pq_artifact.train_pq",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true",
                       help="retrain from the checkpoint's lm_head")
    group.add_argument(
        "--validate-artifact",
        help="validate schema, payload digest, model id, and snapshot revision",
    )
    parser.add_argument(
        "--model-repo",
        default=os.environ.get("DENSESPARK_MODEL", DEFAULT_MODEL_REPO),
        help="checkpoint repository id under the mounted Hugging Face cache",
    )
    parser.add_argument(
        "--snapshot",
        help="explicit checkpoint snapshot directory; overrides --model-repo",
    )
    parser.add_argument("--out", help="output artifact path")
    args = parser.parse_args()

    if args.validate_artifact:
        snapshot = checkpoint_path(args.model_repo, args.snapshot)
        if snapshot is None:
            print(f"FAIL: no checkpoint snapshot found for {args.model_repo}")
            return 1
        try:
            artifact = safe_load(args.validate_artifact)
        except Exception as exc:
            print(f"FAIL: artifact could not be loaded safely: {exc}")
            return 1
        if not artifact_valid(artifact, args.model_repo, snapshot.name):
            print("FAIL: artifact is stale, malformed, or has a bad payload digest")
            return 1
        print(
            f"OK: schema {ARTIFACT_SCHEMA}, model {args.model_repo}, "
            f"revision {snapshot.name}, payload SHA256 verified"
        )
        return 0

    if not args.out:
        parser.error("--out is required with --train")

    artifact = train(args.model_repo, args.snapshot)

    if tuple(artifact["centroids"].shape[:2]) != (M, KSUB):
        raise SystemExit("FAIL: geometry is not M=128/Ksub=256")
    if artifact["codes"].dtype != torch.uint8:
        raise SystemExit("FAIL: codes must be uint8")
    vocab, subspaces = artifact["codes"].shape
    if subspaces != M or artifact["row_scale"].shape[0] != vocab:
        raise SystemExit("FAIL: codes and row_scale disagree")

    artifact["artifact_sha256"] = artifact_digest(artifact)
    # Kept as a display-only alias for the production measurement lab. Runtime
    # validation uses the explicitly versioned artifact_sha256 field above.
    artifact["build_sha256"] = artifact["artifact_sha256"]
    if not artifact_valid(artifact):
        raise SystemExit("FAIL: generated artifact failed its own schema gate")
    directory = os.path.dirname(os.path.abspath(args.out))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = f"{args.out}.tmp"
    torch.save(artifact, temporary)
    os.replace(temporary, args.out)

    total = sum(artifact[key].numel() * artifact[key].element_size()
                for key in KEEP)
    print(f"OK: {args.out}")
    print(f"    vocab {vocab}, {total / 1024 / 1024:.1f} MB of structure")
    print(f"    artifact sha256 {artifact['artifact_sha256']}")
    print(f"    head sha256 {artifact['head_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
