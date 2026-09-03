"""Product-quantized draft head for the MTP proposer.

The MTP layer proposes tokens; the target model verifies them. A proposal that
differs from the one an exact head would emit costs acceptance, never
correctness, because the rejection sampler still emits the target argmax. That
asymmetry is what makes an approximate head admissible in the draft path and
inadmissible in the verify path.

The structure is a product quantization of the deployed INT8 head:

    M = 128 subspaces of 40 dimensions, 256 centroids each
    codes[v, m]   uint8      31,784,960 B
    centroids     fp16        2,621,440 B
    row_scale[v]  fp16          496,640 B

Scoring one query is a 128 x 256 lookup table build, a scan of the code array,
a top-C selection, and a gathered projection of the C candidates from the
deployed INT8 coefficients. The scan reads 31.8 MB where the full head reads
1.271 GB, and the rerank adds C x 5120 B.

Quality is a containment property. On 2409 captured draft calls, the token the
exact head would emit is inside the C = 2048 candidate set on 2400 of them, and
the gathered rerank emits that token on those contained calls. The proposal is
therefore identical on 99.63% of the measured calls. Reduction-order effects
outside this captured set remain an empirical boundary, not a proof.

Set DENSESPARK_PQ_DRAFT=0 to keep the stock draft head at runtime.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sys

import torch
import triton
import triton.language as tl

M = 128
KSUB = 256
SCAN_BLOCK_ROWS = 256
# The batch-flat kernel is validated for B=2,4,8. Larger active-request batches
# use the batch-correct per-query grid rather than compiling unbounded
# specializations.
MAX_BATCH_SCAN = 8
ARTIFACT_SCHEMA = 2
ARTIFACT_VARIANT = "norm_match_m128"
ARTIFACT_DIGEST_ALGORITHM = "sha256-densespark-pq-runtime-v2"
HEAD_DIGEST_ALGORITHM = "sha256-densespark-int8-row-chunks-v1"
HEAD_DIGEST_ROWS = 4096
CANONICAL_TRAINING_ITERS = 8
CANONICAL_TRAINING_SEED = 1


@triton.jit
def _ds_pq_lut(
    out_ptr,
    cent_ptr,
    q_ptr,
    stride_qb,
    n_sub: tl.constexpr,
    n_cent: tl.constexpr,
    dsub: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One lookup table per (subspace, query): LUT[m, k] = <x_m, c[m, k]>."""
    sub = tl.program_id(0)
    qid = tl.program_id(1)
    cent = tl.arange(0, n_cent)
    dim = tl.arange(0, BLOCK_D)
    valid = dim < dsub
    query = tl.load(
        q_ptr + qid * stride_qb + sub * dsub + dim, mask=valid, other=0.0
    ).to(tl.float32)
    centroid = tl.load(
        cent_ptr + sub * n_cent * dsub + cent[:, None] * dsub + dim[None, :],
        mask=valid[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        out_ptr + qid * n_sub * n_cent + sub * n_cent + cent,
        tl.sum(centroid * query[None, :], axis=1),
    )


@triton.jit
def _ds_pq_scan(
    codes_ptr,
    lut_ptr,
    scale_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    n_sub: tl.constexpr,
    n_cent: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """score[v] = row_scale[v] * sum_m LUT[m, codes[m, v]], codes subspace-major."""
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    qid = tl.program_id(1)
    valid = rows < n_rows
    subs = tl.arange(0, n_sub)
    codes = tl.load(
        codes_ptr + subs[:, None] * n_rows + rows[None, :],
        mask=valid[None, :],
        other=0,
    ).to(tl.int32)
    values = tl.load(
        lut_ptr + qid * n_sub * n_cent + subs[:, None] * n_cent + codes,
        mask=valid[None, :],
        other=0.0,
    ).to(tl.float32)
    score = tl.sum(values, axis=0)
    scale = tl.load(scale_ptr + rows, mask=valid, other=0.0).to(tl.float32)
    tl.store(out_ptr + qid * n_rows + rows, score * scale, mask=valid)


@triton.jit
def _ds_pq_scan_batched(
    codes_ptr,
    lut_ptr,
    scale_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    n_sub: tl.constexpr,
    n_cent: tl.constexpr,
    NB: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """The same scan with the code tile loaded once for the whole batch.

    Every query re-gathers from its own 64 KB lookup table, which stays in L2,
    but the 33.3 MiB structure stream is read once instead of once per query. This is
    the head's batched-tl.dot argument applied to the other byte-bound kernel in
    the draft path: the arithmetic is unchanged and the DRAM traffic stops
    scaling with the batch.
    """
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    valid = rows < n_rows
    subs = tl.arange(0, n_sub)
    codes = tl.load(
        codes_ptr + subs[:, None] * n_rows + rows[None, :],
        mask=valid[None, :],
        other=0,
    ).to(tl.int32)
    scale = tl.load(scale_ptr + rows, mask=valid, other=0.0).to(tl.float32)
    for qid in tl.static_range(NB):
        values = tl.load(
            lut_ptr + qid * n_sub * n_cent + subs[:, None] * n_cent + codes,
            mask=valid[None, :],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            out_ptr + qid * n_rows + rows,
            tl.sum(values, axis=0) * scale,
            mask=valid,
        )


@triton.jit
def _ds_pq_rerank(
    out_ptr,
    w_ptr,
    ws_ptr,
    q_ptr,
    idx_ptr,
    stride_qb,
    width: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Gathered deployed-INT8 projection for one candidate row and query."""
    slot = tl.program_id(0)
    qid = tl.program_id(1)
    row = tl.load(idx_ptr + qid * width + slot)
    accumulator = tl.zeros((), dtype=tl.float32)
    for start in range(0, hidden, BLOCK_K):
        column = start + tl.arange(0, BLOCK_K)
        valid = column < hidden
        weight = tl.load(w_ptr + row * hidden + column, mask=valid, other=0).to(
            tl.float32
        )
        query = tl.load(q_ptr + qid * stride_qb + column, mask=valid, other=0.0).to(
            tl.float32
        )
        accumulator += tl.sum(weight * query, axis=0)
    scale = tl.load(ws_ptr + row).to(tl.float32)
    tl.store(out_ptr + qid * width + slot, (accumulator * scale).to(tl.float16))


@triton.jit
def _ds_pq_argmax(
    scores_ptr,
    ids_ptr,
    out_ptr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    qid = tl.program_id(0)
    slot = tl.arange(0, BLOCK)
    valid = slot < width
    score = tl.load(scores_ptr + qid * width + slot, mask=valid, other=-float("inf"))
    token = tl.load(ids_ptr + qid * width + slot, mask=valid, other=0)
    # torch.argmax over the full vocabulary resolves an FP16 tie to the lowest
    # token id. torch.topk(sorted=False) does not preserve that order, so a
    # slot-based argmax can disagree even when the exact winner is contained.
    max_score = tl.max(score, axis=0)
    sentinel = 0x7FFFFFFF
    tied_token = tl.where(valid & (score == max_score), token, sentinel)
    tl.store(out_ptr + qid, tl.min(tied_token, axis=0))


class PQDraftHead:
    """Approximate shortlist followed by a gathered deployed-INT8 rerank."""

    def __init__(self, codes_subspace, centroids, row_scale, head_int8, head_scale,
                 width, batch_scan=True):
        self.codes = codes_subspace          # [M, V] uint8, subspace-major
        self.centroids = centroids           # [M, KSUB, DSUB] fp16
        self.row_scale = row_scale           # [V] fp16
        self.head = head_int8                # [Vpad, H] int8
        self.head_scale = head_scale         # [Vpad] fp16
        self.width = width
        self.batch_scan = batch_scan
        self.vocab = row_scale.shape[0]
        self.hidden = head_int8.shape[1]
        self.dsub = self.hidden // M
        self.calls = 0
        self._capacity = 0
        self._buffers = ()

    def _reserve(self, batch, device):
        if batch <= self._capacity:
            return self._buffers
        lut = torch.empty((batch, M, KSUB), dtype=torch.float16, device=device)
        scores = torch.empty((batch, self.vocab), dtype=torch.float32, device=device)
        cand = torch.empty((batch, self.width), dtype=torch.float16, device=device)
        token = torch.empty(batch, dtype=torch.int32, device=device)
        self._capacity = batch
        self._buffers = (lut, scores, cand, token)
        return self._buffers

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = hidden_states.to(torch.float16).contiguous()
        batch = query.shape[0]
        lut, scores, cand, token = self._reserve(batch, query.device)
        lut, scores = lut[:batch], scores[:batch]
        cand, token = cand[:batch], token[:batch]

        _ds_pq_lut[(M, batch)](
            lut, self.centroids, query, self.hidden,
            n_sub=M, n_cent=KSUB, dsub=self.dsub, BLOCK_D=64,
            num_warps=8, num_stages=1,
        )
        if not _use_batched_scan(batch, self.batch_scan):
            _ds_pq_scan[(triton.cdiv(self.vocab, SCAN_BLOCK_ROWS), batch)](
                self.codes, lut, self.row_scale, scores,
                n_rows=self.vocab, n_sub=M, n_cent=KSUB,
                BLOCK_ROWS=SCAN_BLOCK_ROWS, num_warps=8, num_stages=2,
            )
        else:
            _ds_pq_scan_batched[(triton.cdiv(self.vocab, SCAN_BLOCK_ROWS),)](
                self.codes, lut, self.row_scale, scores,
                n_rows=self.vocab, n_sub=M, n_cent=KSUB, NB=batch,
                BLOCK_ROWS=SCAN_BLOCK_ROWS, num_warps=8, num_stages=2,
            )
        candidate_ids = torch.topk(scores, self.width, dim=-1, sorted=False).indices
        candidate_ids = candidate_ids.to(torch.int32).contiguous()
        _ds_pq_rerank[(self.width, batch)](
            cand, self.head, self.head_scale, query, candidate_ids, self.hidden,
            width=self.width, hidden=self.hidden, BLOCK_K=1024,
            num_warps=4, num_stages=2,
        )
        _ds_pq_argmax[(batch,)](
            cand, candidate_ids, token,
            width=self.width, BLOCK=triton.next_power_of_2(self.width),
            num_warps=8, num_stages=1,
        )
        self.calls += 1
        return token.to(torch.int64)


DEFAULT_ARTIFACT = "/opt/densespark/pq_head_m128.pt"
REQUIRED_KEYS = (
    "schema", "variant", "m", "ksub", "dsub", "vocab", "hidden",
    "codes", "centroids", "row_scale", "row_scale_mode",
    "training_iters", "training_seed", "checkpoint_model",
    "checkpoint_revision", "head_digest_algorithm", "head_sha256",
    "artifact_digest_algorithm", "artifact_sha256",
)
ARTIFACT_METADATA_KEYS = (
    "schema", "variant", "m", "ksub", "dsub", "vocab", "hidden",
    "row_scale_mode", "training_iters", "training_seed",
    "checkpoint_model", "checkpoint_revision", "head_digest_algorithm",
    "head_sha256", "artifact_digest_algorithm",
)
ARTIFACT_TENSOR_KEYS = ("codes", "centroids", "row_scale")


def _log(message: str) -> None:
    print(f"DENSESPARK PQ: {message}", file=sys.stderr, flush=True)


def _update_field(digest, name, value):
    """Add one length-delimited scalar to a portable SHA256 transcript."""
    name_bytes = name.encode("utf-8")
    value_bytes = str(value).encode("utf-8")
    digest.update(len(name_bytes).to_bytes(4, "little"))
    digest.update(name_bytes)
    digest.update(len(value_bytes).to_bytes(8, "little"))
    digest.update(value_bytes)


def _update_tensor(digest, name, tensor):
    """Hash a CPU tensor without changing its dtype or element order."""
    value = tensor.detach().contiguous().cpu()
    _update_field(digest, f"{name}.dtype", str(value.dtype))
    _update_field(digest, f"{name}.shape", ",".join(map(str, value.shape)))
    raw = value.view(torch.uint8).reshape(-1).numpy()
    digest.update(memoryview(raw))


def _artifact_sha256(artifact):
    """Digest the complete runtime payload and its provenance metadata."""
    digest = hashlib.sha256()
    _update_field(digest, "algorithm", ARTIFACT_DIGEST_ALGORITHM)
    for key in ARTIFACT_METADATA_KEYS:
        _update_field(digest, key, artifact[key])
    for key in ARTIFACT_TENSOR_KEYS:
        _update_tensor(digest, key, artifact[key])
    return digest.hexdigest()


def _head_sha256(head_int8, head_scale, vocab):
    """Digest exactly the deployed INT8 coefficients used by the reranker.

    The copy is a one-time startup cost. Hashing all coefficients, rather than
    row norms or a sample, makes sign flips, row permutations, and checkpoint
    substitutions fail closed before the first proposal.
    """
    if head_int8.dtype != torch.int8 or head_scale.dtype != torch.float16:
        raise ValueError("the deployed head must be INT8 with FP16 row scales")
    if head_int8.ndim != 2 or head_scale.ndim != 1:
        raise ValueError("the deployed head has invalid ranks")
    if head_int8.shape[0] < vocab or head_scale.shape[0] < vocab:
        raise ValueError("the deployed head is smaller than the artifact vocabulary")

    digest = hashlib.sha256()
    _update_field(digest, "algorithm", HEAD_DIGEST_ALGORITHM)
    _update_field(digest, "vocab", vocab)
    _update_field(digest, "hidden", head_int8.shape[1])
    _update_field(digest, "weight_dtype", str(head_int8.dtype))
    _update_field(digest, "scale_dtype", str(head_scale.dtype))
    for start in range(0, vocab, HEAD_DIGEST_ROWS):
        end = min(start + HEAD_DIGEST_ROWS, vocab)
        # Interleave coefficients and their scales by row chunk. The format is
        # versioned above and reproduced by the artifact builder.
        weight = head_int8[start:end].detach().contiguous().cpu()
        scale = head_scale[start:end].detach().contiguous().cpu()
        digest.update(memoryview(weight.view(torch.uint8).reshape(-1).numpy()))
        digest.update(memoryview(scale.view(torch.uint8).reshape(-1).numpy()))
    return digest.hexdigest()


def _candidate_argmax_reference(scores, ids):
    """CPU-testable specification for the tie-safe Triton reduction."""
    maximum = scores.max(dim=-1, keepdim=True).values
    sentinel = torch.iinfo(ids.dtype).max
    return torch.where(scores == maximum, ids, sentinel).amin(dim=-1)


def _use_batched_scan(batch, enabled):
    """Use only the batch-flat widths covered by the production GPU matrix."""
    return enabled and 1 < batch <= MAX_BATCH_SCAN


def _validate(codes, centroids, row_scale, head_int8, head_scale,
              sample=4096, seed=0):
    """Check the structure was trained on the head that is actually loaded.

    row_scale[v] is fp16(||exact row|| / ||PQ reconstruction||) by construction,
    so recomputing it from the loaded head reproduces the stored value whenever
    the checkpoint matches, and does not otherwise. Returns (ok, worst, median).
    """
    vocab = row_scale.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.randperm(vocab, generator=generator)[:sample].to(head_int8.device)

    exact = head_int8[rows].float() * head_scale[rows].float()[:, None]
    exact_norm = exact.norm(dim=1)
    subspaces = torch.arange(M, device=codes.device)[None, :]
    reconstruction = centroids[subspaces, codes[rows].long()].reshape(
        rows.shape[0], -1
    )
    reconstruction_norm = reconstruction.float().norm(dim=1).clamp_min(1e-12)
    predicted = (exact_norm / reconstruction_norm).to(torch.float16).float()
    stored = row_scale[rows].float()
    relative = ((predicted - stored).abs() / stored.abs().clamp_min(1e-12))
    worst = float(relative.max())
    median = float(relative.median())
    return worst < 5e-2, worst, median


def build(lm_head, hidden_size):
    """Return a PQDraftHead, or None with a reason logged."""
    if os.environ.get("DENSESPARK_PQ_DRAFT", "0") != "1":
        return None

    path = os.environ.get("DENSESPARK_PQ_ARTIFACT", DEFAULT_ARTIFACT)
    if not os.path.exists(path):
        _log(f"disabled: no artifact at {path}")
        return None

    head_int8 = getattr(lm_head, "_ds_int8", None)
    head_scale = getattr(lm_head, "_ds_scales", None)
    if head_int8 is None or head_scale is None:
        _log("disabled: INT8 head not present (patch 01 did not run on the drafter)")
        return None

    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        _log(f"disabled: artifact could not be loaded safely ({exc})")
        return None
    if not isinstance(artifact, dict):
        _log("disabled: artifact root is not a dictionary")
        return None
    missing = [k for k in REQUIRED_KEYS if k not in artifact]
    if missing:
        _log(f"disabled: artifact missing {missing}")
        return None
    if artifact["schema"] != ARTIFACT_SCHEMA:
        _log(f"disabled: artifact schema {artifact['schema']!r}; "
             f"expected {ARTIFACT_SCHEMA} (rebuild the artifact)")
        return None
    if artifact["variant"] != ARTIFACT_VARIANT:
        _log(f"disabled: unsupported artifact variant {artifact['variant']!r}")
        return None
    if artifact["artifact_digest_algorithm"] != ARTIFACT_DIGEST_ALGORITHM:
        _log("disabled: unsupported artifact digest algorithm")
        return None
    if artifact["head_digest_algorithm"] != HEAD_DIGEST_ALGORITHM:
        _log("disabled: unsupported head digest algorithm")
        return None
    integer_metadata = ("m", "ksub", "dsub", "vocab", "hidden",
                        "training_iters", "training_seed")
    if any(type(artifact[key]) is not int for key in integer_metadata):
        _log("disabled: artifact integer metadata is malformed")
        return None
    if (artifact["training_iters"] != CANONICAL_TRAINING_ITERS
            or artifact["training_seed"] != CANONICAL_TRAINING_SEED):
        _log("disabled: artifact uses an unvalidated training schedule")
        return None
    string_metadata = ("checkpoint_model", "checkpoint_revision",
                       "head_sha256", "artifact_sha256")
    if any(not isinstance(artifact[key], str) or not artifact[key]
           for key in string_metadata):
        _log("disabled: artifact provenance metadata is malformed")
        return None
    if (len(artifact["head_sha256"]) != 64
            or len(artifact["artifact_sha256"]) != 64):
        _log("disabled: artifact SHA256 fields are malformed")
        return None
    if (artifact["m"] != M or artifact["ksub"] != KSUB
            or artifact["dsub"] * M != hidden_size
            or artifact["hidden"] != hidden_size):
        _log("disabled: artifact geometry does not match the loaded model")
        return None
    if artifact["row_scale_mode"] != "norm-matching":
        _log("disabled: unsupported row-scale semantics")
        return None

    codes = artifact["codes"]
    centroids = artifact["centroids"]
    row_scale = artifact["row_scale"]
    if not all(isinstance(value, torch.Tensor)
               for value in (codes, centroids, row_scale)):
        _log("disabled: artifact payload contains non-tensor values")
        return None
    vocab = artifact["vocab"]
    if (not isinstance(vocab, int) or vocab < 1
            or codes.shape != (vocab, M)
            or centroids.shape != (M, KSUB, hidden_size // M)
            or row_scale.shape != (vocab,)):
        _log("disabled: artifact tensor shapes do not match its geometry")
        return None
    if (codes.dtype != torch.uint8 or centroids.dtype != torch.float16
            or row_scale.dtype != torch.float16):
        _log("disabled: artifact tensor dtypes are not uint8/FP16/FP16")
        return None
    try:
        calculated_artifact_sha256 = _artifact_sha256(artifact)
    except Exception as exc:
        _log(f"disabled: artifact digest failed ({exc})")
        return None
    if not hmac.compare_digest(
            str(artifact["artifact_sha256"]), calculated_artifact_sha256):
        _log("disabled: artifact payload digest mismatch")
        return None

    device = head_int8.device
    if (head_int8.shape[0] < vocab or head_int8.shape[1] != hidden_size
            or head_scale.shape[0] < vocab):
        _log(f"disabled: head {tuple(head_int8.shape)} smaller than the structure")
        return None
    try:
        loaded_head_sha256 = _head_sha256(head_int8, head_scale, vocab)
    except Exception as exc:
        _log(f"disabled: loaded head cannot be fingerprinted ({exc})")
        return None
    if not hmac.compare_digest(str(artifact["head_sha256"]), loaded_head_sha256):
        _log("disabled: artifact was trained for a different deployed INT8 head")
        return None

    codes = codes.to(device)
    centroids = centroids.to(device)
    row_scale = row_scale.to(device)

    ok, worst, median = _validate(
        codes, centroids, row_scale, head_int8, head_scale)
    if not ok:
        _log(f"disabled: structure does not match the loaded head "
             f"(worst row-scale error {worst:.4g}, median {median:.4g})")
        return None

    width = int(os.environ.get("DENSESPARK_PQ_CANDIDATES", "2048"))
    if width < 1 or width > vocab:
        _log(f"disabled: candidate width {width} out of range")
        return None

    codes_subspace = codes.t().contiguous()
    head = PQDraftHead(
        codes_subspace, centroids, row_scale, head_int8,
        head_scale.to(torch.float16), width,
        batch_scan=os.environ.get("DENSESPARK_PQ_BATCH_SCAN", "1") != "0",
    )
    megabytes = (
        codes_subspace.numel() + centroids.numel() * 2 + row_scale.numel() * 2
    ) / 1024 / 1024
    _log(f"draft head -> PQ M=128 C={width}, {megabytes:.1f} MB scanned per call "
         f"instead of {head_int8.numel() / 1024 / 1024:.0f} MB "
         f"(artifact/head SHA256 verified; row-scale check: median {median:.2e}, "
         f"worst {worst:.2e})")
    return head
