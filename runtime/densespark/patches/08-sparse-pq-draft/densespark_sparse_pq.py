"""Lab 86 runtime: sparse probabilistic PQ proposal for pinned vLLM 0.27.1.

This module leaves the stock
probabilistic proposer untouched unless ``DENSESPARK_LAB86_SPARSE_PQ=1``.
When enabled, it constructs one proposal distribution by

    PQ retrieval -> gathered deployed-INT8 rerank -> temperature/top-k/top-p
    -> FP32 softmax -> dense FP32 scatter.

The sampled token and the dense tensor returned to vLLM's rejection sampler
are derived from the same ``sparse_probabilities`` tensor.  There is no second
normalization and no logits/probabilities ABI conversion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
import triton
import triton.language as tl


ENV_NAME = "DENSESPARK_LAB86_SPARSE_PQ"
REQUIRED_PQ_ENV = "DENSESPARK_PQ_DRAFT"
DEFAULT_WIDTH = 2_048
SUPPORTED_WIDTHS = (512, 1_024, 2_048)


def _configured_width() -> int:
    raw = os.environ.get("DENSESPARK_PQ_CANDIDATES", str(DEFAULT_WIDTH))
    try:
        width = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"DENSESPARK_PQ_CANDIDATES must be one of {SUPPORTED_WIDTHS}, "
            f"observed {raw!r}"
        ) from exc
    if width not in SUPPORTED_WIDTHS:
        raise RuntimeError(
            f"DENSESPARK_PQ_CANDIDATES must be one of {SUPPORTED_WIDTHS}, "
            f"observed {raw!r}"
        )
    return width


WIDTH = _configured_width()
VOCAB = 248_320
HIDDEN = 5_120
SCAN_BLOCK_ROWS = 256


@dataclass(frozen=True)
class SparseProposal:
    """One q represented sparsely for sampling and densely for rejection."""

    candidate_ids: torch.Tensor
    sparse_probabilities: torch.Tensor
    dense_probabilities: torch.Tensor
    processed_logits: torch.Tensor


@triton.jit
def _ds_lab86_pq_scan_query_fast(
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
    """Lab 78's bit-exact query-fast scheduling of the production PQ scan."""
    pid = tl.program_id(0)
    qid = pid % NB
    row_tile = pid // NB
    rows = row_tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
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
    scale = tl.load(scale_ptr + rows, mask=valid, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + qid * n_rows + rows,
        tl.sum(values, axis=0) * scale,
        mask=valid,
    )


def apply_top_k_top_p_reference(
    logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    """CPU-testable equivalent of vLLM's ascending-sort constraint path."""
    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=False)
    threshold_slots = logits.shape[1] - top_k.to(torch.long)
    thresholds = sorted_logits.gather(1, threshold_slots.unsqueeze(1))
    sorted_logits.masked_fill_(sorted_logits < thresholds, -float("inf"))
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumulative <= 1.0 - top_p.unsqueeze(1)
    remove[:, -1] = False
    sorted_logits.masked_fill_(remove, -float("inf"))
    return torch.empty_like(logits).scatter_(1, sorted_indices, sorted_logits)


def construct_distribution(
    candidate_ids: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    vocab: int,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    constraint: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> SparseProposal:
    """Construct q once, then expose the same values through both ABIs."""
    if candidate_ids.ndim != 2 or candidate_logits.ndim != 2:
        raise RuntimeError("Lab 86 candidate tensors must be [batch,width]")
    if candidate_ids.shape != candidate_logits.shape:
        raise RuntimeError("Lab 86 candidate IDs/logits do not align")
    batch, width = candidate_ids.shape
    if width != WIDTH:
        raise RuntimeError(f"Lab 86 requires width={WIDTH}, observed {width}")
    if candidate_ids.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("Lab 86 candidate IDs must be integral")
    for name, value in (
        ("temperature", temperature),
        ("top_k", top_k),
        ("top_p", top_p),
    ):
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() != batch:
            raise RuntimeError(f"Lab 86 {name} must contain one value per query")

    # Fail closed without a device-to-host synchronization in every draft
    # position.  CUDA executes this assertion before the constraint kernel on
    # the same stream; CPU contract tests observe the same RuntimeError.
    temperature_f32 = temperature.to(
        device=candidate_logits.device, dtype=torch.float32
    )
    top_k_i64 = top_k.to(device=candidate_logits.device, dtype=torch.int64)
    top_p_f32 = top_p.to(device=candidate_logits.device, dtype=torch.float32)
    valid_metadata = (
        torch.isfinite(temperature_f32)
        & (temperature_f32 > 0)
        & (top_k_i64 >= 1)
        & (top_k_i64 <= width)
        & torch.isfinite(top_p_f32)
        & (top_p_f32 > 0)
        & (top_p_f32 <= 1)
    ).all()
    torch._assert_async(
        valid_metadata,
        f"Lab 86 requires finite T>0, top_k in [1,{width}], top_p in (0,1]",
    )

    ids = candidate_ids.to(torch.int64)
    processed = candidate_logits.to(torch.float32) / temperature_f32.unsqueeze(1)
    processed = constraint(processed, top_k_i64, top_p_f32)
    sparse_probabilities = processed.softmax(dim=-1, dtype=torch.float32)
    dense_probabilities = torch.zeros(
        (batch, vocab),
        dtype=torch.float32,
        device=candidate_logits.device,
    )
    # torch.topk produces unique vocabulary indices.  Scatter, rather than a
    # second softmax, makes the reported q byte-identical on its support.
    dense_probabilities.scatter_(1, ids, sparse_probabilities)
    return SparseProposal(
        candidate_ids=ids,
        sparse_probabilities=sparse_probabilities,
        dense_probabilities=dense_probabilities,
        processed_logits=processed,
    )


def sample_with_noise(
    proposal: SparseProposal,
    noise: torch.Tensor,
) -> torch.Tensor:
    """CPU-testable exponential-race sampling from proposal's exact sparse q."""
    if noise.shape != proposal.sparse_probabilities.shape:
        raise RuntimeError("Lab 86 sampling noise shape differs from sparse q")
    if bool((noise <= 0).any()):
        raise RuntimeError("Lab 86 exponential noise must be positive")
    slots = (proposal.sparse_probabilities / noise).argmax(dim=-1)
    return proposal.candidate_ids.gather(1, slots.unsqueeze(1)).squeeze(1)


def fill_exponential_noise(
    noise: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> None:
    """Match vLLM's per-request seeded exponential-noise contract."""
    # Match vLLM 0.27.1: do not consume the global RNG when every row has a
    # request-local generator.  This matters when several seeded requests are
    # interleaved with later unseeded work in the same process.
    if len(generators) != noise.shape[0]:
        noise.exponential_()
    for row, generator in generators.items():
        if type(row) is not int or row < 0 or row >= noise.shape[0]:
            raise RuntimeError(
                f"Lab 86 generator row {row!r} is outside batch={noise.shape[0]}"
            )
        noise[row].exponential_(generator=generator)


class SparseProbabilisticPQ:
    """Production-like wrapper around patch 04's validated artifact/head."""

    def __init__(self, pq_module: Any, pq_head: Any) -> None:
        if pq_head.width != WIDTH:
            raise RuntimeError(
                f"Lab 86 requires DENSESPARK_PQ_CANDIDATES={WIDTH}, "
                f"observed {pq_head.width}"
            )
        if pq_head.vocab != VOCAB or pq_head.hidden != HIDDEN:
            raise RuntimeError(
                "Lab 86 is pinned to the Qwen3.8-27B "
                f"{VOCAB}x{HIDDEN} head, observed "
                f"{pq_head.vocab}x{pq_head.hidden}"
            )
        self.pq = pq_module
        self.head = pq_head

    def __call__(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any,
        *,
        use_fp64_gumbel: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not sampling_metadata.all_random or sampling_metadata.all_greedy:
            raise RuntimeError("Lab 86 supports only an all-random sampling batch")
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[0] < 1
            or hidden_states.shape[1] != HIDDEN
        ):
            raise RuntimeError("Lab 86 received an empty or malformed hidden batch")
        batch = hidden_states.shape[0]
        for name in ("temperature", "top_k", "top_p"):
            value = getattr(sampling_metadata, name, None)
            if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() != batch:
                raise RuntimeError(
                    f"Lab 86 sampling metadata {name} does not align with draft rows"
                )

        query = hidden_states.to(torch.float16).contiguous()
        lut, scores, candidate_logits, _unused_token = self.head._reserve(
            batch, query.device
        )
        lut = lut[:batch]
        scores = scores[:batch]
        candidate_logits = candidate_logits[:batch]
        self.pq._ds_pq_lut[(self.pq.M, batch)](
            lut,
            self.head.centroids,
            query,
            query.stride(0),
            n_sub=self.pq.M,
            n_cent=self.pq.KSUB,
            dsub=self.head.dsub,
            BLOCK_D=64,
            num_warps=8,
            num_stages=1,
        )
        row_tiles = triton.cdiv(self.head.vocab, SCAN_BLOCK_ROWS)
        _ds_lab86_pq_scan_query_fast[(row_tiles * batch,)](
            self.head.codes,
            lut,
            self.head.row_scale,
            scores,
            n_rows=self.head.vocab,
            n_sub=self.pq.M,
            n_cent=self.pq.KSUB,
            NB=batch,
            BLOCK_ROWS=SCAN_BLOCK_ROWS,
            num_warps=8,
            num_stages=2,
        )
        candidate_ids = torch.topk(
            scores, WIDTH, dim=-1, sorted=False
        ).indices
        candidate_ids_i32 = candidate_ids.to(torch.int32).contiguous()
        self.pq._ds_pq_rerank[(WIDTH, batch)](
            candidate_logits,
            self.head.head,
            self.head.head_scale,
            query,
            candidate_ids_i32,
            query.stride(0),
            width=WIDTH,
            hidden=self.head.hidden,
            BLOCK_K=1024,
            num_warps=4,
            num_stages=2,
        )

        from vllm.v1.sample.ops.topk_topp_sampler import (
            apply_top_k_top_p,
            empty_exponential_noise_like,
            sample_with_exponential_noise,
        )

        proposal = construct_distribution(
            candidate_ids,
            candidate_logits,
            vocab=self.head.vocab,
            temperature=sampling_metadata.temperature,
            top_k=sampling_metadata.top_k,
            top_p=sampling_metadata.top_p,
            constraint=apply_top_k_top_p,
        )
        # Sample from the exact sparse tensor that was scattered into the
        # returned dense tensor.  clone() is required because vLLM's sampler
        # divides its probability argument in place.
        noise = empty_exponential_noise_like(
            proposal.sparse_probabilities, use_fp64_gumbel
        )
        fill_exponential_noise(noise, sampling_metadata.generators)
        sampled_slots = sample_with_exponential_noise(
            proposal.sparse_probabilities.clone(), noise
        )
        draft_token_ids = proposal.candidate_ids.gather(
            1, sampled_slots.unsqueeze(1)
        ).squeeze(1)
        self.head.calls += 1
        return draft_token_ids, proposal.dense_probabilities


def _build_sampler(proposer: Any, hidden_states: torch.Tensor) -> SparseProbabilisticPQ:
    if os.environ.get(REQUIRED_PQ_ENV, "0") != "1":
        raise RuntimeError(
            f"{ENV_NAME}=1 also requires {REQUIRED_PQ_ENV}=1 so patch 04 "
            "validates and loads the canonical artifact"
        )
    configured_width = os.environ.get("DENSESPARK_PQ_CANDIDATES", str(DEFAULT_WIDTH))
    if configured_width != str(WIDTH):
        raise RuntimeError(
            f"Lab 86 is pinned to DENSESPARK_PQ_CANDIDATES={WIDTH}, "
            f"observed {configured_width!r}"
        )
    if proposer.method != "mtp":
        raise RuntimeError(f"Lab 86 requires method='mtp', observed {proposer.method!r}")
    if proposer.parallel_drafting:
        raise RuntimeError("Lab 86 is pinned to sequential MTP, not parallel drafting")
    if proposer.use_local_argmax_reduction or proposer.use_heterogeneous_vocab:
        raise RuntimeError("Lab 86 does not support reduced or heterogeneous vocabularies")

    from vllm import _densespark_pq as pq

    lm_head = getattr(proposer.model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Lab 86 could not locate the drafter lm_head")
    if getattr(lm_head, "_ds_int8", None) is None:
        # Patch 01 quantizes lazily inside the first logits call.  Pay this
        # one-time initialization cost before patch 04 fingerprints the head.
        proposer.model.compute_logits(hidden_states[:1])
    pq_head = pq.build(lm_head, hidden_states.shape[-1])
    if pq_head is None:
        raise RuntimeError("Lab 86 patch-04 artifact/head validation failed")
    return SparseProbabilisticPQ(pq, pq_head)


def sample_sparse_probabilistic(
    proposer: Any,
    hidden_states: torch.Tensor,
    sampling_metadata: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hook target for LLMBaseProposer._sample_draft_tokens."""
    if os.environ.get(ENV_NAME, "0") != "1":
        raise RuntimeError("Lab 86 hook was called while its environment gate was disabled")
    sampler = getattr(proposer, "_ds_lab86_sparse_sampler", None)
    if sampler is None:
        sampler = _build_sampler(proposer, hidden_states)
        proposer._ds_lab86_sparse_sampler = sampler
    return sampler(
        hidden_states,
        sampling_metadata,
        use_fp64_gumbel=bool(proposer.use_fp64_gumbel),
    )
