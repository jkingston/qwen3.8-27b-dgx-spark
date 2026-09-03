#!/usr/bin/env python3
"""INT8 LM head for Qwen3.8-27B: batched 2D Triton GEMV.

Qwen3.8-27B keeps an unquantized head. At vocab 248320 x hidden 5120 that is
2.54 GB of BF16 re-read on every decoded token, against roughly 9 GB for the
INT4 language layers — a sixth of the decode budget spent on one matrix.

Quantizing it to INT8 per output channel at load time halves that read. The
kernel is a single 2D launch that loads each weight tile once and reuses it
across the tokens in the batch, instead of one launch per token.

Applies to the vLLM 0.27.x form of LogitsProcessor._get_logits. Patches are
version specific: an unrecognised body aborts without editing the file.

Set DENSESPARK_INT8_LMHEAD=0 in the server environment to keep the stock head
at runtime; the patched image then serves both sides of an A/B unchanged.

Set DENSESPARK_HEAD_AUTOTUNE=0 to pin one tile instead of letting Triton search.
That fixes the fp32 reduction order over K, and it was added to test whether the
autotuner explains the run-to-run spread in serving throughput. It does not, and
in isolation the pin measured about 1.1% slower. The shipped profile pins it
anyway, because that is the setting every published measurement of this
composition was taken with; whether unpinning recovers that 1.1% here has not
been measured on this composition, so it is not claimed either way.

Set DENSESPARK_HEAD_BATCH_DOT=0 to restore the old per-row fallback for
verification batches above four, which is what shipped before and what makes
MTP-4 and longer proposal chains unprofitable.

Set DENSESPARK_HEAD_INTERLEAVED=0 to restore sequential 16-row launches above
batch 16.  The default flattened grid interleaves query panels of the same
vocabulary tile, keeping the 1.27 GB head at one effective memory pass.

Usage:
    python3 patch_int8_lmhead.py                     # patch the importable vLLM
    python3 patch_int8_lmhead.py --vllm-root /path   # patch a specific install
"""

import argparse
import os
import sys

MARKER = "DENSESPARK_INT8_LMHEAD"

# Paths tried when vLLM is not importable from this interpreter, e.g. when the
# script runs during a Docker build against the image's site-packages.
FALLBACK_ROOTS = (
    "/usr/local/lib/python3.12/dist-packages/vllm",
    "/usr/lib/python3/dist-packages/vllm",
    "/opt/venv/lib/python3.12/site-packages/vllm",
)

RELATIVE_TARGET = os.path.join("model_executor", "layers", "logits_processor.py")

# vLLM 0.27.x computes the head through _apply_head, which honours head_dtype.
ANCHOR = '''    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)'''

REPLACEMENT = '''    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # DENSESPARK_INT8_LMHEAD: per-channel INT8 head, one Triton launch per step
        if not hasattr(self, '_densespark_int8_ready'):
            self._densespark_int8_ready = True
            w = lm_head.weight.data
            # DENSESPARK_INT8_LMHEAD=0 keeps the stock head, so the same image
            # serves both sides of an A/B without a rebuild.
            import os as _os
            enabled = _os.environ.get('DENSESPARK_INT8_LMHEAD', '1') != '0'
            # head_dtype other than the model dtype is served by _apply_head:
            # this path emits float16 and cannot satisfy an fp32 head request.
            head_dtype = getattr(self, 'head_dtype', None)
            if (enabled and w.dtype in (torch.bfloat16, torch.float16)
                    and w.shape[0] > 100000
                    and head_dtype in (None, hidden_states.dtype)):
                scales = (w.float().abs().amax(dim=1) / 127.0).clamp(min=1e-12)
                w_int8 = (w.float() / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
                lm_head._ds_int8 = w_int8
                lm_head._ds_scales = scales.to(torch.float16)
                saved_mb = (w.numel() * w.element_size() - w_int8.numel()) // 1024 // 1024
                lm_head.weight.data = torch.empty(0, device=w.device, dtype=w.dtype)
                import sys as _sys
                print(f"DENSESPARK: LM head -> INT8 Triton {list(w_int8.shape)}, "
                      f"{saved_mb} MB less to read per token", file=_sys.stderr, flush=True)
                import triton
                import triton.language as tl

                # Autotuned per (M, K, NUM_BATCH). The arithmetic is identical
                # across configs; only the launch geometry changes.
                _CONFIGS = [
                    triton.Config({'BLOCK_M': 64,  'BLOCK_K': 256}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 512}, num_warps=8, num_stages=2),
                    triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
                    triton.Config({'BLOCK_M': 256, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
                ]

                @triton.autotune(configs=_CONFIGS, key=['M', 'K', 'NUM_BATCH'])
                @triton.jit
                def _ds_head(out_ptr, w_ptr, x_ptr, s_ptr, M, K,
                             stride_ob, stride_xb, NUM_BATCH: tl.constexpr,
                             BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
                    # One program per row block. The weight tile is loaded once
                    # and reused for every token in the batch.
                    pid_m = tl.program_id(0)
                    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                    rmask = rows < M
                    acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    for ks in range(0, K, BLOCK_K):
                        co = ks + tl.arange(0, BLOCK_K)
                        km = co < K
                        w = tl.load(w_ptr + rows[:, None] * K + co[None, :],
                                    mask=rmask[:, None] & km[None, :], other=0).to(tl.float32)
                        x0 = tl.load(x_ptr + 0 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                        acc0 += tl.sum(w * x0[None, :], axis=1)
                        if NUM_BATCH > 1:
                            x1 = tl.load(x_ptr + 1 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc1 += tl.sum(w * x1[None, :], axis=1)
                        if NUM_BATCH > 2:
                            x2 = tl.load(x_ptr + 2 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc2 += tl.sum(w * x2[None, :], axis=1)
                        if NUM_BATCH > 3:
                            x3 = tl.load(x_ptr + 3 * stride_xb + co, mask=km, other=0.0).to(tl.float32)
                            acc3 += tl.sum(w * x3[None, :], axis=1)
                    s = tl.load(s_ptr + rows, mask=rmask, other=1.0).to(tl.float32)
                    tl.store(out_ptr + 0 * stride_ob + rows, (acc0 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 1:
                        tl.store(out_ptr + 1 * stride_ob + rows, (acc1 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 2:
                        tl.store(out_ptr + 2 * stride_ob + rows, (acc2 * s).to(tl.float16), mask=rmask)
                    if NUM_BATCH > 3:
                        tl.store(out_ptr + 3 * stride_ob + rows, (acc3 * s).to(tl.float16), mask=rmask)

                lm_head._ds_kernel = _ds_head
                lm_head._ds_pinned = {'BLOCK_M': 128, 'BLOCK_K': 256,
                                      'num_warps': 8, 'num_stages': 2}
                lm_head._ds_dot_ok = _os.environ.get(
                    'DENSESPARK_HEAD_BATCH_DOT', '1') != '0'
                # Above 16 rows the shared-weight kernel is still the right one;
                # it only has to be called once per 16 rows. Set
                # DENSESPARK_HEAD_CHUNK16=0 to restore the old per-row fallback
                # so both sides of that comparison stay servable from one image.
                lm_head._ds_chunk16 = _os.environ.get(
                    'DENSESPARK_HEAD_CHUNK16', '1') != '0'
                # Above 16 rows, flatten (vocabulary tile, query panel) into one
                # launch. Adjacent program IDs cover panels of the same tile,
                # allowing their identical weight requests to reuse L2. Lab 77
                # measured 2.0x at B=32 and 3.85x at B=64, bit-identical under
                # the matched fixed configuration.
                lm_head._ds_interleaved = _os.environ.get(
                    'DENSESPARK_HEAD_INTERLEAVED', '1') != '0'
                # DENSESPARK_HEAD_AUTOTUNE=0 pins one tile instead of letting
                # Triton search, so the fp32 reduction order over K is fixed.
                # Pinning measured slightly slower in isolation. The shipped
                # profile pins it regardless, to match the configuration its
                # published numbers were measured with.
                lm_head._ds_autotune = _os.environ.get(
                    'DENSESPARK_HEAD_AUTOTUNE', '1') != '0'

                # Verification batches above four. The scalar kernel carries four
                # accumulators, so a fifth token fell into a per-row loop that
                # re-read the whole matrix once per row: 26.716 ms at batch 5
                # against 5.542 ms at batch 4. That cliff is the dominant
                # measured explanation for the MTP-4 regression.
                #
                # Routing the batch dimension through tl.dot instead gives one
                # 2-D accumulator and one weight pass for any batch: 5.451 ms at
                # batch 4 and 5.519 ms at batch 8 on this part. Batches of four
                # or fewer keep the scalar path, which stays bit-identical to
                # what shipped before.
                _DOT_CONFIGS = [
                    triton.Config({'BLOCK_M': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=4, num_stages=2),
                    triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
                    triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
                    triton.Config({'BLOCK_M': 256, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
                ]

                @triton.autotune(configs=_DOT_CONFIGS, key=['M', 'K', 'NB'])
                @triton.jit
                def _ds_head_dot(out_ptr, w_ptr, x_ptr, s_ptr, M, K,
                                 stride_ob, stride_xb, NB: tl.constexpr,
                                 PADB: tl.constexpr,
                                 BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
                    pid_m = tl.program_id(0)
                    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                    rmask = rows < M
                    bidx = tl.arange(0, PADB)
                    bmask = bidx < NB
                    acc = tl.zeros((BLOCK_M, PADB), dtype=tl.float32)
                    for ks in range(0, K, BLOCK_K):
                        co = ks + tl.arange(0, BLOCK_K)
                        km = co < K
                        w = tl.load(w_ptr + rows[:, None] * K + co[None, :],
                                    mask=rmask[:, None] & km[None, :],
                                    other=0).to(tl.float16)
                        xt = tl.load(x_ptr + bidx[None, :] * stride_xb + co[:, None],
                                     mask=km[:, None] & bmask[None, :],
                                     other=0.0).to(tl.float16)
                        acc += tl.dot(w, xt, out_dtype=tl.float32)
                    s = tl.load(s_ptr + rows, mask=rmask, other=1.0).to(tl.float32)
                    res = (acc * s[:, None]).to(tl.float16)
                    tl.store(out_ptr + bidx[None, :] * stride_ob + rows[:, None], res,
                             mask=rmask[:, None] & bmask[None, :])

                lm_head._ds_kernel_dot = _ds_head_dot

                @triton.autotune(
                    configs=_DOT_CONFIGS, key=['M', 'K', 'NB', 'PANELS'])
                @triton.jit
                def _ds_head_dot_interleaved(
                        out_ptr, w_ptr, x_ptr, s_ptr, M, K,
                        stride_ob, stride_xb, NB: tl.constexpr,
                        PANELS: tl.constexpr, PADB: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
                    flat_pid = tl.program_id(0)
                    panel = flat_pid % PANELS
                    pid_m = flat_pid // PANELS
                    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                    rmask = rows < M
                    bidx = panel * PADB + tl.arange(0, PADB)
                    bmask = bidx < NB
                    acc = tl.zeros((BLOCK_M, PADB), dtype=tl.float32)
                    for ks in range(0, K, BLOCK_K):
                        co = ks + tl.arange(0, BLOCK_K)
                        km = co < K
                        w = tl.load(w_ptr + rows[:, None] * K + co[None, :],
                                    mask=rmask[:, None] & km[None, :],
                                    other=0).to(tl.float16)
                        xt = tl.load(
                            x_ptr + bidx[None, :] * stride_xb + co[:, None],
                            mask=km[:, None] & bmask[None, :],
                            other=0.0).to(tl.float16)
                        acc += tl.dot(w, xt, out_dtype=tl.float32)
                    s = tl.load(s_ptr + rows, mask=rmask, other=1.0).to(tl.float32)
                    res = (acc * s[:, None]).to(tl.float16)
                    tl.store(
                        out_ptr + bidx[None, :] * stride_ob + rows[:, None],
                        res, mask=rmask[:, None] & bmask[None, :])

                lm_head._ds_kernel_dot_interleaved = _ds_head_dot_interleaved

        if hasattr(lm_head, '_ds_int8'):
            M, K = lm_head._ds_int8.shape
            x = hidden_states.reshape(-1, K)
            batch = x.shape[0]
            if batch > 0:
                out = torch.empty(batch, M, dtype=torch.float16, device=x.device)
                grid = lambda meta: ((M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)
                xf = x.to(torch.float16).contiguous()
                # A pinned launch bypasses the autotuner by calling the
                # underlying JIT function with the tile as an argument, so the
                # fp32 reduction order over K is the same on every restart. If a
                # future Triton changes that interface, fall back to the search
                # rather than failing the request.
                pin = None if (getattr(lm_head, '_ds_autotune', True)
                               or getattr(lm_head, '_ds_pin_broken', False)) \
                    else dict(lm_head._ds_pinned)

                def _launch(pin):
                    g = grid if pin is None else \
                        ((M + pin['BLOCK_M'] - 1) // pin['BLOCK_M'],)
                    extra = pin or {}
                    if batch <= 4:
                        k = lm_head._ds_kernel if pin is None else lm_head._ds_kernel.fn
                        k[g](out, lm_head._ds_int8, xf, lm_head._ds_scales, M, K,
                             out.stride(0), xf.stride(0), NUM_BATCH=batch, **extra)
                    elif getattr(lm_head, '_ds_dot_ok', False) and batch <= 16:
                        # Verification batch of an MTP-k step, k >= 4. One weight
                        # pass regardless of k.
                        k = lm_head._ds_kernel_dot if pin is None else lm_head._ds_kernel_dot.fn
                        k[g](out, lm_head._ds_int8, xf, lm_head._ds_scales, M, K,
                             out.stride(0), xf.stride(0), NB=batch, PADB=16, **extra)
                    elif getattr(lm_head, '_ds_dot_ok', False) and \
                            getattr(lm_head, '_ds_interleaved', True):
                        # Query-fast flattened grid. One launch covers every
                        # 16-row panel while adjacent CTAs request the same
                        # weight tile. Unlike sequential chunking, the measured
                        # cost stays near one head pass through B=64.
                        panels = (batch + 15) // 16
                        gi = ((M + (pin['BLOCK_M'] if pin else 128) - 1) //
                              (pin['BLOCK_M'] if pin else 128) * panels,) \
                            if pin is not None else \
                            (lambda meta: (((M + meta['BLOCK_M'] - 1) //
                                            meta['BLOCK_M']) * panels,))
                        k = (lm_head._ds_kernel_dot_interleaved if pin is None
                             else lm_head._ds_kernel_dot_interleaved.fn)
                        k[gi](out, lm_head._ds_int8, xf, lm_head._ds_scales,
                              M, K, out.stride(0), xf.stride(0), NB=batch,
                              PANELS=panels, PADB=16, **extra)
                    elif getattr(lm_head, '_ds_dot_ok', False) and \
                            getattr(lm_head, '_ds_chunk16', True):
                        # More than 16 rows. The old fallback launched one
                        # single-row kernel per row, so each row re-read the
                        # whole 1.271 GB head: a C=16 MTP-6 verify call presents
                        # 16 * 7 = 112 rows and requested about 142 GB, or 605 ms
                        # at this part's 235.47 GB/s. Seven 16-row calls request
                        # 8.9 GB. The batch dimension is the only thing that
                        # changes, so each chunk is bit-identical to the same
                        # rows submitted as their own batch <= 16 call.
                        #
                        # This also matters without speculation: the
                        # non-speculative head batch is one row per running
                        # request, so the cliff was at C = 17, one request above
                        # the concurrency this project reports.
                        k = lm_head._ds_kernel_dot if pin is None else lm_head._ds_kernel_dot.fn
                        for start in range(0, batch, 16):
                            nb = batch - start
                            if nb > 16:
                                nb = 16
                            k[g](out[start:start + nb], lm_head._ds_int8,
                                 xf[start:start + nb], lm_head._ds_scales, M, K,
                                 out.stride(0), xf.stride(0), NB=nb, PADB=16,
                                 **extra)
                    else:
                        # Reachable only with DENSESPARK_HEAD_BATCH_DOT=0 or
                        # DENSESPARK_HEAD_CHUNK16=0. The per-row loop re-reads
                        # the weights for every row, so it is a correctness
                        # fallback and not a fast path.
                        k = lm_head._ds_kernel if pin is None else lm_head._ds_kernel.fn
                        for b in range(batch):
                            k[g](out[b:b + 1], lm_head._ds_int8, xf[b:b + 1],
                                 lm_head._ds_scales, M, K,
                                 out.stride(0), xf.stride(0), NUM_BATCH=1, **extra)

                try:
                    _launch(pin)
                except TypeError:
                    if pin is None:
                        raise
                    lm_head._ds_pin_broken = True
                    import sys as _sys2
                    print('DENSESPARK: pinned head launch unavailable, using autotune',
                          file=_sys2.stderr, flush=True)
                    _launch(None)
                logits = out.view(hidden_states.shape[:-1] + (M,))
                if embedding_bias is not None:
                    logits = logits + embedding_bias
                if lm_head.tp_size > 1:
                    logits = self._gather_logits(logits)
                if logits is not None:
                    logits = logits[..., : self.org_vocab_size]
                return logits

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)'''


def find_target(vllm_root=None):
    """Return the logits_processor.py to patch, or None if no vLLM was found."""
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
        candidate = os.path.join(root, RELATIVE_TARGET)
        if os.path.exists(candidate):
            return candidate
    return None


def apply(target):
    with open(target, encoding="utf-8") as handle:
        content = handle.read()

    if MARKER in content:
        print(f"SKIP: already applied to {target}")
        return 0

    if ANCHOR not in content:
        print(f"FAIL: _get_logits in {target} is not the vLLM 0.27.x form")
        print("      Patches are version specific; re-check against your vLLM.")
        return 1

    with open(target, "w", encoding="utf-8") as handle:
        handle.write(content.replace(ANCHOR, REPLACEMENT))
    print(f"OK: INT8 LM head applied to {target}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm-root", help="path to the vllm package to patch")
    args = parser.parse_args()

    target = find_target(args.vllm_root)
    if target is None:
        print("FAIL: no vLLM installation found")
        print("      Pass --vllm-root /path/to/site-packages/vllm")
        return 1
    return apply(target)


if __name__ == "__main__":
    sys.exit(main())
