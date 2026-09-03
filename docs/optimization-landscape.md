# Qwen3.8-27B optimization landscape — 2026-09-03

## Executive decision

Keep RadixArk NVFP4 + SGLang DSpark as the default today. DenseSpark 1.3 is the
most credible route to a meaningful speed increase and is included in this
repository for qualification, but it has not passed the local concurrency and
agent-correctness gates. DFlash2 is a useful SGLang-side alternative, especially
if preserving Radix prefix caching is a requirement, but its current public
recipe also relies on a patched image and has variable boot-time tuning.

## What is available

| Stack | Target quantization | Drafter | Reported or measured result | Main trade-off |
|---|---|---|---:|---|
| Existing production | RadixArk NVFP4 | DSpark K7 | 39.60 tok/s C1; 163.30 tok/s C8 local | Fast, stable, 131K and prefix-cached |
| DFlash2 SGLang | RadixArk NVFP4 | DFlash2 K8/K10 | 42.04 tok/s dedicated C1; 114.5 tok/s C8 upstream | Patched image; auto-tune boot variance |
| DenseSpark 1.3 | AutoRound GPTQ INT4, group 128 | native MTP K8/K6/K3 | 49.1 token-weighted C1; 260.2 tok/s C16 upstream | No deterministic prefix cache; complex patched runtime |
| DenseSpark local C16 profile | same | native MTP K3 | 38.76 tok/s warm C1 | Eight-way qualification did not finish |

The rows do not use one universal benchmark protocol, so they show promising
operating points rather than a perfectly controlled ranking. The repository's
next qualification run must use fixed prompts, seeds and sampling across all
three candidates.

## Quantization

DenseSpark uses `Frozenlock/Qwen3.8-27B-int4-AutoRound`: symmetric GPTQ INT4
with group size 128 and BF16 activations. This is not the same representation
as the structured NVFP4 checkpoints used by the SGLang and earlier vLLM
profiles. Its upstream numerical gate compares optimized kernels with exact
dequantization of the same INT4 checkpoint and reports a mean NLL delta of
-0.00012 nat/token against a 0.005 threshold. That is strong evidence that the
runtime transforms preserve the quantized model, but it does **not** measure
the checkpoint's quality loss against BF16 or NVFP4.

The existing local task and agent evaluation therefore remains mandatory.
Kernel-equivalence measurements cannot replace end-to-end coding, tool-call,
instruction-following and long-context checks.

DenseSpark's default KV cache is BF16. It measured 710,106 cache tokens at 0.90
GPU utilization, compared with 909,193 tokens for the production SGLang FP8 KV
profile at a lower 0.80 static-memory fraction. An FP8 DenseSpark KV cache may
increase capacity, but it has not passed this repository's quality gate and is
not enabled by either wrapper.

## Drafting

- DSpark K7 is the known-good default. It has the best local evidence: short,
  coding, concurrent, long-prefix and tool-call tests.
- DFlash2 can improve single-request output, with published K10 at 42.04 tok/s,
  but its own results show a roughly 42-versus-33 tok/s fast/slow boot split.
  A production launcher would need a startup performance gate and restart loop.
- DenseSpark uses Qwen's native MTP layer. Its measured policy selects K8 for
  concurrency 1, K6 for concurrency 2 and K3 from concurrency 3 upward. Deep
  drafts make sense while the GPU would otherwise wait for one stream; shallow
  drafts win once a batch supplies enough target work.

This is why the new wrapper offers workload profiles rather than one global
draft setting.

## DenseSpark performance mechanisms

The speedup is compositional rather than one flag:

1. Humming 0.1.13 kernels target GB10/SM121 INT4 linears.
2. Three-way dispatch chooses Humming, Marlin or CUTLASS according to matrix
   shape instead of forcing one kernel over prefill and decode.
3. FlashInfer handles GDN prefill.
4. The output projection is INT8 and uses a product-quantized draft-head search.
5. Native MTP depth is selected by intended concurrency.
6. The compiled vLLM cache is persisted and keyed by relevant tuning controls.

The cost is a large patch surface over pinned vLLM 0.27.1. Updating vLLM means
reapplying and revalidating nine transforms; using an unpinned image is unsafe.

## Context and multi-agent implications

For repeated repository prefixes, SGLang remains materially better: a local
55,670-token repeated prompt fell from 41.93 seconds TTFT to 0.37 seconds via
Radix reuse. DenseSpark disables prefix caching because upstream found
nondeterministic output on the hybrid GDN cache path.

At the measured DenseSpark capacity, five independent agents can each occupy a
full 131K context. A 16-agent deployment can admit 16 requests, but they cannot
all hold 64K simultaneously in the 710K-token BF16 cache; the arithmetic limit
is 10.8 full 64K contexts. Real agent contexts are usually below their declared
maximum, but the distinction must be visible in capacity planning.

For multi-agent coding today:

- use SGLang when agents share a large stable repository prefix or require the
  full 131K window;
- qualify DenseSpark `agents` when aggregate decode throughput dominates and
  average live context stays below roughly 44K at 16 active requests;
- use DenseSpark `interactive` only for latency-sensitive one-agent work after
  its K8 result and correctness suite pass locally.

## Promotion gates

DenseSpark replaces SGLang only after all of these pass on this Spark:

1. Three warm runs each at concurrency 1, 8 and 16, with cold compilation
   reported separately.
2. The existing Pi agent suite and deterministic tool-call tests.
3. Exact retrieval at 32K, 64K, 96K and 128K.
4. A sustained mixed prompt/decode soak with no engine death, preemption or host
   memory failure.
5. Restart reproducibility from its persistent compile cache.

## Sources

- [DenseSpark repository](https://github.com/albond/DenseSpark-Qwen3.8-27B)
- [DenseSpark NVIDIA forum report](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-single-dgx-spark-1-9x-stock-nvfp4-2-5x-fp8-at-16-concurrent-50-tok-s-on-a-single-request/381923)
- [DFlash2 DGX Spark repository](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark)
- [SGLang DFlash2 tracking issue](https://github.com/sgl-project/sglang/issues/35860)
- [Earlier SGLang NVFP4/DSpark/DFlash2 experiments](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang)
