# Qwen3.8-27B on DGX Spark

This repository serves Qwen3.8-27B on one NVIDIA DGX Spark. The validated
production profile uses RadixArk's NVFP4 target, its native DSpark drafter and
SGLang. A newer, substantially more ambitious DenseSpark 1.3 profile is now
included as an opt-in candidate. It combines a symmetric AutoRound GPTQ INT4
checkpoint, native MTP, an INT8 output head, product-quantized draft-head
search, SM121 Humming kernels, three-way linear dispatch and FlashInfer GDN
prefill on vLLM 0.27.1.

The SGLang launcher pins its container image and checkpoints. DenseSpark is
vendored at upstream release 1.3, commit
`9ae122f757bbae28c875d00c8b186c4837187434`, so this repository remains a
complete checkout rather than requiring a Git submodule.

## Requirements

- NVIDIA DGX Spark, or another GB10 system with `sm_121a` support
- Docker with NVIDIA Container Toolkit configured
- the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli),
  or `uvx` as a fallback

The default model directory is `$HOME/models/qwen3.8-27b`:

```text
qwen3.8-27b/
├── bf16/
├── fp8/
├── nvfp4/
├── radix-nvfp4/
└── radix-dspark/
```

Only the directory for the profile you run is required. If it is missing, the
launcher downloads the selected checkpoints at pinned revisions before
starting the server. Hugging Face's local-directory metadata makes interrupted
and repeated downloads resumable. Existing checkpoints are left alone.

## Running it

The recommended production command is:

```bash
runtime/run-sglang.sh
```

This selects the RadixArk NVFP4 target and DSpark drafter at block size 7,
an FP8 KV cache, FlashInfer attention, CUDA graphs for decode and verification,
Radix prefix caching, 8,192-token chunked prefill, and eight request slots.

The endpoint is available at `http://127.0.0.1:18083/v1`. For example:

```bash
curl http://127.0.0.1:18083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
```

Use `runtime/stop-sglang.sh` to stop the container cleanly.

### DenseSpark candidate

Build it once (about 18 GiB of model weights plus the container image):

```bash
runtime/densespark/install.sh --fast --concurrency 16 --no-launch
```

Then select a workload rather than hand-tuning the draft depth:

```bash
# 16-request, 64K multi-agent profile; native MTP depth 3
runtime/run-densespark.sh agents

# 131K latency profile; native MTP depth 8
runtime/run-densespark.sh interactive
```

Both expose model `qwen3.8-27b` at `http://127.0.0.1:18083/v1`. Stop either
with `runtime/stop-densespark.sh`. The profile name describes the concurrency
used to choose MTP depth; vLLM admission is capped at 16 only in the agent
profile.

DenseSpark currently defaults to a BF16 KV cache and disables prefix caching.
At 131K, the measured cache held 710,106 tokens, or 5.42 completely full
contexts. Choose SGLang when repeated long prefixes or validated full-context
capacity matter more than peak decode throughput.

## Runtime profiles

`runtime/run-sglang.sh` remains the promoted profile. `runtime/run-densespark.sh`
is the faster experimental path. `runtime/run-vllm.sh` remains available for
the BF16, FP8, and Unsloth NVFP4 checkpoints; its second argument is `none` or
`mtp`.

The promoted profile has:

- a 131,072-token context window
- an FP8 KV cache and 909,193 measured cache-token capacity
- target-verification and DSpark-draft CUDA graphs
- Radix prefix caching and chunked prefill
- Qwen3 reasoning and Qwen3 Coder tool-call parsers
- at most eight concurrent requests

Explicit API sampling values are left to the client. Qwen recommends these
presets in the [official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices):

| Mode | Thinking | Temperature | Top-p | Top-k | Min-p | Presence penalty | Repetition penalty |
|---|---:|---:|---:|---:|---:|---:|---:|
| Thinking | enabled | `1.0` | `0.95` | `20` | `0.0` | `0.0` | `1.0` |
| Instruct | disabled | `0.7` | `0.8` | `20` | `0.0` | `1.5` | `1.0` |

Set `chat_template_kwargs.enable_thinking` to `true` or `false` in the request
body. Do not include prior reasoning content in subsequent messages.

SGLang's native Responses endpoint uses `reasoning.effort: "none"` rather than
`chat_template_kwargs` to select non-thinking mode. The parent DGX Spark
unified API performs this mapping automatically. Its non-thinking streaming
path also translates SGLang Chat Completions events to Responses events to
work around a serializer bug in the pinned image; text and function-call
streaming are covered by integration tests.

## Speed on DGX Spark

These are three-run measurements from one DGX Spark with a 131,072-token
context, temperature 0, and top-p 1. The 16-bit baseline is the official BF16
checkpoint; Qwen does not publish this checkpoint as FP16.

| Checkpoint | Speculation | Short decode | Coding decode | Long prefill |
|---|---:|---:|---:|---:|
| Qwen BF16 | none | 4.41 tok/s | 4.40 tok/s | 1,206 tok/s |
| Qwen FP8 | none | 7.87 tok/s | 7.84 tok/s | 906 tok/s |
| Unsloth NVFP4 | none | 11.20 tok/s | 11.20 tok/s | 1,315 tok/s |
| Unsloth NVFP4 | MTP, width 2 | 23.91 tok/s | 21.44 tok/s | 1,248 tok/s |
| Unsloth NVFP4 | MTP, width 3 | 26.84 tok/s | 22.98 tok/s | — |
| RadixArk NVFP4 | SGLang DSpark, block 7 | 39.60 tok/s | 30.16 tok/s | — |
| AutoRound GPTQ INT4 | DenseSpark, C16/MTP3 | 38.76 tok/s | — | — |

The first five rows use the older vLLM profiles. The RadixArk row is the
promoted SGLang profile and achieved
163.30 aggregate tok/s at concurrency 8. On a 55,670-token repository prompt,
prefix reuse reduced TTFT from 41.93 seconds to 0.37 seconds and the repeated
request completed in 7.25 seconds. A Qwen Coder tool-call smoke test passed.

The DenseSpark row is the median of two warm 1,000-input/1,000-output runs on
the profile tuned for concurrency 16, not its faster concurrency-1/MTP8
profile. Its first concurrency-8 shape did not complete during local
qualification, so DenseSpark has not replaced SGLang. See the
[evaluation record](results/2026-09-03-densespark-evaluation.md) for cold-start,
memory and failure details.

Upstream DenseSpark reports 44.9 tok/s mean decode and 49.1 token-weighted
single-request throughput, plus 260.2 aggregate tok/s for its balanced
16-request test. Those are promising targets, not results reproduced by this
repository. The other active optimization line is
[DFlash2 on SGLang](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark),
which retains the Radix/SGLang architecture and reports 114.5 aggregate tok/s
at concurrency 8. It currently depends on a patched image and its auto-tuned
single-stream result has a documented fast/slow boot lottery.

Pinned checkpoint revisions used for the comparison:

- `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- `Qwen/Qwen3.8-27B-FP8` at `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- `unsloth/Qwen3.8-27B-NVFP4` at `16b6615af3548b88e2d8e382457bc705b00479cf`
- `RadixArk/Qwen3.8-27B-NVFP4` at `52d1adc5f38aa5ebf099c29ed7025ba34cfbb854`
- `RadixArk/Qwen3.8-27B-DSpark` at `85ef153be924f17ce4bf62726954eeaa4a73e854`

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `QWEN38_MODELS_ROOT` | `$HOME/models/qwen3.8-27b` | checkpoint directories |
| `QWEN38_SGLANG_IMAGE` | pinned digest | SGLang container image |
| `QWEN38_CACHE` | `${XDG_CACHE_HOME:-$HOME/.cache}/qwen3.8-sglang` | compiler and kernel cache |
| `QWEN38_PORT` | `18083` | local API port |
| `QWEN38_CONTEXT` | `131072` | maximum model length |
| `QWEN38_MEMORY_FRACTION` | `0.80` | SGLang static memory fraction |
| `QWEN38_MAX_REQUESTS` | `8` | maximum concurrent requests |
| `QWEN38_CHUNKED_PREFILL_SIZE` | `8192` | prefill chunk size |
| `QWEN38_KV_CACHE_DTYPE` | `fp8_e4m3` | KV cache data type |
| `QWEN38_DSPARK_BLOCK_SIZE` | `7` | DSpark proposal block size |
| `QWEN38_AUTO_DOWNLOAD` | `1` | set to `0` to require preloaded weights |

DenseSpark's primary deployment controls are:

| Variable | Agent default | Interactive default | Purpose |
|---|---:|---:|---|
| `DENSESPARK_CONCURRENCY` | `16` | `1` | selects the measured MTP policy row |
| `DENSESPARK_MAX_LEN` | `65536` | `131072` | maximum sequence length |
| `DENSESPARK_MAX_NUM_SEQS` | `16` | vLLM default | admission cap |
| `DENSESPARK_PORT` | `18083` | `18083` | loopback API port |
| `DENSESPARK_GPU_UTIL` | `0.90` | `0.90` | vLLM GPU-memory utilization |
| `DENSESPARK_KV_DTYPE` | auto/BF16 | auto/BF16 | KV-cache precision |
| `DENSESPARK_SPEC_TOKENS` | policy (`3`) | policy (`8`) | explicit MTP-depth override |

The vendored upstream README documents the lower-level experimental switches.
Do not turn on prefix caching without also accepting its documented
nondeterministic-output failure on this hybrid GDN model.

The pinned image digest is the only one tested here. The launcher uses host
networking and binds SGLang to `127.0.0.1`; put an authenticated proxy in front
of it if remote access is needed. The public checkpoints do not require
authentication; if that changes, the downloader honors the Hugging Face CLI's
saved login or `HF_TOKEN`.

## Notes

Model licenses are not covered by this repository's license. Review the terms
for each checkpoint before downloading or redistributing weights. The launcher
also enables vLLM's `--trust-remote-code`; inspect checkpoint code when changing
away from the pinned revisions above.
