# Qwen3.8-27B on DGX Spark

This repository contains the small runtime I use to serve Qwen3.8-27B on a
single NVIDIA DGX Spark. The recommended profile uses Unsloth's native NVFP4
checkpoint and Qwen's native MTP draft model. It exposes an OpenAI-compatible
API through vLLM and binds it to localhost.

The launcher is deliberately narrow in scope. It pins the container image,
keeps model weights outside the repository, and makes the settings that are
useful to tune available as environment variables.

## Requirements

- NVIDIA DGX Spark, or another GB10 system with `sm_121a` support
- Docker with NVIDIA Container Toolkit configured
- one of the supported Qwen3.8-27B checkpoints downloaded locally

The default model directory is `$HOME/models/qwen3.8-27b`:

```text
qwen3.8-27b/
├── bf16/
├── fp8/
└── nvfp4/
```

Only the directory for the profile you run is required. Each directory should
be a complete Hugging Face checkpoint and contain `config.json`.

## Running it

The recommended production command is:

```bash
QWEN38_KV_CACHE_DTYPE=float8_e4m3fn \
QWEN38_CONTEXT=131072 \
QWEN38_GPU_MEMORY_UTILIZATION=0.70 \
QWEN38_MAX_SEQS=8 \
QWEN38_MAX_BATCHED_TOKENS=16384 \
QWEN38_MTP_TOKENS=3 \
runtime/run-vllm.sh nvfp4 mtp
```

This selects the NVFP4 checkpoint, native MTP at width 3, the calibrated FP8
KV cache, CUDA graphs, prefix caching, chunked prefill, and the scheduler limits
used for the promoted benchmark profile.

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

Use `runtime/stop-vllm.sh` to stop the container cleanly.

## Profiles

The first launcher argument selects `bf16`, `fp8`, or `nvfp4`. The second is
either `none` or `mtp`. MTP is useful with the NVFP4 checkpoint; set
`QWEN38_MTP_TOKENS` to change its draft width.

The promoted profile has:

- a 131,072-token context window
- configurable KV cache dtype; the measured production profile used FP8
- CUDA graphs, async scheduling, prefix caching, and chunked prefill
- Qwen3 reasoning and Qwen3 Coder tool-call parsers
- at most eight concurrent sequences

Explicit API sampling values are left to the client. Qwen recommends these
presets in the [official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices):

| Mode | Thinking | Temperature | Top-p | Top-k | Min-p | Presence penalty | Repetition penalty |
|---|---:|---:|---:|---:|---:|---:|---:|
| Thinking | enabled | `1.0` | `0.95` | `20` | `0.0` | `0.0` | `1.0` |
| Instruct | disabled | `0.7` | `0.8` | `20` | `0.0` | `1.5` | `1.0` |

For vLLM, set `chat_template_kwargs.enable_thinking` to `true` or `false` in
the request body. The launcher sets `preserve_thinking` to `false`, so prior
reasoning is not carried into later turns.

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

All profiles used the same vLLM optimizations listed above. The `none` rows
show precision changes without speculative decoding; the MTP rows isolate the
additional decode optimization. Width 3 produced the best observed time to a
short answer at 3.47 seconds. Wider drafts increased the reported short-token
rate but made the complete answer slower and reduced sustained coding speed.

Pinned checkpoint revisions used for the comparison:

- `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- `Qwen/Qwen3.8-27B-FP8` at `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`
- `unsloth/Qwen3.8-27B-NVFP4` at `16b6615af3548b88e2d8e382457bc705b00479cf`

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `QWEN38_MODELS_ROOT` | `$HOME/models/qwen3.8-27b` | checkpoint directories |
| `QWEN38_CACHE` | `${XDG_CACHE_HOME:-$HOME/.cache}/qwen3.8-vllm` | compiler and kernel cache |
| `QWEN38_PORT` | `18083` | local API port |
| `QWEN38_CONTEXT` | `131072` | maximum model length |
| `QWEN38_GPU_MEMORY_UTILIZATION` | `0.70` | vLLM GPU memory target |
| `QWEN38_MAX_SEQS` | `8` | maximum concurrent sequences |
| `QWEN38_MAX_BATCHED_TOKENS` | `16384` | scheduler token budget |
| `QWEN38_KV_CACHE_DTYPE` | `auto` | KV cache data type |
| `QWEN38_MTP_TOKENS` | `3` | native MTP draft width |
| `QWEN38_CUDAGRAPHS` | `1` | set to `0` to disable CUDA graphs |

The image can be changed with `QWEN38_IMAGE`, but the pinned digest is the only
one tested here. The launcher uses host networking and binds vLLM to
`127.0.0.1`; put an authenticated proxy in front of it if remote access is
needed.

## Notes

Model licenses are not covered by this repository's license. Review the terms
for each checkpoint before downloading or redistributing weights. The launcher
also enables vLLM's `--trust-remote-code`; inspect checkpoint code when changing
away from the pinned revisions above.
