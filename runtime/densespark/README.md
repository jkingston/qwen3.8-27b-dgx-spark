# DenseSpark — Qwen3.8-27B, fast, on one DGX Spark

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/NVIDIA-DGX_Spark-76B900?style=flat&logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97-Qwen3.8--27B-yellow)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![vLLM](https://img.shields.io/badge/vLLM-0.27.1-red?style=flat)](https://docs.vllm.ai/)
[![Single request](https://img.shields.io/badge/single_request-58_tok%2Fs-brightgreen?style=flat)](#what-you-get)
[![16 parallel](https://img.shields.io/badge/16_parallel-260_tok%2Fs-brightgreen?style=flat)](#what-you-get)

**Qwen3.8-27B is a very good model that is painfully slow to run at home. This
makes it fast on a single DGX Spark, and it takes one command.**

A dense 27B reads all 27 billion parameters for every single token it writes.
That is what makes it good, and it is what makes it slow: out of the box on one
Spark it is slow enough that people install it, try it, and stop using it.
DenseSpark ships a measured configuration that removes most of that cost, and
every number below was measured on one machine with the protocol stated beside
it.

## Why this model

On the independent
[Artificial Analysis Intelligence Index](https://artificialanalysis.ai/models/qwen3-8-27b)
v4.1.1 — nine evaluations including GPQA Diamond, Humanity's Last Exam, SciCode
and Terminal-Bench — a 27B dense model lands four points below a 180B one and
above a 122B one:

| model | size | Intelligence Index |
|---|---|---:|
| Qwen3.8-Flash-Next | 180B total, 6B active | 56 |
| **Qwen3.8 27B** (xhigh reasoning) | **27B dense** | **52** |
| Qwen3.8 27B (medium) | 27B dense | 44 |
| Qwen3.8 27B (low) | 27B dense | 43 |
| Qwen3.8 27B | 27B dense | 35 |
| Qwen3.5 122B A10B | 125B total, 10B active | 33 |

Those are Artificial Analysis's numbers, not measurements from this project.

Read the first two rows together: the model at the top needs 180 billion
parameters resident, and the one below it needs 27 billion — a size a single
DGX Spark holds whole — to come within four points of it.

The catch is that those four points cost reasoning tokens. Going from 35 to 52
means the model thinks at length before answering, and thinking at length is
only usable if the tokens arrive quickly. That is what the rest of this
repository is for.

## Try it

You need a DGX Spark (or a compatible GB10 machine) and about 60 GB free in
your home directory — 19 GB of that is the model itself, the rest is the
container image and build scratch. The installer checks before it downloads
anything.

```bash
git clone https://github.com/albond/DenseSpark-Qwen3.8-27B
cd DenseSpark-Qwen3.8-27B
./install.sh
```

By default the API is published on loopback only. To reach it from another
machine — a chat UI on your LAN, say — name the interface explicitly, and read
[SECURITY.md](SECURITY.md) first, because there is no authentication:

```bash
DENSESPARK_BIND_HOST=0.0.0.0 ./configs/launch-densespark.sh
```

Tool calling works out of the box: the server is started with vLLM's automatic
tool choice and the parser this checkpoint's chat template needs, so agent
clients that ask for `tool_choice: "auto"` are served. Those are capability
gates rather than a mode — a request without a `tools` field takes the ordinary
chat path and nothing about generation changes — so there is no tool variant to
choose between and no separate server to run.

It asks **how many requests you expect to run at the same time** — 1 if it is
just you, more if a team or an application will use it. That tunes the server;
it does not limit it, and more requests than that still work. Then it offers to
start; say no and start it yourself whenever you want:

```bash
./configs/launch-densespark.sh
```

That is the whole thing. It serves an OpenAI-compatible API on port 8000, so
any tool that talks to OpenAI talks to this:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"densespark-qwen3.8-27b","messages":[{"role":"user","content":"Hello"}]}'
```

You do not have to read anything below this line to use it.

## What you get

**One request decodes at 44.9 tokens per second** — five prompts of different
lengths, three runs each, decode alone, measured with the checkpoint's reasoning
mode off so the figure is decode speed and nothing else. That is the speed you
feel when you are the only one talking to it. Token-weighted it is 49.1.

On NVIDIA's three tests, output tokens per second and time to first token:

| NVIDIA test | one request | 16 at once |
|---|---:|---:|
| balanced (1,000 in / 1,000 out) | **49.1** · 0.66 s | **260.2** · 7.57 s |
| prompt-heavy (8,000 in / 1,000 out) | **48.1** · 4.08 s | **111.2** · 37.97 s |
| decode-heavy (1,000 in / 8,000 out) | **65.2** · 0.66 s | **227.0** · 7.56 s |

### Against stock, at sixteen requests

Stock FP8 and stock NVFP4 as published on the
[NVIDIA developer forum](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dgx-spark-using-vllm-nvfp4-vs-fp8-performance/380258),
same model, same three tests, no speculative decoding. Output tokens per second.

| NVIDIA test | stock FP8 | stock NVFP4 | **DenseSpark** | vs NVFP4 |
|---|---:|---:|---:|---:|
| balanced (1,000 in / 1,000 out) | 104.44 | 134.41 | **260.2** | **1.94x** |
| prompt-heavy (8,000 in / 1,000 out) | 65.58 | 87.91 | **111.2** | **1.26x** |
| decode-heavy (1,000 in / 8,000 out) | 99.47 | 132.07 | **227.0** | **1.72x** |

**Stock is faster to the first token, on every one of the three**, and that is a
systematic trade rather than one awkward row:

| time to first token, 16 parallel | stock NVFP4 | DenseSpark |
|---|---:|---:|
| balanced | 4.05 s | 7.57 s |
| prompt-heavy | 29.81 s | 37.97 s |
| decode-heavy | 4.05 s | 7.56 s |

The path that would have closed that gap was built, measured, and dropped: it
fails the quality gate, and nothing that fails it gets shipped. If your workload
is many short requests where the wait to the first token dominates, that is the
trade to weigh.

The DenseSpark column is the median of three retained runs on the image
`./install.sh` builds, with the settings the launcher picks from the concurrency
you gave it. Spread is 1.6% on balanced and 5.3% on prompt-heavy; two cells
carry one low run among three and spread 32% and 40%. Reproduce it with the
same protocol — one warm-up pass over every shape, then three retained runs,
with the generation seed fixed:

```bash
for s in balanced prompt-heavy decode-heavy; do
  ./bench_serving.py --scenario "$s" --concurrency 16 --runs 1 \
      --generation-seed 20260829 --result-dir "out/warmup/$s"
done
for s in balanced prompt-heavy decode-heavy; do
  ./bench_serving.py --scenario "$s" --concurrency 16 --runs 3 \
      --generation-seed 20260829 --result-dir "out/retained/$s"
done
```

Each run needs its own directory: the benchmark refuses to overwrite results it
already wrote, which is the behaviour you want and the reason for the `$s` in
those paths.

Without `--runs 3` and a fixed seed you are looking at a single sample of a
distribution whose spread reaches 40% in two of these cells.

## Why the number you pick at install matters

Speculative decoding guesses several tokens ahead and has the model check them.
A long guess pays off when the machine would otherwise sit waiting on one
request, and wastes work when sixteen requests already keep it busy. So the
profile holds a setting per concurrency and your answer picks the row:

| requests you expect at once | tokens guessed ahead |
|---|---:|
| 1 | 8 |
| 2 | 6 |
| 3 or more | 3 |

The ends of that table are measured firmly, the middle less so: one request
rests on fifteen measurements per candidate depth, three and up on an
eighty-cell sweep where deeper chains lose outright, and two on a direction two
cells agree about inside a spread too wide to call the margin.

Get it wrong and nothing breaks — set `DENSESPARK_CONCURRENCY`, or re-run
`./install.sh`, and the server comes back tuned for that instead.

## What is under the hood

**INT4 weights.** A dense model's speed is set by how many bytes it reads per
token, so storing the weights in 4-bit AutoRound instead of 16-bit is most of
the win by itself.

**Guessing ahead, then checking.** The checkpoint ships a small extra layer that
guesses the next few tokens cheaply; the full model verifies them in one pass
instead of writing each one separately. Because it checks every guess, this
cannot change what the model would have said.

**Kernels built for this chip.** Several routines vLLM would otherwise pick are
compiled for a different architecture than the GB10's SM121. This replaces the
ones that matter — the output projection, the linear attention that 48 of the 64
layers use, and the backbone matrix multiplications — each dispatched by size so
the right routine runs at the right shape.

Nine build steps apply that. Each refuses to run if the code it expects has
changed and re-checks its own result, so a half-applied build fails while you
are installing rather than while you are using it.

## Does it change the answers?

Half of that question has an exact answer and half has a measured one, and they
are worth separating.

**The speculative decoding does not change the distribution they come from.**
The full model verifies every guessed token, and rejection sampling is built so
that what survives is distributed exactly as the model's own output would be.
That is structural rather than statistical — but it is a statement about the
distribution, not about a particular reply: with greedy decoding you get the
same text, while with sampling you get a draw from the same distribution, which
at a given seed need not be the same draw, because the two paths consume the
random stream differently.

**The quantization and the kernels are measured, not proven.** On a frozen
long-prompt gate — 23,997 tokens of 8,000-token prompts, on the image
`install.sh` builds — the shipped stack's **paired mean-NLL delta** against exact
dequantization is **-0.00012 nat/token**, against a threshold of 0.005. The
honest reading is *no measurable degradation on this gate*. That delta is a
difference of average losses, not a distance between distributions, and it does
not certify that every answer is identical. Its sign is not a win either:
landing marginally below the reference is what being inside the noise looks
like.

Paired mean-NLL deltas, nat/token, against the 0.005 threshold:

| | delta | vs threshold |
|---|---:|---|
| the shipped profile, against exact dequantization | **-0.00012** | passes |
| the linear-attention prefill backend | -0.000487 | passes |
| the long-prompt prefill route | -0.000700 | passes |
| structured NVFP4 — **not shipped** | +0.0156 to +0.0159 | **exceeds it threefold** |

That last row is the design decision: the fastest prefill path measured here,
and not in the build, because it is the only one that fails the gate. It is
compared against the threshold rather than against the shipped figure, since
dividing two signed loss differences produces a number with no meaning once one
of them is negative.

This is a gate on the model's own predictive distribution, not a task benchmark,
and no task-accuracy claim is made anywhere.

## What didn't work

Everything here was measured and then not shipped. It is the part of this
project worth the most to anyone else, because each line is an evening someone
does not have to spend.

| Tried | Result |
|---|---|
| **Structured NVFP4 for prefill** | Works, and its paired mean-NLL delta against exact dequantization is **+0.0156**, which is 3.1x the 0.005 quality gate — the only measured path here that fails it. The fastest thing built here and deliberately dropped. The fused SwiGLU goes with it: it fuses into the NVFP4 quantizer and cannot be taken alone. |
| **A sparse candidate scan in the draft head** | Withdrawn, and not for speed. It needs real proposal probabilities for its rejection step and refuses when a request asks for deterministic output — which takes down the engine, not the request. It was also slower than the plain head at one request. |
| **A retrieval tail on the proposal chain** | −7%. Offline it raised accepted tokens per step 4.44 → 4.74, exactly as designed. The lookup must read back the tokens MTP just proposed, and that one blocking copy stalls vLLM's async scheduling. Free in bytes is not free. |
| **Chains longer than eight drafts** | Unresolved, and reported as such. Two rounds of measurement disagree about whether the optimum moves outward, and neither settles it. |
| **Dead-tail state elision** | −0.8% unweighted, −3.9% token-weighted. |
| **A wider PQ candidate set** | No effect, at cost. Accepted tokens per step is identical to four decimals at 2,048, 8,192 and 32,768 candidates while the step grows 4.2 ms. Whatever the loss is, the true argmax is not falling outside the shortlist. |
| **Rolling native NVFP4 out layer by layer** | Layers 0 and 1 clear the error gate; 2, 3 and the aggregate miss it. |
| **2:4 structured sparsity on the down projection** | Exact, beats a matched dense control, and still loses: the installed kernel is 11.8–12.8% faster throughout, and widening to N64 leaves 15.9–16.7%. |
| **Natural activation sparsity in FP4** | Bounded at nothing. 0.0057% of K128 tiles are eligible, capping the ideal speedup at 1.0000287x. That is a property of the model, so no better kernel changes it. |
| **Low-rank recovery of the quantization residual** | Rank-64 retains 95.02% of the error a plain diagonal affine already leaves. |
| **Distributed shared memory across the SM cluster** | Exact, and 2.73x slower. |
| **Certified adaptive reranking in the PQ head** | 0 of 128 certificates, at every group size and width tested. |
| **Partitioned batch-flat K64 composition** | Exact, and 0.883x. |
| **FlashInfer attention backend** | Zero. Full attention is 5.7% of this model's decode read — 48 of its 64 layers hold no KV cache. There is nothing there to win. |
| **Building vLLM from source for `sm_121a`** | Zero, and slightly negative at baseline. The stock image is the default for a reason. |
| **Disabling vLLM's startup autotuning** | No effect on reproducibility. Two restarts still agreed on only 2 of 6 prompts. What did make the difference is pinning the torch.compile cache instead of letting each container start with an empty one — five consecutive restarts then came out byte-identical. |
| **The legacy video-SIMD PTX** (`vadd4`, `vmax4`, `vavrg4`) | 0.13x to 0.51x of scalar `IMAD`. They lower to up to 35 instructions. The packed byte opcodes exist in this silicon; that PTX is not the door to them. |
| **`rcp.approx.f32` and `rsqrt.approx.f32` as single instructions** | They are not: 6.9 and 3.8 instructions, at 0.12x and 0.13x of `FFMA`. RMSNorm is built on the second one. |
| **Entropy-coded INT4 weights** | Bounded, not merely unimplemented. A 4-bit alphabet yields so little entropy per symbol that a fused decoder would need a symbol rate this silicon does not have. |
| **Token trees over the recurrent layers** | +7.40 ms per fork. Every branch must fork the Gated DeltaNet recurrent state. |
| **Instruction-level tuning of the target pass** | Nothing left to win. The INT4 body runs at 98.3% of the measured 235.5 GB/s streaming ceiling, about 2 instructions per weight byte against roughly 33 slots per streamed byte. |

## Model and checkpoint

| Property | Value |
|---|---|
| Architecture | `Qwen3_5ForConditionalGeneration` |
| Layers | 64: 48 linear-attention and 16 full-attention |
| MLP | Dense, intermediate size 17,408 |
| Hidden size | 5,120 |
| Vocabulary | 248,320, untied LM head |
| Maximum context | 262,144 tokens |
| Speculation | Built-in MTP layer |
| Checkpoint | `Frozenlock/Qwen3.8-27B-int4-AutoRound` |
| Quantization | Symmetric INT4 AutoRound, group size 128 |

**Why that checkpoint and not Intel's.** AutoRound is Intel's method, and Intel
does publish an AutoRound build of this model — a 2.8 bits-per-weight mixed
INT2/INT4 checkpoint in safetensors, whose card documents serving it with vLLM
from an unmerged pull request. What Intel does not publish for this model is a
checkpoint in the plain 4-bit format the vLLM release pinned here can load. So
the default is a community AutoRound build at 4 bits, group size 128.

**What you can swap it for.** A different quantization of Qwen3.8-27B that meets
the kernels' contract. That contract is narrow, and every line of it is checked
at load time with a `RuntimeError` rather than a fallback:

| Requirement | Refusal if not met |
|---|---|
| symmetric GPTQ, no zero points | `requires symmetric weights without zero points` |
| no act-order | `does not support GPTQ act-order` |
| 4-bit weights | `requires a 4-bit MPLinear weight type` |
| group size 128 | `requires group_size=128` |
| BF16 activations | `requires BF16 activations` |
| no bias on the quantized linears | `no verified bias-preserving dual-layout path` |
| this architecture's geometry | five exact weight shapes, and exactly 260 quantized linears — 256 backbone plus the four of one MTP layer, counted as they load |

So an INT4 build of this model is not automatically enough: one with act-order,
or asymmetric with zero points, or at group size 64, is refused at startup. A
model of another architecture is refused for the geometry. Speculation cannot be
turned off for the same reason — those four MTP linears only exist when it is
on.

If your checkpoint does meet it, point both steps at it. The installer reads the
same variable as the launcher, so this is enough:

```bash
DENSESPARK_MODEL=your-org/your-qwen3.8-27b-int4 ./install.sh
DENSESPARK_MODEL=your-org/your-qwen3.8-27b-int4 ./configs/launch-densespark.sh
```

Run the installer, not only the launcher: the draft head's structure is trained
from the checkpoint's own output layer, and at startup the server hashes the
output layer it actually loaded and compares it against the one the structure
was built from. A mismatch is not fatal — it logs it and serves without the
draft head, costing speed and nothing else — but that is a silent loss of the
thing that makes one request fast.

And every number on this page was measured on the checkpoint above. A different
quantization is a different model as far as those numbers go.

The profiles are text-only by default even though the checkpoint carries a
vision tower. Set `DENSESPARK_LIMIT_MM` to enable image or video input.

## Settings you might want

| Variable | Default | Effect |
|---|---|---|
| `DENSESPARK_CONCURRENCY` | your install answer | How many requests you expect at once. Selects the tuned settings; not a limit. |
| `DENSESPARK_SPEC_TOKENS` | from the table | Speculation depth, overriding the row. |
| `DENSESPARK_PORT` | `8000` | Server port. |
| `DENSESPARK_MODEL` | `Frozenlock/Qwen3.8-27B-int4-AutoRound` | A checkpoint meeting the contract below. Read by both `install.sh` and the launcher. |
| `DENSESPARK_LIMIT_MM` | unset | Enable image or video input. |
| `DENSESPARK_BIND_HOST` | `127.0.0.1` | Interface the API is published on. `0.0.0.0` or a LAN address exposes it — the server has no authentication. |
| `DENSESPARK_TOOL_CALL_PARSER` | `qwen3_xml` | Parser for tool calls. Matches this checkpoint's template; change it only for a checkpoint that emits a different format. |
| `DENSESPARK_MAX_NUM_SEQS` | unset | Hard-cap the running batch, so extra requests queue. Costs throughput at one request. |

## What you need, and what gets installed

`./install.sh` checks every prerequisite before it downloads anything, reports
all of the missing ones at once, and prints the exact command to fix each. You
need these on the machine:

| Prerequisite | Tested with | If it is missing |
|---|---|---|
| NVIDIA driver | 580.173.02 (CUDA 13.0) | install your distribution's driver |
| Docker Engine, usable without `sudo` | 29.2.1 | [docs.docker.com](https://docs.docker.com/engine/install/ubuntu/), then `sudo usermod -aG docker $USER` |
| NVIDIA Container Toolkit | — | `sudo apt install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker` |
| python3 | 3.12.3 | `sudo apt install -y python3` |
| python3-venv, ensurepip | 3.12.3 | `sudo apt install -y python3-venv python3-pip` |
| git | 2.43.0 | `sudo apt install -y git` |
| coreutils (`sha256sum`, `df`) | — | `sudo apt install -y coreutils` |

If any of those is missing, the installer names all of them at once with the
command that fixes each — it does not stop at the first one. It also asks the
Hugging Face hub about the checkpoint before starting a 19 GB transfer, so a
misspelt name, a gated repository or a machine with no route out is reported in
seconds with what to do about it, rather than as a download that fails later.
No Hugging Face CLI on the machine is fine: one is installed into a private
virtualenv beside the script. And if a run fails
anywhere, it writes the failure and a snapshot of the machine to a single file
and tells you where it is: attach that to an issue instead of a screenshot of
the last three lines.

Nothing else is installed on the host. Everything the server runs lives in the
container image, pinned:

| Inside the image | Version |
|---|---|
| vLLM | 0.27.1 (`vllm/vllm-openai:v0.27.1`, arm64) |
| PyTorch | 2.13.0+cu130 |
| torchvision | 0.28.0+cu130 |
| Triton | 3.7.1 |
| transformers | 5.15.0 |
| FlashInfer | 0.6.16.post3 |
| humming-kernels | 0.1.13 |
| NumPy | 2.2.6 |
| xgrammar | 0.2.3 |
| Starlette / FastAPI | 1.6.0 / 0.136.3 |
| CUDA (torch) | 13.0 |

The only thing that lands outside the image is the checkpoint,
`Frozenlock/Qwen3.8-27B-int4-AutoRound`, under `~/.cache/huggingface`, and the
draft head's structure under `~/.cache/densespark`.

## Tested environment, and what is not claimed

| Component | Tested value |
|---|---|
| Hardware | NVIDIA DGX Spark, GB10, SM121, 128 GB unified memory |
| OS | Ubuntu 24.04.4 LTS, aarch64, kernel 6.17 |

Everything above was measured on one machine, with the protocol stated next to
the numbers. What is **not** measured, and so is not claimed:

- Concurrency above 16.
- Context beyond the 9,000 tokens these tests reach; the model's own limit is
  262,144.
- Multimodal requests.
- Task accuracy. The quality evidence bounds how far the profile moves the
  model's own predictive distribution. It is not a benchmark score.
- Any other machine.

## Repository layout

```text
.
├── install.sh                one command: checkpoint, image, patches, tuning
├── bench_serving.py          the three-shape serving benchmark used above
├── bench_densespark.py       single-request decode measurement
├── configs/
│   ├── launch-densespark.sh  the profile — the one to use
│   ├── _common.sh            shared container plumbing
│   └── launch-*.sh           single-change A/B arms kept for reproduction
├── docker/Dockerfile         the pinned runtime image and every build step
└── patches/                  one directory per build step, each self-verifying
```

## Contributing and security

Performance changes should come with the prompt shape, the concurrency, the
warm-up policy, how many runs, and before/after numbers. State what you did not
measure. See [CONTRIBUTING.md](CONTRIBUTING.md).

The launch scripts publish the vLLM API on loopback only, without
authentication. `DENSESPARK_BIND_HOST` changes that; read
[SECURITY.md](SECURITY.md) before you do.

This model thinks before it answers, and its chat template opens the reasoning
block itself, so the model emits only the closing tag. The profile therefore
serves with vLLM's Qwen3 reasoning parser, without which every OpenAI-compatible
client renders the reasoning as part of the reply with a stray `</think>` in the
middle. Known limitation: the parser removes the reasoning rather than
forwarding it, so `reasoning_content` stays empty and a client cannot show it as
a collapsible block — the tokens are still generated and still cost latency.

## Acknowledgments

This is a thin layer over other people's work: the vLLM serving engine, the
Triton compiler, the AutoRound quantization method and the published INT4
checkpoint it produced, and the Qwen model itself.

## License

Project code is [MIT](LICENSE). The checkpoint, the vLLM runtime, the base
images and all transitive dependencies keep their own licenses.
