# DenseSpark 1.3 evaluation — 2026-09-03

## Candidate

- Upstream: `albond/DenseSpark-Qwen3.8-27B`
- Commit: `9ae122f757bbae28c875d00c8b186c4837187434`
- Runtime: vLLM 0.27.1 on the upstream `--fast` image
- Checkpoint: `Frozenlock/Qwen3.8-27B-int4-AutoRound`
- Checkpoint revision: `b4c61732c4f2d8af323d75ba5702b5c7f3361539`
- Context: 131,072 tokens
- GPU memory utilization: 0.90
- Profile selector: concurrency 16, therefore native MTP depth 3
- Sampling: temperature 0, top-p 1, fixed generation seed

Installation, all nine image transforms, the INT8 output head and the PQ-head
artifact verification completed successfully. The loaded target and drafter
occupied 45.83 GiB. vLLM allocated a 47.96 GiB KV cache holding 710,106 tokens:
5.42 completely full 131,072-token contexts. Prefix caching was disabled, as
required by this upstream profile's deterministic-output gate.

## Results retained

The 1,000-input/1,000-output single-stream test produced:

| Run | Output throughput | TTFT | TPOT | MTP acceptance | Acceptance length |
|---|---:|---:|---:|---:|---:|
| Cold first shape | 17.28 tok/s | 33.37 s | 24.54 ms | 67.07% | 3.01 |
| Warm 1 | 38.83 tok/s | 619 ms | 25.16 ms | 65.38% | 2.96 |
| Warm 2 | 38.69 tok/s | 619 ms | 25.25 ms | 64.71% | 2.94 |

The cold run included 57.88 seconds of one-time shape autotuning and is not a
steady-state speed result. The median warm output throughput was 38.76 tok/s.
This is intentionally not the single-stream-tuned DenseSpark setting: the
concurrency-16 profile uses MTP depth 3. The upstream policy selects depth 8 at
concurrency 1.

## Failed gate

The first concurrency-8 balanced run did not complete after more than ten
minutes and produced no result. The client was terminated. Until a warmed
concurrency sweep and the agent correctness suite both pass locally, DenseSpark
remains an opt-in candidate rather than replacing the established SGLang
DSpark deployment.

This failure also exposed a benchmark-harness integration bug: the upstream
script used one URL both from the host and from inside the container, so a
server published as host port 18083 to container port 8000 could not pass both
readiness and benchmark requests. The vendored harness now has separate
`--base-url` and `--container-base-url` options.

## Decision

DenseSpark is the strongest current route to substantially higher 27B decode
throughput, but the local evidence does not yet justify making it the default.
Use `runtime/run-densespark.sh interactive` for an opt-in latency experiment or
`runtime/run-densespark.sh agents` for a 16-request/64K experiment. Keep
`runtime/run-sglang.sh` for the validated 131K, prefix-cached service.
