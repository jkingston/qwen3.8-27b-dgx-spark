#!/usr/bin/env python3
"""Run the fixed-length, 16-request vLLM serving throughput protocol.

This complements ``bench_densespark.py``.  The existing benchmark measures
single-request decode latency on meaningful prompts; this one reproduces the
finite open-loop burst and three synthetic shapes used in the referenced DGX
Spark comparison.  A capped refill mode is also available, but its ramp and
drain remain in the reported wall time and it is not a steady-state window.

The official vLLM 0.27.1 benchmark client is executed inside the already running
DenseSpark container.  That keeps tokenizer and client semantics pinned to the
server version and requires no host-side vLLM installation.

Examples:
    ./bench_serving.py --scenario balanced
    ./bench_serving.py --scenario all --runs 3 --result-dir benchmark-output
"""

from __future__ import annotations

import argparse
import datetime
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request


PROTOCOL_SOURCE = (
    "https://forums.developer.nvidia.com/t/"
    "qwen3-8-27b-on-dgx-spark-using-vllm-nvfp4-vs-fp8-performance/380258"
)
DEFAULT_CONTAINER = "densespark"
DEFAULT_TOKENIZER_MODEL = "Frozenlock/Qwen3.8-27B-int4-AutoRound"
DEFAULT_SERVED_MODEL = "densespark-qwen3.8-27b"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_CONTAINER_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_CONCURRENCY = 16
DEFAULT_REQUEST_RATE = 10000
SUMMARY_METRICS = (
    "duration",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "max_output_tokens_per_s",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
)
SPECULATIVE_METRICS = (
    "spec_decode_acceptance_length",
    "spec_decode_acceptance_rate",
    "spec_decode_accepted_tokens",
    "spec_decode_draft_tokens",
    "spec_decode_num_drafts",
    "spec_decode_per_position_acceptance_rates",
)
SAFE_CONTAINER_ENV_NAMES = {
    "CUDA_VERSION",
    "HF_HUB_OFFLINE",
    "TORCH_CUDA_ARCH_LIST",
}
SAFE_CONTAINER_ENV_PREFIXES = ("DENSESPARK_", "VLLM_")


@dataclass(frozen=True)
class Scenario:
    """One fixed input/output shape in the serving protocol."""

    name: str
    input_tokens: int
    output_tokens: int


SCENARIOS = (
    Scenario("prompt-heavy", 8000, 1000),
    Scenario("decode-heavy", 1000, 8000),
    Scenario("balanced", 1000, 1000),
)
SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


def build_bench_command(
    *,
    container: str,
    tokenizer_model: str,
    served_model: str,
    base_url: str,
    scenario: Scenario,
    concurrency: int,
    request_rate: int,
    result_dir: str,
    result_filename: str,
    label: str,
    sustained: int = 0,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    generation_seed: int | None = None,
) -> list[str]:
    """Build the pinned vLLM client command for one measured run.

    ``sustained`` selects the second protocol. The forum protocol sends exactly
    ``concurrency`` requests at once and measures wall time, so its duration is
    the completion time of the slowest request. That is the right measurement
    for a closed batch and the wrong one for a server under load: any change
    that raises per-request variance - speculation above all - is reported by
    its unluckiest draw rather than by its rate. With ``sustained = n`` the run
    issues ``n * concurrency`` prompts and caps in-flight requests at
    ``concurrency``, so the batch refills as requests complete.  This reduces
    tail sensitivity, but the reported duration still includes its initial ramp
    and final drain.  A true steady-state service-rate experiment needs a much
    longer run and an explicitly trimmed interior window.  The two modes answer
    different questions, and neither substitutes for that third protocol.
    """

    extra: list[str] = []
    if sustained:
        prompts = concurrency * sustained
        extra = ["--max-concurrency", str(concurrency)]
    else:
        prompts = concurrency

    sampling: list[str] = []
    if temperature is not None:
        sampling.extend(("--temperature", str(temperature)))
    if top_p is not None:
        sampling.extend(("--top-p", str(top_p)))
    if top_k is not None:
        sampling.extend(("--top-k", str(top_k)))
    if generation_seed is not None:
        # ``--seed`` below controls random dataset construction.  The OpenAI
        # request seed is a separate field and must travel in the request body.
        sampling.extend(("--extra-body", json.dumps({"seed": generation_seed})))

    return [
        "docker",
        "exec",
        container,
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--endpoint",
        "/v1/completions",
        "--base-url",
        base_url,
        "--model",
        tokenizer_model,
        "--served-model-name",
        served_model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(scenario.input_tokens),
        "--random-output-len",
        str(scenario.output_tokens),
        "--random-range-ratio",
        "0",
        "--request-rate",
        str(request_rate),
        "--num-prompts",
        str(prompts),
        *extra,
        "--ignore-eos",
        "--seed",
        "0",
        *sampling,
        "--label",
        label,
        "--save-result",
        "--save-detailed",
        "--result-dir",
        result_dir,
        "--result-filename",
        result_filename,
    ]


def selected_scenarios(name: str) -> tuple[Scenario, ...]:
    if name == "all":
        return SCENARIOS
    return (SCENARIO_BY_NAME[name],)


def require_running_container(container: str) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("docker is not installed or is not on PATH")
    completed = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", container],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or completed.stdout.strip() != "true":
        detail = completed.stderr.strip() or f"container {container!r} is not running"
        raise RuntimeError(detail)


def require_served_model(base_url: str, served_model: str) -> None:
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot query {base_url}/v1/models: {error}") from error
    model_ids = {item.get("id") for item in payload.get("data", [])}
    if served_model not in model_ids:
        raise RuntimeError(
            f"server exposes {sorted(model_ids)!r}, not {served_model!r}"
        )


def filter_safe_container_environment(values: list[str]) -> dict[str, str]:
    """Retain experiment controls without ever serializing credentials."""

    safe: dict[str, str] = {}
    forbidden_segments = {"TOKEN", "PASSWORD", "SECRET", "CREDENTIAL", "KEY"}
    for item in values:
        name, separator, value = item.partition("=")
        segments = set(name.upper().split("_"))
        if not separator or segments.intersection(forbidden_segments):
            continue
        if name in SAFE_CONTAINER_ENV_NAMES or name.startswith(
            SAFE_CONTAINER_ENV_PREFIXES
        ):
            safe[name] = value
    return dict(sorted(safe.items()))


def redact_server_command(values: list[str] | None) -> list[str] | None:
    """Redact known credential-bearing CLI options from captured argv."""

    if values is None:
        return None
    secret_flags = {"--api-key", "--hf-token", "--token", "--password"}
    redacted: list[str] = []
    hide_next = False
    for value in values:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        flag = value.split("=", 1)[0]
        if flag in secret_flags:
            if "=" in value:
                redacted.append(f"{flag}=<redacted>")
            else:
                redacted.append(value)
                hide_next = True
        else:
            redacted.append(value)
    return redacted


def capture_checkpoint_generation_config(
    container: str, model_reference: str
) -> dict[str, object]:
    """Capture the resolved Hub revision and generation defaults, if local."""

    script = r"""
import hashlib
import json
from pathlib import Path
import sys

reference = sys.argv[1]
base = Path('/root/.cache/huggingface/hub') / ('models--' + reference.replace('/', '--'))
ref = base / 'refs' / 'main'
if not ref.is_file():
    print(json.dumps({'available': False, 'reason': 'refs/main not found'}))
    raise SystemExit(0)
revision = ref.read_text(encoding='utf-8').strip()
path = base / 'snapshots' / revision / 'generation_config.json'
if not path.is_file():
    print(json.dumps({'available': False, 'revision': revision,
                      'reason': 'generation_config.json not found'}))
    raise SystemExit(0)
blob = path.read_bytes()
print(json.dumps({
    'available': True,
    'revision': revision,
    'generation_config_sha256': hashlib.sha256(blob).hexdigest(),
    'generation_config': json.loads(blob),
}, sort_keys=True))
"""
    completed = subprocess.run(
        ["docker", "exec", container, "python3", "-c", script, model_reference],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "available": False,
            "reason": "container probe failed",
            "returncode": completed.returncode,
        }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"available": False, "reason": "container probe returned invalid JSON"}
    return payload if isinstance(payload, dict) else {
        "available": False,
        "reason": "container probe returned a non-object",
    }


def capture_server_provenance(
    container: str, model_reference: str | None = None
) -> dict[str, object]:
    """Capture the exact running image, argv, and safe experiment controls."""

    completed = subprocess.run(
        ["docker", "inspect", container],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError(f"docker inspect returned {len(payload)!r} records")
    record = payload[0]
    config = record.get("Config") or {}
    provenance = {
        "container_name": container,
        "image_reference": config.get("Image"),
        "image_id": record.get("Image"),
        "entrypoint": config.get("Entrypoint"),
        "command": redact_server_command(config.get("Cmd")),
        "environment": filter_safe_container_environment(config.get("Env") or []),
    }
    if model_reference is not None:
        provenance["checkpoint"] = capture_checkpoint_generation_config(
            container, model_reference
        )
    return provenance


def capture_harness_provenance() -> dict[str, object]:
    """Record the loaded harness bytes and Git context when available."""

    source = Path(__file__).resolve()
    provenance: dict[str, object] = {
        "path": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    root = source.parent
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode == 0:
        provenance["git_head"] = head.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
        provenance["git_dirty"] = status.returncode != 0 or bool(
            status.stdout.strip()
        )
    return provenance


@dataclass(frozen=True)
class EngineCounters:
    """One scrape of the engine gauges a capacity gate needs to see.

    ``waiting`` and ``preemptions`` are optional rather than required. Section
    21.12 asks for "no preemption, and stable server waiting depth", which the
    earlier three-gauge scrape could not answer at all; a server that does not
    export them should still be measurable for everything else, and a gate that
    needs them says so by finding ``None``.
    """

    running: int
    prompt_tokens: float
    generation_tokens: float
    at: float
    waiting: int | None = None
    preemptions: float | None = None


def read_engine_counters(base_url: str) -> EngineCounters:
    """Read the active batch, the token counters, and the capacity gauges."""

    metrics_url = f"{base_url}/metrics"
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as response:
            text = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError, UnicodeDecodeError) as error:
        raise RuntimeError(f"cannot query {metrics_url}: {error}") from error

    values: dict[str, float] = {}
    required = (
        "num_requests_running",
        "prompt_tokens_total",
        "generation_tokens_total",
    )
    optional = (
        "num_requests_waiting",
        "num_preemptions_total",
    )
    for line in text.splitlines():
        if not line.startswith("vllm:") or line.startswith("#"):
            continue
        for name in required + optional:
            if line.startswith(f"vllm:{name}{{") or line.startswith(
                f"vllm:{name} "
            ):
                try:
                    values[name] = float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError) as error:
                    raise RuntimeError(f"malformed Prometheus sample: {line!r}") from error
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"{metrics_url} is missing {', '.join(missing)}")
    running = values["num_requests_running"]
    if not running.is_integer() or running < 0:
        raise RuntimeError(f"invalid running-request gauge {running!r}")
    waiting = values.get("num_requests_waiting")
    if waiting is not None and (not waiting.is_integer() or waiting < 0):
        raise RuntimeError(f"invalid waiting-request gauge {waiting!r}")
    return EngineCounters(
        running=int(running),
        prompt_tokens=values["prompt_tokens_total"],
        generation_tokens=values["generation_tokens_total"],
        at=time.monotonic(),
        waiting=None if waiting is None else int(waiting),
        preemptions=values.get("num_preemptions_total"),
    )


def engine_decode_phase_ready(
    *,
    running: int,
    concurrency: int,
    prompt_tokens: float,
    baseline_prompt_tokens: float,
    expected_prompt_tokens: int,
    generation_tokens: float,
    baseline_generation_tokens: float,
) -> bool:
    """Return whether a counter sample proves full-C post-prefill decode."""

    return (
        running == concurrency
        and prompt_tokens - baseline_prompt_tokens >= expected_prompt_tokens
        and generation_tokens > baseline_generation_tokens
    )


def engine_window_is_retained(
    *, running_min_sampled: int, running_max_sampled: int, concurrency: int,
    generation_tokens_delta: float, elapsed: float
) -> bool:
    """Apply the sampled-full-C retention contract."""

    return (
        running_min_sampled == concurrency
        and running_max_sampled == concurrency
        and generation_tokens_delta >= 0
        and elapsed > 0
    )


def refill_window_is_retained(
    *, running_max_sampled: int, concurrency: int,
    generation_tokens_delta: float, elapsed: float,
    prompt_tokens_end: float, baseline_prompt_tokens: float,
    total_expected_prompt_tokens: int,
) -> bool:
    """Retain an interior capped-refill service window, including handovers."""

    return (
        running_max_sampled == concurrency
        and generation_tokens_delta >= 0
        and elapsed > 0
        # Once every prompt has entered the engine, the final drain has begun.
        and prompt_tokens_end - baseline_prompt_tokens < total_expected_prompt_tokens
    )


def capture_engine_windows(
    *,
    process: subprocess.Popen,
    base_url: str,
    concurrency: int,
    window_seconds: float,
    max_windows: int,
    wait_seconds: float,
    baseline_prompt_tokens: float,
    baseline_generation_tokens: float,
    expected_prompt_tokens: int,
    sample_seconds: float = 0.25,
    refill_expected_prompt_tokens: int | None = None,
) -> list[dict[str, object]]:
    """Measure emitted tokens in phase-conditioned interior windows.

    This observes an already-running official client.  A window is retained
    only after the prompt counter has advanced by ``expected_prompt_tokens``
    and generation has begun. Closed-burst windows require the running-request
    gauge to equal C at every sample. For capped refill the caller sets the
    phase gate to ``C + 1`` prompts, excluding the initial fill; handover dips
    are intentionally retained as part of service cost, and a window is kept
    only while some prompts have not yet entered, excluding the final drain.
    C=1 is deliberately unsupported: the client's one-request readiness probe
    is indistinguishable from the measured request in the gauge alone.
    """

    if concurrency < 2:
        return []
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline and process.poll() is None:
        sample = read_engine_counters(base_url)
        if engine_decode_phase_ready(
            running=sample.running,
            concurrency=concurrency,
            prompt_tokens=sample.prompt_tokens,
            baseline_prompt_tokens=baseline_prompt_tokens,
            expected_prompt_tokens=expected_prompt_tokens,
            generation_tokens=sample.generation_tokens,
            baseline_generation_tokens=baseline_generation_tokens,
        ):
            break
        time.sleep(0.25)
    else:
        return []

    # A discarded window used to end the series. Under capped refill that made
    # a six-block gate unreachable whenever one handover dipped the batch below
    # C, however steady the engine actually was: the run would stop at block
    # three and report three. Keep sampling instead, retain every attempt with
    # its verdict, and stop on the client, the drain, or an attempt cap.
    windows: list[dict[str, object]] = []
    retained_count = 0
    attempt = 0
    max_attempts = max_windows * 3
    while retained_count < max_windows and attempt < max_attempts:
        attempt += 1
        index = attempt
        start = read_engine_counters(base_url)
        while start.running != concurrency and process.poll() is None:
            if (
                refill_expected_prompt_tokens is None
                or start.prompt_tokens - baseline_prompt_tokens
                >= refill_expected_prompt_tokens
            ):
                return windows
            time.sleep(min(sample_seconds, 0.1))
            start = read_engine_counters(base_url)
        if process.poll() is not None:
            break
        running0, prompt0, tokens0, started = (
            start.running, start.prompt_tokens, start.generation_tokens, start.at
        )
        running_samples = [running0]
        waiting_samples = [] if start.waiting is None else [start.waiting]
        preemptions_start = start.preemptions
        preemptions_end = start.preemptions
        ended = started
        prompt1 = prompt0
        tokens1 = tokens0
        while ended - started < window_seconds and process.poll() is None:
            remaining = window_seconds - (ended - started)
            time.sleep(min(sample_seconds, remaining))
            sample = read_engine_counters(base_url)
            running1, prompt1, tokens1, ended = (
                sample.running, sample.prompt_tokens, sample.generation_tokens, sample.at
            )
            running_samples.append(running1)
            if sample.waiting is not None:
                waiting_samples.append(sample.waiting)
            if sample.preemptions is not None:
                preemptions_end = sample.preemptions
        elapsed = ended - started
        delta = tokens1 - tokens0
        if refill_expected_prompt_tokens is None:
            valid = engine_window_is_retained(
                running_min_sampled=min(running_samples),
                running_max_sampled=max(running_samples),
                concurrency=concurrency,
                generation_tokens_delta=delta,
                elapsed=elapsed,
            )
            window_kind = "sampled-full-c-decode"
        else:
            valid = refill_window_is_retained(
                running_max_sampled=max(running_samples),
                concurrency=concurrency,
                generation_tokens_delta=delta,
                elapsed=elapsed,
                prompt_tokens_end=prompt1,
                baseline_prompt_tokens=baseline_prompt_tokens,
                total_expected_prompt_tokens=refill_expected_prompt_tokens,
            )
            window_kind = "capped-refill-interior-service"
        row = {
            "window": index,
            "running_start": running0,
            "running_end": running_samples[-1],
            "running_min_sampled": min(running_samples),
            "running_max_sampled": max(running_samples),
            "running_samples": len(running_samples),
            "running_sample_period_seconds": sample_seconds,
            "prompt_tokens_start": prompt0,
            "prompt_tokens_end": prompt1,
            "seconds": elapsed,
            "generation_tokens_delta": delta,
            "output_tokens_per_second": delta / elapsed if valid else None,
            "retained": valid,
            "window_kind": window_kind,
            # Section 21.12 asks for no preemption and a stable server waiting
            # depth. Both are recorded per window rather than judged here; a
            # missing gauge stays null rather than being read as a pass.
            "waiting_min_sampled": min(waiting_samples) if waiting_samples else None,
            "waiting_max_sampled": max(waiting_samples) if waiting_samples else None,
            "waiting_mean_sampled": (
                sum(waiting_samples) / len(waiting_samples) if waiting_samples else None
            ),
            "waiting_samples": len(waiting_samples),
            "preemptions_start": preemptions_start,
            "preemptions_end": preemptions_end,
            "preemptions_delta": (
                None
                if preemptions_start is None or preemptions_end is None
                else preemptions_end - preemptions_start
            ),
        }
        windows.append(row)
        if valid:
            retained_count += 1
        print(
            "engine window "
            f"{index}: C={running0}->{running_samples[-1]}, {delta:.0f} tokens / "
            f"{elapsed:.3f} s"
            + (f" = {delta / elapsed:.2f} tok/s" if valid else " (discarded)"),
            flush=True,
        )
    return windows


def validate_result(
    payload: object, *, scenario: Scenario, concurrency: int, sustained: int = 0
) -> dict[str, object]:
    """Reject incomplete runs instead of reporting misleading throughput."""

    if not isinstance(payload, dict):
        raise ValueError("benchmark result must be a JSON object")

    required = (
        "completed",
        "failed",
        "total_input_tokens",
        "total_output_tokens",
        "max_concurrent_requests",
        "input_lens",
        "output_lens",
        "errors",
        *SUMMARY_METRICS,
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"benchmark result is missing fields: {', '.join(missing)}")

    prompts = concurrency * sustained if sustained else concurrency
    expected = {
        "completed": prompts,
        "failed": 0,
        "total_input_tokens": prompts * scenario.input_tokens,
        "total_output_tokens": prompts * scenario.output_tokens,
    }
    if not sustained:
        expected["max_concurrent_requests"] = concurrency
    for key, expected_value in expected.items():
        if payload[key] != expected_value:
            raise ValueError(
                f"benchmark result has {key}={payload[key]!r}; "
                f"expected {expected_value!r}"
            )

    if sustained:
        # The real cap is the client's semaphore. `max_concurrent_requests` is
        # not an instantaneous count: vllm/benchmarks/serve.py:699-706 buckets
        # each request into every whole second its [start, start+latency]
        # interval touches and takes the maximum, so a request finishing at
        # 10.1 s and its replacement starting at 10.2 s both land in bucket 10.
        # Under a sustained protocol that overshoots by the number of handovers
        # inside one second, which is exactly what the protocol is designed to
        # produce. The meaningful check is therefore that the batch reached the
        # cap at all - a run that never filled it measured something else.
        observed = payload["max_concurrent_requests"]
        if not isinstance(observed, int) or observed < concurrency:
            raise ValueError(
                f"benchmark result has max_concurrent_requests={observed!r}; "
                f"a sustained run must reach {concurrency}"
            )

    expected_vectors = {
        "input_lens": scenario.input_tokens,
        "output_lens": scenario.output_tokens,
    }
    for key, expected_value in expected_vectors.items():
        values = payload[key]
        if not isinstance(values, list) or values != [expected_value] * prompts:
            raise ValueError(
                f"benchmark result {key} must contain {prompts} copies of "
                f"{expected_value}"
            )

    errors = payload["errors"]
    if not isinstance(errors, list) or len(errors) != prompts or any(errors):
        raise ValueError("benchmark result contains missing or non-empty request errors")

    for key in SUMMARY_METRICS:
        value = payload[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"benchmark result has invalid {key}={value!r}")

    return payload


def load_validated_result(
    path: Path, *, scenario: Scenario, concurrency: int, sustained: int = 0
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_result(payload, scenario=scenario, concurrency=concurrency,
                           sustained=sustained)


def result_metrics(payload: dict[str, object]) -> dict[str, object]:
    wanted = (
        "completed",
        "failed",
        "total_input_tokens",
        "total_output_tokens",
        "max_concurrent_requests",
        *SUMMARY_METRICS,
        *SPECULATIVE_METRICS,
    )
    metrics = {key: payload[key] for key in wanted if key in payload}
    itls = payload.get("itls")
    total_output = payload.get("total_output_tokens")
    if (
        isinstance(itls, list)
        and isinstance(total_output, int)
        and total_output > 0
        and all(isinstance(row, list) for row in itls)
    ):
        # vLLM records one timestamp per streamed response chunk. Speculative
        # decoding can return several accepted tokens in one chunk, so the
        # client's one-second histogram counts events, not tokens. Aggregate
        # throughput and TPOT use the true output lengths and remain valid.
        stream_events = sum(1 + len(row) for row in itls)
        coverage = stream_events / total_output
        metrics["stream_events"] = stream_events
        metrics["stream_event_coverage"] = coverage
        if (
            not math.isclose(coverage, 1.0, rel_tol=0.0, abs_tol=1.0e-9)
            and "max_output_tokens_per_s" in metrics
        ):
            reported = metrics.pop("max_output_tokens_per_s")
            metrics["max_stream_events_per_s"] = reported
            metrics["client_peak_metric_valid_for_tokens"] = False
        else:
            metrics["client_peak_metric_valid_for_tokens"] = True
    return metrics


def write_summary(
    path: Path,
    *,
    container: str,
    tokenizer_model: str,
    served_model: str,
    concurrency: int,
    request_rate: int,
    records: list[dict[str, object]],
    sustained: int = 0,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    generation_seed: int | None = None,
    engine_window_seconds: float = 0.0,
    engine_windows: int = 0,
    server_provenance: dict[str, object] | None = None,
    harness_provenance: dict[str, object] | None = None,
) -> None:
    payload = {
        "schema": 3,
        "protocol": "capped-refill" if sustained else "forum-open-loop-burst",
        "prompts_per_run": concurrency * sustained if sustained else concurrency,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol_source": PROTOCOL_SOURCE,
        "client": "vllm bench serve",
        "container": container,
        "tokenizer_model": tokenizer_model,
        "served_model": served_model,
        "concurrency": concurrency,
        "request_rate": request_rate,
        "ignore_eos": True,
        "random_range_ratio": 0,
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "generation_seed": generation_seed,
            "client_sampling_overrides_omitted": (
                temperature is None and top_p is None and top_k is None
                and generation_seed is None
            ),
        },
            "engine_window": {
            "seconds": engine_window_seconds,
            "maximum_windows": engine_windows,
            "metric": "delta(vllm:generation_tokens_total)/monotonic_seconds",
            "retention": (
                "after the first refill and before all prompts have entered "
                "(the final drain), the sampled running gauge reaches C; brief "
                "handover dips remain part of the measured service rate"
                if sustained
                else "the full input batch is prefetched, generation has begun, "
                "and every sampled running-request gauge value is C"
            ),
        },
        "client_metric_caveat": (
            "When stream_event_coverage < 1, speculative decoding bundled "
            "multiple accepted tokens into response chunks. The official "
            "client's max one-second value and ITL then describe stream events, "
            "not individual tokens; aggregate output_throughput and TPOT remain "
            "token-correct."
        ),
        "provenance": {
            "server": server_provenance,
            "harness": harness_provenance,
        },
        "records": records,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario",
        choices=("all", *SCENARIO_BY_NAME),
        default="all",
    )
    parser.add_argument("--runs", type=positive_int, default=1)
    parser.add_argument(
        "--sustained", type=int, default=0, metavar="N",
        help="issue N*concurrency prompts and cap in-flight requests at "
             "concurrency, so the batch refills as requests complete. The "
             "default closed protocol sends exactly `concurrency` requests at "
             "once and therefore reports the slowest one; that hides any gain "
             "which raises per-request variance. Ramp and drain remain in the "
             "result, so this is capped refill rather than a trimmed steady-state "
             "window. N=4 is a short diagnostic default.")
    parser.add_argument("--concurrency", type=positive_int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--request-rate", type=positive_int, default=DEFAULT_REQUEST_RATE)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="explicit request temperature; omit for the forum-compatible server default",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="explicit request top-p; omit for the forum-compatible server default",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="explicit request top-k; omit for the forum-compatible server default",
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=None,
        help="OpenAI request seed (distinct from the fixed random-dataset seed)",
    )
    parser.add_argument(
        "--engine-window-seconds",
        type=float,
        default=0.0,
        help="also sample the server token counter over full-batch interior windows",
    )
    parser.add_argument(
        "--engine-windows",
        type=positive_int,
        default=3,
        help="maximum full-batch interior windows per run (default: 3)",
    )
    parser.add_argument("--container", default=os.environ.get("DENSESPARK_CONTAINER", DEFAULT_CONTAINER))
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL)
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="host-visible server URL used for readiness and metrics",
    )
    parser.add_argument(
        "--container-base-url",
        default=DEFAULT_CONTAINER_BASE_URL,
        help="server URL visible inside the benchmark container (default: %(default)s)",
    )
    parser.add_argument("--result-dir", type=Path, default=Path("benchmark-output"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without requiring a running server",
    )
    args = parser.parse_args()
    if args.engine_window_seconds < 0:
        parser.error("--engine-window-seconds must be non-negative")
    result_dir = args.result_dir.resolve()
    remote_dir = f"/tmp/densespark-serving-{os.getpid()}"
    work: list[tuple[Scenario, int, Path, str, list[str]]] = []
    for scenario in selected_scenarios(args.scenario):
        for run_index in range(1, args.runs + 1):
            suffix = f"-s{args.sustained}" if args.sustained else ""
            stem = (f"densespark-{scenario.name}-c{args.concurrency}"
                    f"{suffix}-run{run_index}")
            local_path = result_dir / f"{stem}.json"
            remote_filename = f"{stem}.json"
            command = build_bench_command(
                container=args.container,
                tokenizer_model=args.tokenizer_model,
                served_model=args.served_model_name,
                base_url=args.container_base_url,
                scenario=scenario,
                concurrency=args.concurrency,
                request_rate=args.request_rate,
                result_dir=remote_dir,
                result_filename=remote_filename,
                label=stem,
                sustained=args.sustained,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                generation_seed=args.generation_seed,
            )
            work.append((scenario, run_index, local_path, remote_filename, command))

    for _, _, _, _, command in work:
        print(shlex.join(command))
    if args.dry_run:
        return 0

    try:
        require_running_container(args.container)
        require_served_model(args.base_url, args.served_model_name)
    except RuntimeError as error:
        parser.error(str(error))
    server_provenance = capture_server_provenance(
        args.container, args.tokenizer_model
    )
    harness_provenance = capture_harness_provenance()

    existing = [local_path for _, _, local_path, _, _ in work if local_path.exists()]
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        existing.append(summary_path)
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        parser.error(f"refusing to overwrite {joined}; use --overwrite or another directory")

    result_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["docker", "exec", args.container, "mkdir", "-p", remote_dir], check=True
    )

    records: list[dict[str, object]] = []
    for scenario, run_index, local_path, remote_filename, command in work:
        print(
            f"\n=== {scenario.name}: {scenario.input_tokens} input / "
            f"{scenario.output_tokens} output, C={args.concurrency}, "
            f"run {run_index}/{args.runs} ===",
            flush=True,
        )
        engine_baseline = None
        if args.engine_window_seconds:
            engine_baseline_sample = read_engine_counters(args.base_url)
            engine_baseline = (
                engine_baseline_sample.running,
                engine_baseline_sample.prompt_tokens,
                engine_baseline_sample.generation_tokens,
                engine_baseline_sample.at,
            )
        process = subprocess.Popen(command)
        engine_windows = []
        if args.engine_window_seconds:
            engine_windows = capture_engine_windows(
                process=process,
                base_url=args.base_url,
                concurrency=args.concurrency,
                window_seconds=args.engine_window_seconds,
                max_windows=args.engine_windows,
                wait_seconds=120.0,
                baseline_prompt_tokens=engine_baseline[1],
                baseline_generation_tokens=engine_baseline[2],
                expected_prompt_tokens=scenario.input_tokens * (
                    args.concurrency + 1 if args.sustained else args.concurrency
                ),
                refill_expected_prompt_tokens=(
                    scenario.input_tokens * args.concurrency * args.sustained
                    if args.sustained else None
                ),
            )
        returncode = process.wait()
        if returncode != 0:
            raise SystemExit(
                f"vLLM benchmark failed for {scenario.name} run {run_index} "
                f"with exit status {returncode}"
            )
        subprocess.run(
            [
                "docker",
                "cp",
                f"{args.container}:{remote_dir}/{remote_filename}",
                str(local_path),
            ],
            check=True,
        )
        try:
            payload = load_validated_result(
                local_path,
                scenario=scenario,
                concurrency=args.concurrency,
                sustained=args.sustained,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise SystemExit(
                f"invalid vLLM result for {scenario.name} run {run_index}: {error}"
            ) from error
        record: dict[str, object] = {
            "scenario": scenario.name,
            "input_tokens_per_request": scenario.input_tokens,
            "output_tokens_per_request": scenario.output_tokens,
            "run": run_index,
            "result_file": local_path.name,
            "engine_full_batch_windows": engine_windows,
            "client_command": command,
            "dataset_seed": 0,
        }
        record.update(result_metrics(payload))
        records.append(record)

    write_summary(
        summary_path,
        container=args.container,
        tokenizer_model=args.tokenizer_model,
        served_model=args.served_model_name,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        records=records,
        sustained=args.sustained,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        generation_seed=args.generation_seed,
        engine_window_seconds=args.engine_window_seconds,
        engine_windows=args.engine_windows,
        server_provenance=server_provenance,
        harness_provenance=harness_provenance,
    )
    print(f"\nRaw results: {result_dir}")
    print(f"Protocol summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
