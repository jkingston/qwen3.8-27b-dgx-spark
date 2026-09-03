#!/usr/bin/env python3
"""Measure DenseSpark decode speed against a running server.

Streams every request so time-to-first-token and per-token decode time are
reported separately. A dense model is decode bound, and a single tok/s figure
that folds prefill into the average hides exactly the number this project is
trying to move.

Prints a markdown table that can be pasted into the README as measured.

Usage:
    ./bench_densespark.py --label baseline
    ./bench_densespark.py --label mtp-2 --runs 5
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request

PROMPTS = [
    ("Q&A 256", "What are the main differences between TCP and UDP? Be concise.", 256),
    ("Code 512", "Write a Python function that implements binary search on a sorted "
                 "list. Include type hints and a docstring.", 512),
    ("JSON 1024", "Generate a JSON array of 10 fictional employees with fields: name, "
                  "age, department, salary, email, skills (array of 3). Output ONLY "
                  "valid JSON, no explanation.", 1024),
    ("Math 64", "What is 7823 * 4519? Show only the answer.", 64),
    ("LongCode 2048", "Write a complete Python implementation of a red-black tree with "
                      "insert, delete, search, and in-order traversal. Include all "
                      "rotation methods.", 2048),
]


def stream_once(base_url, model, prompt, max_tokens, timeout, think=False):
    """Return timing, token count, and output identity for one completion."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})

    started = time.perf_counter()
    first_token_at = None
    last_token_at = started
    counted = 0
    usage_tokens = None
    pieces = []

    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage_tokens = chunk["usage"].get("completion_tokens")
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            # A reasoning parser moves the thinking trace to its own field. Time
            # it like any other decoded token where it is streamed at all.
            reasoning = delta.get("reasoning_content")
            if not content and not reasoning:
                continue
            if content:
                pieces.append(content)
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            last_token_at = now
            counted += 1

    if first_token_at is None:
        raise RuntimeError(
            "the server produced no output (reasoning consumed the whole "
            "max_tokens budget? drop --think, or raise max_tokens for this prompt)")
    generated = usage_tokens or counted
    output = "".join(pieces).encode("utf-8")
    return (
        first_token_at - started,
        last_token_at - first_token_at,
        generated,
        hashlib.sha256(output).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="unlabeled", help="name for this run in the output")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="densespark-qwen3.8-27b")
    parser.add_argument("--runs", type=int, default=3, help="repeats per prompt")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--warmup", action="store_true",
                        help="discard one silent run first, to pay CUDA graph and "
                             "autotune costs before measuring")
    parser.add_argument("--think", action="store_true",
                        help="leave the checkpoint's default reasoning mode on. Off "
                             "by default: this is a decode-speed measurement, and on "
                             "a Qwen3.5-hybrid checkpoint (thinking on by default) "
                             "the reasoning trace alone can burn the whole max_tokens "
                             "budget on the shorter prompts (Math 64, the --warmup "
                             "probe) before any content token is emitted, which the "
                             "server reports as zero content rather than a timeout.")
    parser.add_argument(
        "--json-out",
        type=Path,
        help="write unrounded per-run timing, token counts, and output hashes",
    )
    args = parser.parse_args()

    print(f"DenseSpark benchmark — {args.label}")
    print(f"server {args.base_url}, model {args.model}, {args.runs} runs per prompt")
    if args.think:
        # A reasoning parser that withholds the thinking trace leaves its tokens
        # in the server's completion count while its time falls outside the
        # decode window, which reports a rate the run never reached.
        print("note: --think counts every generated token, but a server whose "
              "reasoning\n      parser withholds the trace times only the visible "
              "tail. Rates are not\n      comparable with the default.")
    print()

    if args.warmup:
        print("warmup ...", flush=True)
        try:
            stream_once(args.base_url, args.model, "Say hello.", 16, args.timeout,
                        think=args.think)
        except Exception as error:
            raise SystemExit(f"warmup failed: {error}")

    rows = []
    all_decode_rates = []
    records = []
    total_decode_tokens = 0
    total_decode_seconds = 0.0
    for name, prompt, max_tokens in PROMPTS:
        rates, ttfts = [], []
        for run_index in range(args.runs):
            try:
                ttft, decode_s, generated, output_sha256 = stream_once(
                    args.base_url, args.model, prompt, max_tokens, args.timeout,
                    think=args.think)
            except urllib.error.URLError as error:
                raise SystemExit(f"cannot reach {args.base_url}: {error}")
            except Exception as error:
                raise SystemExit(f"[{name}] failed: {error}")
            if generated < 2:
                raise SystemExit(f"[{name}] only {generated} tokens; nothing to measure")
            # Decode rate excludes the first token, which is prefill bound.
            decode_tokens = generated - 1
            rate = decode_tokens / decode_s if decode_s > 0 else float("nan")
            rates.append(rate)
            ttfts.append(ttft)
            total_decode_tokens += decode_tokens
            total_decode_seconds += decode_s
            records.append({
                "prompt": name,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "max_tokens": max_tokens,
                "run": run_index + 1,
                "ttft_s": ttft,
                "decode_s": decode_s,
                "generated_tokens": generated,
                "decode_tokens": decode_tokens,
                "decode_tokens_per_s": rate,
                "output_sha256": output_sha256,
            })
        spread = statistics.stdev(rates) if len(rates) > 1 else 0.0
        rows.append((name, statistics.mean(rates), statistics.median(rates), spread,
                     statistics.mean(ttfts)))
        all_decode_rates.append(statistics.mean(rates))
        print(f"  {name:<15} {rows[-1][1]:6.1f} tok/s decode   ttft {rows[-1][4]*1000:6.0f} ms")

    print(f"\n### {args.label}\n")
    print("| Prompt | Decode tok/s | Median | Std | TTFT ms |")
    print("|---|---:|---:|---:|---:|")
    for name, mean, median, spread, ttft in rows:
        print(f"| {name} | {mean:.1f} | {median:.1f} | {spread:.2f} | {ttft*1000:.0f} |")
    overall = statistics.mean(all_decode_rates)
    weighted = total_decode_tokens / total_decode_seconds
    print(f"\nCross-prompt mean decode rate: **{overall:.1f} tok/s** "
          f"({args.runs} runs per prompt)")
    print(f"Token-weighted decode rate: **{weighted:.1f} tok/s** "
          f"({total_decode_tokens} decode tokens / {total_decode_seconds:.6f} s)")

    if args.json_out:
        payload = {
            "schema": 1,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "label": args.label,
            "server": args.base_url,
            "model": args.model,
            "runs_per_prompt": args.runs,
            "warmup": args.warmup,
            "think": args.think,
            "cross_prompt_mean_decode_tokens_per_s": overall,
            "token_weighted_decode_tokens_per_s": weighted,
            "total_decode_tokens": total_decode_tokens,
            "total_decode_s": total_decode_seconds,
            "records": records,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Raw record: {args.json_out}")


if __name__ == "__main__":
    main()
