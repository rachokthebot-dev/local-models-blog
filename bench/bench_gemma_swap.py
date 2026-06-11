#!/usr/bin/env python3
"""Compare gemma4 q8_0 GGUF vs nvfp4 MLX — same prompts, think=off, gemma sampling defaults."""

import json
import subprocess
import sys

MODELS = [
    ("q8_0 GGUF", "gemma4:26b-a4b-it-q8_0"),
    ("nvfp4 MLX", "gemma4:26b-nvfp4"),
]

PROMPTS = [
    ("short", "Explain memoization in one paragraph."),
    ("medium", "Write a Python LRU cache class with get/put methods. Include docstrings."),
]

OPTS = {"num_ctx": 8192, "num_predict": 2048, "temperature": 1.0, "top_p": 0.95, "top_k": 64}


def call(model: str, prompt: str) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": OPTS,
    }
    r = subprocess.run(
        ["curl", "-s", "http://localhost:11434/api/chat", "-d", json.dumps(body)],
        capture_output=True, text=True, timeout=600,
    )
    return json.loads(r.stdout)


def fmt_row(label, ptag, d):
    out = d.get("eval_count", 0)
    eval_ns = d.get("eval_duration", 1)
    prompt_eval = d.get("prompt_eval_count", 0)
    prompt_eval_ns = d.get("prompt_eval_duration", 1)
    total_ns = d.get("total_duration", 0)
    decode = out / (eval_ns / 1e9) if eval_ns else 0
    prefill = prompt_eval / (prompt_eval_ns / 1e9) if prompt_eval_ns else 0
    wall = total_ns / 1e9
    return f"{label:12} {ptag:8} {out:>8} {decode:>10.1f} {prefill:>11.1f} {wall:>8.1f}"


def main():
    print(f"{'model':12} {'prompt':8} {'out tok':>8} {'decode/s':>10} {'prefill/s':>11} {'wall s':>8}")
    print("-" * 64)
    for label, model in MODELS:
        # warm
        sys.stderr.write(f"warming {model}…\n")
        call(model, "hi")
        for ptag, prompt in PROMPTS:
            d = call(model, prompt)
            print(fmt_row(label, ptag, d), flush=True)


if __name__ == "__main__":
    main()
