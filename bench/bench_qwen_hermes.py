#!/usr/bin/env python3
"""Measure throughput on qwen-hermes via Ollama /api/chat.
Three regimes: short, medium, long (simulates hermes' 13k-token system prompt)."""

import json
import subprocess
import sys

MODEL = "qwen-hermes"

PROMPTS = [
    ("short", "What is 2+2? Answer in one word."),
    ("medium", "Write a Python LRU cache class with get/put. Include type hints and a brief docstring."),
    # ~13k tokens — simulates hermes' system prompt overhead
    ("long-13k", "You are an AI agent with extensive tool-calling capabilities. " * 800 +
     "\n\nFinal user task: List five Python testing frameworks and one sentence each."),
]

OPTS = {
    "num_ctx": 65536,
    "num_predict": 512,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
    "repeat_penalty": 1.05,
}


def call(prompt: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "keep_alive": "1h",
        "options": OPTS,
    }
    r = subprocess.run(
        ["curl", "-s", "http://127.0.0.1:11434/api/chat", "-d", json.dumps(body)],
        capture_output=True, text=True, timeout=600,
    )
    return json.loads(r.stdout)


def fmt_row(label, d):
    out_tok = d.get("eval_count", 0)
    eval_ns = d.get("eval_duration", 1)
    in_tok = d.get("prompt_eval_count", 0)
    prefill_ns = d.get("prompt_eval_duration", 1)
    total_ns = d.get("total_duration", 0)
    decode = out_tok / (eval_ns / 1e9) if eval_ns else 0
    prefill = in_tok / (prefill_ns / 1e9) if prefill_ns else 0
    wall = total_ns / 1e9
    return f"{label:10} {in_tok:>7} {out_tok:>7} {prefill:>10.1f} {decode:>10.1f} {wall:>8.2f}"


def main():
    sys.stderr.write(f"warming {MODEL}…\n")
    call("hi")
    print(f"{'regime':10} {'in_tok':>7} {'out_tok':>7} {'prefill/s':>10} {'decode/s':>10} {'wall_s':>8}")
    print("-" * 65)
    for label, prompt in PROMPTS:
        d = call(prompt)
        print(fmt_row(label, d), flush=True)


if __name__ == "__main__":
    main()
