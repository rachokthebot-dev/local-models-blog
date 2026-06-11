#!/usr/bin/env python3
"""Cross-server TTFT + decode bench: Ollama vs Rapid-MLX.

Measures client-side time-to-first-token with a stable ~13k-token system
prompt (simulates Hermes' system prompt) and a varying user turn. Calls the
same model 3x: call #1 is cold (prefix not cached), #2 and #3 reuse the
identical system prefix -> tests prefix-cache / pin-system-prompt TTFT win.
"""
import json
import sys
import time
import urllib.request

# ~13k-token stable system prefix (the thing a prefix cache should reuse)
SYSTEM = (
    "You are an autonomous AI agent with extensive tool-calling capabilities. "
    "You can read files, run shell commands, search the web, and edit code. "
    "Always think step by step and prefer structured tool calls over prose. "
) * 260

USERS = [
    "What is 2+2? One word.",
    "Name three Python web frameworks. One line.",
    "What's the capital of France? One word.",
]


def ollama_stream(model, system, user, port=11434):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": True,
        "think": False,
        "keep_alive": "1h",
        "options": {"num_ctx": 65536, "num_predict": 128, "temperature": 0.7,
                    "top_p": 0.8, "top_k": 20, "min_p": 0, "repeat_penalty": 1.05},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    out_tok = 0
    in_tok = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            msg = d.get("message", {}).get("content", "")
            if msg and ttft is None:
                ttft = time.time() - t0
            if msg:
                out_tok += 1
            if d.get("done"):
                in_tok = d.get("prompt_eval_count", 0)
                out_tok = d.get("eval_count", out_tok)
    wall = time.time() - t0
    decode = out_tok / (wall - ttft) if ttft and wall > ttft else 0
    return ttft, decode, in_tok, out_tok, wall


def openai_stream(model, system, user, port):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": True,
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.8,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    out_tok = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            raw = raw.strip()
            if not raw or not raw.startswith(b"data:"):
                continue
            chunk = raw[5:].strip()
            if chunk == b"[DONE]":
                break
            d = json.loads(chunk)
            delta = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta and ttft is None:
                ttft = time.time() - t0
            if delta:
                out_tok += 1
    wall = time.time() - t0
    decode = out_tok / (wall - ttft) if ttft and wall > ttft else 0
    return ttft, decode, 0, out_tok, wall


def run(label, fn, model, port):
    print(f"\n=== {label} (model={model}, port={port}) ===")
    # warm with a tiny prompt (load weights, no big prefix)
    sys.stderr.write("  warming...\n")
    fn(model, "You are a helpful assistant.", "hi", port)
    approx_tok = len(SYSTEM) // 4
    print(f"  system prefix: {len(SYSTEM)} chars (~{approx_tok} tok)")
    print(f"  {'call':14} {'in_tok':>7} {'out':>5} {'TTFT_s':>8} {'decode/s':>9} {'wall_s':>8}")
    for i, u in enumerate(USERS):
        tag = "cold(call1)" if i == 0 else f"cached(call{i+1})"
        ttft, dec, intok, out, wall = fn(model, SYSTEM, u, port)
        ttft_s = f"{ttft:.2f}" if ttft else "n/a"
        print(f"  {tag:14} {intok:>7} {out:>5} {ttft_s:>8} {dec:>9.1f} {wall:>8.2f}", flush=True)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "ollama"
    if target == "ollama":
        run("OLLAMA /api/chat", ollama_stream, "qwen-hermes", 11434)
    elif target == "rapid":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 11600
        run("RAPID-MLX /v1", openai_stream, "qwen3.6", port)
