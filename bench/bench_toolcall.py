#!/usr/bin/env python3
"""Tool-calling fidelity + perf bench for Hermes-candidate models.

Serves each model via Rapid-MLX (OpenAI /v1) with the right tool-call parser,
then scores single-turn function-calling (BFCL-style) plus TTFT/decode on a
Hermes-sized prompt. Run one model at a time (32GB = one model resident).

Usage: python3 bench_toolcall.py <served_model_name> [port]
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 11600
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.6"

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read the contents of a file at a path",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Execute a shell command and return stdout",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "send_email",
        "description": "Send an email",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "subject": {"type": "string"},
            "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}},
]

# (prompt, expected_fn or None, {arg: substring_expected})
CASES = [
    ("What's the weather in Tokyo in celsius?", "get_weather",
     {"location": "tokyo", "unit": "celsius"}),
    ("Read the file at /etc/hosts for me.", "read_file", {"path": "/etc/hosts"}),
    ("List the files in the current directory.", "run_shell", {"command": "ls"}),
    ("What is 47 times 89?", "calculator", {"expression": "47"}),
    ("Search the web for the latest Rust release.", "web_search", {"query": "rust"}),
    ("Email alice@example.com with subject Standup and body Running late.",
     "send_email", {"to": "alice@example.com", "subject": "standup"}),
    ("Just say hello to me in one word. Do not call any tools.", None, {}),
]


def chat(messages, tools=None, max_tokens=256, stream=False):
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.7, "top_p": 0.8, "stream": stream}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    if not stream:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())
    # streaming -> return (ttft, decode_tps, out_tok, wall)
    t0 = time.time(); ttft = None; out = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            raw = raw.strip()
            if not raw.startswith(b"data:"):
                continue
            c = raw[5:].strip()
            if c == b"[DONE]":
                break
            d = json.loads(c)
            delta = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta and ttft is None:
                ttft = time.time() - t0
            if delta:
                out += 1
    wall = time.time() - t0
    dec = out / (wall - ttft) if ttft and wall > ttft else 0
    return ttft, dec, out, wall


def score_toolcalls():
    passed = 0; name_ok = 0; total = len(CASES); details = []
    for prompt, exp_fn, exp_args in CASES:
        resp = chat([{"role": "user", "content": prompt}], tools=TOOLS, max_tokens=256)
        msg = resp.get("choices", [{}])[0].get("message", {})
        tcs = msg.get("tool_calls") or []
        if exp_fn is None:
            ok = len(tcs) == 0
            details.append(("no-tool", ok, "(none)" if ok else f"called {len(tcs)}"))
            passed += ok
            continue
        if not tcs:
            details.append((exp_fn, False, "NO tool_call emitted"))
            continue
        fn = tcs[0].get("function", {})
        got_name = fn.get("name", "")
        name_match = got_name == exp_fn
        name_ok += name_match
        try:
            args = json.loads(fn.get("arguments", "{}"))
            args_str = json.dumps(args).lower()
            args_match = all(str(v).lower() in args_str for v in exp_args.values())
        except Exception:
            args_match = False
        ok = name_match and args_match
        passed += ok
        details.append((exp_fn, ok,
                        f"got={got_name} args_ok={args_match}"))
    return passed, name_ok, total, details


def decode_rate():
    """Non-streaming: short prompt (negligible prefill) + 200 out tokens.
    decode ~= completion_tokens / wall. Bandwidth-bound, model-dependent."""
    t0 = time.time()
    resp = chat([{"role": "user", "content":
                  "Count from 1 to 100 with a word after each number."}],
                max_tokens=200, stream=False)
    wall = time.time() - t0
    out = resp.get("usage", {}).get("completion_tokens", 0)
    return (out / wall if wall else 0), out, wall


def perf():
    # ~13k-token Hermes-ish system prefix; measure TTFT (cold vs cached prefix)
    sysmsg = ("You are an autonomous AI agent with tool-calling capabilities. "
              "Think step by step and prefer structured tool calls. ") * 450
    msgs = [{"role": "system", "content": sysmsg},
            {"role": "user", "content": "Write a haiku about local LLMs."}]
    cold = chat(msgs, max_tokens=40, stream=True)
    return cold, None, len(sysmsg) // 4


if __name__ == "__main__":
    print(f"### MODEL: {MODEL} (port {PORT})")
    sys.stderr.write("scoring tool calls...\n")
    p, n, t, det = score_toolcalls()
    print(f"\nTool-calling fidelity: {p}/{t} passed  (correct fn name: {n}/{t-1})")
    for fn, ok, info in det:
        print(f"  [{'PASS' if ok else 'FAIL'}] {fn:12} {info}")
    sys.stderr.write("measuring decode rate...\n")
    dr, dout, dwall = decode_rate()
    print(f"\nDecode rate: {dr:.1f} tok/s ({dout} tok in {dwall:.2f}s)")
    sys.stderr.write("measuring TTFT...\n")
    cold, _, approx = perf()
    ttft_c = cold[0] if cold[0] else 0
    print(f"Cold prefill TTFT on ~{approx}-tok prefix: {ttft_c:.2f}s")
