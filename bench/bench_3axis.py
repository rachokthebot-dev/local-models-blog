#!/usr/bin/env python3
"""
3-axis local LLM benchmark — long-doc summary, vision/UI describe, multi-step reasoning.

Hits the OpenAI-compatible API exposed by rapid-mlx at http://localhost:8000.
Captures tok/s + raw outputs to JSON for later LLM-as-judge scoring.

Usage:
    python3 bench_3axis.py --model mlx-community/Qwen3-14B-4bit --axes doc,reasoning
    python3 bench_3axis.py --model mlx-community/Qwen3-VL-8B-Instruct-4bit --axes vision
"""

import argparse, base64, json, pathlib, sys, time, urllib.request

API_DEFAULT = "http://localhost:8000/v1/chat/completions"
API = API_DEFAULT  # may be overridden via --api
HOME = pathlib.Path.home()
RESULTS_DIR = HOME / "claude" / "bench_results"
DATA_DIR = HOME / "claude" / "bench_data"

DOC_URL = "https://www.gutenberg.org/cache/epub/84/pg84.txt"  # Frankenstein
DOC_CACHE = DATA_DIR / "frankenstein.txt"
DOC_TARGET_CHARS = 200_000  # ≈ 50K tokens at ~4 chars/token

VISION_IMAGE = HOME / "claude" / "lickbank-songs-portrait.png"

REASONING_PROMPT = """A train leaves City A at 8:00 AM heading east at 60 mph. \
Another train leaves City B (240 miles east of A) at 9:00 AM heading west at 80 mph.

At 9:30 AM, both trains hit construction zones that reduce their speeds by 25% \
for exactly 20 minutes. After that, they resume their original speeds.

At what time and how many miles from City A will they meet?

Show every step of your reasoning: positions at 9:00 AM, 9:30 AM, and 9:50 AM, \
then compute the meeting time and meeting distance from City A. State the final \
answer on its own line as: FINAL: <time>, <distance> miles from A."""


def fetch_doc() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DOC_CACHE.exists():
        print(f"  fetching {DOC_URL} ...", file=sys.stderr)
        with urllib.request.urlopen(DOC_URL, timeout=60) as r:
            raw = r.read().decode("utf-8", errors="replace")
        start = raw.find("*** START")
        end = raw.find("*** END")
        if 0 < start < end:
            raw = raw[raw.find("\n", start) + 1 : end]
        DOC_CACHE.write_text(raw)
    return DOC_CACHE.read_text()[:DOC_TARGET_CHARS]


def call(model: str, messages: list, max_tokens: int) -> dict:
    """Streaming POST. Streaming avoids rapid-mlx's 5-min request timeout on long
    prefills and lets us separate TTFT (prefill cost) from decode rate.

    Two API shapes supported (auto-detected from URL):
      - /v1/chat/completions (OpenAI-compatible — rapid-mlx, llama.cpp, Ollama OAI)
      - /api/chat            (Ollama native — supports `think:false` to skip Gemma 4's
                              built-in thinking, matching the apples-to-apples non-thinking
                              comparison we used for Qwen3-14B yesterday)"""
    if API.endswith("/api/chat"):
        return _call_ollama_native(model, messages, max_tokens)
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    content_chunks = []
    reasoning_chunks = []
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {})
                # rapid-mlx splits Qwen3 <think>...</think> blocks into `reasoning`;
                # final answer lands in `content`. Capture both so throughput is honest
                # and the judge sees the final answer when it exists.
                # rapid-mlx → `reasoning`; llama.cpp → `reasoning_content`. Capture both.
                for field, bucket in (
                    ("reasoning", reasoning_chunks),
                    ("reasoning_content", reasoning_chunks),
                    ("content", content_chunks),
                ):
                    piece = delta.get(field) or ""
                    if piece:
                        if ttft is None:
                            ttft = time.time() - t0
                        bucket.append(piece)
    elapsed = time.time() - t0
    reasoning = "".join(reasoning_chunks)
    content = "".join(content_chunks)
    # If model never emitted content (ran out of budget mid-think), surface that.
    output = content if content else f"[NO FINAL CONTENT — only reasoning, len={len(reasoning)} chars]"
    comp = usage.get("completion_tokens", 0)
    decode_s = max(elapsed - (ttft or 0), 1e-6)
    return {
        "output": output,
        "reasoning": reasoning,  # may be empty for non-thinking models
        "elapsed_s": round(elapsed, 2),
        "ttft_s": round(ttft, 2) if ttft else None,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": comp,
        "tok_s_end_to_end": round(comp / elapsed, 2) if elapsed > 0 else 0,
        "tok_s_decode": round(comp / decode_s, 2) if comp and ttft else 0,
    }


def _call_ollama_native(model: str, messages: list, max_tokens: int) -> dict:
    """Ollama /api/chat — honors `think:false` for Gemma 4. Streams JSONL chunks."""
    # Ollama vision: native API wants images as a separate `images` list of base64
    # (not OpenAI's content-array image_url shape). Translate if needed.
    ollama_messages = []
    for m in messages:
        if isinstance(m["content"], list):
            text_parts, images = [], []
            for c in m["content"]:
                if c.get("type") == "text":
                    text_parts.append(c["text"])
                elif c.get("type") == "image_url":
                    url = c["image_url"]["url"]
                    if url.startswith("data:"):
                        images.append(url.split(",", 1)[1])
            ollama_messages.append({"role": m["role"], "content": " ".join(text_parts), "images": images})
        else:
            ollama_messages.append(m)
    body = json.dumps({
        "model": model,
        "messages": ollama_messages,
        "stream": True,
        "think": False,
        "options": {"num_predict": max_tokens, "temperature": 0.7},
    }).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    content_chunks = []
    reasoning_chunks = []
    prompt_tokens = 0
    completion_tokens = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = obj.get("message", {})
            for field, bucket in (("reasoning", reasoning_chunks), ("content", content_chunks)):
                piece = msg.get(field) or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    bucket.append(piece)
            if obj.get("done"):
                prompt_tokens = obj.get("prompt_eval_count", 0)
                completion_tokens = obj.get("eval_count", 0)
    elapsed = time.time() - t0
    reasoning = "".join(reasoning_chunks)
    content = "".join(content_chunks)
    output = content if content else f"[NO FINAL CONTENT — only reasoning, len={len(reasoning)} chars]"
    decode_s = max(elapsed - (ttft or 0), 1e-6)
    return {
        "output": output,
        "reasoning": reasoning,
        "elapsed_s": round(elapsed, 2),
        "ttft_s": round(ttft, 2) if ttft else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tok_s_end_to_end": round(completion_tokens / elapsed, 2) if elapsed > 0 else 0,
        "tok_s_decode": round(completion_tokens / decode_s, 2) if completion_tokens and ttft else 0,
    }


def axis_doc(model: str) -> dict:
    text = fetch_doc()
    prompt = (
        "Summarize the following document in fewer than 300 words. "
        "Identify the three central themes and the narrative arc.\n\n"
        f"---\n{text}\n---\n\nSummary:"
    )
    return call(model, [{"role": "user", "content": prompt}], max_tokens=1024)


def axis_vision(model: str) -> dict:
    img_bytes = VISION_IMAGE.read_bytes()
    b64 = base64.b64encode(img_bytes).decode()
    ext = VISION_IMAGE.suffix.lstrip(".").lower()
    ext = "jpeg" if ext == "jpg" else ext
    question = (
        "Describe this screenshot in detail. Identify the type of app, the layout, "
        "the main UI elements (search, toggles, buttons, tiles), and what the user "
        "appears to be doing. Be specific about visible text, content type, and structure."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
            {"type": "text", "text": question},
        ],
    }]
    return call(model, messages, max_tokens=1024)


def axis_reasoning(model: str) -> dict:
    # 8192 so Qwen3-thinking has room for ~4K of <think> + final answer.
    return call(model, [{"role": "user", "content": REASONING_PROMPT}], max_tokens=8192)


AXES = {"doc": axis_doc, "vision": axis_vision, "reasoning": axis_reasoning}


def main():
    global API
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--axes", default="doc,vision,reasoning")
    ap.add_argument("--label", default=None)
    ap.add_argument("--api", default=API_DEFAULT,
                    help="OpenAI-compatible chat-completions endpoint "
                         "(default %(default)s; Ollama is http://localhost:11434/v1/chat/completions)")
    args = ap.parse_args()
    API = args.api

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    label = args.label or args.model.rsplit("/", 1)[-1]
    results = {
        "model": args.model,
        "label": label,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "axes": {},
    }

    for axis in [a.strip() for a in args.axes.split(",") if a.strip()]:
        if axis not in AXES:
            print(f"unknown axis: {axis}", file=sys.stderr)
            continue
        print(f"=== {label} :: {axis} ===")
        try:
            r = AXES[axis](args.model)
            print(f"  prompt_tok={r['prompt_tokens']} comp_tok={r['completion_tokens']} "
                  f"ttft={r['ttft_s']}s total={r['elapsed_s']}s "
                  f"decode={r['tok_s_decode']} tok/s end2end={r['tok_s_end_to_end']} tok/s")
            results["axes"][axis] = r
        except Exception as e:
            print(f"  ERROR: {e}")
            results["axes"][axis] = {"error": str(e)}

    out_path = RESULTS_DIR / f"{label}_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
