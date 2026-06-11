#!/usr/bin/env python3
"""Bench qwen3.6:35b-a3b-coding-nvfp4 (MLX) against the same 5 prompts."""

import json
import re
import subprocess
import time

MODEL = "qwen3.6:35b-a3b-coding-nvfp4"

PROMPTS = [
    ("Swift struct + Codable",
     "Write a Swift struct called PracticeSession that tracks: song name, BPM, duration in seconds, accuracy percentage, and date. Include a computed property that returns a difficulty rating (easy/medium/hard) based on BPM and accuracy. Make it Codable and Identifiable."),
    ("SwiftUI animated view",
     "Write a SwiftUI view called MetronomeView that displays a circular BPM indicator with a tap tempo button. Use @State for BPM (default 120), animate a pendulum dot around the circle, and include a slider to adjust BPM from 40 to 240. Use .animation(.easeInOut) for smooth transitions."),
    ("Core Data async manager",
     "Write a Core Data model manager in Swift for a guitar practice app. It should handle CRUD operations for PracticeSession entities (songName: String, bpm: Int16, accuracy: Double, date: Date). Include a method that returns weekly practice statistics grouped by day. Use async/await."),
    ("MVVM refactor",
     "Refactor this SwiftUI code to use MVVM pattern. The view currently has inline URLSession calls, UserDefaults access, and business logic mixed with UI code. Show the ViewModel with @Published properties, the protocol for dependency injection, and the cleaned-up View. Keep it concise."),
    ("Docker Compose backend",
     "Write a Docker Compose file for a Swift Vapor backend that serves a REST API for syncing guitar practice sessions. Include: the Vapor app, a PostgreSQL database, Redis for caching, and nginx as reverse proxy. Add health checks and volume mounts for persistence."),
]

OUT = "/tmp/local_bench_mlx_results.txt"

def call(prompt):
    body = {"model": MODEL, "prompt": prompt, "stream": False}
    r = subprocess.run(
        ["curl", "-s", "http://localhost:11434/api/generate", "-d", json.dumps(body)],
        capture_output=True, text=True, timeout=900,
    )
    return json.loads(r.stdout)

with open(OUT, "w") as f:
    header = (
        "============================================\n"
        f"  MLX bench — {MODEL}\n"
        f"  {time.strftime('%c')}\n"
        "  Ollama 0.21.2, MLX engine v0.31.2, Metal GPU\n"
        "============================================\n"
    )
    print(header, end="")
    f.write(header)

    total_eval_rate = 0.0
    n = 0

    for i, (name, prompt) in enumerate(PROMPTS, 1):
        d = call(prompt)
        eval_count = d.get("eval_count", 0)
        eval_dur_ns = d.get("eval_duration", 1)
        prompt_eval_count = d.get("prompt_eval_count", 0)
        prompt_eval_dur_ns = d.get("prompt_eval_duration", 1)
        total_dur_ns = d.get("total_duration", 0)
        eval_rate = eval_count * 1e9 / eval_dur_ns if eval_dur_ns else 0
        prompt_rate = prompt_eval_count * 1e9 / prompt_eval_dur_ns if prompt_eval_dur_ns else 0

        line = (
            f"  [{i}/5] {name:<25} "
            f"out={eval_count:5d} tok  decode={eval_rate:6.2f} tok/s  "
            f"prefill={prompt_rate:6.1f} tok/s  total={total_dur_ns/1e9:6.1f}s\n"
        )
        print(line, end="")
        f.write(line)
        total_eval_rate += eval_rate
        n += 1

    avg = total_eval_rate / n if n else 0
    summary = f"\n  ── avg decode {avg:.2f} tok/s across {n} prompts ──\n"
    print(summary, end="")
    f.write(summary)
