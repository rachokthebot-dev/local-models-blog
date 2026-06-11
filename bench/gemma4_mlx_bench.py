#!/usr/bin/env python3
"""Gemma 4 MLX throughput benchmark — same coding tasks as Ollama test."""

import json, time, urllib.request

API = "http://localhost:8000/v1/chat/completions"
MODEL = "unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit"

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

print("=" * 50)
print("  Gemma 4 MLX (rapid-mlx) Throughput Benchmark")
print(f"  {time.strftime('%c')}")
print(f"  Model: {MODEL}")
print("=" * 50)
print()

total_tps = 0
count = 0

for i, (name, prompt) in enumerate(PROMPTS):
    print(f"[{i+1}/{len(PROMPTS)}] {name}")

    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.7,
        "stream": False,
    }).encode()

    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"    ERROR: {e}\n")
        continue
    elapsed = time.time() - start

    usage = data.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)

    if comp_tokens > 0:
        tps = round(comp_tokens / elapsed, 2)
        total_tps += tps
        count += 1
        print(f"    Tokens: {comp_tokens} | Time: {elapsed:.1f}s | Rate: {tps} tok/s")
    else:
        print(f"    No tokens in response")
    print()

if count > 0:
    avg = round(total_tps / count, 2)
    print("=" * 50)
    print(f"  MLX Average: {avg} tok/s across {count} prompts")
    print(f"  Ollama Q8_0: ~14.2 tok/s (baseline)")
    print(f"  Speedup:     ~{round(avg / 14.2, 1)}x")
    print("=" * 50)
