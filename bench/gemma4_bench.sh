#!/bin/bash
# Gemma 4 throughput benchmark — coding tasks similar to Shreddy (Swift/iPad app)

MODELS=("gemma4:26b-a4b-it-q8_0" "gemma4:31b-it-q4_K_M")

PROMPTS=(
  "Write a Swift struct called PracticeSession that tracks: song name, BPM, duration in seconds, accuracy percentage, and date. Include a computed property that returns a difficulty rating (easy/medium/hard) based on BPM and accuracy. Make it Codable and Identifiable."

  "Write a SwiftUI view called MetronomeView that displays a circular BPM indicator with a tap tempo button. Use @State for BPM (default 120), animate a pendulum dot around the circle, and include a slider to adjust BPM from 40 to 240. Use .animation(.easeInOut) for smooth transitions."

  "Write a Core Data model manager in Swift for a guitar practice app. It should handle CRUD operations for PracticeSession entities (songName: String, bpm: Int16, accuracy: Double, date: Date). Include a method that returns weekly practice statistics grouped by day. Use async/await."

  "Refactor this SwiftUI code to use MVVM pattern. The view currently has inline URLSession calls, UserDefaults access, and business logic mixed with UI code. Show the ViewModel with @Published properties, the protocol for dependency injection, and the cleaned-up View. Keep it concise."

  "Write a Docker Compose file for a Swift Vapor backend that serves a REST API for syncing guitar practice sessions. Include: the Vapor app, a PostgreSQL database, Redis for caching, and nginx as reverse proxy. Add health checks and volume mounts for persistence."
)

PROMPT_NAMES=(
  "Swift struct + Codable"
  "SwiftUI animated view"
  "Core Data async manager"
  "MVVM refactor"
  "Docker Compose backend"
)

echo "============================================"
echo "  Gemma 4 Coding Throughput Benchmark"
echo "  $(date)"
echo "  Hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
echo "  RAM: $(sysctl -n hw.memsize | awk '{print $1/1073741824 " GB"}')"
echo "============================================"
echo ""

for model in "${MODELS[@]}"; do
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Model: $model"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  total_tps=0
  prompt_count=0

  for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    name="${PROMPT_NAMES[$i]}"

    echo ""
    echo "  [$((i+1))/${#PROMPTS[@]}] $name"

    # Run ollama and capture the full verbose output
    start_time=$(python3 -c "import time; print(time.time())")

    result=$(ollama run "$model" "$prompt" --verbose 2>&1)

    # Extract metrics from verbose output
    eval_count=$(echo "$result" | grep "eval count:" | awk '{print $3}')
    eval_duration=$(echo "$result" | grep "eval duration:" | grep -oP '[\d.]+s' | tr -d 's' 2>/dev/null)
    eval_rate=$(echo "$result" | grep "eval rate:" | awk '{print $3}')
    total_duration=$(echo "$result" | grep "total duration:" | head -1)

    if [ -n "$eval_rate" ]; then
      printf "       Tokens: %s | Rate: %s tok/s | %s\n" "$eval_count" "$eval_rate" "$total_duration"
      total_tps=$(python3 -c "print($total_tps + $eval_rate)")
      prompt_count=$((prompt_count + 1))
    else
      # Fallback: parse differently
      echo "       (parsing metrics...)"
      echo "$result" | tail -8 | grep -E "eval|total|prompt"
    fi
  done

  if [ $prompt_count -gt 0 ]; then
    avg_tps=$(python3 -c "print(round($total_tps / $prompt_count, 2))")
    echo ""
    echo "  ── Average: $avg_tps tok/s across $prompt_count prompts ──"
  fi
  echo ""
done

echo "============================================"
echo "  Benchmark complete"
echo "============================================"
