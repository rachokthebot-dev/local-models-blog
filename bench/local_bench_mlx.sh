#!/bin/bash
# Same 5 prompts as local_bench.sh, just the new MLX-tagged variant.
MODEL="qwen3.6:35b-a3b-coding-nvfp4"

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

OUT=/tmp/local_bench_mlx_results.txt
{
  echo "============================================"
  echo "  MLX bench — $MODEL"
  echo "  $(date)"
  echo "  Ollama: $(ollama --version 2>&1 | grep -i version | head -1)"
  echo "============================================"
} | tee "$OUT"

total_tps=0
prompt_count=0

for i in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$i]}"
  name="${PROMPT_NAMES[$i]}"

  result=$(ollama run "$MODEL" "$prompt" --verbose 2>&1)
  eval_count=$(echo "$result" | grep "eval count:" | sed -n 's/.*eval count: *\([0-9]*\).*/\1/p' | head -1)
  eval_rate=$(echo "$result" | grep "eval rate:" | sed -n 's/.*eval rate: *\([0-9.]*\).*/\1/p' | head -1)
  total_dur=$(echo "$result" | grep "total duration:" | head -1 | sed 's/total duration://;s/^ *//')

  line=$(printf "  [%d/%d] %-25s tokens=%-6s rate=%-7s tot=%s" \
    $((i+1)) "${#PROMPTS[@]}" "$name" "${eval_count:-?}" "${eval_rate:-?}" "${total_dur:-?}")
  echo "$line" | tee -a "$OUT"

  if [ -n "$eval_rate" ]; then
    total_tps=$(python3 -c "print($total_tps + $eval_rate)")
    prompt_count=$((prompt_count + 1))
  fi
done

if [ $prompt_count -gt 0 ]; then
  avg_tps=$(python3 -c "print(round($total_tps / $prompt_count, 2))")
  echo "  ── avg ${avg_tps} tok/s across $prompt_count prompts ──" | tee -a "$OUT"
fi
echo "Done." | tee -a "$OUT"
