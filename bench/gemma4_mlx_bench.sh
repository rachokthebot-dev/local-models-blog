#!/bin/bash
# Gemma 4 MLX throughput benchmark — same coding tasks as Ollama test
# Uses the OpenAI-compatible API from rapid-mlx

API="http://localhost:8000/v1/chat/completions"
MODEL="mlx-community/gemma-4-26b-a4b-it-4bit"

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
echo "  Gemma 4 MLX (rapid-mlx) Throughput Bench"
echo "  $(date)"
echo "  Model: $MODEL"
echo "============================================"
echo ""

total_tps=0
prompt_count=0

for i in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$i]}"
  name="${PROMPT_NAMES[$i]}"

  echo "[$((i+1))/${#PROMPTS[@]}] $name"

  # Call the API with streaming off, measure time
  start=$(python3 -c "import time; print(time.time())")

  response=$(curl -s "$API" \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "
import json
print(json.dumps({
    'model': '$MODEL',
    'messages': [{'role': 'user', 'content': $(python3 -c "import json; print(json.dumps('''$prompt'''))")}],
    'max_tokens': 4096,
    'temperature': 0.7,
    'stream': False
}))
")")

  end=$(python3 -c "import time; print(time.time())")

  # Parse response
  completion_tokens=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens','?'))" 2>/dev/null)
  prompt_tokens=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('prompt_tokens','?'))" 2>/dev/null)
  elapsed=$(python3 -c "print(round($end - $start, 2))")

  if [ "$completion_tokens" != "?" ] && [ -n "$completion_tokens" ]; then
    tps=$(python3 -c "print(round($completion_tokens / $elapsed, 2))")
    total_tps=$(python3 -c "print($total_tps + $tps)")
    prompt_count=$((prompt_count + 1))
    printf "    Tokens: %s | Time: %ss | Rate: %s tok/s\n" "$completion_tokens" "$elapsed" "$tps"
  else
    echo "    ERROR: $response" | head -c 200
    echo ""
  fi
  echo ""
done

if [ $prompt_count -gt 0 ]; then
  avg_tps=$(python3 -c "print(round($total_tps / $prompt_count, 2))")
  echo "============================================"
  echo "  Average: $avg_tps tok/s across $prompt_count prompts"
  echo "============================================"
fi

echo ""
echo "Compare: Ollama Q8_0 averaged ~14.2 tok/s"
