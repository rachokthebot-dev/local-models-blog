# Benchmark scripts

The scripts that produced every number on the [blog post](https://rachokthebot-dev.github.io/local-models-blog/). Run them on a Mac mini M4 (32 GB) for reproducibility; the harnesses are written against [rapid-mlx](https://github.com/Blaizzy/rapid-mlx)'s OpenAI-compatible `/v1` endpoint and Ollama's `/api/chat` and `/api/generate`.

## Scripts

| File | What it does | Used for |
|---|---|---|
| [`bench_3axis.py`](bench_3axis.py) | Reasoning + vision + long-doc decode/TTFT capture against `localhost:8000/v1/chat/completions`. Streams SSE, captures content + reasoning. | Reasoning, vision, long-doc tables in the post |
| [`judge_3axis.py`](judge_3axis.py) | LLM-as-judge wrapper around the local `claude -p` CLI. Scores doc/vision/reasoning outputs against a rubric. | Quality columns (5/5/5 etc.) |
| [`bench_toolcall.py`](bench_toolcall.py) | 7-case single-turn function-calling fidelity suite + decode + cold prefill on a Hermes-sized prompt. Serves through rapid-mlx on `:11600`. | Tool-call fidelity chart |
| [`bench_server_ttft.py`](bench_server_ttft.py) | Cold prefill timing — measures TTFT on a ~13K system-prompt prefix as a model swaps in. | Cold-prefill column |
| [`bench_gemma_swap.py`](bench_gemma_swap.py) | GGUF q8_0 vs NVFP4 MLX delta on a single coding prompt. Hits Ollama at `:11434`. | "10.7× short / 7.1× medium" finding |
| [`bench_qwen_hermes.py`](bench_qwen_hermes.py) | Tool-call sampling sweep for the Qwen team's published recipe. | qwen-hermes Modelfile params |
| [`local_bench_mlx.py`](local_bench_mlx.py) | 5-prompt coding bench against Ollama MLX-backend. | Initial MLX backend comparison |
| [`local_bench_mlx.sh`](local_bench_mlx.sh) | Bash orchestrator for `local_bench_mlx.py`. | — |
| [`gemma4_mlx_bench.py`](gemma4_mlx_bench.py) | Gemma 4 12 B MLX harness. | Pre-26B-QAT generation |
| [`gemma4_mlx_bench.sh`](gemma4_mlx_bench.sh) | Bash orchestrator for `gemma4_mlx_bench.py`. | — |
| [`gemma4_bench.sh`](gemma4_bench.sh) | Original GGUF-only Gemma 4 bench from April 2026. | Historical baseline |

## Raw data

[`results/`](results/) holds the JSON outputs and the markdown reports the post draws from:

- [`REPORT.md`](results/REPORT.md) — original 3-axis run (Qwen3-14B / Qwen3-VL-8B, 2026-06-03)
- [`REPORT_v2_gemma4_vs_qwen3.md`](results/REPORT_v2_gemma4_vs_qwen3.md) — Gemma 4 12B vs Qwen3 head-to-head (2026-06-04)
- [`REPORT_v3_throughput_hermes.md`](results/REPORT_v3_throughput_hermes.md) — qwen-hermes / gemma-hermes throughput (2026-06-04)
- [`REPORT_v4_gemma4_26b_a4b_qat.md`](results/REPORT_v4_gemma4_26b_a4b_qat.md) — new Gemma 4 26B-A4B QAT NVFP4 across all axes (2026-06-06)
- `*.json` — raw run dumps; each has `prompt`, `comp_tok`, `ttft_s`, `total_s`, `decode_tps`, and the full streamed content for re-judging.

## Running them

### Reasoning + vision + long-doc

```bash
# Start rapid-mlx with the model under test
rapid-mlx serve mlx-community/gemma-4-26B-A4B-it-qat-nvfp4 --mllm --no-thinking --port 8000

# In another shell — run the axes you want
python3 bench_3axis.py --model gemma-4-26B-A4B-it-qat-nvfp4 \
    --axes reasoning,vision,doc \
    --out-dir bench_results/

# Score the runs
python3 judge_3axis.py --results bench_results/*.json
```

### Tool-call fidelity

```bash
# Per model — rapid-mlx with the right tool-call parser
rapid-mlx serve mlx-community/Qwen3.6-35B-A3B-4bit --tool-parser qwen --port 11600

python3 bench_toolcall.py Qwen3.6-35B-A3B-4bit 11600
```

### GGUF vs MLX swap

```bash
# Ollama daemon must be running with both q8_0 GGUF and NVFP4 MLX variants pulled
python3 bench_gemma_swap.py
```

## Caveats

- N=1 per axis per prompt. Decode tok/s numbers are reliable (physics). TTFT cold-load contaminated. See the [Methodology section](https://rachokthebot-dev.github.io/local-models-blog/#methods) of the post.
- Versions: rapid-mlx 0.4.2, mlx-vlm 0.4.4, Ollama 0.30.4 MLX backend, MLX engine v0.31.2 — as of June 2026.
- Hardware: Mac mini M4, 32 GB unified memory.
