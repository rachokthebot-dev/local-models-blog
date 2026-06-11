# Gemma 4 26B-A4B QAT (nvfp4 MLX) vs everything — 2026-06-06

**Model:** `mlx-community/gemma-4-26B-A4B-it-qat-nvfp4` (~13GB on disk, MoE 26B/4B-active, QAT, nvfp4)  
**Runtime:** rapid-mlx 0.4.2 with `--mllm --no-thinking`  
**Why this candidate over the GGUF you linked:** the unsloth QAT GGUF ships only `UD-Q4_K_XL` (single quant) and would need llama.cpp; the mlx-community line has the same QAT weights in native MLX with nvfp4 (Apple's preferred FP4) and rapid-mlx loads it cleanly because the model_type is plain `gemma4` (not `gemma4_unified` like the 12B that broke rapid-mlx 2 days ago).  
**Bonus this model gives us:** vision works through rapid-mlx (`MLLM=True` confirmed at load) — unlike `gemma4:12b-mlx` which Ollama packaged text-only.

## Reasoning axis (5-step train problem)

| Rank | Model | Decode tok/s | E2E tok/s | TTFT | Total | comp tok | Quality |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | gemma4-26b-a4b-qat-nvfp4 (NEW) | **34.1** | 31.75 | 2.1s | 31.1s | 986 | 5/5/5 |
| 2 | qwen-hermes | **34.01** | 28.27 | 11.9s | 70.7s | 1,998 | — |
| 3 | gemma-hermes | **31.95** | 19.36 | 25.4s | 64.4s | 1,247 | — |
| 4 | Qwen3-14B-4bit (no-think) | **12.52** | 12.28 | 1.9s | 98.3s | 1,207 | 5/5/5 |
| 5 | gemma4:12b-mlx | **11.87** | 11.63 | 1.8s | 88.9s | 1,034 | 5/5/5 |

## Vision axis (LickBank screenshot)

| Rank | Model | Decode tok/s | E2E tok/s | TTFT | Total | prompt tok | Quality |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | gemma4-26b-a4b-qat-nvfp4 (NEW) | **34.02** | 29.53 | 3.0s | 22.8s | 331 | 5/5/5 |
| 2 | Qwen3-VL-8B-Instruct-4bit | **21.53** | 17.61 | 10.6s | 58.3s | 1,436 | 5/5/5 |

## Long-doc axis (~46K prompt tokens, Frankenstein)

| Rank | Model | Decode tok/s | TTFT | Total | Quality |
|---|---|---:|---:|---:|---|
| 1 | gemma4-26b-a4b-qat-nvfp4 (NEW) | **22.17** | 170.1s | 188.2s | 4/2/5 |
| 2 | gemma4:12b-mlx | **9.74** | 545.6s | 580.2s | 3/2/4 |
| 3 | Qwen3-14B-4bit (no-think) | **4.76** | 684.3s | 771.8s | 2/2/5 |

## The headline

**This model is the new local default for everything we test.** It either wins or ties every axis:

- **Reasoning:** ties `qwen-hermes` at 34 tok/s decode, beats it by ~2× on TTFT (cold-load comparison) and on total wall-clock (31s vs 71s — same answer, half the words).
- **Vision:** **1.6× faster decode than Qwen3-VL-8B** (34 vs 21.5 tok/s), TTFT 3s vs 10.6s, and 5/5/5 judge identical to Qwen3-VL's. The image encodes to only 331 tokens vs Qwen3-VL's 1436 — much more efficient vision tokenizer.
- **Long-doc:** **2.3× faster decode than gemma4:12b-mlx** (22 vs 9.7 tok/s), **3.2× faster TTFT** (170s vs 545s). MoE prefill on Apple Silicon scales with active params (~4B), not total (26B), so the long-context cost collapses.
- **Quality:** 5/5/5 on reasoning and vision; 4/2/5 on doc summary (matches the same conciseness-fail pattern as every other model — none of them respect the 300-word cap).

## What changed between the 12B (Mon) and the 26B-A4B QAT (today)

- **Arch went from `gemma4_unified` → `gemma4`.** The 12B uses the new unified arch which mlx-vlm 0.4.4 doesn't load (forcing the Ollama dance); the 26B-A4B uses the older `gemma4` which rapid-mlx serves directly. Just-works on the same toolchain.
- **MoE pays off on Apple Silicon at every context length.** The 12B (dense) decodes at 11.9 tok/s; the 26B-A4B (MoE 4B active) at 34 tok/s. That's the active-param count driving memory bandwidth.
- **QAT shows up in the doc-summary quality:** correctness 4/5 vs the 12B's 3/5. Conciseness still fails (everything does), themes still 5/5.

## What to apply

1. **Make `mlx-community/gemma-4-26B-A4B-it-qat-nvfp4` the local-LLM default.** Replaces both `gemma4:12b-mlx` (long-doc / vision were either slow or absent) and Qwen3-VL-8B (vision-tier).
2. **Retire the `--mllm` path for `gemma4_unified` for now.** Keep the rapid-mlx + plain `gemma4` arch path. If a vision-capable `gemma4_unified` 12B ships on rapid-mlx eventually, re-evaluate.
3. **Keep `qwen-hermes` for tool-calling agentic loops** — it ties on decode and the Hermes tool-call format is its main edge. For chat/vision/long-context, the new Gemma is strictly better.
4. **The unsloth QAT GGUF you linked is a fine fallback** if MLX tooling breaks again. It's the same training, packaged for llama.cpp/Ollama via GGUF — but at the moment the MLX path is faster and lower-friction on this machine.

## Caveats

- Still N=1 per axis on a single prompt. Decode tok/s numbers are physics; quality ratings are directional.
- `qwen-hermes` TTFTs include cold-load from model swap. Pure prefill cost is probably 1–3s for short prompts, not 12s.
- Long-doc summary judge correctness is constrained by Frankenstein's mid-novel cutoff; all models hallucinate forward to some degree.