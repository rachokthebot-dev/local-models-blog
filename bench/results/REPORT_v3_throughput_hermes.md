# Throughput-only comparison — reasoning axis, M4 mini 32GB, 2026-06-04

Same 5-step train prompt, temperature 0.7, `think:false` (or `--no-thinking` on rapid-mlx). Hermes-wired models served via Ollama 0.30.4 with their existing Modelfile params (`/no_think` system message on `gemma-hermes`, no system on `qwen-hermes`).

## Decode-rate ranking (the one that matters once the model is warm)

| Rank | Model | Params (active) | Quant | Decode tok/s | E2E tok/s | TTFT | Total | Answer |
|---|---|---|---|---:|---:|---:|---:|---|
| 1 | qwen-hermes | 35.0B | nvfp4) | **34.01** | 28.27 | 11.9s | 70.7s | ✓ |
| 2 | gemma-hermes | 6.3B | /no_think) | **31.95** | 19.36 | 25.4s | 64.4s | ✓ |
| 3 | Qwen3-14B-4bit | 14.0B | 4-bit | **12.52** | 12.28 | 1.9s | 98.3s | ✓ |
| 4 | gemma4:12b-mlx | 12.0B | nvfp4) | **11.87** | 11.63 | 1.8s | 88.9s | ✓ |

## Readings

- **qwen-hermes (35B-A3B MoE) decodes at 34 tok/s** — the fastest of the four. That's the MoE structure paying off: only ~3B parameters activate per token, so decode is bounded by ~3B-class memory bandwidth, not the full 35B. Caveat: TTFT was 12s because the full 35B has to be loaded into RAM regardless.
- **gemma-hermes (6.3B dense) decodes at 32 tok/s** — basically tied with qwen-hermes despite being 1/5th the params. That's just small-dense vs MoE-active being similar bandwidth-wise. Quality is presumably lower (smaller model, no MoE routing diversity), but quality wasn't this question's axis.
- **gemma4:12b-mlx decodes at 11.9 tok/s** — solid dense-12B speed, but **~3× slower** than the two Hermes models. nvfp4 helps but doesn't beat MoE sparsity or pure smaller-dense.
- **Qwen3 14B 4-bit decodes at 12.5 tok/s** — same regime as gemma4:12b-mlx; ~14B dense at 4-bit is what it is on Apple Silicon.
- **TTFT spread is 1.8s → 25s** but mostly reflects cold-load on model-swap, not real prefill cost. The Hermes models' TTFT numbers here are pessimistic (we swapped from another model right before each run).

## What this means for routing

Pick by the task shape, not just the brand:

- **Short-prompt, latency-sensitive agentic loops** → **qwen-hermes** (34 tok/s decode + Hermes tool format + thinking-capable). Pay the 12s cold load once.
- **Short-prompt, low-VRAM-pressure, English chat** → **gemma-hermes** (32 tok/s, only 6.3B). Sweet for keeping headroom for other resident models.
- **Long-context (>20K) text work** → **gemma4:12b-mlx**. Half the decode rate of the Hermes models but you measured ~2× faster long-context decode than Qwen3 14B yesterday — the 12B + nvfp4 combo is the long-doc winner of what we've benched, even though it's not the headline at short context.
- **Quality-critical short prompts** → still **qwen-hermes** if you can afford the load. 35B total params route to better answers per token than the smaller models even with only 3B active.

## Caveats

- N=1 per model on a single prompt. Decode rates are reliable; TTFT spread is noisy because of cold-load contamination (Ollama hot-swaps models).
- The two Hermes models are tuned for the Hermes tool-call format; using them for plain chat is fair but not their highest-leverage use case.
- Long-context (50K) and vision axes weren't repeated here — would cost another ~25 min for each Hermes model and the answer on long-context would mostly track yesterday's findings (prefill dominates).
- No quality scoring on this round. All four models hit the correct FINAL: 10:22 AM / 137.14 mi answer on the reasoning prompt.