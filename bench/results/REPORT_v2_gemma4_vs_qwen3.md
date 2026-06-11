# Gemma 4 12B (MLX via Ollama) vs Qwen3 — 2026-06-04

**Hardware:** M4 mini 32GB  
**Sampling:** temperature 0.7. Gemma 4 served with `think:false` for apples-to-apples with yesterday's Qwen3 `--no-thinking` run.  
**Doc:** Project Gutenberg *Frankenstein* truncated to ~200K chars (≈ 46K tokens).  
**Judge:** Claude (via local `claude -p`), same rubric as yesterday's report.

## Results

| Model | Axis | Runtime | Quant | prompt tok | comp tok | TTFT | total | decode tok/s | end-to-end tok/s | Quality |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Gemma 4 12B | reasoning | Ollama 0.30.4 (MLX backend) | nvfp4 | 182 | 1,034 | 1.8s | 88.9s | 11.87 | 11.63 | 5/5/5 |
| Qwen3 14B | reasoning | rapid-mlx 0.4.2 (--no-thinking) | 4-bit | 178 | 1,207 | 1.9s | 98.3s | 12.52 | 12.28 | 5/5/5 |
| Gemma 4 12B | doc 50K | Ollama 0.30.4 (MLX backend) | nvfp4 | 45,887 | 336 | 545.6s | 580.2s | 9.74 | 0.58 | 3/2/4 |
| Qwen3 14B | doc 50K | rapid-mlx 0.4.2 (--no-thinking) | 4-bit | 46,725 | 416 | 684.3s | 771.8s | 4.76 | 0.54 | 2/2/5 |
| Qwen3-VL 8B | vision | rapid-mlx 0.4.2 (--mllm) | 4-bit | 1,436 | 1,026 | 10.6s | 58.3s | 21.53 | 17.61 | 5/5/5 |
| Gemma 4 12B | vision | — (no vision in Ollama MLX pkg yet) | — | — | — | — | — | — | — | — |

## Head-to-head readings

- **Long-context decode (~46K prompt):** Gemma 4 12B nvfp4 runs **~2× faster** than Qwen3 14B 4-bit on decode (9.74 vs 4.76 tok/s). nvfp4 is doing real work on Apple Silicon at long context — likely the headline finding.
- **Long-context prefill:** Gemma 4 12B TTFT **545s** (~9 min) vs Qwen3 14B's 684s (~11 min). ~20% faster, smaller param count helps. Still in 'route to cloud' territory for any interactive use.
- **Short-context decode (~180 prompt):** Gemma 4 12B 11.87 tok/s vs Qwen3 14B 12.52 tok/s. Effectively a tie; the Qwen 14B's larger size + Gemma's nvfp4 cancel out at this regime.
- **Reasoning quality:** both 5/5/5. Gemma 4 hit the `FINAL:` line cleanly in 1034 tokens; Qwen3 reached the answer in 1207 tokens but also clean. Functionally equivalent on this prompt.
- **Doc summary quality:** Gemma 4 3/2/4 vs Qwen3 2/2/5. Gemma got more facts right (3 vs 2), Qwen3 covered themes better (5 vs 4). Both ran long and both hallucinated forward past the 50K cut. Neither is reliable for unsupervised long-doc summarization without prompt scaffolding.
- **Vision: not comparable yet.** Ollama's `gemma4:12b-mlx` ships text-only despite the underlying model being unified text+vision+audio. Qwen3-VL-8B's 5/5/5 on the LickBank screenshot stands as the local vision-tier default until Ollama (or rapid-mlx) ships the Gemma 4 vision projector. Tooling-wise: mlx-vlm 0.4.4 doesn't handle `gemma4_unified` weights, mlx-vlm 0.6.1 needs newer mlx-metal than brew supplies, and llama.cpp 9430 supports gemma4 text but not the multimodal projector.

## What to apply

1. **Gemma 4 12B nvfp4 is the new local default for long-context text.** ~2× decode advantage over Qwen3 14B 4-bit at 46K context, ~20% faster prefill. If you do any local long-doc work, swap in `gemma4:12b-mlx`.
2. **For short-context reasoning/chat, Gemma 4 12B and Qwen3 14B are interchangeable** on throughput; quality is a wash on this single test. Pick by ecosystem fit (Ollama vs rapid-mlx) and by tool-calling parser quality.
3. **Wait on Gemma 4 vision.** No clean Mac path today; expect Ollama or mlx-vlm to ship the vision projector for `gemma4_unified` within weeks given how fresh the release is. For now, Qwen3-VL-8B remains the vision tier.
4. **Always disable thinking by default for benches and bounded-latency agents.** Both Gemma 4 and Qwen3 14B run away in thinking mode. Ollama: `think: false` on `/api/chat`. rapid-mlx: `--no-thinking`.

## Tooling notes (what I learned trying to get here)

- **mlx-community/gemma-4-12B-it-mxfp4** was my first pick (smallest, mxfp4 has rep for quality > flat 4-bit). It failed to load: mlx-vlm 0.4.4 (rapid-mlx's pin) doesn't have a handler for the `gemma4_unified` model_type, and patching `MODEL_REMAPPING` got past the type check but hit a weight-layout mismatch (`vision_embedder.patch_dense.*` unknown to the old gemma4 module).
- **mlx-vlm 0.6.1** does support gemma4_unified, but it requires a newer mlx-metal than brew's `mlx-metal 0.31.1`. Brew installs mlx-metal without a RECORD file, so pip refuses to upgrade and `--no-deps` install breaks on `mlx.core.new_thread_local_stream`.
- **llama.cpp HEAD (b9430)** does support `gemma4` arch (cleanly loads the unsloth GGUF), but the Ollama desktop daemon was pinned to v0.24, which uses an older bundled llama.cpp that errors `unknown model architecture: 'gemma4'`. Upgrading via `brew upgrade ollama` to 0.30.4 and restarting the daemon unblocked everything.
- **Ollama 0.30.4 ships `gemma4:12b-mlx` as a first-class MLX-backed model.** This is the cleanest Mac path today — sidesteps the mlx-vlm/mlx-metal version conflict entirely.
- **Field naming gotcha:** rapid-mlx streams Qwen3 thinking as `delta.reasoning`; Ollama streams Gemma 4 thinking as `delta.reasoning` (same); llama.cpp 9430 streams as `delta.reasoning_content` (different). The bench parser now captures all three.

## Caveats

- Sample size still N=1 per axis. tok/s numbers are reliable, quality scores are directional.
- Quants are different (Gemma nvfp4 vs Qwen 4-bit), which is part of why Gemma 4 looks faster on long-context decode. Some of the win is architectural (smaller model, newer attention pattern), some is quant-format. Not separable from this bench alone.
- Vision axis on Gemma 4 is unbenched, not 'failed' — pending upstream tooling support.