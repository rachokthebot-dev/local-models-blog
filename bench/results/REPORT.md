# Local-Qwen 3-axis benchmark — 2026-06-03

**Hardware:** M4 mini 32GB · **Serving:** rapid-mlx 0.4.2 (mlx-vlm 0.4.4)  
**Sampling:** temperature 0.7, top_p default. Qwen3-14B served with `--no-thinking` for apples-to-apples (thinking mode tested separately, see Caveats).  
**Doc source:** Project Gutenberg *Frankenstein* truncated to ~50K tokens.  
**Vision image:** `~/claude/lickbank-songs-portrait.png` (LickBank guitar-song library, portrait grid).  
**Judge:** Claude (via local `claude -p`).

## Results

| Model | Axis | prompt tok | comp tok | TTFT | total | decode tok/s | end-to-end tok/s | Quality (Claude judge) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3-14B-4bit-nothink | reasoning | 178 | 1,207 | 1.9s | 98.3s | 12.52 | 12.28 | final 5/5, steps 5/5, clarity 5/5 |
| Qwen3-14B-4bit-nothink | doc | 46,725 | 416 | 684.3s | 771.8s | 4.76 | 0.54 | correct 2/5, concise 2/5, themes 5/5 |
| Qwen3-VL-8B-Instruct-4bit | vision | 1,436 | 1,026 | 10.6s | 58.3s | 21.53 | 17.61 | correct 5/5, specific 5/5, complete 5/5 |

## Judge notes

- **Qwen3-14B-4bit-nothink / reasoning:** All checkpoints match reference; final time and distance within tolerance.
- **Qwen3-14B-4bit-nothink / doc:** Misidentifies Walton as encountering 'the creature Victor Frankenstein,' covers events beyond the first 50K tokens, and exceeds 300 words.
- **Qwen3-VL-8B-Instruct-4bit / vision:** Accurately identifies the guitar library app with detailed naming of search, tabs, import button, and grid tiles including titles.

## Throughput readings

- **Short-context decode (Qwen3-14B-4bit, 178 prompt tok, 1.2K answer):** ~**12.5 tok/s**. Matches the rough range we've seen for dense 14B 4-bit on M4. Not far off the gemma4_mlx_bench prior of ~14 tok/s on Q8_0 Ollama — MLX 4-bit on Apple gets a similar order of magnitude despite the smaller bit width.
- **Long-context decode (Qwen3-14B-4bit, 46.7K prompt tok):** decode drops to **~4.8 tok/s**. Attention cost over 47K KV cache is the killer.
- **Long-context prefill TTFT:** **684s** (~11 min) for 46.7K prompt tokens → prefill throughput ≈ **68 tok/s**. This is the dominant cost of any long-doc task locally; you'll feel it on every cold cache hit.
- **Vision (Qwen3-VL-8B-Instruct-4bit, 1.4K vision+text prompt tok, 1K answer):** **21.5 tok/s decode**, TTFT 10.6s (image encode + prefill).

## Quality readings

- **Reasoning, no-think 5/5/5:** Qwen3-14B without thinking still nailed the 5-step train problem (10:22 AM, ~137 mi). Confirms the model has enough capacity for this class of problem in non-thinking mode — turning thinking *off* was a win on this prompt (the thinking run blew the 8K-token budget without ever closing `<think>`).
- **Doc summary 2/2/5:** Captured the three themes well, but hallucinated forward past the 50K truncation point and exceeded 300 words. Probably needs stricter prompt scaffolding ("summarize ONLY what you read; do not extrapolate") plus an explicit word cap.
- **Vision 5/5/5:** Qwen3-VL-8B identified LickBank correctly, named every UI region (search, tab toggle, import button, tile grid), and pulled out specific song titles. Real signal that the VL 8B variant is strong for UI/screen understanding on Apple Silicon at ~5GB weights + 21 tok/s.

## What to apply

1. **Qwen3-VL-8B-Instruct-4bit is a keeper for screen/UI work.** 21.5 tok/s decode, 10s TTFT, perfect judge on the LickBank screenshot. Add it as the vision-tier default in any local agentic pipeline that needs to read screenshots.
2. **For Qwen3-14B short-context tasks, default `--no-thinking`.** Thinking mode runs away to 8K+ tokens without closing the block on multi-step word problems; the model is capable enough in non-thinking mode. Save the thinking variant for prompts where you can budget ≥16K completion tokens and verifiably need the deeper search.
3. **Avoid local 50K-token summaries unless you can wait ~13 min.** TTFT of ~11 min is structural (~70 tok/s prefill on dense 14B 4-bit). For long docs, chunk + map-reduce or route to cloud — matches the recent `reference_qwen_mtp_throughput.md` lesson that low-bit quant is the throughput lever, not architectural tricks.
4. **rapid-mlx vision extras aren't installed by default.** `pip install 'rapid-mlx[vision]'` into the brew-managed env at `/opt/homebrew/Cellar/rapid-mlx/0.4.2/libexec/bin/python` was required. Worth a memory.

## Caveats

- **Gemma 4 wasn't re-benched** in this run — cache was empty and the comparison would have cost another ~13GB pull + ~25 min of bench. Numbers above stand on their own; head-to-head against Gemma is a follow-up.
- **One image, one document, one math problem.** N=1 per axis. Treat the tok/s numbers as solid (they're physics) but the quality scores as directional.
- **Judge = Claude.** Same-family bias is possible. The reasoning axis has a ground-truth answer (10:22 AM, 137 mi) so the 5/5/5 is grounded; the doc summary and vision scores rely on Claude's read.
- **Qwen3-14B is not 'Qwen 3.5 9B' from the XDA article.** Different generation, different size. rapid-mlx already ships `qwen3.5-9b` and `qwen3-vl-4b` aliases that would let you re-run the exact article comparison.

## Files

- `~/claude/bench_3axis.py` — bench runner (streaming SSE, captures content + reasoning)
- `~/claude/judge_3axis.py` — LLM-as-judge wrapper around `claude -p`
- `~/claude/run_3axis.sh` — orchestrator (start server / bench / swap)
- `~/claude/bench_data/frankenstein.txt` — cached long-doc input
- `~/claude/bench_results/*.json` — raw run + judge data
- `~/claude/bench_results/.discarded/` — failed earlier runs (504s, empty-content streams)