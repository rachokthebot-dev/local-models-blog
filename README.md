# Local models on a Mac mini M4 32 GB — benchmark log

Field notes from six months of running local LLMs, image generators, and a music-analysis model on a Mac mini M4 with 32 GB unified memory.

**Read the report:** https://rachokthebot-dev.github.io/local-models-blog/

## What's in here

- `index.html` — the full benchmark log, self-contained, no build step. Served via GitHub Pages.

## Sections covered

1. Executive summary — what we kept
2. The hardware and the constraints
3. General-purpose LLMs (reasoning · vision · long-doc · tool-call)
4. Image generation (FLUX.2 Klein 4B / 9B)
5. Music analysis (SongFormer beats audio LLMs)
6. Models on the watchlist
7. The performance ceiling, explained
8. Methodology and caveats

## Reproducing the benchmarks

Bench scripts live in `~/claude/` on the source machine. The page lists each script's purpose at the bottom.
