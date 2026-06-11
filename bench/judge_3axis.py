#!/usr/bin/env python3
"""
LLM-as-judge for 3-axis benchmark outputs. Uses the local `claude` CLI
(non-interactive `-p` mode) so the user's existing Claude Code subscription
is the credential — no API key needed.

Usage:
    python3 judge_3axis.py --results bench_results/*.json
"""

import argparse, json, pathlib, re, subprocess, sys

JUDGE_PROMPTS = {
    "doc": """You are evaluating a summary of the first ~50K tokens of Mary Shelley's \
*Frankenstein* (Project Gutenberg edition). Score the summary on three axes (1-5 each):

- correctness: Does it accurately represent the text?
- conciseness: Is it under 300 words, tight, and free of fluff?
- themes: Does it surface three central themes (e.g., ambition/hubris, isolation, \
creation and responsibility, knowledge and consequence)?

Respond with ONLY a JSON object on a single line:
{"correctness": N, "conciseness": N, "themes": N, "notes": "<one short sentence>"}

Summary to evaluate:
---
{output}
---""",

    "vision": """You are evaluating a model's description of a UI screenshot. The image \
shows a music-learning library app called "LickBank" displayed in portrait orientation. \
Visible elements include: a search bar at the top, "Licks/Songs" toggle, an "Import from \
YouTube" button, and a grid of song tiles with YouTube-style video thumbnails. Tile titles \
reference guitar lessons (e.g., "Hotel California", "Smells Like Teen Spirit", "Sweet Child \
O' Mine", "Comfortably Numb", "Easy Guitar Solo #4", "Black Sabbath - Paranoid", etc.).

Score the description on three axes (1-5 each):
- correctness: Does it correctly identify a music/guitar library app with a tile grid?
- specificity: Are visible UI elements (search, toggles, import button, tile grid, thumbnails) named?
- completeness: Does it mention the content type (guitar songs/lessons), grid structure, and any visible titles?

Respond with ONLY a JSON object on a single line:
{"correctness": N, "specificity": N, "completeness": N, "notes": "<one short sentence>"}

Description to evaluate:
---
{output}
---""",

    "reasoning": """You are grading a multi-step word problem. Two trains: Train A leaves \
City A at 8:00 AM eastbound at 60 mph; Train B leaves City B (240 mi east of A) at 9:00 AM \
westbound at 80 mph. At 9:30 AM both hit a 20-minute construction zone reducing speeds by 25%. \
Then they resume original speeds.

Reference solution checkpoints:
- 9:00 AM: A at mile 60, B at mile 240
- 9:30 AM: A at mile 90, B at mile 200 (gap = 110)
- 9:50 AM: A at mile 105, B at mile 180 (gap = 75)
- Closing speed after resume = 140 mph; time to close 75 mi = 75/140 h ≈ 32.14 min
- FINAL: ~10:22 AM, ~137.14 miles from City A

Score on three axes (1-5 each):
- final_answer: Is the time and distance within ±2 min and ±2 miles of the reference?
- steps: Are the 9:30 and 9:50 positions correct?
- clarity: Is the reasoning followable and well-organized?

Respond with ONLY a JSON object on a single line:
{"final_answer": N, "steps": N, "clarity": N, "notes": "<one short sentence>"}

Answer to evaluate:
---
{output}
---""",
}


def judge(axis: str, output: str) -> dict:
    prompt = JUDGE_PROMPTS[axis].replace("{output}", output)
    try:
        r = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"error": "judge timeout"}
    raw = r.stdout.strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"error": "no JSON in judge output", "raw": raw[:500]}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse: {e}", "raw": raw[:500]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    args = ap.parse_args()

    for rf in args.results:
        rp = pathlib.Path(rf)
        data = json.loads(rp.read_text())
        label = data.get("label", rp.stem)
        for axis, r in data.get("axes", {}).items():
            if axis not in JUDGE_PROMPTS or "output" not in r:
                continue
            print(f"=== judging {label} :: {axis} ===")
            verdict = judge(axis, r["output"])
            print(f"  {verdict}")
            r["judge"] = verdict
        rp.write_text(json.dumps(data, indent=2))
        print(f"updated {rp}\n")


if __name__ == "__main__":
    main()
