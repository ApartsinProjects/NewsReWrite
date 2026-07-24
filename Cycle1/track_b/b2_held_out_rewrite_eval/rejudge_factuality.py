"""Root-cause the hallucination jump: re-judge rewrites with a rubric that
separates a GENUINE new verifiable fact from a rhetorical question / framing.

The graded judge's `hallucinates` flag fires on "any factual claim not in the
original", which over-counts the target tactics themselves: a Provocative
Question ("Are the foundations at risk?") or Sensationalist framing ("chaos
erupts") is flagged even though it asserts no new verifiable fact. This script
re-judges each rewrite with a stricter definition and FORCES the judge to quote
the specific added fact, which filters out the rhetorical-question false
positives. It reports, per (alpha,beta) cell, the corrected NEW-FACT rate next
to the original hallucination rate.

Reads results/<dir>/per_item_scores.csv, writes:
    results/<dir>/factuality_refined.json
    results/<dir>/factuality_refined.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.artifacts import JsonlWriter
from common.paths import TACTIC_NAMES
from score_rewrites import _judge_api


def _factuality_prompt(neutral: str, edited: str) -> str:
    return (
        "Compare a REWRITE of a news headline to the ORIGINAL.\n\n"
        f"ORIGINAL: \"{neutral}\"\n"
        f"REWRITE:  \"{edited}\"\n\n"
        "Does the REWRITE assert a NEW, VERIFIABLE FACT that is not stated or "
        "directly implied by the ORIGINAL? A verifiable fact is a specific "
        "named entity, number, quote, date, place, or event presented as true. "
        "A rhetorical question, an opinion, emotional or dramatic framing, a "
        "metaphor, or vivid wording is NOT a new fact, even if striking. Only "
        "count something that a fact-checker could verify as added and false or "
        "unsupported.\n"
        "Return ONLY strict JSON: {\"new_fact\": bool, "
        "\"added_fact\": \"the specific new fact, or empty if none\"}."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--max-per-cell", type=int, default=150,
                    help="rows re-judged per (alpha,beta) cell (0 = all)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    from collections import defaultdict

    in_dir = Path(args.in_dir)
    df = pd.read_csv(in_dir / "per_item_scores.csv")
    df = df[df.get("is_empty", False) != True].copy()

    # Subsample per cell for cost control (deterministic: first N per cell).
    keep = []
    percell: dict = {}
    for idx, row in df.iterrows():
        cell = (row["alpha"], row["beta"])
        if args.max_per_cell and percell.get(cell, 0) >= args.max_per_cell:
            continue
        percell[cell] = percell.get(cell, 0) + 1
        keep.append(idx)
    sub = df.loc[keep]
    print(f"[factuality] re-judging {len(sub)} rows "
          f"(<= {args.max_per_cell}/cell) across {sub.groupby(['alpha','beta']).ngroups} cells",
          flush=True)

    def _one(item):
        idx, row = item
        j = _judge_api(_factuality_prompt(str(row["neutral"]), str(row["edited"])))
        nf = bool(j.get("new_fact")) if isinstance(j, dict) and "new_fact" in j else None
        return idx, row, j, nf

    out_jsonl = in_dir / "factuality_refined.jsonl"
    newfact = defaultdict(list)
    orig_hall = defaultdict(list)
    with JsonlWriter(out_jsonl) as w, \
            ThreadPoolExecutor(max_workers=max(args.workers, 1)) as ex:
        for idx, row, j, nf in ex.map(_one, list(sub.iterrows())):
            cell = (float(row["alpha"]), float(row["beta"]))
            w.write({"item_id": int(row["item_id"]), "alpha": row["alpha"],
                     "beta": row["beta"], "tactic_label": row["tactic_label"],
                     "neutral": row["neutral"], "edited": row["edited"],
                     "orig_hallucinates": (None if pd.isna(row.get("judge_hallucinates"))
                                           else bool(row.get("judge_hallucinates"))),
                     "new_fact": nf, "added_fact": (j.get("added_fact") if isinstance(j, dict) else None),
                     "judge_response": j, "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
            if nf is not None:
                newfact[cell].append(1.0 if nf else 0.0)
            oh = row.get("judge_hallucinates")
            if not pd.isna(oh):
                orig_hall[cell].append(float(oh))

    summary = {}
    for cell in sorted(newfact):
        nf = newfact[cell]
        oh = orig_hall.get(cell, [])
        summary[f"a{cell[0]}_b{cell[1]}"] = {
            "n": len(nf),
            "orig_hallucination_rate": (sum(oh) / len(oh)) if oh else None,
            "refined_new_fact_rate": (sum(nf) / len(nf)) if nf else None,
        }
    (in_dir / "factuality_refined.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("[factuality] per-cell (orig hallucination -> refined new-fact):", flush=True)
    for k, v in summary.items():
        oh = v["orig_hallucination_rate"]; nf = v["refined_new_fact_rate"]
        print(f"  {k:12s} n={v['n']:4d}  orig={oh:.3f}  refined_newfact={nf:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
