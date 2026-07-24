"""B3: Aggregate the three pre-existing sweep CSVs into one tidy dataset and
draw a Pareto plot of engagement vs. fidelity.

Reviewer concern
----------------
The paper's alpha,beta sweep exists but is spread across three separate CSVs
and lacks a clean Pareto-front visual. This script re-aggregates them and,
if the pre-existing CSVs already carry engagement and fidelity metrics,
produces the plot directly. If they only carry rewrites (no metrics), it
tells the human to run b2's scorer instead.

ETA           : seconds.
Cost          : $0.

Inputs        : (default) the three files at
                E:/Projects/Submitted/NewsReWrite/prefix_generation_results_{1,2,3} (1).csv

Outputs       : results/b3/sweep_all.csv
                results/b3/pareto_engagement_vs_fidelity.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import RESULTS_DIR, TRACK_B_ROOT, ensure_dirs

DEFAULT_INPUTS = [
    Path("E:/Projects/Submitted/NewsReWrite/prefix_generation_results_1 (1).csv"),
    Path("E:/Projects/Submitted/NewsReWrite/prefix_generation_results_2 (1).csv"),
    Path("E:/Projects/Submitted/NewsReWrite/prefix_generation_results_3 (1).csv"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", default=[str(p) for p in DEFAULT_INPUTS])
    ap.add_argument("--out-csv", default=str(RESULTS_DIR / "b3" / "sweep_all.csv"))
    ap.add_argument("--out-plot", default=str(RESULTS_DIR / "b3" / "pareto_engagement_vs_fidelity.png"))
    ap.add_argument("--engagement-col", default=None,
                    help="column name for engagement metric; if omitted, autodetects")
    ap.add_argument("--fidelity-col", default=None,
                    help="column name for fidelity metric; if omitted, autodetects")
    args = ap.parse_args()

    ensure_dirs()
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_plot = Path(args.out_plot); out_plot.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    frames = []
    for i, p in enumerate(args.inputs, start=1):
        path = Path(p)
        if not path.exists():
            print(f"[b3] missing: {path}")
            continue
        df = pd.read_csv(path)
        df["source_file"] = path.name
        df["source_idx"] = i
        frames.append(df)
    if not frames:
        print("[b3] no input CSVs found; nothing to aggregate.")
        return 1

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[b3] wrote {out_csv} ({len(all_df)} rows)")

    # Auto-detect metric columns by EXACT name (case-insensitive). A prefix
    # match previously bound "tac" to the legacy INPUT column "tactics" (the
    # requested-tactics knob, not a measured engagement metric), which would
    # plot a control parameter as if it were an outcome. Exact names only.
    lowered = {c.lower(): c for c in all_df.columns}
    eng_candidates = ["engagement", "attr_realised_frac", "attribute_realization_rate"]
    fid_candidates = ["sts", "sts_similarity", "sts_cos", "bertscore_f1", "fidelity"]

    def _find(col_list):
        for name in col_list:
            if name in lowered:
                return lowered[name]
        return None

    eng = args.engagement_col or _find(eng_candidates)
    fid = args.fidelity_col or _find(fid_candidates)

    if eng is None or fid is None:
        print(
            "[b3] Pre-existing CSVs do not contain the engagement/fidelity metrics "
            "needed for a Pareto plot.\n"
            "    Re-run through b2 (run_rewrites.py then score_rewrites.py) and\n"
            "    then call this script on the resulting per_item_scores.csv."
        )
        return 0

    print(f"[b3] engagement column: {eng}")
    print(f"[b3] fidelity   column: {fid}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gb_cols = [c for c in ("alpha", "beta") if c in all_df.columns]
        if gb_cols:
            grouped = all_df.groupby(gb_cols)[[eng, fid]].mean().reset_index()
        else:
            grouped = all_df[[eng, fid]].copy()

        plt.figure(figsize=(6, 5))
        plt.scatter(grouped[fid], grouped[eng], s=40)
        if gb_cols:
            for _, r in grouped.iterrows():
                label = ",".join(f"{c}={r[c]}" for c in gb_cols)
                plt.annotate(label, (r[fid], r[eng]), fontsize=7)
        plt.xlabel(f"fidelity ({fid})")
        plt.ylabel(f"engagement ({eng})")
        plt.title("B3 Pareto: engagement vs fidelity")
        plt.tight_layout()
        plt.savefig(out_plot, dpi=150)
        print(f"[b3] wrote {out_plot}")
    except Exception as e:
        print(f"[b3] plot failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
