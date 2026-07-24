"""B7: Reuters neutrality spot-check.

Reviewer concern
----------------
R2: your "neutral" label is a training artefact of the synthetic split;
it may not reflect what a real editorial newsroom outputs.

Answer
------
Run the trained clickbait detector over EVERY headline in ISOT True.csv
(Reuters wire copy, curated as "real news" by the ISOT team). Report:
  * fraction predicted as clickbait at threshold 0.5,
  * distribution of predicted probabilities (deciles),
  * top 20 highest-probability Reuters headlines (for a qualitative check).

If the detector is well-calibrated for real-world neutrality, the fraction
above 0.5 should be low.

ETA (CPU) : ~2 min for 21k rows. ETA (GPU): ~10 s.
Cost      : $0.

Outputs   : results/b7/reuters_neutrality.md
            results/b7/reuters_scores.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import ISOT_DIR, RESULTS_DIR, ensure_dirs
from common.scorers import ClickbaitScorer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--isot-csv", default=str(ISOT_DIR / "True.csv"))
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-md", default=str(RESULTS_DIR / "b7" / "reuters_neutrality.md"))
    ap.add_argument("--out-csv", default=str(RESULTS_DIR / "b7" / "reuters_scores.csv"))
    args = ap.parse_args()

    ensure_dirs()
    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)

    src = Path(args.isot_csv)
    if not src.exists():
        print(f"[b7] missing {src}. Run data/download_isot_reuters.py first.")
        return 1

    import pandas as pd

    df = pd.read_csv(src)
    text_col = "title" if "title" in df.columns else df.columns[0]
    all_titles = df[text_col].astype(str).str.strip()
    n_all = len(all_titles)
    # Deduplicate: ISOT True.csv repeats some titles; scoring duplicates would
    # bias the neutrality fraction toward whatever the repeated headlines are.
    texts = all_titles.drop_duplicates().tolist()
    n_dup = n_all - len(texts)
    print(f"[b7] {n_all} rows, {n_dup} duplicate titles removed, "
          f"scoring {len(texts)} unique Reuters headlines")
    print("[b7] ETA CPU ~2 min, GPU ~10 s")

    device = "cpu"
    if args.gpu:
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass

    scorer = ClickbaitScorer(device=device)
    probs = scorer.prob(texts, batch_size=args.batch_size)

    scored = pd.DataFrame({"title": texts, "clickbait_prob": probs})
    scored.to_csv(out_csv, index=False, encoding="utf-8-sig")

    deciles = np.quantile(probs, np.linspace(0.1, 0.9, 9))
    frac_cb = float((probs >= 0.5).mean())

    top = scored.sort_values("clickbait_prob", ascending=False).head(20)

    lines = ["# B7 Reuters neutrality spot-check", "",
             f"- n unique = {len(texts)} (from {n_all} rows, {n_dup} duplicates removed)",
             f"- clickbait fraction at threshold 0.5: {frac_cb:.4f}",
             f"- mean prob: {probs.mean():.4f}",
             f"- median prob: {np.median(probs):.4f}",
             "",
             "## Decile boundaries of predicted clickbait probability",
             ""]
    for i, v in enumerate(deciles, start=1):
        lines.append(f"- D{i*10}: {v:.4f}")
    lines += ["", "## Top 20 highest-scoring Reuters headlines", ""]
    for _, r in top.iterrows():
        lines.append(f"- {r['clickbait_prob']:.3f} :: {r['title']}")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[b7] wrote {out_md} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
