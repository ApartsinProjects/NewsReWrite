"""B1-prefix: validate the PREFIX clickbait guide on REAL data at partial
prefix lengths -- the regime it actually operates in during FUDGE decoding.

Why this exists
---------------
b1 (eval_clickbait_scorer.py) validates the prefix-finetuned guide on real
human data, but only on FULL headlines. During FUDGE decoding the guide scores
PARTIAL prefixes (word-ratios 0.3/0.5/0.7). That partial regime had only ever
been validated on SYNTHETIC prefixes (the training test split). This script
closes the gap: it truncates each REAL labelled headline (Webis-17,
Chakraborty-16) to the same fixed ratios the guide was trained on, scores those
REAL prefixes with the guide, and asks two questions:

  1. AUROC at each ratio: does an early real prefix already predict the real
     full-headline clickbait label? (predictive prefix behaviour on real data)
  2. Score separation + monotonicity: does the guide's clickbait probability
     rise as a real clickbait headline is revealed word by word, while staying
     low for real non-clickbait? (the steering signal it feeds FUDGE)

The prefix label is inherited from its parent headline -- exactly the guide's
operating assumption (a clickbait headline's prefixes trend clickbait), so
testing it on real text is a direct validation of that assumption.

Outputs
-------
results/b1_prefix_on_real.json
results/b1_prefix_on_real.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bootstrap_ci import bootstrap_metric_ci
from common.paths import CHAKRABORTY_DIR, RESULTS_DIR, WEBIS_DIR, ensure_dirs
from common.scorers import ClickbaitScorer

# Import the real-data loaders from b1 so the corpora are read identically.
from eval_clickbait_scorer import load_chakraborty, load_webis

# Must match build_prefix_datasets.py exactly, so we test the guide on the
# same prefix construction it was trained on.
PREFIX_RATIOS = [0.3, 0.5, 0.7, 1.0]
MIN_WORDS = 2


def make_prefix(text: str, r: float) -> str:
    words = str(text).strip().split()
    n = len(words)
    if n == 0:
        return ""
    k = max(MIN_WORDS, int(n * r))
    k = min(k, n)
    p = " ".join(words[:k])
    if r == 1.0 and not p.endswith((".", "?", "!")):
        p += "."
    return p


def eval_corpus(name: str, texts, y, scorer: ClickbaitScorer,
                n_boot: int, max_n: int) -> dict:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    if max_n and len(texts) > max_n:
        # Deterministic stratified-ish subsample for speed on huge corpora.
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(texts))[:max_n]
        texts = [texts[i] for i in idx]
        y = y[idx]
    print(f"[prefix-real] {name}: {len(texts)} headlines, "
          f"positive_rate={y.mean():.3f}", flush=True)

    per_ratio = []
    for r in PREFIX_RATIOS:
        prefixes = [make_prefix(t, r) for t in texts]
        probs = np.asarray(scorer.prob(prefixes), dtype=float)
        try:
            auroc = float(roc_auc_score(y, probs))
        except Exception:
            auroc = float("nan")

        def _auroc(idx, _y=y, _p=probs):
            try:
                return float(roc_auc_score(_y[idx], _p[idx]))
            except Exception:
                return float("nan")

        lo, hi = bootstrap_metric_ci(_auroc, len(y), n_boot=n_boot)
        mean_pos = float(probs[y == 1].mean()) if (y == 1).any() else float("nan")
        mean_neg = float(probs[y == 0].mean()) if (y == 0).any() else float("nan")
        rec = {
            "ratio": r,
            "auroc": auroc,
            "auroc_ci95": {"lo": lo, "hi": hi},
            "mean_prob_clickbait": mean_pos,
            "mean_prob_nonclickbait": mean_neg,
            "separation": mean_pos - mean_neg,
        }
        per_ratio.append(rec)
        print(f"[prefix-real] {name} r={r}: AUROC={auroc:.4f} "
              f"(sep={rec['separation']:+.4f})", flush=True)

    aurocs = [x["auroc"] for x in per_ratio]
    seps = [x["separation"] for x in per_ratio]
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "per_ratio": per_ratio,
        # Monotone non-decreasing AUROC as the prefix grows is the expected
        # signature of a well-behaved prefix guide.
        "auroc_monotone_nondecreasing": bool(
            all(aurocs[i] <= aurocs[i + 1] + 1e-6 for i in range(len(aurocs) - 1))),
        "separation_monotone_nondecreasing": bool(
            all(seps[i] <= seps[i + 1] + 1e-6 for i in range(len(seps) - 1))),
        "auroc_at_earliest_ratio": aurocs[0],
        "auroc_at_full": aurocs[-1],
    }


def _write_md(report: dict, path: Path) -> None:
    lines = ["# B1-prefix: prefix clickbait guide on REAL data at partial lengths",
             "",
             "AUROC = prefix score vs the real full-headline label. "
             "sep = mean(prob | clickbait) - mean(prob | non-clickbait).", ""]
    for corpus, res in report.items():
        lines.append(f"## {corpus}")
        lines.append(f"- n = {res['n']}, positive rate = {res['positive_rate']:.3f}")
        lines.append(f"- AUROC monotone as prefix grows: "
                     f"{res['auroc_monotone_nondecreasing']}")
        lines.append("")
        lines.append("| ratio | AUROC | 95% CI | mean_prob cb | mean_prob non | sep |")
        lines.append("|---|---|---|---|---|---|")
        for x in res["per_ratio"]:
            ci = x["auroc_ci95"]
            lines.append(
                f"| {x['ratio']} | {x['auroc']:.4f} | "
                f"{ci['lo']:.4f}-{ci['hi']:.4f} | "
                f"{x['mean_prob_clickbait']:.4f} | "
                f"{x['mean_prob_nonclickbait']:.4f} | {x['separation']:+.4f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--max-n", type=int, default=8000,
                    help="cap headlines per corpus for speed (0 = all). The "
                         "score is over 4 prefixes each, so 8000 -> 32k scorings.")
    ap.add_argument("--webis-dir", default=str(WEBIS_DIR))
    ap.add_argument("--chakraborty-dir", default=str(CHAKRABORTY_DIR))
    ap.add_argument("--out-json", default=str(RESULTS_DIR / "b1_prefix_on_real.json"))
    ap.add_argument("--out-md", default=str(RESULTS_DIR / "b1_prefix_on_real.md"))
    args = ap.parse_args()

    ensure_dirs()

    device = "cpu"
    if args.gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
    print(f"[prefix-real] device = {device}", flush=True)

    scorer = ClickbaitScorer(device=device)
    report = {}

    try:
        wt, wy, _ = load_webis(Path(args.webis_dir))
        report["webis17"] = eval_corpus("webis17", wt, wy, scorer,
                                         args.n_boot, args.max_n)
    except Exception as e:
        print(f"[prefix-real] WARN: Webis skipped: {e}", flush=True)

    try:
        ct, cy = load_chakraborty(Path(args.chakraborty_dir))
        report["chakraborty16"] = eval_corpus("chakraborty16", ct, cy, scorer,
                                               args.n_boot, args.max_n)
    except Exception as e:
        print(f"[prefix-real] WARN: Chakraborty skipped: {e}", flush=True)

    if not report:
        raise SystemExit("[prefix-real] no corpus evaluated")

    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(report, Path(args.out_md))
    print(f"[prefix-real] wrote {args.out_json} and {args.out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
