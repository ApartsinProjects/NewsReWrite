"""B1: External-benchmark validation of the tactics (engagement-attribute) model.

Reviewer concern
----------------
Does each of the 10 attributes carry a signal that generalises to a
human-labelled clickbait corpus, or are they only picking up
GPT-4o-mini's generation quirks?

Method
------
Run the repo's `bert_tactics_prefix_finetuned` over every Webis-Clickbait-17
headline and report:
  (i)  per-attribute mean probability on clickbait vs non-clickbait rows,
       with a two-sample unpaired label-permutation significance test
       (the two groups are different Webis rows of unequal size, so an
       unpaired test is the correct choice).
  (ii) per-attribute Spearman rank correlation between the attribute score
       and Webis's graded truthMean score.

ETA (CPU) : ~4 min. ETA (GPU) : under 1 min.
Cost      : $0.

Outputs
-------
results/b1_tactics_external.json
results/b1_tactics_external.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import RESULTS_DIR, TACTIC_NAMES, WEBIS_DIR, ensure_dirs
from common.scorers import TacticsScorer

# Re-use Webis loader.
from b1_external_benchmarks.eval_clickbait_scorer import load_webis  # noqa: E402


def two_sample_boot_pvalue(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """Two-sided bootstrap p-value on the difference in means."""
    obs = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        perm = rng.permutation(pooled)
        diffs[i] = perm[:n_a].mean() - perm[n_a:].mean()
    p = float((np.abs(diffs) >= abs(obs)).mean())
    return obs, p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--webis-dir", default=str(WEBIS_DIR))
    ap.add_argument("--out-json", default=str(RESULTS_DIR / "b1_tactics_external.json"))
    ap.add_argument("--out-md", default=str(RESULTS_DIR / "b1_tactics_external.md"))
    args = ap.parse_args()

    ensure_dirs()

    print("[b1-tac] ETA: ~4 min CPU, <1 min GPU.")

    device = "cpu"
    if args.gpu:
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass

    scorer = TacticsScorer(device=device)
    texts, y_bin, y_grd = load_webis(Path(args.webis_dir))
    print(f"[b1-tac] Webis rows: {len(texts)}")

    t0 = time.time()
    probs = scorer.probs(texts, batch_size=args.batch_size)  # (N, 10)
    print(f"[b1-tac] scoring done in {time.time()-t0:.1f}s")

    from scipy.stats import spearmanr

    report = {"tactic_names": TACTIC_NAMES, "attributes": []}
    for k, name in enumerate(TACTIC_NAMES):
        col = probs[:, k]
        pos = col[y_bin == 1]
        neg = col[y_bin == 0]
        diff, pval = two_sample_boot_pvalue(pos, neg, n_boot=args.n_boot)
        r_s, r_p = spearmanr(y_grd, col)
        report["attributes"].append(
            {
                "idx": k,
                "name": name,
                "mean_clickbait": float(pos.mean()),
                "mean_non_clickbait": float(neg.mean()),
                "mean_diff": diff,
                "boot_pvalue": pval,
                "spearman_r_vs_graded": float(r_s),
                "spearman_p_vs_graded": float(r_p),
                "n_pos": int(pos.size),
                "n_neg": int(neg.size),
            }
        )

    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# B1 tactics scorer, Webis-Clickbait-17", ""]
    lines.append("| # | attribute | mean_cb | mean_non | diff | p_boot | Spearman r | p |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for a in report["attributes"]:
        lines.append(
            f"| {a['idx']} | {a['name']} | {a['mean_clickbait']:.3f} | "
            f"{a['mean_non_clickbait']:.3f} | {a['mean_diff']:+.3f} | "
            f"{a['boot_pvalue']:.3g} | {a['spearman_r_vs_graded']:+.3f} | "
            f"{a['spearman_p_vs_graded']:.2e} |"
        )
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[b1-tac] wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
