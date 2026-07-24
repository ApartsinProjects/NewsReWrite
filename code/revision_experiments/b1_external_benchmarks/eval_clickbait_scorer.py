"""B1: External-benchmark validation of the clickbait scoring model.

Reviewer concern
----------------
R1 asked whether the clickbait discriminator generalises beyond the paper's
own synthetic corpus. This script scores the repo's `bert_clickbait_prefix_finetuned`
on two independent public benchmarks and reports AUROC, F1, Precision, Recall
with 95% percentile bootstrap CIs (n=1000).

Datasets
--------
1. Webis-Clickbait-17 (graded regression + binary at threshold 0.5).
2. Chakraborty 2016 "Stop Clickbait" (binary).

ETA (CPU)  : Webis ~4 min, Chakraborty ~6 min at batch 32.
ETA (GPU)  : under 1 min each.
Cost       : $0 (local).

Outputs
-------
results/b1_clickbait_external.json
results/b1_clickbait_external.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bootstrap_ci import bootstrap_metric_ci
from common.paths import (
    CHAKRABORTY_DIR,
    RESULTS_DIR,
    WEBIS_DIR,
    ensure_dirs,
)
from common.scorers import ClickbaitScorer


def load_webis(root: Path):
    """Return (texts, y_binary, y_graded).

    y_binary uses truthClass ("clickbait" -> 1, "no-clickbait" -> 0).
    y_graded uses truthMean in [0,1].
    """
    import pandas as pd

    inst_path = root / "instances.jsonl"
    truth_path = root / "truth.jsonl"
    if not inst_path.exists() or not truth_path.exists():
        raise FileNotFoundError(
            f"Webis files missing under {root}. Run data/download_webis.py first."
        )

    inst = pd.read_json(inst_path, lines=True)
    truth = pd.read_json(truth_path, lines=True)
    merged = inst.merge(truth, on="id", how="inner")

    def flatten_post(x):
        if isinstance(x, list):
            return " ".join(str(t) for t in x)
        return str(x)

    texts = merged["postText"].map(flatten_post).tolist()
    y_bin = (merged["truthClass"] == "clickbait").astype(int).to_numpy()
    y_grd = merged["truthMean"].to_numpy(dtype=float)
    return texts, y_bin, y_grd


def load_chakraborty(root: Path):
    cb = (root / "clickbait_data").read_text(encoding="utf-8", errors="ignore").splitlines()
    nc = (root / "non_clickbait_data").read_text(encoding="utf-8", errors="ignore").splitlines()
    cb = [t.strip() for t in cb if t.strip()]
    nc = [t.strip() for t in nc if t.strip()]
    texts = cb + nc
    y = np.array([1] * len(cb) + [0] * len(nc), dtype=int)
    return texts, y


def eval_binary(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000):
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_prob >= 0.5).astype(int)
    point = {
        "n": int(y_true.size),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_true.mean()),
    }
    n = y_true.size

    def make_metric(fn):
        def wrap(idx):
            try:
                return float(fn(y_true[idx], y_prob[idx]))
            except Exception:
                return float("nan")

        return wrap

    def make_binmetric(fn):
        def wrap(idx):
            try:
                return float(fn(y_true[idx], y_pred[idx], zero_division=0))
            except Exception:
                return float("nan")

        return wrap

    ci = {}
    ci["auroc"] = bootstrap_metric_ci(make_metric(roc_auc_score), n, n_boot=n_boot)
    ci["f1"] = bootstrap_metric_ci(make_binmetric(f1_score), n, n_boot=n_boot)
    ci["precision"] = bootstrap_metric_ci(
        make_binmetric(precision_score), n, n_boot=n_boot
    )
    ci["recall"] = bootstrap_metric_ci(
        make_binmetric(recall_score), n, n_boot=n_boot
    )
    point["ci95"] = {k: {"lo": v[0], "hi": v[1]} for k, v in ci.items()}
    return point


def eval_regression(y_graded: np.ndarray, y_prob: np.ndarray):
    from scipy.stats import pearsonr, spearmanr

    r_s, p_s = spearmanr(y_graded, y_prob)
    r_p, p_p = pearsonr(y_graded, y_prob)
    return {
        "spearman_r": float(r_s),
        "spearman_p": float(p_s),
        "pearson_r": float(r_p),
        "pearson_p": float(p_p),
        "n": int(y_graded.size),
    }


def _write_md(report: dict, path: Path) -> None:
    lines = ["# B1 clickbait scorer, external benchmarks", ""]
    for corpus, res in report.items():
        lines.append(f"## {corpus}")
        if "binary" in res:
            b = res["binary"]
            lines.append(f"- n = {b['n']}, positive rate = {b['positive_rate']:.3f}")
            for k in ("auroc", "f1", "precision", "recall"):
                ci = b["ci95"][k]
                lines.append(
                    f"- {k} = {b[k]:.4f}  (95% CI {ci['lo']:.4f} to {ci['hi']:.4f})"
                )
        if "regression" in res:
            r = res["regression"]
            lines.append(
                f"- graded Spearman r = {r['spearman_r']:.4f} (p={r['spearman_p']:.2e}, n={r['n']})"
            )
            lines.append(
                f"- graded Pearson  r = {r['pearson_r']:.4f} (p={r['pearson_p']:.2e})"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", action="store_true", help="use CUDA if available")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--webis-dir", default=str(WEBIS_DIR))
    ap.add_argument("--chakraborty-dir", default=str(CHAKRABORTY_DIR))
    ap.add_argument("--out-json", default=str(RESULTS_DIR / "b1_clickbait_external.json"))
    ap.add_argument("--out-md", default=str(RESULTS_DIR / "b1_clickbait_external.md"))
    args = ap.parse_args()

    ensure_dirs()

    print("[b1] ETA: ~4-6 min per corpus on CPU, ~30s each on GPU.")

    device = "cpu"
    if args.gpu:
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
    print(f"[b1] device = {device}")

    scorer = ClickbaitScorer(device=device)
    report: dict = {}

    # Webis
    try:
        texts, y_bin, y_grd = load_webis(Path(args.webis_dir))
        print(f"[b1] Webis: {len(texts)} rows")
        t0 = time.time()
        probs = scorer.prob(texts, batch_size=args.batch_size)
        print(f"[b1] Webis scoring done in {time.time()-t0:.1f}s")
        report["webis17"] = {
            "binary": eval_binary(y_bin, probs, n_boot=args.n_boot),
            "regression": eval_regression(y_grd, probs),
        }
    except FileNotFoundError as e:
        print(f"[b1] SKIP Webis: {e}")

    # Chakraborty
    try:
        texts, y = load_chakraborty(Path(args.chakraborty_dir))
        print(f"[b1] Chakraborty: {len(texts)} rows")
        t0 = time.time()
        probs = scorer.prob(texts, batch_size=args.batch_size)
        print(f"[b1] Chakraborty scoring done in {time.time()-t0:.1f}s")
        report["chakraborty16"] = {
            "binary": eval_binary(y, probs, n_boot=args.n_boot),
        }
    except FileNotFoundError as e:
        print(f"[b1] SKIP Chakraborty: {e}")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(report, Path(args.out_md))
    print(f"[b1] wrote {out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
