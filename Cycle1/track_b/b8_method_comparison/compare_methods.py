"""Cross-method comparison for reviewer concern R2-M4.

Loads the per_item_scores.csv produced by score_rewrites.py for several
methods (FUDGE, prompt-only baseline, DExperts, GeDi), aligns them on the
SAME held-out items and tactic configs, and produces:

  1. A tradeoff table: per (method, setting) mean + 95% bootstrap CI of
     fidelity (BERTScore, STS), independent clickbait probability, and
     attribute realization. This traces each method's fidelity-vs-clickbait
     curve so FUDGE's dual-signal control can be read against the single-axis
     DExperts/GeDi controllers and the plain prompt.
  2. A Pareto scatter (fidelity on x, independent clickbait on y), one point
     per (method, setting), so the frontier is visible at a glance.
  3. Pairwise Wilcoxon signed-rank tests between a REFERENCE setting (default:
     the FUDGE full cell) and every other method's representative setting, on
     item-and-tactic-matched pairs, for fidelity / clickbait / attribute
     realization. This gives significance, not just overlapping CIs.

Every method's per_item_scores.csv shares the schema written by
score_rewrites.py; the sweep parameter lives in the `alpha` column (FUDGE
also uses `beta`; DExperts uses alpha; GeDi puts omega in alpha; prompt-only
uses the string "prompt_only").

Usage:
  python compare_methods.py \
     --inputs results/b2/per_item_scores.csv results/b4/per_item_scores.csv \
              results/b6_dexperts/per_item_scores.csv results/b6_gedi/per_item_scores.csv \
     --labels fudge prompt_only dexperts gedi \
     --out-dir results/method_comparison
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FIDELITY = "bertscore_f1"
CLICKBAIT = "clickbait_prob_external"
ATTR = "attr_realised_frac_guide_circular"
JUDGE_ATTR = "judge_attr_confirmed_frac"


def boot_ci(x, n_boot=1000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _setting(row):
    a, b = row.get("alpha"), row.get("beta")
    try:
        a = float(a); b = float(b)
        return f"a{a}_b{b}"
    except (TypeError, ValueError):
        return str(a)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="per_item_scores.csv paths, one per method")
    ap.add_argument("--labels", nargs="+", required=True,
                    help="method label per input, same order")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ref-label", default="fudge",
                    help="method whose representative setting is the reference")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    if len(args.inputs) != len(args.labels):
        raise SystemExit("--inputs and --labels must be the same length")

    import pandas as pd

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for path, label in zip(args.inputs, args.labels):
        p = Path(path)
        if not p.exists():
            print(f"[cmp] WARN: {p} missing; method '{label}' skipped")
            continue
        d = pd.read_csv(p)
        if "is_empty" in d.columns:
            d = d[~d["is_empty"].astype(bool)]
        d["method"] = label
        d["setting"] = d.apply(_setting, axis=1)
        frames.append(d)
    if not frames:
        raise SystemExit("no usable inputs")
    alld = pd.concat(frames, ignore_index=True)

    metrics = [m for m in (FIDELITY, "sts", CLICKBAIT, ATTR, JUDGE_ATTR)
               if m in alld.columns]

    # 1. tradeoff table: per (method, setting)
    table = []
    for (method, setting), grp in alld.groupby(["method", "setting"], sort=True):
        row = {"method": method, "setting": setting, "n": int(len(grp))}
        for m in metrics:
            mean, lo, hi = boot_ci(grp[m].values, n_boot=args.n_boot)
            row[m] = round(mean, 4)
            row[f"{m}_ci"] = [round(lo, 4), round(hi, 4)]
        table.append(row)

    # 2. Pareto scatter (fidelity vs independent clickbait)
    plotted = False
    if FIDELITY in alld.columns and CLICKBAIT in alld.columns:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 5))
            for method, grp in alld.groupby("method"):
                pts = grp.groupby("setting")[[FIDELITY, CLICKBAIT]].mean()
                ax.scatter(pts[FIDELITY], pts[CLICKBAIT], label=method, s=60, alpha=0.8)
                for setting, r in pts.iterrows():
                    ax.annotate(setting, (r[FIDELITY], r[CLICKBAIT]), fontsize=6)
            ax.set_xlabel("fidelity (BERTScore-F1 vs source, higher better)")
            ax.set_ylabel("independent clickbait probability (lower better)")
            ax.set_title("Method comparison: fidelity vs clickbait tradeoff")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "pareto_fidelity_vs_clickbait.png", dpi=130)
            plotted = True
        except Exception as e:
            print(f"[cmp] WARN: Pareto plot failed: {e}")

    # 3. pairwise Wilcoxon vs the reference method's representative setting.
    # Reference setting = the FUDGE cell with the largest alpha AND beta (full
    # dual guidance); other methods use the setting closest to the median of
    # their sweep.
    from scipy.stats import wilcoxon

    def representative(method_df):
        settings = method_df["setting"].unique().tolist()
        # prefer a "full" fudge cell if alpha/beta numeric
        best = None
        for s in settings:
            sub = method_df[method_df["setting"] == s]
            try:
                a = float(sub["alpha"].iloc[0]); b = float(sub["beta"].iloc[0])
                score = a + b
            except (TypeError, ValueError):
                score = -1
            if best is None or score > best[1]:
                best = (s, score)
        return best[0] if best else (settings[0] if settings else None)

    sig = {}
    ref_df = alld[alld["method"] == args.ref_label]
    if len(ref_df):
        ref_setting = representative(ref_df)
        ref = ref_df[ref_df["setting"] == ref_setting].set_index(["item_id", "tactic_label"])
        for method, mdf in alld.groupby("method"):
            if method == args.ref_label:
                continue
            m_setting = representative(mdf)
            cur = mdf[mdf["setting"] == m_setting].set_index(["item_id", "tactic_label"])
            common = ref.index.intersection(cur.index)
            for m in metrics:
                try:
                    x = ref.loc[common, m].to_numpy(dtype=float)
                    y = cur.loc[common, m].to_numpy(dtype=float)
                    ok = ~(np.isnan(x) | np.isnan(y))
                    x, y = x[ok], y[ok]
                    if len(x) >= 6 and np.any(x != y):
                        st, p = wilcoxon(x, y)
                        sig[f"{args.ref_label}({ref_setting}) vs {method}({m_setting}) [{m}]"] = {
                            "p": round(float(p), 5), "n": int(len(x)),
                            "delta_mean_ref_minus_other": round(float(np.mean(x - y)), 4)}
                except Exception:
                    pass

    report = {
        "methods": args.labels,
        "reference": args.ref_label,
        "metrics": metrics,
        "tradeoff_table": table,
        "significance_vs_reference": sig,
        "pareto_plot": str(out_dir / "pareto_fidelity_vs_clickbait.png") if plotted else None,
    }
    (out_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # markdown table
    md = ["# Method comparison (R2-M4)", "",
          f"Reference: {args.ref_label}. Metrics: {', '.join(metrics)}.", "",
          "| method | setting | n | " + " | ".join(metrics) + " |",
          "|" + "---|" * (3 + len(metrics))]
    for r in table:
        md.append("| " + " | ".join(
            [r["method"], r["setting"], str(r["n"])]
            + [f"{r.get(m, float('nan')):.3f}" for m in metrics]) + " |")
    md += ["", "## Significance vs reference (Wilcoxon signed-rank, item-matched)", ""]
    for k, v in sig.items():
        md.append(f"- {k}: p={v['p']}, n={v['n']}, delta={v['delta_mean_ref_minus_other']}")
    (out_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[cmp] wrote {out_dir}/comparison.json + comparison.md"
          + (" + pareto_fidelity_vs_clickbait.png" if plotted else ""))
    print(json.dumps(report, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
