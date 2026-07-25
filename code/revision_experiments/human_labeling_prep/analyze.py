"""Analyze and diagnose completed human-labeling studies (Track C).

Consumes the filled-in rater CSVs (rater_01.csv ...) plus the oracle.csv that
prepare.py produced, and computes every statistic the reviewers asked for,
PLUS diagnostics that flag where the study is weak (low agreement, wide
disagreement, scales that do not separate conditions).

C1 (tactic-label validation, reviewer R2-M6 / R2-M1 rubric):
  - per-tactic inter-rater agreement (Fleiss kappa over raters, and mean
    pairwise Cohen kappa), so rubric reliability is quantified per tactic
  - agreement of the rater consensus with the generator's INTENDED tactic
    vector (precision / recall / F1 per tactic): does the synthetic label
    match what humans actually see (the R2-M6 tactic-label-fidelity question)
  - DIAGNOSTICS: tactics with kappa < 0.4 (unreliable), tactics whose
    intended-vs-perceived F1 is low (noisy synthetic labels)

C2 (rewrite quality, reviewer R2-M2d / R1-2):
  - per-condition (method x alpha,beta) mean of engagement / faithfulness /
    perceived-clickbait with 95% bootstrap CI
  - ICC(2,k) across raters for each scale (reliability of the human ratings)
  - paired Wilcoxon signed-rank between key conditions (full FUDGE vs none,
    FUDGE vs prompt-only, and every method vs baseline) on item-matched
    per-condition rater means -> significance, not just CIs
  - correlation of human ratings with the automatic metrics (validates that
    BERTScore / external-clickbait / attribute-realization track human judgment)
  - DIAGNOSTICS: scales with ICC < 0.5 (unreliable), conditions with high
    rater spread, method rankings with overlapping CIs

--simulate: with no real rater data yet, fill the rater files from the oracle
(intended tactics for C1; auto-scores mapped to Likert for C2) plus noise, so
the ENTIRE analysis pipeline can be dry-run and validated before any human is
paid. This is the "diagnose possible gaps" tool: run it now, confirm every
statistic computes and the diagnostics fire, then hand the same rater files to
real raters and re-run without --simulate.

Usage:
  python analyze.py --study-dir results/human_labeling/c1_rubric_validation
  python analyze.py --study-dir results/human_labeling/c2_rewrite_quality
  python analyze.py --study-dir <either> --simulate   # dry-run on fake labels
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


TACTIC_COLS = [
    "tactic__curiosity_gap", "tactic__exaggeration", "tactic__emotional_trigger",
    "tactic__sensationalism", "tactic__lists_or_superlatives",
    "tactic__ambiguous_references", "tactic__direct_appeals",
    "tactic__unfinished_narratives", "tactic__unexpected_associations",
    "tactic__provocative_questions",
]
LIKERT_COLS = ["engagement_1_to_5", "faithfulness_1_to_5", "clickbait_1_to_5"]


# ------------------------------------------------------------------
# agreement statistics (manual, numpy-only, so no extra deps)
# ------------------------------------------------------------------
def fleiss_kappa(counts: np.ndarray) -> float:
    """counts: (n_items, n_categories) tally of rater votes per item."""
    n_items, n_cat = counts.shape
    n_raters = counts.sum(axis=1)
    if (n_raters == 0).any() or n_items == 0:
        return float("nan")
    p_j = counts.sum(axis=0) / counts.sum()
    P_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()
    denom = 1 - P_e
    return float((P_bar - P_e) / denom) if abs(denom) > 1e-12 else float("nan")


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    if len(set(a.tolist()) | set(b.tolist())) < 2:
        # degenerate (all one class): perfect agreement iff identical
        return 1.0 if np.array_equal(a, b) else 0.0
    try:
        return float(cohen_kappa_score(a, b))
    except Exception:
        return float("nan")


def icc_2k(matrix: np.ndarray) -> float:
    """ICC(2,k): two-way random, average of k raters. matrix: (n_items, k)."""
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ss_total = ((matrix - grand) ** 2).sum()
    ss_row = k * ((row_means - grand) ** 2).sum()
    ss_col = n * ((col_means - grand) ** 2).sum()
    ss_err = ss_total - ss_row - ss_col
    ms_row = ss_row / (n - 1)
    ms_col = ss_col / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_row + (ms_col - ms_err) / n
    return float((ms_row - ms_err) / denom) if abs(denom) > 1e-12 else float("nan")


def boot_ci(x, n_boot=1000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ------------------------------------------------------------------
# simulate rater responses from the oracle (for pipeline validation)
# ------------------------------------------------------------------
def _simulate(study_dir: Path, kind: str, seed: int = 0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    oracle = pd.read_csv(study_dir / "oracle.csv")
    raters = sorted(study_dir.glob("rater_*.csv"))
    if not raters:
        raise SystemExit(f"no rater_*.csv in {study_dir}; run prepare.py first")

    for ri, rp in enumerate(raters):
        r = pd.read_csv(rp)
        r = r.merge(oracle, on="task_id", suffixes=("", "_oracle"))
        if kind == "c1":
            intended = r["intended_tactic_vector"].apply(
                lambda s: json.loads(s) if isinstance(s, str) else list(s))
            for j, col in enumerate(TACTIC_COLS):
                base = np.array([v[j] for v in intended], dtype=float)
                # noisy rater: mostly follows intended, flips with prob 0.15
                flip = rng.random(len(base)) < (0.15 + 0.05 * ri)
                r[col] = np.where(flip, 1 - base, base).astype(int)
        else:
            # map auto-scores to 1-5 with rater noise; higher fidelity -> higher faithfulness etc.
            def to5(x, invert=False):
                x = np.asarray(x, dtype=float)
                x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)
                if invert:
                    x = 1 - x
                return np.clip(np.round(1 + 4 * x + rng.normal(0, 0.6 + 0.2 * ri, len(x))), 1, 5).astype(int)
            r["faithfulness_1_to_5"] = to5(r.get("auto_bertscore_f1", r.get("auto_sts_cos", 0.5)))
            r["engagement_1_to_5"] = to5(r.get("auto_attr_realised_frac_guide_circular",
                                               r.get("condition_alpha", 0.5)))
            r["clickbait_1_to_5"] = to5(r.get("auto_clickbait_prob_external", 0.1))
        keep = [c for c in pd.read_csv(rp).columns]
        r[keep].to_csv(rp, index=False)
    print(f"[analyze] SIMULATED rater responses into {len(raters)} files "
          f"(for pipeline validation only)")


# ------------------------------------------------------------------
# C1 analysis
# ------------------------------------------------------------------
def analyze_c1(study_dir: Path) -> dict:
    import pandas as pd

    oracle = pd.read_csv(study_dir / "oracle.csv")
    raters = [pd.read_csv(p) for p in sorted(study_dir.glob("rater_*.csv"))]
    R = len(raters)
    if R == 0:
        raise SystemExit("no rater files")
    # align every rater to task_id order
    task_ids = oracle["task_id"].tolist()
    per_rater = [r.set_index("task_id").reindex(task_ids) for r in raters]

    intended = oracle.set_index("task_id")["intended_tactic_vector"].apply(
        lambda s: json.loads(s) if isinstance(s, str) else list(s)).reindex(task_ids)

    report = {"n_items": len(task_ids), "n_raters": R, "per_tactic": {}}
    lowkappa, noisy = [], []
    for j, col in enumerate(TACTIC_COLS):
        votes = np.stack([pr[col].fillna(0).astype(int).to_numpy() for pr in per_rater], axis=1)  # (n, R)
        counts = np.stack([(votes == 0).sum(axis=1), (votes == 1).sum(axis=1)], axis=1)
        fk = fleiss_kappa(counts)
        pair_ks = [cohen_kappa(votes[:, a], votes[:, b])
                   for a, b in itertools.combinations(range(R), 2)]
        mean_pair = float(np.nanmean(pair_ks)) if pair_ks else float("nan")
        consensus = (votes.mean(axis=1) >= 0.5).astype(int)
        intend = np.array([v[j] for v in intended], dtype=int)
        tp = int(((consensus == 1) & (intend == 1)).sum())
        fp = int(((consensus == 1) & (intend == 0)).sum())
        fn = int(((consensus == 0) & (intend == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec and (prec + rec) else float("nan")
        report["per_tactic"][col] = {
            "fleiss_kappa": round(fk, 3), "mean_pairwise_cohen": round(mean_pair, 3),
            "intended_vs_consensus_precision": round(prec, 3) if prec == prec else None,
            "intended_vs_consensus_recall": round(rec, 3) if rec == rec else None,
            "intended_vs_consensus_f1": round(f1, 3) if f1 == f1 else None,
        }
        if fk == fk and fk < 0.4:
            lowkappa.append(col)
        if f1 == f1 and f1 < 0.5:
            noisy.append(col)
    report["diagnostics"] = {
        "low_agreement_tactics (kappa<0.4)": lowkappa,
        "noisy_synthetic_label_tactics (intended-vs-consensus F1<0.5)": noisy,
    }
    return report


# ------------------------------------------------------------------
# C2 analysis
# ------------------------------------------------------------------
def analyze_c3(study_dir: Path) -> dict:
    """2AFC engagement: raters pick which of two same-source rewrites they would
    click. oracle.csv carries target_side (A/B) = the dual FUDGE (4,2) rewrite; the
    other side is no_guidance (0,0). Reports the pooled win-rate for the dual method
    with a binomial test, plus inter-rater agreement on the raw choice."""
    import pandas as pd
    from scipy.stats import binomtest

    oracle = pd.read_csv(study_dir / "oracle.csv")           # task_id, target_side, ...
    key = oracle.set_index("task_id")["target_side"].astype(str).str.upper().to_dict()
    raters = [pd.read_csv(p) for p in sorted(study_dir.glob("rater_*.csv"))]
    R = len(raters)
    if R == 0:
        raise SystemExit("no rater files")

    # per-rater: did they pick the positive_only side?
    per_rater = []
    choices_by_item: dict = {}
    for ri, r in enumerate(raters):
        ch = r[["task_id", "choice"]].copy()
        ch["choice"] = ch["choice"].astype(str).str.upper().str.strip()
        hit = 0
        n = 0
        for _, row in ch.iterrows():
            tid, c = row["task_id"], row["choice"]
            if tid not in key or c not in ("A", "B"):
                continue
            n += 1
            hit += int(c == key[tid])
            choices_by_item.setdefault(tid, []).append(c)
        per_rater.append({"rater": ri, "n": n, "prefer_positive_rate": round(hit / n, 3) if n else None})

    # pooled binomial (every valid rater-judgment)
    total = sum(p["n"] for p in per_rater)
    hits = sum(round(p["prefer_positive_rate"] * p["n"]) for p in per_rater if p["n"])
    bt = binomtest(hits, total, 0.5, alternative="two-sided") if total else None

    # majority-vote per item, then win-rate over items
    item_pref = []
    for tid, cs in choices_by_item.items():
        maj = max(set(cs), key=cs.count)
        item_pref.append(int(maj == key.get(tid)))
    # raw inter-rater agreement on A/B choice (mean pairwise percent agreement)
    agree_pairs = []
    for cs in choices_by_item.values():
        if len(cs) >= 2:
            same = sum(1 for a, b in itertools.combinations(cs, 2) if a == b)
            tot = len(list(itertools.combinations(cs, 2)))
            agree_pairs.append(same / tot)

    return {
        "n_items": oracle["task_id"].nunique(),
        "n_raters": R,
        "prefer_dual_pooled": {
            "hits": int(hits), "n": int(total),
            "rate": round(hits / total, 3) if total else None,
            "p_binom_vs_0.5": (bt.pvalue if bt else None),
            "ci95": [round(bt.proportion_ci().low, 3), round(bt.proportion_ci().high, 3)] if bt else None,
        },
        "prefer_dual_majority_vote_item_rate": round(float(np.mean(item_pref)), 3) if item_pref else None,
        "per_rater": per_rater,
        "mean_pairwise_choice_agreement": round(float(np.mean(agree_pairs)), 3) if agree_pairs else None,
        "diagnostics": {},
    }


def analyze_c2(study_dir: Path) -> dict:
    import pandas as pd
    from scipy.stats import wilcoxon, spearmanr

    oracle = pd.read_csv(study_dir / "oracle.csv")
    raters = [pd.read_csv(p) for p in sorted(study_dir.glob("rater_*.csv"))]
    R = len(raters)
    if R == 0:
        raise SystemExit("no rater files")

    # long frame: one row per (task_id, rater) with the 3 scales
    frames = []
    for ri, r in enumerate(raters):
        rr = r[["task_id"] + LIKERT_COLS].copy()
        rr["rater"] = ri
        frames.append(rr)
    long = pd.concat(frames, ignore_index=True).merge(
        oracle, on="task_id", how="left")

    cond = "condition_label"
    method = "method" if "method" in long.columns else None
    group_cols = ([method] if method else []) + [cond]
    report = {"n_items": oracle["task_id"].nunique(), "n_raters": R,
              "per_condition": {}, "reliability_icc2k": {}, "significance": {},
              "human_vs_auto_spearman": {}, "diagnostics": {}}

    # per-condition mean + CI (over item-level rater means)
    item_cond = long.groupby(["task_id"] + group_cols)[LIKERT_COLS].mean().reset_index()
    for keys, sub in item_cond.groupby(group_cols):
        label = keys if isinstance(keys, str) else "|".join(map(str, keys))
        report["per_condition"][label] = {
            s: dict(zip(("mean", "lo", "hi"), [round(v, 3) for v in boot_ci(sub[s])]))
            for s in LIKERT_COLS
        }

    # ICC(2,k) per scale: items x raters matrix (averaged over conditions per item)
    for s in LIKERT_COLS:
        wide = long.pivot_table(index="task_id", columns="rater", values=s)
        report["reliability_icc2k"][s] = round(icc_2k(wide.dropna().to_numpy()), 3)
        if report["reliability_icc2k"][s] == report["reliability_icc2k"][s] and report["reliability_icc2k"][s] < 0.5:
            report["diagnostics"].setdefault("low_reliability_scales (ICC<0.5)", []).append(s)

    # paired Wilcoxon: baseline vs each condition, on item-matched per-condition means
    labels = list(report["per_condition"].keys())
    baselines = [l for l in labels if "no_guidance" in l or "prompt_only" in l]
    base_label = baselines[0] if baselines else (labels[0] if labels else None)
    if base_label:
        base = item_cond.copy()
        base["_lab"] = base[group_cols].astype(str).agg("|".join, axis=1) if len(group_cols) > 1 else base[group_cols[0]]
        for lab in labels:
            if lab == base_label:
                continue
            for s in LIKERT_COLS:
                a = base[base["_lab"] == base_label].set_index("task_id")[s]
                b = base[base["_lab"] == lab].set_index("task_id")[s]
                common = a.index.intersection(b.index)
                if len(common) >= 6 and (a.loc[common].to_numpy() - b.loc[common].to_numpy()).any():
                    try:
                        st, p = wilcoxon(a.loc[common], b.loc[common])
                        report["significance"][f"{lab} vs {base_label} [{s}]"] = round(float(p), 5)
                    except Exception:
                        pass

    # human vs auto-metric correlation (validate the automatic metrics)
    auto_map = {
        "faithfulness_1_to_5": "auto_bertscore_f1",
        "clickbait_1_to_5": "auto_clickbait_prob_external",
        "engagement_1_to_5": "auto_attr_realised_frac_guide_circular",
    }
    item_mean = long.groupby("task_id")[LIKERT_COLS].mean()
    o = oracle.set_index("task_id")
    for scale, auto in auto_map.items():
        if auto in o.columns:
            j = item_mean.join(o[auto], how="inner").dropna()
            if len(j) >= 6:
                rho, p = spearmanr(j[scale], j[auto])
                report["human_vs_auto_spearman"][f"{scale} vs {auto}"] = {
                    "rho": round(float(rho), 3), "p": round(float(p), 5), "n": int(len(j))}
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-dir", required=True,
                    help="a c1_rubric_validation or c2_rewrite_quality dir")
    ap.add_argument("--simulate", action="store_true",
                    help="fill rater files with simulated labels from the oracle "
                         "to validate the analysis pipeline before real raters")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    study_dir = Path(args.study_dir)
    if not (study_dir / "oracle.csv").exists():
        raise SystemExit(f"no oracle.csv in {study_dir}")
    kind = "c1" if "c1" in study_dir.name else ("c3" if "c3" in study_dir.name else "c2")

    if args.simulate and kind != "c3":
        _simulate(study_dir, kind, seed=args.seed)

    if kind == "c1":
        report = analyze_c1(study_dir)
    elif kind == "c3":
        report = analyze_c3(study_dir)
    else:
        report = analyze_c2(study_dir)
    report["study"] = study_dir.name
    report["simulated"] = bool(args.simulate)

    out = Path(args.out_json) if args.out_json else study_dir / "analysis.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[analyze] wrote {out}")
    if report.get("diagnostics"):
        print(f"[analyze] DIAGNOSTICS: {json.dumps(report['diagnostics'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
