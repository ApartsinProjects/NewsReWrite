"""Prepare Track C human-labeling packages.

Two independent studies. Each produces a self-contained package with
(1) an oracle CSV that keeps every provenance and auto-score column for
the analyst, (2) N per-rater task CSVs with only the fields the rater
needs, blinded IDs, and position-shuffled rewrites, and (3) a codebook.md
with rating instructions and tactic definitions.

C1: Tactic-label validation (reviewer concern R2-M6)
    Given a neutral source and a synthetic clickbait rewrite of it,
    raters mark which of the 10 engagement tactics are actually realized
    in the rewrite. Compare to the tactic vector the generator was
    instructed to activate. Report per-tactic Cohen's kappa and overall
    agreement.

C2: Rewrite quality (reviewer concerns R2-M2d, R1-2)
    Given a source and one FUDGE-produced rewrite, raters rate
    engagement / faithfulness / perceived clickbait on 1-5 Likert
    scales. Raters see rewrites from all four (alpha, beta) conditions
    without knowing which cell each rewrite came from. Compare
    per-condition means; report ICC(2,k) across raters.

Inputs
------
--synthetic-csv  For C1. Written by regenerate_synthetic on Modal.
                 Expected columns: original, clickbait, methods_vector.
                 methods_vector may be a Python-list-literal or JSON list of 10 ints.
--per-item-csv   For C2. Written by score_rewrites.py.
                 Expected columns: source_id, source, edited, alpha,
                 beta, top_k, seed, tactic_config, tactic_ids,
                 bertscore_f1, sts_cos, nli_entail_fwd, nli_entail_rev,
                 clickbait_prob_external, attribute_realization_rate.
                 Missing metric columns are silently omitted from the
                 oracle; the study still runs.

Both flags may be given together. Skipping either disables that study.

Outputs
-------
<out-dir>/c1_rubric_validation/
    oracle.csv
    rater_{k}.csv                for k in 1..n_raters
    codebook.md
<out-dir>/c2_rewrite_quality/
    oracle.csv
    rater_{k}.csv                for k in 1..n_raters
    codebook.md

Design
------
- Every rater sees every item (full overlap) so we can compute pairwise
  agreement without imputation. Set --overlap partial to move to a
  fractional rotation later.
- task_id is a random 12-char token; the mapping task_id -> source_id
  lives only in oracle.csv.
- For C2 the (source, rewrite) presentation order is shuffled
  independently per rater to counter primacy effects.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import secrets
from pathlib import Path
from typing import Iterable


TACTIC_NAMES = [
    "curiosity_gap",
    "exaggeration",
    "emotional_trigger",
    "sensationalism",
    "lists_or_superlatives",
    "ambiguous_references",
    "direct_appeals",
    "unfinished_narratives",
    "unexpected_associations",
    "provocative_questions",
]

# One-line rater-facing gloss for each tactic. Full definitions live in the
# codebook that ships with each study package.
TACTIC_ONELINERS = {
    "curiosity_gap":
        "Signals that a specific but unnamed piece of information is being withheld.",
    "exaggeration":
        "Amplifies importance or magnitude with intensity modifiers, no new facts.",
    "emotional_trigger":
        "Uses explicit emotional wording (fear, outrage, hope, concern...).",
    "sensationalism":
        "Heightens drama or spectacle without explicit emotional wording.",
    "lists_or_superlatives":
        "Uses list scaffolding or extreme ranking words (top, first, largest).",
    "ambiguous_references":
        "Deploys vague pronouns or indefinite references (this, they, something).",
    "direct_appeals":
        "Addresses the reader or a specific audience directly (you, voters, parents).",
    "unfinished_narratives":
        "Presents the event as ongoing or unresolved (what happens next...).",
    "unexpected_associations":
        "Links two normally disjoint concepts or domains in a surprising way.",
    "provocative_questions":
        "Uses interrogative phrasing that challenges an assumption.",
}


def _mint_task_id(*parts) -> str:
    """Deterministic per-rater-invariant task id from any set of parts."""
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


# Maps the column names produced by score_rewrites.py (the actual pipeline)
# to the canonical names _prepare_c2 uses internally. score_rewrites.py
# carries `neutral`, `sts`, `clickbait_prob`, `attr_realised_frac`,
# `nli_neutral_entails_edited`, `nli_edited_entails_neutral`, `tactic_label`,
# and an integer `item_id`, none of which match the canonical schema.
_C2_COLUMN_ALIASES = {
    "neutral": "source",
    "sts": "sts_cos",
    "clickbait_prob": "clickbait_prob_external",
    "attr_realised_frac": "attribute_realization_rate",
    "nli_neutral_entails_edited": "nli_entail_fwd",
    "nli_edited_entails_neutral": "nli_entail_rev",
    "tactic_label": "tactic_config",
}


def _normalize_c2_frame(d, method_label: str):
    """Rename score_rewrites.py columns to the canonical C2 schema and
    synthesize source_id when the pipeline only provides item_id.

    Handles the prompt-only baseline, whose alpha/beta are the string
    'prompt_only' rather than floats.
    """
    d = d.rename(columns={k: v for k, v in _C2_COLUMN_ALIASES.items()
                          if k in d.columns})
    if "source_id" not in d.columns:
        if "item_id" in d.columns:
            d["source_id"] = d["item_id"].apply(lambda x: f"item_{x}")
        elif "source" in d.columns:
            d["source_id"] = (
                "src_" + d["source"].astype(str).map(
                    lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
                )
            )
    return d


def _parse_vector(x) -> list[int]:
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        try:
            v = json.loads(s)
        except Exception:
            try:
                v = ast.literal_eval(s)
            except Exception:
                return []
        return [int(v) for v in v] if isinstance(v, Iterable) else []
    return []


# ------------------------------------------------------------------
# C1: Tactic-label validation
# ------------------------------------------------------------------
def _prepare_c1(
    synthetic_csv: Path,
    out_dir: Path,
    n_items: int,
    n_raters: int,
    seed: int,
) -> None:
    import pandas as pd

    df = pd.read_csv(synthetic_csv)
    need = {"original", "clickbait", "methods_vector"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"C1 input missing columns: {missing}")

    df["_vec"] = df["methods_vector"].apply(_parse_vector)
    df = df[df["_vec"].apply(lambda v: len(v) == len(TACTIC_NAMES))].reset_index(drop=True)

    # Stratify by number of activated tactics so we do not oversample the
    # dominant "one tactic activated" bucket. Shuffle first, then take an
    # even slice per bucket, then top up from whatever rows were not already
    # taken. All bookkeeping is done on the shuffled df's own index so the
    # top-up can never re-include an already-sampled row.
    df["_n_active"] = df["_vec"].apply(sum)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    per_bucket = max(1, n_items // max(1, df["_n_active"].nunique()))
    taken_idx = (
        df.groupby("_n_active", group_keys=False)[df.columns.tolist()]
          .apply(lambda g: g.head(per_bucket))
          .index
    )
    taken_idx = list(taken_idx)[:n_items]
    if len(taken_idx) < n_items:
        remaining_idx = [i for i in df.index if i not in set(taken_idx)]
        need = min(n_items - len(taken_idx), len(remaining_idx))
        # deterministic top-up: remaining_idx is already in shuffled order
        taken_idx += remaining_idx[:need]
    sample = df.loc[taken_idx].reset_index(drop=True)

    # Oracle rows carry everything the analyst may want later.
    oracle_rows = []
    for i, r in sample.iterrows():
        vec = r["_vec"]
        intended = [TACTIC_NAMES[k] for k, v in enumerate(vec) if v == 1]
        task_id = _mint_task_id("c1", i, r["original"], r["clickbait"])
        row = {
            "task_id": task_id,
            "study": "c1_rubric_validation",
            "source_headline": r["original"],
            "rewrite": r["clickbait"],
            "intended_tactic_vector": vec,
            "intended_tactic_names": intended,
            "n_intended_tactics": int(sum(vec)),
        }
        oracle_rows.append(row)

    oracle_df = pd.DataFrame(oracle_rows)
    (out_dir / "c1_rubric_validation").mkdir(parents=True, exist_ok=True)
    oracle_df.to_json(
        out_dir / "c1_rubric_validation" / "oracle.jsonl",
        orient="records", lines=True, force_ascii=False,
    )
    oracle_df.assign(
        intended_tactic_vector=oracle_df["intended_tactic_vector"].astype(str),
        intended_tactic_names=oracle_df["intended_tactic_names"].astype(str),
    ).to_csv(out_dir / "c1_rubric_validation" / "oracle.csv", index=False)

    # Per-rater task files: same items, order shuffled per rater. Rater-facing
    # columns strip the intended-tactic ground truth and add 10 empty checkbox
    # columns named as the tactics.
    for k in range(1, n_raters + 1):
        rng2 = pd.Series(range(len(oracle_df))).sample(frac=1, random_state=seed + k).tolist()
        rater = oracle_df.iloc[rng2].reset_index(drop=True)
        cols = {
            "task_id": rater["task_id"],
            "presentation_order": range(1, len(rater) + 1),
            "source_headline": rater["source_headline"],
            "rewrite": rater["rewrite"],
        }
        for t in TACTIC_NAMES:
            cols[f"tactic__{t}"] = ""
        cols["rater_notes"] = ""
        pd.DataFrame(cols).to_csv(
            out_dir / "c1_rubric_validation" / f"rater_{k:02d}.csv",
            index=False,
        )

    _write_codebook_c1(out_dir / "c1_rubric_validation" / "codebook.md",
                       n_items=len(oracle_df), n_raters=n_raters)
    print(f"[c1] wrote {len(oracle_df)} items x {n_raters} raters to "
          f"{out_dir / 'c1_rubric_validation'}")


def _write_codebook_c1(path: Path, n_items: int, n_raters: int) -> None:
    lines = [
        "# C1 Codebook: Tactic-label validation",
        "",
        "Study: rater sees a neutral source headline and a rewrite. For each ",
        "of ten engagement tactics, mark 1 if the rewrite realizes the tactic ",
        "and 0 otherwise. Multiple tactics may co-occur.",
        "",
        f"Items per rater: {n_items}. Independent raters: {n_raters}. Every rater ",
        "sees every item (full overlap). Presentation order is randomized per ",
        "rater; do not compare across raters by row position, only by task_id.",
        "",
        "## Rating rules",
        "",
        "- Base your judgment on the text of the rewrite alone, not on the ",
        "  source. The source is provided only for context.",
        "- A tactic is realized only if its distinguishing linguistic cue is ",
        "  actually present in the rewrite text. Do not infer intent.",
        "- If the rewrite copies the source verbatim (or nearly so), mark all ",
        "  tactics 0.",
        "- If in doubt between two overlapping tactics (e.g. Curiosity Gap vs. ",
        "  Ambiguous References), pick the one whose canonical cue is closer to ",
        "  the surface wording. See the tactic definitions below.",
        "- Use the rater_notes column to flag ambiguous cases; leave blank ",
        "  otherwise.",
        "",
        "## Tactic definitions",
        "",
    ]
    for t in TACTIC_NAMES:
        lines.append(f"### {t}")
        lines.append(TACTIC_ONELINERS[t])
        lines.append("")
    lines += [
        "## File format",
        "",
        "One row per item. Columns:",
        "- task_id: opaque 12-character identifier.",
        "- presentation_order: your rater-specific row order (already shuffled).",
        "- source_headline, rewrite: the two headlines to compare.",
        "- tactic__<name>: put 1 or 0 in each of the ten tactic columns.",
        "- rater_notes: free-text, optional.",
        "",
        "Save the file with the same name; do not add or remove rows.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# C2: Rewrite quality
# ------------------------------------------------------------------
def _prepare_c2(
    per_item_csvs: list[Path],
    method_labels: list[str] | None,
    out_dir: Path,
    n_items: int,
    n_raters: int,
    seed: int,
) -> None:
    import pandas as pd

    if method_labels is None:
        method_labels = [p.parent.name for p in per_item_csvs]
    if len(method_labels) != len(per_item_csvs):
        raise SystemExit(
            f"--method-labels ({len(method_labels)}) must match "
            f"--per-item-csv ({len(per_item_csvs)}) in length"
        )

    # Load and stack every method into one long dataframe with a method column.
    frames = []
    for csv_path, label in zip(per_item_csvs, method_labels):
        d = pd.read_csv(csv_path)
        d = _normalize_c2_frame(d, label)
        need = {"source_id", "source", "edited"}
        missing = need - set(d.columns)
        if missing:
            raise SystemExit(
                f"C2 input {csv_path} missing columns {missing} "
                f"(after alias normalization; available: {sorted(d.columns)})"
            )
        # alpha/beta may be absent for prompt-only or other non-FUDGE methods.
        for col, default in (("alpha", 0.0), ("beta", 0.0)):
            if col not in d.columns:
                d[col] = default
        d = d.copy()
        d["method"] = label
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # Keep at most one rewrite per (method, source_id, alpha, beta, seed) so a
    # single source under a single method is never overrepresented.
    dedup_cols = [
        c for c in ("method", "source_id", "alpha", "beta", "seed")
        if c in df.columns
    ]
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    # Sample source headlines ONCE (from the intersection across methods so
    # every picked source is guaranteed to have a rewrite under every method).
    # If intersection is smaller than n_items, fall back to the union.
    per_method_sources = [set(d["source_id"].unique()) for d in frames]
    common = set.intersection(*per_method_sources) if per_method_sources else set()
    pool = pd.Series(sorted(common)) if len(common) >= n_items \
        else pd.Series(sorted(set().union(*per_method_sources)))
    src_pool = pool.sample(
        min(n_items, len(pool)), random_state=seed,
    ).tolist()
    sample = df[df["source_id"].isin(src_pool)].reset_index(drop=True)

    metric_cols = [
        c for c in (
            "bertscore_f1", "sts_cos",
            "nli_entail_fwd", "nli_entail_rev",
            "clickbait_prob_external", "attribute_realization_rate",
        ) if c in sample.columns
    ]
    prov_cols = [
        c for c in (
            "top_k", "seed", "tactic_config", "tactic_ids",
        ) if c in sample.columns
    ]

    oracle_rows = []
    for i, r in sample.iterrows():
        task_id = _mint_task_id(
            "c2", r["method"], r["source_id"],
            r.get("alpha"), r.get("beta"), r.get("seed"),
        )
        row = {
            "task_id": task_id,
            "study": "c2_rewrite_quality",
            "method": r["method"],
            "source_id": r["source_id"],
            "source_headline": r["source"],
            "rewrite": r["edited"],
            "condition_alpha": _as_float(r["alpha"]),
            "condition_beta": _as_float(r["beta"]),
            "condition_label": _cell_label(r["alpha"], r["beta"]),
        }
        for c in prov_cols:
            row[f"gen_{c}"] = r[c]
        for c in metric_cols:
            row[f"auto_{c}"] = r[c]
        oracle_rows.append(row)

    oracle_df = pd.DataFrame(oracle_rows)
    (out_dir / "c2_rewrite_quality").mkdir(parents=True, exist_ok=True)
    oracle_df.to_json(
        out_dir / "c2_rewrite_quality" / "oracle.jsonl",
        orient="records", lines=True, force_ascii=False,
    )
    oracle_df.to_csv(out_dir / "c2_rewrite_quality" / "oracle.csv", index=False)

    # Per-rater task files: shuffle rows; blind the rater to condition and to
    # every automatic score.
    for k in range(1, n_raters + 1):
        rater = oracle_df.sample(frac=1, random_state=seed + k).reset_index(drop=True)
        cols = {
            "task_id": rater["task_id"],
            "presentation_order": range(1, len(rater) + 1),
            "source_headline": rater["source_headline"],
            "rewrite": rater["rewrite"],
            "engagement_1_to_5": "",
            "faithfulness_1_to_5": "",
            "clickbait_1_to_5": "",
            "rater_notes": "",
        }
        pd.DataFrame(cols).to_csv(
            out_dir / "c2_rewrite_quality" / f"rater_{k:02d}.csv",
            index=False,
        )

    _write_codebook_c2(out_dir / "c2_rewrite_quality" / "codebook.md",
                       n_items=len(oracle_df), n_raters=n_raters)
    print(f"[c2] wrote {len(oracle_df)} items x {n_raters} raters to "
          f"{out_dir / 'c2_rewrite_quality'}")


def _as_float(x) -> float:
    """Best-effort float; non-numeric condition tags map to NaN."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _cell_label(alpha, beta) -> str:
    # Non-FUDGE methods (e.g. prompt-only) tag alpha/beta with a string
    # such as "prompt_only"; surface that verbatim as the condition label.
    a, b = _as_float(alpha), _as_float(beta)
    if a != a or b != b:  # NaN check: at least one was non-numeric
        tag = str(alpha)
        return tag if tag == str(beta) else f"{alpha}_{beta}"
    if a == 0 and b == 0:
        return "no_guidance"
    if a > 0 and b == 0:
        return "positive_only"
    if a == 0 and b > 0:
        return "negative_only"
    return "full_fudge"


def _write_codebook_c2(path: Path, n_items: int, n_raters: int) -> None:
    lines = [
        "# C2 Codebook: Rewrite quality",
        "",
        "Study: rater sees a neutral source headline and a rewritten headline. ",
        "Rate the REWRITE on three independent 1-5 Likert scales. You are NOT ",
        "told which generation condition or which method produced the rewrite; ",
        "multiple methods and multiple parameter settings are mixed together ",
        "and shuffled per rater.",
        "",
        f"Items per rater: {n_items}. Independent raters: {n_raters}. Every rater ",
        "sees every item (full overlap). Presentation order is randomized per ",
        "rater; do not compare rewrites across raters by row position, only by ",
        "task_id.",
        "",
        "## Scales",
        "",
        "### engagement_1_to_5",
        "How likely would a busy reader click this rewrite in a news feed, ",
        "relative to the source?",
        "",
        "  1 = far less engaging than the source",
        "  2 = somewhat less engaging",
        "  3 = about the same as the source",
        "  4 = somewhat more engaging",
        "  5 = far more engaging",
        "",
        "### faithfulness_1_to_5",
        "How well does the rewrite preserve the factual content of the source? ",
        "Introducing a new noun, entity, number, or event counts as a fidelity ",
        "loss even if it sounds plausible.",
        "",
        "  1 = the rewrite contradicts or adds facts absent from the source",
        "  2 = one clear novel claim or spin present",
        "  3 = mild embellishment, no new facts",
        "  4 = fully faithful with different wording",
        "  5 = fully faithful and preserves emphasis",
        "",
        "### clickbait_1_to_5",
        "How much does the rewrite feel like clickbait, in the sense of ",
        "manipulative or misleading engagement optimization?",
        "",
        "  1 = pure informational rewrite, no manipulation",
        "  2 = mild rhetorical framing but responsible",
        "  3 = borderline",
        "  4 = noticeable clickbait patterns",
        "  5 = clearly manipulative clickbait",
        "",
        "## Rating rules",
        "",
        "- Rate the rewrite as it stands, without imagining a better version.",
        "- The three scales are independent: a rewrite can score high on ",
        "  engagement AND high on faithfulness AND low on clickbait, or any ",
        "  other combination.",
        "- If the rewrite is identical to the source, mark engagement 3 and ",
        "  faithfulness 5.",
        "- Use rater_notes for anything you want to flag; leave blank otherwise.",
        "",
        "## File format",
        "",
        "One row per item. Columns:",
        "- task_id: opaque 12-character identifier.",
        "- presentation_order: your rater-specific row order (already shuffled).",
        "- source_headline, rewrite: the two headlines.",
        "- engagement_1_to_5, faithfulness_1_to_5, clickbait_1_to_5: integers 1-5.",
        "- rater_notes: free-text, optional.",
        "",
        "Save the file with the same name; do not add or remove rows.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic-csv",
                    help="C1 input: synthetic_clickbait.csv from regenerate_synthetic")
    ap.add_argument("--per-item-csv", nargs="+", default=None,
                    help="C2 input(s): one or more per_item_scores.csv files "
                         "written by score_rewrites.py. Pass multiple to run "
                         "the multi-method design where each source headline "
                         "is rated once per method under blinded conditions.")
    ap.add_argument("--method-labels", nargs="+", default=None,
                    help="Optional labels for each --per-item-csv entry, in "
                         "the same order. If omitted, the parent directory "
                         "name of each CSV is used (e.g. 'b2', 'b4').")
    ap.add_argument("--out-dir", required=True,
                    help="Root output dir (subdirs c1_* and c2_* will be created)")
    ap.add_argument("--c1-n-items", type=int, default=150,
                    help="C1 sample size")
    ap.add_argument("--c2-n-items", type=int, default=100,
                    help="C2 sample size in DISTINCT source headlines. Each "
                         "source spawns one rewrite per method x condition "
                         "cell present in --per-item-csv, so the actual "
                         "number of rewrites in each rater file is "
                         "c2_n_items x (methods x cells).")
    ap.add_argument("--n-raters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260723)
    args = ap.parse_args()

    if not args.synthetic_csv and not args.per_item_csv:
        raise SystemExit(
            "give at least one of --synthetic-csv (C1) or --per-item-csv (C2)"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic_csv:
        _prepare_c1(Path(args.synthetic_csv), out_dir,
                    n_items=args.c1_n_items, n_raters=args.n_raters,
                    seed=args.seed)

    if args.per_item_csv:
        _prepare_c2(
            [Path(p) for p in args.per_item_csv],
            args.method_labels,
            out_dir,
            n_items=args.c2_n_items, n_raters=args.n_raters,
            seed=args.seed,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
