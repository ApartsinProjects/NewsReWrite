# -*- coding: utf-8 -*-
"""Ingest filled human annotation workbooks into per-study rater CSVs.

Each annotator returns one filled copy of rater_annotation_TEMPLATE.xlsx (with
the two sheets C1_tactic_labelling and C2_rewrite_quality). This adapter splits
every workbook into the two per-study rater_NN.csv files that analyze.py reads,
renames the C1 tactic columns to the `tactic__` prefix analyze.py expects, and
stamps rater_source=human.

Usage
-----
  python ingest_workbooks.py results/human_labeling/incoming/
  python ingest_workbooks.py results/human_labeling/incoming/ --dry-run

Then, per study:
  python analyze.py --study-dir results/human_labeling/c1_rubric_validation
  python analyze.py --study-dir results/human_labeling/c2_rewrite_quality
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

# repo root = three levels up from this file (.../code/revision_experiments/human_labeling_prep)
REPO = Path(__file__).resolve().parents[3]
HL = REPO / "results" / "human_labeling"

BARE_TACTICS = [
    "curiosity_gap", "exaggeration", "emotional_trigger", "sensationalism",
    "lists_or_superlatives", "ambiguous_references", "direct_appeals",
    "unfinished_narratives", "unexpected_associations", "provocative_questions",
]
TACTIC_RENAME = {b: f"tactic__{b}" for b in BARE_TACTICS}
LIKERT_COLS = ["engagement_1_to_5", "faithfulness_1_to_5", "clickbait_1_to_5"]

C1_SHEET, C2_SHEET = "C1_tactic_labelling", "C2_rewrite_quality"
C1_DIR = HL / "c1_rubric_validation"
C2_DIR = HL / "c2_rewrite_quality"


def _validate(df, cols, name, valid, warnings):
    """Report blank / out-of-range annotation cells (does not abort)."""
    for c in cols:
        if c not in df.columns:
            warnings.append(f"  [{name}] missing column '{c}'")
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        n_blank = int(s.isna().sum())
        n_bad = int((~s.isna() & ~s.isin(valid)).sum())
        if n_blank:
            warnings.append(f"  [{name}] '{c}': {n_blank} blank cell(s)")
        if n_bad:
            warnings.append(f"  [{name}] '{c}': {n_bad} out-of-range value(s) (allowed {sorted(valid)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="folder of filled rater_annotation_*.xlsx workbooks")
    ap.add_argument("--dry-run", action="store_true", help="validate and report, write nothing")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_absolute():
        in_dir = REPO / in_dir
    books = sorted(p for p in in_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if not books:
        raise SystemExit(f"no *.xlsx workbooks in {in_dir}")

    print(f"Found {len(books)} workbook(s) in {in_dir}")
    if not args.dry_run:
        C1_DIR.mkdir(parents=True, exist_ok=True)
        C2_DIR.mkdir(parents=True, exist_ok=True)

    mapping, all_warnings = [], []
    for i, book in enumerate(books, start=1):
        rid = f"rater_{i:02d}"
        warnings = []
        # --- C1 ---
        c1 = pd.read_excel(book, sheet_name=C1_SHEET)
        _validate(c1, BARE_TACTICS, f"{book.name}:C1", {0, 1}, warnings)
        c1 = c1.rename(columns=TACTIC_RENAME)
        c1["rater_source"] = "human"
        if "rater_notes" not in c1.columns:
            c1["rater_notes"] = ""
        # --- C2 ---
        c2 = pd.read_excel(book, sheet_name=C2_SHEET)
        _validate(c2, LIKERT_COLS, f"{book.name}:C2", {1, 2, 3, 4, 5}, warnings)
        if "notes" in c2.columns and "rater_notes" not in c2.columns:
            c2 = c2.rename(columns={"notes": "rater_notes"})
        c2["rater_source"] = "human"

        mapping.append({"rater_id": rid, "source_workbook": book.name,
                        "c1_rows": len(c1), "c2_rows": len(c2)})
        status = "DRY-RUN" if args.dry_run else "written"
        print(f"  {rid}  <- {book.name}   C1={len(c1)}  C2={len(c2)}  [{status}]")
        for w in warnings:
            print(w)
        all_warnings += warnings

        if not args.dry_run:
            c1.to_csv(C1_DIR / f"{rid}.csv", index=False)
            c2.to_csv(C2_DIR / f"{rid}.csv", index=False)

    if not args.dry_run:
        pd.DataFrame(mapping).to_csv(HL / "rater_workbook_mapping.csv", index=False)
        print(f"\nWrote {len(books)} rater file(s) to each study dir; "
              f"mapping -> {HL / 'rater_workbook_mapping.csv'}")
        print("Next:")
        print("  python code/revision_experiments/human_labeling_prep/analyze.py "
              "--study-dir results/human_labeling/c1_rubric_validation")
        print("  python code/revision_experiments/human_labeling_prep/analyze.py "
              "--study-dir results/human_labeling/c2_rewrite_quality")

    if all_warnings:
        print(f"\n{len(all_warnings)} validation warning(s) above "
              f"(blank/out-of-range cells) — review before analysis.", file=sys.stderr)


if __name__ == "__main__":
    main()
