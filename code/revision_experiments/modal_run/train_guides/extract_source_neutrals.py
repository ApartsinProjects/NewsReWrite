"""Extract source_neutrals.csv from ISOT True.csv.

Reads the ISOT True.csv Reuters half (columns: title, text, subject, date) and
emits a compact CSV with just what the downstream synthetic-generation and
rewrite scripts consume: a title column plus lightweight provenance.

Output schema
-------------
    source_id       stable identifier: isot_{row_index}
    title           the neutral headline (deduplicated, whitespace-normalised)
    subject         ISOT topic tag (politicsNews / worldnews)
    date            original publish date (kept for downstream stratification)

Deduplication is exact on the normalised title. Rows with an empty title after
stripping are dropped. If --max-rows N is given, an even sample across subjects
is taken via pandas `groupby(subject).sample`.

Usage
-----
    python extract_source_neutrals.py \
        --isot-csv /workspace/store/data/isot/True.csv \
        --out-csv  /workspace/store/data/source_neutrals.csv \
        [--max-rows 12600] [--seed 42]
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--isot-csv", required=True,
                    help="Path to ISOT True.csv")
    ap.add_argument("--out-csv", required=True,
                    help="Where to write source_neutrals.csv")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="If set, subsample stratified by subject (default: keep all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pandas as pd  # local import so --help works without pandas

    in_path = Path(args.isot_csv)
    out_path = Path(args.out_csv)
    if not in_path.exists():
        raise SystemExit(f"missing input: {in_path}")

    print(f"[extract] reading {in_path}")
    df = pd.read_csv(in_path)
    print(f"[extract] read {len(df)} rows, columns={df.columns.tolist()}")

    if "title" not in df.columns:
        raise SystemExit("input must have a 'title' column")

    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates(subset=["title"])
    print(f"[extract] deduplicated {before} -> {len(df)} rows")

    if args.max_rows and len(df) > args.max_rows:
        if "subject" in df.columns:
            n_subj = df["subject"].nunique()
            per_subj = max(1, args.max_rows // n_subj)
            df = (
                df.groupby("subject", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), per_subj),
                                            random_state=args.seed))
                  .reset_index(drop=True)
            )
            print(f"[extract] stratified subsample to {len(df)} rows "
                  f"({per_subj} per subject across {n_subj} subjects)")
        else:
            df = df.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)
            print(f"[extract] random subsample to {len(df)} rows")

    df = df.reset_index(drop=True)
    df["source_id"] = [f"isot_{i:06d}" for i in range(len(df))]

    keep = ["source_id", "title"]
    for c in ("subject", "date"):
        if c in df.columns:
            keep.append(c)
    out = df[keep]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[extract] wrote {out_path}  ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
