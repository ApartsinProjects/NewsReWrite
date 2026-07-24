"""Build prefix training + test CSVs for both guide models in one pass.

Input:
    --headlines-csv  CSV with columns (original, clickbait, methods_vector)
                     where methods_vector is a JSON list of 10 ints.

We split at the SOURCE HEADLINE level (each row of the input file becomes
either a trainval row or a test row, never both). This prevents any prefix
leakage between splits.

For each source headline we emit prefixes at the same fixed length ratios the
published pipeline used (create_prefix_dataset_train_val.py): word-count
ratios PREFIX_RATIOS = [0.3, 0.5, 0.7, 1.0] with MIN_WORDS = 2, and the
full-length (ratio 1.0) prefix gets a trailing period. This keeps the guide
models identical in construction to the paper's guide models, so their
reported per-tactic F1 and clickbait AUROC are directly comparable to the
manuscript's Figure 1. Both the neutral 'original' and the 'clickbait' side
of the row become their own prefix families:
    - the neutral side has label 0 and tactics vector [0]*10
    - the clickbait side has label 1 and tactics vector from methods_vector

Every emitted row carries:
    text, label, tactics_vector, weight, source_id, source_kind

'weight' is length-aware: min(1.0, word_count / 6.0). Short prefixes get
down-weighted so the trainer does not overfit to two-word stubs, implementing
the length-aware weighted loss the paper describes.

Four output CSVs:
    --out-trainval-binary, --out-test-binary       (text, label, weight)
    --out-trainval-tactics, --out-test-tactics     (text, tactics_vector, weight)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Fixed prefix scheme matching the published pipeline. Do not change without
# noting it in the paper: the guide models are only comparable to Figure 1 if
# they are trained on the same prefix construction.
PREFIX_RATIOS = [0.3, 0.5, 0.7, 1.0]
MIN_WORDS = 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--headlines-csv", required=True)
    p.add_argument("--out-trainval-binary", required=True)
    p.add_argument("--out-test-binary", required=True)
    p.add_argument("--out-trainval-tactics", required=True)
    p.add_argument("--out-test-tactics", required=True)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def ratio_prefixes(text: str) -> list[tuple[str, int, int]]:
    """Return [(prefix_text, word_count, is_full), ...] at the paper's fixed
    ratios. is_full is 1 only for the r==1.0 prefix (the complete headline).

    Matches create_prefix_dataset_train_val.py: for each ratio r, take the
    first max(MIN_WORDS, int(N*r)) words; the r==1.0 prefix (the full
    headline) is period-terminated. Duplicate prefixes from adjacent ratios
    on short headlines are kept, exactly as the published script emitted
    them, so the training distribution is reproduced faithfully.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    words = text.strip().split()
    n = len(words)
    out: list[tuple[str, int, int]] = []
    for r in PREFIX_RATIOS:
        k = max(MIN_WORDS, int(n * r))
        k = min(k, n)  # never exceed the headline length
        prefix = " ".join(words[:k])
        is_full = 1 if r == 1.0 else 0
        if is_full and not prefix.endswith((".", "?", "!")):
            prefix = prefix + "."
        out.append((prefix, k, is_full))
    return out


def length_weight(k: int) -> float:
    return min(1.0, k / 6.0)


def emit_rows(source_id: int, kind: str, text: str, label: int, vector: list[int]):
    rows = []
    for prefix, k, is_full in ratio_prefixes(text):
        rows.append({
            "text": prefix,
            "label": label,
            "tactics_vector": json.dumps(vector),
            "weight": length_weight(k),
            "is_full": is_full,
            "source_id": source_id,
            "source_kind": kind,
        })
    return rows


def main() -> None:
    args = parse_args()

    import numpy as np
    import pandas as pd

    df = pd.read_csv(args.headlines_csv)
    required = {"original", "clickbait", "methods_vector"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"headlines CSV missing columns: {missing}")

    # Parse the methods_vector cell whether it was stored as JSON or Python
    # repr. A single malformed cell must not abort the whole build: bad rows
    # are dropped and counted rather than raising.
    def parse_vec(cell):
        if isinstance(cell, list):
            v = cell
        else:
            s = str(cell).strip()
            try:
                v = json.loads(s)
            except Exception:
                try:
                    v = json.loads(s.replace("'", '"'))
                except Exception:
                    return None
        try:
            v = [int(x) for x in v]
        except Exception:
            return None
        return v if len(v) == 10 else None

    df = df.dropna(subset=["original", "clickbait"]).reset_index(drop=True)
    df["methods_vector"] = df["methods_vector"].apply(parse_vec)
    n_before = len(df)
    df = df[df["methods_vector"].notna()].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[build] dropped {n_dropped} rows with malformed methods_vector",
              flush=True)

    # Deduplicate on the clickbait rewrite: GPT occasionally emits the same
    # rewrite for two different source headlines, which (with a source-level
    # split) would otherwise put an identical full headline in both splits.
    n_before = len(df)
    df = df.drop_duplicates(subset=["clickbait"]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"[build] dropped {n_before - len(df)} duplicate-clickbait rows",
              flush=True)
    print(f"[build] loaded {len(df)} source headline rows", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = int(round(len(df) * args.test_frac))
    test_ids = set(idx[:n_test].tolist())
    print(f"[build] split: trainval={len(df) - n_test}  test={n_test}", flush=True)

    # Build trainval first, collect its full-headline (period-terminated)
    # strings, then when building test drop any source row whose neutral OR
    # clickbait full headline already appears in trainval. This guarantees a
    # leakage-free split by construction, even across the neutral/clickbait
    # boundary (a clickbait rewrite that equals a different row's neutral).
    def _full(text):
        # Must match exactly how ratio_prefixes emits the r=1.0 prefix:
        # whitespace-normalized via split()/join, then a trailing period unless
        # it already ends in sentence punctuation. A raw strip() would miss
        # collisions on headlines with double/newline whitespace.
        words = str(text).strip().split()
        p = " ".join(words)
        if not p.endswith((".", "?", "!")):
            p += "."
        return p

    trainval_rows: list[dict] = []
    trainval_full: set = set()
    for source_id, row in df.iterrows():
        if source_id in test_ids:
            continue
        trainval_rows.extend(emit_rows(source_id, "neutral",
                                       row["original"], 0, [0] * 10))
        trainval_rows.extend(emit_rows(source_id, "clickbait",
                                       row["clickbait"], 1, row["methods_vector"]))
        trainval_full.add(_full(row["original"]))
        trainval_full.add(_full(row["clickbait"]))

    test_rows: list[dict] = []
    n_leak_dropped = 0
    for source_id, row in df.iterrows():
        if source_id not in test_ids:
            continue
        if _full(row["original"]) in trainval_full or \
                _full(row["clickbait"]) in trainval_full:
            n_leak_dropped += 1
            continue
        test_rows.extend(emit_rows(source_id, "neutral",
                                   row["original"], 0, [0] * 10))
        test_rows.extend(emit_rows(source_id, "clickbait",
                                   row["clickbait"], 1, row["methods_vector"]))
    if n_leak_dropped:
        print(f"[build] dropped {n_leak_dropped} test rows colliding with "
              f"trainval full headlines (leakage-free split)", flush=True)

    tv = pd.DataFrame(trainval_rows)
    te = pd.DataFrame(test_rows)
    print(f"[build] emitted {len(tv)} trainval prefixes, {len(te)} test prefixes",
          flush=True)

    for path in (
        args.out_trainval_binary, args.out_test_binary,
        args.out_trainval_tactics, args.out_test_tactics,
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    tv[["text", "label", "weight", "is_full"]].to_csv(args.out_trainval_binary, index=False)
    te[["text", "label", "weight", "is_full"]].to_csv(args.out_test_binary, index=False)
    tv[["text", "tactics_vector", "weight", "is_full"]].to_csv(
        args.out_trainval_tactics, index=False)
    te[["text", "tactics_vector", "weight", "is_full"]].to_csv(
        args.out_test_tactics, index=False)
    print("[build] wrote all four CSVs", flush=True)


if __name__ == "__main__":
    main()
