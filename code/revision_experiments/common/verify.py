"""Step-level sanity checks for the Track B pipeline.

Every expensive Modal step calls one of these verifiers on its OWN output
before returning, and the next step calls the matching `require_*` on its
INPUT before doing any GPU work. A failed check raises SystemExit with a
specific message, so the pipeline halts at the first bad artifact instead
of propagating silently.

Design goals:
- Cheap: only reads headers / row counts / value ranges, never reloads models.
- Specific: the message names the file, the expectation, and what was seen.
- Range-aware: catches the failure modes that do not raise on their own,
  e.g. a CSV that exists but has 0 rows, an all-NaN metric column, a
  probability column outside [0, 1], a tactic vector of the wrong width.
"""
from __future__ import annotations

import json
from pathlib import Path


class CheckError(SystemExit):
    """Raised when a sanity check fails. Subclass of SystemExit so it
    halts a Modal function with a non-zero exit."""


def _fail(msg: str):
    raise CheckError(f"[verify] FAIL: {msg}")


def _ok(msg: str):
    print(f"[verify] OK: {msg}", flush=True)


def require_file(path, min_bytes: int = 1):
    p = Path(path)
    if not p.exists():
        _fail(f"missing file: {p}")
    size = p.stat().st_size
    if size < min_bytes:
        _fail(f"file too small ({size} B < {min_bytes} B): {p}")
    _ok(f"{p} present ({size/1024:.1f} KB)")
    return p


def _read_csv(path):
    import pandas as pd
    return pd.read_csv(path)


# ------------------------------------------------------------------
# Dataset checks
# ------------------------------------------------------------------
def check_source_neutrals(path, min_rows: int = 100):
    df = _read_csv(require_file(path))
    if "title" not in df.columns:
        _fail(f"source_neutrals missing 'title' column: {list(df.columns)}")
    n = len(df)
    if n < min_rows:
        _fail(f"source_neutrals has {n} rows < {min_rows}")
    empties = int(df["title"].isna().sum() + (df["title"].astype(str).str.strip() == "").sum())
    if empties:
        _fail(f"source_neutrals has {empties} empty titles")
    dups = int(df["title"].duplicated().sum())
    _ok(f"source_neutrals: {n} rows, 0 empty, {dups} duplicate titles")
    return df


def check_synthetic(path, source_path=None, min_rows: int = 100):
    df = _read_csv(require_file(path))
    need = {"original", "clickbait", "methods_vector"}
    missing = need - set(df.columns)
    if missing:
        _fail(f"synthetic missing columns {missing}: {list(df.columns)}")
    n = len(df)
    if n < min_rows:
        _fail(f"synthetic has {n} rows < {min_rows}")

    # Vector width + activation-count invariant (1..3 ones per row).
    bad_width = bad_count = 0
    for v in df["methods_vector"]:
        try:
            vec = json.loads(v) if isinstance(v, str) else list(v)
        except Exception:
            bad_width += 1
            continue
        if len(vec) != 10:
            bad_width += 1
        elif not (1 <= sum(int(x) for x in vec) <= 3):
            bad_count += 1
    if bad_width:
        _fail(f"synthetic: {bad_width} rows with malformed / non-10 methods_vector")
    if bad_count:
        _fail(f"synthetic: {bad_count} rows whose vector does not have 1..3 ones")

    # Empty generations.
    empty_cb = int((df["clickbait"].astype(str).str.strip() == "").sum() + df["clickbait"].isna().sum())
    if empty_cb:
        _fail(f"synthetic: {empty_cb} empty clickbait rewrites")

    # Coverage vs source (catches the resume desync failure mode).
    if source_path is not None and Path(source_path).exists():
        src = _read_csv(source_path)
        n_src = len(src)
        cov = n / max(n_src, 1)
        dup_orig = int(df["original"].duplicated().sum())
        if dup_orig > 0.02 * n:
            _fail(f"synthetic: {dup_orig} duplicated 'original' headlines "
                  f"({dup_orig/n:.1%}); resume desync suspected")
        _ok(f"synthetic: {n} rows, coverage {cov:.1%} of {n_src} sources, "
            f"{dup_orig} duplicate originals")
    else:
        _ok(f"synthetic: {n} rows, all vectors valid, no empty rewrites")
    return df


def check_prefix_split(trainval_path, test_path, kind: str):
    tv = _read_csv(require_file(trainval_path))
    te = _read_csv(require_file(test_path))
    if "text" not in tv.columns:
        _fail(f"{kind} trainval missing 'text' column: {list(tv.columns)}")
    if len(tv) == 0 or len(te) == 0:
        _fail(f"{kind} split empty: trainval={len(tv)}, test={len(te)}")

    # Leakage guard: no full source headline shared across splits. Prefer the
    # explicit is_full column (marks the ratio-1.0 complete headline); the
    # endswith heuristic is only a fallback and misfires on abbreviation
    # prefixes like "U.S." or "Mr." that legitimately recur across splits.
    if "is_full" in tv.columns and "is_full" in te.columns:
        full_tv = set(tv.loc[tv["is_full"] == 1, "text"])
        full_te = set(te.loc[te["is_full"] == 1, "text"])
    else:
        def _is_full(s):
            return s.astype(str).str.endswith((".", "?", "!"))
        full_tv = set(tv.loc[_is_full(tv["text"]), "text"])
        full_te = set(te.loc[_is_full(te["text"]), "text"])
    leak = full_tv & full_te
    if leak:
        _fail(f"{kind} LEAKAGE: {len(leak)} full headlines in both trainval and test")

    # Label sanity.
    if kind == "binary" and "label" in tv.columns:
        classes = sorted(set(tv["label"].unique()))
        if set(classes) != {0, 1}:
            _fail(f"binary trainval labels not {{0,1}}: {classes}")
        pos = float((tv["label"] == 1).mean())
        if not (0.1 < pos < 0.9):
            _fail(f"binary trainval class balance suspicious: {pos:.1%} positive")
    _ok(f"{kind} split: trainval={len(tv)}, test={len(te)}, no full-headline leakage")
    return tv, te


# ------------------------------------------------------------------
# Model checks
# ------------------------------------------------------------------
def check_model_dir(path, expect_labels: int | None = None):
    p = Path(path)
    cfg = p / "config.json"
    require_file(cfg, min_bytes=10)
    has_weights = any((p / w).exists() for w in
                      ("model.safetensors", "pytorch_model.bin"))
    if not has_weights:
        _fail(f"model dir has config but no weights: {p}")
    if expect_labels is not None:
        conf = json.loads(cfg.read_text())
        n = conf.get("num_labels") or len(conf.get("id2label", {})) or None
        if n is not None and n != expect_labels:
            _fail(f"model {p} has num_labels={n}, expected {expect_labels}")
    _ok(f"model dir {p} has config + weights"
        + (f" (num_labels={expect_labels})" if expect_labels else ""))
    return p


# ------------------------------------------------------------------
# Result checks
# ------------------------------------------------------------------
def check_rewrites(out_dir, expect_min_files: int = 1):
    d = Path(out_dir)
    files = sorted(d.glob("rewrites_*.csv"))
    if len(files) < expect_min_files:
        _fail(f"expected >= {expect_min_files} rewrites_*.csv in {d}, found {len(files)}")
    total = 0
    empty_edits = 0
    for f in files:
        df = _read_csv(f)
        for col in ("neutral", "edited"):
            if col not in df.columns:
                _fail(f"{f} missing '{col}' column")
        total += len(df)
        empty_edits += int((df["edited"].astype(str).str.strip() == "").sum()
                           + df["edited"].isna().sum())
    if total == 0:
        _fail(f"rewrites in {d} contain 0 rows")
    frac_empty = empty_edits / max(total, 1)
    if frac_empty > 0.05:
        _fail(f"{frac_empty:.1%} of rewrites are empty in {d} "
              f"({empty_edits}/{total}); generation likely broke")
    _ok(f"rewrites {d}: {len(files)} files, {total} rows, {empty_edits} empty edits")


def check_per_item_scores(path, min_rows: int = 1):
    df = _read_csv(require_file(path))
    n = len(df)
    if n < min_rows:
        _fail(f"per_item_scores has {n} rows < {min_rows}")

    # Probability-like columns must sit in [0, 1] (ignoring NaN). NOTE: sts is
    # a COSINE similarity in [-1, 1], NOT a probability -- strongly divergent
    # rewrites (e.g. heavy GeDi/DExperts drift) legitimately go slightly
    # negative, so it is checked on [-1, 1] separately, not here.
    prob_cols = [c for c in ("clickbait_prob", "clickbait_prob_external",
                             "attr_realised_frac", "attr_realised_frac_guide_circular",
                             "nli_neutral_entails_edited",
                             "nli_edited_entails_neutral") if c in df.columns]
    for c in prob_cols:
        s = df[c].dropna()
        if len(s) and (s.min() < -1e-6 or s.max() > 1 + 1e-6):
            _fail(f"{c} outside [0,1]: min={s.min()}, max={s.max()}")
    if "sts" in df.columns:
        s = df["sts"].dropna()
        if len(s) and (s.min() < -1 - 1e-6 or s.max() > 1 + 1e-6):
            _fail(f"sts (cosine) outside [-1,1]: min={s.min()}, max={s.max()}")

    # A fully-NaN metric column means a scorer silently failed.
    metric_cols = [c for c in ("sts", "bertscore_f1", "clickbait_prob",
                               "attr_realised_frac") if c in df.columns]
    for c in metric_cols:
        if df[c].isna().all():
            _fail(f"metric column '{c}' is entirely NaN; scorer failed")
    _ok(f"per_item_scores: {n} rows, metric columns in range, none all-NaN")
    return df


def check_json_report(path, required_keys=()):
    p = require_file(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    for k in required_keys:
        if k not in obj:
            _fail(f"{p} missing key '{k}'")
    _ok(f"{p} valid JSON with keys {list(obj.keys())[:6]}")
    return obj


def check_human_labeling(out_dir):
    d = Path(out_dir)
    found = []
    for study in ("c1_rubric_validation", "c2_rewrite_quality"):
        sd = d / study
        if not sd.exists():
            continue
        oracle = sd / "oracle.csv"
        if not oracle.exists():
            _fail(f"{study}: missing oracle.csv")
        raters = sorted(sd.glob("rater_*.csv"))
        if not raters:
            _fail(f"{study}: no rater files")
        odf = _read_csv(oracle)
        rdf = _read_csv(raters[0])
        if len(rdf) != len(odf):
            _fail(f"{study}: rater file has {len(rdf)} rows, oracle has {len(odf)}")
        # Rater files must NOT leak ground truth.
        leak_cols = [c for c in rdf.columns
                     if c.startswith(("auto_", "gen_", "intended_", "condition_", "method"))]
        if leak_cols:
            _fail(f"{study}: rater file leaks ground-truth columns {leak_cols}")
        found.append(f"{study}({len(odf)} items, {len(raters)} raters)")
    if not found:
        _fail(f"no human-labeling studies produced under {d}")
    _ok("human labeling: " + ", ".join(found))
