"""Control-set validation for two things at once:

(c) NEGATIVE CONTROL for the hallucination metric. Generate, per source, two
    KNOWN-FAITHFUL rephrasings -- a plain paraphrase and a rhetorical-QUESTION
    form (the tactic that trips the naive judge) -- plus one KNOWN-UNFAITHFUL
    rephrasing that inserts a fabricated fact. The refined new-fact judge should
    score the faithful ones ~0 and the unfaithful one ~1. If so, the metric is
    validated (it measures facts, not rhetorical form), and we report it as
    calibration -- NOT as an offset subtracted from the score.

(b) NLI model A/B. Score every (source, rephrasing) pair with BOTH
    roberta-large-mnli and the DeBERTa-v3 multi-dataset NLI, using the
    faithful/unfaithful labels as ground truth. The better model has higher
    AUROC separating faithful from unfaithful, and -- critically -- does not
    tank the faithful QUESTION form. Keep whichever wins.

Outputs results/control/control_validation.json (+ per-row jsonl).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.artifacts import JsonlWriter
from score_rewrites import _judge_api, _nli
from rejudge_factuality import _factuality_prompt

ROBERTA = "roberta-large-mnli"
DEBERTA = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

VARIANTS = {
    "faithful_paraphrase": (
        "Rewrite this news headline in different words. Do NOT add, remove, or "
        "change any facts, names, numbers, or events. Return ONLY the rewritten "
        "headline, no quotes.\n\nHeadline: {h}"),
    "faithful_question": (
        "Rewrite this news headline as a single rhetorical or provocative "
        "QUESTION. Do NOT add any new fact, name, number, place, or event that "
        "is not in the original. Return ONLY the rewritten headline, no "
        "quotes.\n\nHeadline: {h}"),
    "unfaithful_fact": (
        "Rewrite this news headline and INSERT exactly one specific fabricated "
        "detail (a fake number, name, place, or event) that is NOT in the "
        "original. Return ONLY the rewritten headline, no quotes.\n\n"
        "Headline: {h}"),
}
IS_FAITHFUL = {"faithful_paraphrase": 1, "faithful_question": 1, "unfaithful_fact": 0}


def _gen(prompt: str) -> str:
    import requests
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"},
        json={"model": os.environ.get("OPENAI_API_MODEL", "gpt-4o-mini"),
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.7},
        timeout=60,
    )
    return r.json()["choices"][0]["message"]["content"].strip().strip('"').strip()


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.src_csv)
    col = "title" if "title" in df.columns else "neutral"
    srcs = df[col].dropna().astype(str).tolist()[:args.n]
    print(f"[control] {len(srcs)} sources x {len(VARIANTS)} variants", flush=True)

    # 1. Generate the control rephrasings (OpenAI), concurrently.
    jobs = [(i, s, v, tmpl.format(h=s)) for i, s in enumerate(srcs)
            for v, tmpl in VARIANTS.items()]

    def _do(job):
        i, s, v, prompt = job
        try:
            return i, s, v, _gen(prompt)
        except Exception as e:
            return i, s, v, f""

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, s, v, edited in ex.map(_do, jobs):
            rows.append({"item_id": i, "variant": v, "is_faithful": IS_FAITHFUL[v],
                         "neutral": s, "edited": edited})
    rows = [r for r in rows if r["edited"].strip()]
    print(f"[control] generated {len(rows)} rephrasings", flush=True)

    # 2. Refined new-fact judge on each (metric validation, part c).
    def _judge(r):
        j = _judge_api(_factuality_prompt(r["neutral"], r["edited"]))
        nf = bool(j.get("new_fact")) if isinstance(j, dict) and "new_fact" in j else None
        return r, nf, j
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        judged = list(ex.map(_judge, rows))
    for r, nf, j in judged:
        r["new_fact"] = nf
        r["added_fact"] = j.get("added_fact") if isinstance(j, dict) else None

    # 3. NLI with both models (part b). entailment = source -> rewrite.
    device = "cpu"
    if args.gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
    pairs = [(r["neutral"], r["edited"]) for r in rows]
    for tag, name in (("roberta", ROBERTA), ("deberta", DEBERTA)):
        print(f"[control] NLI {tag} ({name})...", flush=True)
        ent_fwd, _ = _nli(pairs, device, name=name)
        for r, e in zip(rows, ent_fwd):
            r[f"nli_{tag}"] = float(e)

    # 4. Metrics.
    with JsonlWriter(out_dir / "control_rows.jsonl") as w:
        for r in rows:
            r["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
            w.write(r)

    def rate(vname, key):
        vals = [r[key] for r in rows if r["variant"] == vname and r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    y = np.array([r["is_faithful"] for r in rows])
    summary = {
        "n_rows": len(rows),
        "c_new_fact_rate": {v: rate(v, "new_fact") for v in VARIANTS},
        "b_nli_auroc_faithful_vs_unfaithful": {
            "roberta": _auroc(y, np.array([r["nli_roberta"] for r in rows])),
            "deberta": _auroc(y, np.array([r["nli_deberta"] for r in rows])),
        },
        "b_nli_mean_on_faithful_question": {
            "roberta": float(np.mean([r["nli_roberta"] for r in rows if r["variant"] == "faithful_question"])),
            "deberta": float(np.mean([r["nli_deberta"] for r in rows if r["variant"] == "faithful_question"])),
        },
        "b_nli_mean_on_faithful_paraphrase": {
            "roberta": float(np.mean([r["nli_roberta"] for r in rows if r["variant"] == "faithful_paraphrase"])),
            "deberta": float(np.mean([r["nli_deberta"] for r in rows if r["variant"] == "faithful_paraphrase"])),
        },
    }
    (out_dir / "control_validation.json").write_text(json.dumps(summary, indent=2),
                                                     encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
