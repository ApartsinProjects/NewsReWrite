"""B4: Prompt-only baseline.

Reviewer concern
----------------
R1: how much of the reported effect is prompt engineering rather than the
FUDGE guides? Answer by generating the same held-out set with plain
Llama-3-8B-Instruct given a single-sentence rewrite instruction, then
scoring with the same b2 scorer.

Prompt (kept deliberately minimal):

    Rewrite the following news headline to be more engaging but not
    clickbait. Preserve all facts. Output only the rewritten headline.

ETA           : ~2 s/item on GPU; 300 items x 3 seeds -> ~30 min on 4090.
Cost          : ~$1 on Modal A10G.

Outputs       : results/b4/rewrites_prompt_only_s{seed}.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.artifacts import JsonlWriter
from common.paths import BASE_LLM_ID, RESULTS_DIR, ensure_dirs, find_repo_test_csv

PROMPT = (
    "Rewrite the following news headline to be more engaging but not clickbait. "
    "Preserve all facts. Output only the rewritten headline."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-csv", default=str(find_repo_test_csv()))
    ap.add_argument("--n-items", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out-dir", default=str(RESULTS_DIR / "b4"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Crash-safe per-datapoint sidecar: one flushed JSONL line per generation
    # (including per-item failures), opened once and appended across seeds.
    gen_writer = JsonlWriter(out_dir / "generations.jsonl")
    print(f"[b4] raw generations -> {out_dir / 'generations.jsonl'}")

    import pandas as pd

    df = pd.read_csv(args.test_csv)
    text_col = "title" if "title" in df.columns else ("neutral" if "neutral" in df.columns else df.columns[0])
    df = df.head(args.n_items).reset_index(drop=True)
    total = len(df) * len(args.seeds)
    print(f"[b4] planned generations: {total}")
    print(f"[b4] ETA GPU 2 s/item: {total*2/60:.1f} min; CPU 10 s/item: {total*10/60:.1f} min")

    if args.dry_run:
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE_LLM_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_LLM_ID,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).eval()

    t0 = time.time(); done = 0
    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        out_csv = out_dir / f"rewrites_prompt_only_s{seed}.csv"
        print(f"[b4] -> {out_csv}")
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "seed", "alpha", "beta", "top_k",
                        "tactic_ids", "tactic_label", "neutral", "edited"])
            for i, row in df.iterrows():
                neutral = str(row[text_col])
                # Per-item guard: a single bad headline must not kill the run;
                # log it, write an empty rewrite (scoring flags it and excludes
                # it), and continue.
                failed = False
                err = None
                try:
                    messages = [
                        {"role": "system", "content": "You are an experienced news editor."},
                        {"role": "user", "content": f"{PROMPT}\n\nHeadline: {neutral}"},
                    ]
                    ids = tok.apply_chat_template(
                        messages, add_generation_prompt=True, return_tensors="pt"
                    ).to(model.device)
                    with torch.no_grad():
                        out = model.generate(
                            ids,
                            max_new_tokens=args.max_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=tok.eos_token_id,
                        )
                    edited = tok.decode(out[0, ids.shape[-1] :], skip_special_tokens=True).strip()
                except Exception as e:
                    print(f"[b4] WARN item {i} FAILED: {type(e).__name__}: {e}", flush=True)
                    edited = ""
                    failed = True
                    err = f"{type(e).__name__}: {e}"
                w.writerow([i, seed, "prompt_only", "prompt_only", 0,
                            "[]", "prompt_only", neutral, edited])
                f.flush()
                gen_writer.write({
                    "item_id": i, "seed": seed, "method": "prompt_only",
                    "neutral": neutral, "edited": edited, "failed": failed,
                    "error": err, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                done += 1
                if done % 25 == 0:
                    rate = done / (time.time() - t0)
                    eta = (total - done) / max(rate, 1e-9)
                    print(f"[b4] {done}/{total} rate={rate:.2f}/s ETA={eta/60:.1f} min")
    gen_writer.close()
    print(f"[b4] done in {(time.time()-t0)/60:.1f} min")
    print("[b4] score with: python b2_held_out_rewrite_eval/score_rewrites.py --in-dir results/b4 "
          "--out-json results/b4/summary.json --out-md results/b4/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
