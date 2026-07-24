"""B6 DExperts: decoding-time controlled rewriting to compare against FUDGE (b2).

Reviewer R2-M4 asked for a DExperts comparison. DExperts (Liu et al. 2021)
steers the base LM's next-token logits with a product-of-experts:

    final_logits = base_logits + alpha * (expert_logits - antiexpert_logits)

Here:
    base       = meta-llama/Meta-Llama-3-8B-Instruct   (the rewrite generator)
    expert     = small Llama-3.2-1B fine-tuned on CLICKBAIT headlines (engagement)
    antiexpert = small Llama-3.2-1B fine-tuned on NEUTRAL source headlines

All three SHARE the Llama-3 tokenizer/vocabulary, which is what makes the
per-token logit ensemble well defined. alpha traces the same fidelity-vs-
engagement tradeoff axis as FUDGE's alpha, so results drop straight into
b2/score_rewrites.py (identical CSV schema).

Decoding is deterministic: we restrict to the base model's top-k candidate set
(standard DExperts filtering, so the anti-expert cannot promote tokens the base
LM would never emit), apply the ensemble within that set, and take greedy
argmax. Single seed; extra seeds only replicate identical rows.

Outputs
    results/b6_dexperts/rewrites_dexperts_a{alpha}_s{seed}.csv
        columns: item_id, seed, alpha, beta, top_k, tactic_ids, tactic_label,
                 neutral, edited   (beta is always 0 for DExperts)
    results/b6_dexperts/generations.jsonl   (crash-safe sidecar)

Heavy torch/transformers imports happen AFTER argparse so --help stays fast.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.artifacts import JsonlWriter
from common.paths import BASE_LLM_ID, RESULTS_DIR, TACTIC_NAMES, ensure_dirs, find_repo_test_csv


def _dexperts_generate(
    tokenizer,
    base,
    expert,
    antiexpert,
    create_prompt,
    neutral: str,
    tactic_ids: list[int],
    alpha: float,
    top_k: int,
    max_tokens: int,
) -> str:
    """One DExperts rewrite with KV cache on all three models.

    final_logits = base_logits + alpha * (expert_logits - antiexpert_logits),
    restricted to the base model's top-k, then greedy argmax.
    """
    import torch

    tactic_names = [TACTIC_NAMES[i] for i in tactic_ids]
    prompt = create_prompt(neutral, tactic_names)
    messages = [
        {"role": "system", "content": "You are an experienced news editor."},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(base.device)

    # Shared vocab; guard against any off-by-one head width mismatch.
    vocab = min(base.config.vocab_size, expert.config.vocab_size,
                antiexpert.config.vocab_size)
    eos_id = tokenizer.eos_token_id

    generated: list[int] = []
    past_b = past_e = past_a = None
    step_b = input_ids
    step_e = input_ids.to(expert.device)
    step_a = input_ids.to(antiexpert.device)

    with torch.no_grad():
        for _ in range(max_tokens):
            ob = base(input_ids=step_b, past_key_values=past_b, use_cache=True)
            oe = expert(input_ids=step_e, past_key_values=past_e, use_cache=True)
            oa = antiexpert(input_ids=step_a, past_key_values=past_a, use_cache=True)
            past_b, past_e, past_a = ob.past_key_values, oe.past_key_values, oa.past_key_values

            base_logits = ob.logits[:, -1, :vocab].float()
            exp_logits = oe.logits[:, -1, :vocab].float().to(base_logits.device)
            anti_logits = oa.logits[:, -1, :vocab].float().to(base_logits.device)

            final_logits = base_logits + alpha * (exp_logits - anti_logits)

            # Restrict the candidate set to the base model's top-k, then argmax
            # the ensemble within it (canonical DExperts top-k filtering).
            top_idx = torch.topk(base_logits[0], k=min(top_k, vocab)).indices
            cand_scores = final_logits[0][top_idx]
            best_id = int(top_idx[int(torch.argmax(cand_scores))].item())

            if best_id == eos_id:
                break
            generated.append(best_id)
            step_b = torch.tensor([[best_id]], device=base.device)
            step_e = torch.tensor([[best_id]], device=expert.device)
            step_a = torch.tensor([[best_id]], device=antiexpert.device)

    text = tokenizer.decode(generated).strip()
    if text and not text.endswith((".", "?", "!")):
        text += "."
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-csv", default=str(find_repo_test_csv()),
                    help="held-out neutral headlines CSV (needs 'title' or 'neutral' column)")
    ap.add_argument("--n-items", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="DExperts decoding is greedy/deterministic; extra seeds "
                         "only replicate identical rows.")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--expert-dir", required=True, help="clickbait expert LM dir (from train_dexperts.py)")
    ap.add_argument("--antiexpert-dir", required=True, help="neutral antiexpert LM dir (from train_dexperts.py)")
    ap.add_argument("--alphas", default="0.5,1,2,4",
                    help="comma-separated DExperts alpha values to sweep.")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR / "b6_dexperts"))
    # Sharding for parallel fan-out: each container processes a disjoint item
    # range and writes to shard-tagged filenames, so K containers can run
    # concurrently against the same volume dir with no collision. item_id stays
    # the GLOBAL row index so scoring/comparison align across shards.
    ap.add_argument("--item-start", type=int, default=0)
    ap.add_argument("--item-end", type=int, default=0,
                    help="exclusive end; 0 means use --n-items from --item-start")
    ap.add_argument("--shard-tag", default="",
                    help="suffix appended to output filenames to keep shards distinct")
    ap.add_argument("--dry-run", action="store_true",
                    help="print planned work then exit without loading models")
    args = ap.parse_args()

    ensure_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(x) for x in str(args.alphas).split(",") if str(x).strip()]

    # create_prompt and the tactic-config list are reused verbatim from the FUDGE
    # runner so the per-tactic breakdown lines up with b2.
    from b2_held_out_rewrite_eval.run_rewrites import DEFAULT_TACTIC_CONFIGS, create_prompt

    import pandas as pd

    df = pd.read_csv(args.test_csv)
    text_col = "title" if "title" in df.columns else ("neutral" if "neutral" in df.columns else df.columns[0])
    # Select the item range. For a full run this is the first n_items. For a
    # shard it is [item_start, item_end). item_id is taken from the GLOBAL row
    # index (no reset_index), so shards emit disjoint item_ids that align with
    # every other method and shard.
    if args.item_end and args.item_end > args.item_start:
        df = df.iloc[args.item_start:args.item_end]
    else:
        df = df.iloc[args.item_start:args.item_start + args.n_items]
    print(f"[b6-dexperts] test set: {args.test_csv} column={text_col} n={len(df)} "
          f"(item_id {df.index.min()}..{df.index.max()})")
    print(f"[b6-dexperts] alphas={alphas} top_k={args.top_k} seeds={args.seeds}")
    print(f"[b6-dexperts] expert={args.expert_dir}")
    print(f"[b6-dexperts] antiexpert={args.antiexpert_dir}")

    total = len(df) * len(alphas) * len(args.seeds) * len(DEFAULT_TACTIC_CONFIGS)
    print(f"[b6-dexperts] planned rewrites: {total}")
    print(f"[b6-dexperts] ETA @ 10 s/rewrite on GPU: {total*10/3600:.1f} h "
          f"(3 model forwards/step vs FUDGE's 1; use gpu2modal/gpu2vast).")

    if args.dry_run:
        return 0

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[b6-dexperts] device = {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_LLM_ID)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_LLM_ID,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).eval()
        expert = AutoModelForCausalLM.from_pretrained(
            args.expert_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        antiexpert = AutoModelForCausalLM.from_pretrained(
            args.antiexpert_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
    except Exception as e:
        print("[b6-dexperts] ERROR loading a model. If this is a gated-model "
              "license error for the Llama-3.2 experts, accept the license at "
              "https://huggingface.co/meta-llama/Llama-3.2-1B (and the base model "
              "license) then re-run.")
        print(f"[b6-dexperts] {type(e).__name__}: {e}")
        return 2

    gen_writer = JsonlWriter(out_dir / "generations.jsonl")
    print(f"[b6-dexperts] raw generations -> {out_dir / 'generations.jsonl'}")

    t0 = time.time()
    n_done = 0

    for seed in args.seeds:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        for alpha in alphas:
            out_csv = out_dir / f"rewrites_dexperts_a{alpha}_s{seed}{args.shard_tag}.csv"

            # Resume: skip (item_id, tactic_label) pairs already in this CSV.
            done_keys: set = set()
            header_needed = True
            if out_csv.exists():
                try:
                    prev = pd.read_csv(out_csv)
                    done_keys = set(zip(prev["item_id"].tolist(),
                                        prev["tactic_label"].tolist()))
                    header_needed = False
                except Exception:
                    done_keys, header_needed = set(), False
            print(f"[b6-dexperts] -> {out_csv} (resume: {len(done_keys)} rows present)")

            with open(out_csv, "a", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                if header_needed:
                    w.writerow(
                        ["item_id", "seed", "alpha", "beta", "top_k",
                         "tactic_ids", "tactic_label", "neutral", "edited"]
                    )
                for i, row in df.iterrows():
                    neutral = str(row[text_col])
                    for tactic_ids, label in DEFAULT_TACTIC_CONFIGS:
                        if (i, label) in done_keys:
                            continue
                        # Per-item guard: one bad headline must not kill the run.
                        failed = False
                        err = None
                        try:
                            edited = _dexperts_generate(
                                tokenizer, base, expert, antiexpert, create_prompt,
                                neutral, tactic_ids, alpha, args.top_k, args.max_tokens,
                            )
                        except Exception as e:
                            print(f"[b6-dexperts] WARN item {i} tactic '{label}' "
                                  f"(alpha={alpha}) FAILED: {type(e).__name__}: {e}",
                                  flush=True)
                            edited = ""
                            failed = True
                            err = f"{type(e).__name__}: {e}"
                        w.writerow(
                            [i, seed, alpha, 0, args.top_k,
                             json.dumps(tactic_ids), label, neutral, edited]
                        )
                        f.flush()
                        gen_writer.write({
                            "item_id": i, "alpha": alpha, "tactic_label": label,
                            "neutral": neutral, "edited": edited,
                            "failed": failed, "error": err,
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        })
                        n_done += 1
                        if n_done % 25 == 0:
                            dt = time.time() - t0
                            rate = n_done / dt
                            eta = (total - n_done) / max(rate, 1e-9)
                            print(f"[b6-dexperts] {n_done}/{total} rate={rate:.2f}/s "
                                  f"ETA={eta/60:.1f} min", flush=True)

    gen_writer.close()
    print(f"[b6-dexperts] done in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
