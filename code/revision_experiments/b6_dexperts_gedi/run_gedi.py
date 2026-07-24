"""B6: GeDi decoding-time controlled generation, baseline vs the paper's FUDGE.

Reviewer R2-M4 asked for a GeDi comparison. GeDi (Krause et al. 2021) steers a
large base LM with a small class-conditional LM. At each decoding step, for each
candidate next token GeDi computes, via Bayes' rule, the posterior probability
that the token belongs to the DESIRED class, and adds omega * log P(desired) to
the base LM log-prob. Here the desired class is 'clickbait' (the engagement-
bearing class); omega traces the control/fluency tradeoff, comparable to FUDGE's
alpha.

GeDi Bayes rule (this script)
-----------------------------
Let the class-conditional LM (trained by train_gedi.py) be run twice on the
running rewrite, once under each literal control-code prefix:

    lp_d(v) = log P_gedi(v | code_desired + rewrite)   # next-token log-prob
    lp_o(v) = log P_gedi(v | code_other   + rewrite)

For a candidate token v the class posterior comes from a softmax over the two
class-conditional next-token log-probs (uniform class prior cancels):

    log P(desired | rewrite, v) = log_softmax([lp_d(v), lp_o(v)])[desired]
                                = lp_d(v) - logsumexp(lp_d(v), lp_o(v))

The re-scored candidate score is:

    score(v) = log P_base(v | prompt, rewrite) + omega * log P(desired | rewrite, v)

Greedy argmax over the base top-k candidates. Deterministic, single seed.

Output schema (SHARED with run_rewrites.py so score_rewrites.py ingests it
unchanged):
    item_id, seed, alpha, beta, top_k, tactic_ids, tactic_label, neutral, edited
omega is written into the 'alpha' column; 'beta' is always 0.

Heavy torch/transformers imports happen AFTER argparse so `--help` is cheap.
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
from common.paths import (
    BASE_LLM_ID,
    RESULTS_DIR,
    TACTIC_NAMES,
    ensure_dirs,
    find_repo_test_csv,
)

# The three tactic configs that shipped in the FUDGE results CSVs; reused here
# so the GeDi and FUDGE outputs cover the identical (item, tactic) cells.
DEFAULT_TACTIC_CONFIGS = [
    ([0, 9], "Curiosity Gap + Provocative Questions"),
    ([2, 3], "Emotional Trigger + Sensationalism"),
    ([3, 9], "Sensationalism + Question"),
]

# Fallback control-code strings, overridden by gedi_control_codes.json if present.
DEFAULT_CODES = {"clickbait": "<clickbait> ", "neutral": "<neutral> "}


def create_prompt(neutral_headline: str, tactics: list[str]) -> str:
    """Identical rewrite prompt to run_rewrites.py (kept in sync verbatim)."""
    joined = ", ".join(tactics)
    return f"""

  Rewrite the following neutral news headline.
  The wording may be lightly influenced by the following style: {joined}.
  The influence should be subtle.

IMPORTANT:
- Do NOT mention the name of the tactic in the output.
- The tactic must be expressed only through wording style and tone.

STRICT RULES:
- Do NOT add any new facts
- Do NOT add any new people, places, or events
- Do NOT change the meaning
- Do NOT introduce claims not present in the original headline
- Only rephrase the wording

Output:
One headline only. No explanations.

Neutral headline:

{neutral_headline}
  """


def _load_control_codes(gedi_dir: Path) -> dict:
    p = gedi_dir / "gedi_control_codes.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return {"clickbait": d["clickbait"], "neutral": d["neutral"]}
        except Exception:
            pass
    return dict(DEFAULT_CODES)


def _gedi_generate(
    llm_tok,
    llm,
    gedi,
    device: str,
    neutral: str,
    tactic_names: list[str],
    code_desired: str,
    code_other: str,
    omega: float,
    top_k: int,
    max_tokens: int,
) -> str:
    """GeDi decoding for one (neutral, tactic_config, omega) cell.

    Base LM is KV-cached. The GeDi class-conditional LM is recomputed each step
    over the short running rewrite under both control-code prefixes (batched as
    two rows in one forward).
    """
    import torch

    prompt = create_prompt(neutral, tactic_names)
    messages = [
        {"role": "system", "content": "You are an experienced news editor."},
        {"role": "user", "content": prompt},
    ]
    input_ids = llm_tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(llm.device)

    @torch.no_grad()
    def gedi_class_logprobs(running_text: str):
        # Two class-conditional next-token distributions over the shared vocab.
        seqs = [code_desired + running_text, code_other + running_text]
        enc = gedi_tok(seqs, return_tensors="pt", padding=True,
                       truncation=True, max_length=64).to(device)
        out = gedi(**enc)
        # Last real (non-pad) position for each row via the attention mask.
        last = enc["attention_mask"].sum(dim=1) - 1
        logits = out.logits[torch.arange(2), last, :]      # [2, vocab]
        lp = torch.log_softmax(logits, dim=-1)             # [2, vocab]
        return lp[0], lp[1]                                # desired, other

    gedi_tok = llm_tok  # shared tokenizer (Llama-3 128k)
    generated: list[int] = []
    past = None
    step_input = input_ids
    with torch.no_grad():
        for _ in range(max_tokens):
            out = llm(input_ids=step_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]
            base_lp = torch.log_softmax(next_logits, dim=-1)[0]     # [vocab]
            top_vals, top_idx = torch.topk(next_logits, k=top_k, dim=-1)
            top_idx = top_idx[0]

            running_text = llm_tok.decode(
                generated, clean_up_tokenization_spaces=False)
            lp_d_all, lp_o_all = gedi_class_logprobs(running_text)

            # Bayes rule per candidate token: posterior of the DESIRED class is
            # a 2-class log_softmax over [lp_desired, lp_other] (prior cancels).
            lp_d = lp_d_all[top_idx]                                 # [top_k]
            lp_o = lp_o_all[top_idx]                                 # [top_k]
            stacked = torch.stack([lp_d, lp_o], dim=0)               # [2, top_k]
            post_desired = torch.log_softmax(stacked, dim=0)[0]      # [top_k]

            cand_scores = base_lp[top_idx] + omega * post_desired
            best_id = int(top_idx[int(torch.argmax(cand_scores).item())].item())

            if best_id == llm_tok.eos_token_id:
                break
            generated.append(best_id)
            step_input = torch.tensor([[best_id]], device=llm.device)

    text = llm_tok.decode(generated).strip()
    if text and not text.endswith((".", "?", "!")):
        text += "."
    return text


def _load_models(gedi_dir: Path, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        llm_tok = AutoTokenizer.from_pretrained(BASE_LLM_ID)
        llm = AutoModelForCausalLM.from_pretrained(
            BASE_LLM_ID,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).eval()
    except Exception as e:
        print(f"[gedi] FAILED to load gated base '{BASE_LLM_ID}': "
              f"{type(e).__name__}: {e}", flush=True)
        print("[gedi] request access and log in (huggingface-cli login). See "
              "https://huggingface.co/meta-llama/Llama-3.2-1B", flush=True)
        raise SystemExit(2)

    try:
        gedi = AutoModelForCausalLM.from_pretrained(
            str(gedi_dir),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
    except Exception as e:
        print(f"[gedi] FAILED to load GeDi LM from '{gedi_dir}': "
              f"{type(e).__name__}: {e}", flush=True)
        print("[gedi] train it first with train_gedi.py. The base model is "
              "gated: https://huggingface.co/meta-llama/Llama-3.2-1B", flush=True)
        raise SystemExit(2)

    if llm_tok.pad_token is None:
        llm_tok.pad_token = llm_tok.eos_token
    return llm_tok, llm, gedi


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-csv", default=str(find_repo_test_csv()),
                    help="held-out neutral headlines CSV (needs 'title' or 'neutral')")
    ap.add_argument("--n-items", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--gedi-dir", required=True,
                    help="dir of the class-conditional LM from train_gedi.py")
    ap.add_argument("--omegas", default="5,10,20,30",
                    help="comma-separated GeDi strengths (written to alpha column)")
    ap.add_argument("--desired-class", default="clickbait",
                    choices=["clickbait", "neutral"])
    ap.add_argument("--out-dir", default=str(RESULTS_DIR / "b6_gedi"))
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

    gedi_dir = Path(args.gedi_dir)
    codes = _load_control_codes(gedi_dir)
    other_class = "neutral" if args.desired_class == "clickbait" else "clickbait"
    code_desired = codes[args.desired_class]
    code_other = codes[other_class]
    omegas = [float(x) for x in args.omegas.split(",") if x.strip()]

    import pandas as pd

    df = pd.read_csv(args.test_csv)
    text_col = "title" if "title" in df.columns else (
        "neutral" if "neutral" in df.columns else df.columns[0])
    # Select the item range. For a full run this is the first n_items. For a
    # shard it is [item_start, item_end). item_id is taken from the GLOBAL row
    # index (no reset_index), so shards emit disjoint item_ids that align with
    # every other method and shard.
    if args.item_end and args.item_end > args.item_start:
        df = df.iloc[args.item_start:args.item_end]
    else:
        df = df.iloc[args.item_start:args.item_start + args.n_items]

    total = len(df) * len(DEFAULT_TACTIC_CONFIGS) * len(args.seeds) * len(omegas)
    print(f"[gedi] test set: {args.test_csv} column={text_col} n={len(df)} "
          f"(item_id {df.index.min()}..{df.index.max()})")
    print(f"[gedi] desired='{args.desired_class}' code={code_desired!r} "
          f"other='{other_class}' code={code_other!r}")
    print(f"[gedi] omegas={omegas} seeds={args.seeds} top_k={args.top_k}")
    print(f"[gedi] planned rewrites: {total}")
    print(f"[gedi] ETA @ 10 s/rewrite on GPU: {total*10/3600:.1f} h")

    if args.dry_run:
        return 0

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gedi] device = {device}")

    llm_tok, llm, gedi = _load_models(gedi_dir, device)

    gen_writer = JsonlWriter(out_dir / "generations.jsonl")
    print(f"[gedi] raw generations -> {out_dir / 'generations.jsonl'}")

    t0 = time.time()
    n_done = 0

    for seed in args.seeds:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        for omega in omegas:
            out_csv = out_dir / f"rewrites_gedi_o{omega}_s{seed}{args.shard_tag}.csv"

            # Resume: skip (item_id, tactic_label) pairs already in this CSV.
            done_keys: set = set()
            header_needed = not out_csv.exists()
            if out_csv.exists():
                try:
                    prev = pd.read_csv(out_csv)
                    done_keys = set(zip(prev["item_id"].tolist(),
                                        prev["tactic_label"].tolist()))
                except Exception:
                    done_keys = set()
            print(f"[gedi] -> {out_csv} (resume: {len(done_keys)} rows present)")

            with open(out_csv, "a", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                if header_needed:
                    w.writerow(
                        ["item_id", "seed", "alpha", "beta", "top_k",
                         "tactic_ids", "tactic_label", "neutral", "edited"])
                for i, row in df.iterrows():
                    neutral = str(row[text_col])
                    for tactic_ids, label in DEFAULT_TACTIC_CONFIGS:
                        if (i, label) in done_keys:
                            continue
                        tactic_names = [TACTIC_NAMES[t] for t in tactic_ids]
                        failed = False
                        err = None
                        try:
                            edited = _gedi_generate(
                                llm_tok, llm, gedi, device, neutral,
                                tactic_names, code_desired, code_other,
                                omega, args.top_k, args.max_tokens)
                        except SystemExit:
                            raise
                        except Exception as e:
                            print(f"[gedi] WARN item {i} tactic '{label}' "
                                  f"(omega={omega}) FAILED: "
                                  f"{type(e).__name__}: {e}", flush=True)
                            edited = ""
                            failed = True
                            err = f"{type(e).__name__}: {e}"
                        # omega -> alpha column, beta always 0 (shared schema).
                        w.writerow(
                            [i, seed, omega, 0, args.top_k,
                             json.dumps(tactic_ids), label, neutral, edited])
                        f.flush()
                        gen_writer.write({
                            "item_id": i, "omega": omega, "tactic_label": label,
                            "neutral": neutral, "edited": edited,
                            "failed": failed, "error": err,
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        })
                        n_done += 1
                        if n_done % 25 == 0:
                            dt = time.time() - t0
                            rate = n_done / dt
                            eta = (total - n_done) / max(rate, 1e-9)
                            print(f"[gedi] {n_done}/{total} rate={rate:.2f}/s "
                                  f"ETA={eta/60:.1f} min", flush=True)

    gen_writer.close()
    print(f"[gedi] done in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
