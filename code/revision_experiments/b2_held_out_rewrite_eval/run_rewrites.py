"""B2: Run FUDGE rewrites on a held-out neutral test set, 4-cell ablation grid.

Grid cells (alpha, beta):
    (0.0, 0.0)  no guidance         -- pure Llama baseline
    (0.7, 0.0)  positive only       -- tactics push, no clickbait brake
    (0.0, 0.7)  negative only       -- clickbait brake, no push
    (0.7, 0.7)  full FUDGE          -- the paper's headline config

Reviewer concerns
-----------------
- R1: paper reports only positive alpha,beta cells. This gives us a real
  ablation and lets us report the delta each guide contributes.
- R2: paper's engagement/clickbait numbers came from a small,
  cherry-illustrated set. Here we run N=300 held-out neutrals so we can
  attach a CI to every table cell.

Decoding is greedy (argmax) by default, which is DETERMINISTIC: extra seeds
would only replicate identical rows, so replication is over ITEMS via the
bootstrap in score_rewrites.py, not over seeds. The default --seeds is a single
seed [42]. Pass --sample to switch to temperature sampling, which makes
multiple --seeds produce genuinely different rows.

Design choice: this script only PRODUCES rewrites. Metrics are computed
by score_rewrites.py so we can re-score without re-generating.

ETA           : ~7 s/rewrite on a 4090, ~30 s/rewrite on CPU.
                300 items x 4 cells x 1 seed x 3 tactic configs = 3600 rewrites.
                => ~7 h on a single 4090, ~30 h on CPU. USE gpu2modal / gpu2vast.
Cloud cost    : ~$3-5 on Modal A10G for the full grid.
                Cut items to 100 for a smoke run.

Outputs       : results/b2/rewrites_a{alpha}_b{beta}_s{seed}.csv
                one row per (item, tactic_config), columns:
                item_id, seed, alpha, beta, tactic_ids, tactic_label,
                neutral, edited, top_k
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.artifacts import JsonlWriter
from common.paths import (
    BASE_LLM_ID,
    CLICKBAIT_MODEL_DIR,
    RESULTS_DIR,
    TACTIC_NAMES,
    TACTICS_MODEL_DIR,
    ensure_dirs,
    find_repo_test_csv,
)

# Grid for the ablation. We keep top_k fixed at 50 (the smaller of the two
# TOP_K_VALUES in the repo) to keep runtime tractable; expose --top-k for
# sensitivity.
# Numerical floor for logs, and a scale for the EOS length bonus in log mode
# (per-step scores are sums of log-probs of magnitude ~2-10, so the small
# prob-scale bonuses must be scaled up to stay influential). Both, along with
# the on-weights below, should be re-tuned per objective via --sweep.
LOG_EPS = 1e-6
LENGTH_BONUS_SCALE_LOG = 5.0

# Characters outside the Latin script (CJK, Hangul, kana, Cyrillic, Hebrew,
# Arabic). Used by the --latin-only language-consistency mask to stop the
# clickbait brake from reward-hacking the English-only guide by code-switching
# on a multilingual base model.
import re as _re
_NONLATIN_RE = _re.compile(
    "[぀-ヿ㐀-鿿가-힯Ѐ-ӿ֐-׿؀-ۿ]"
)

# Default "on" weight for the 4-cell ablation, per objective. The prob-domain
# value (0.7) was the paper's; the log-domain value is a starting point that
# MUST be confirmed by a --sweep before the ablation is reported.
DEFAULT_ON = {"prob": 0.7, "log": 2.0}

# The 4-cell ablation grid is built in main() from the chosen on-weight:
#   (0,0) no guidance, (on,0) positive only, (0,on) negative only, (on,on) both.
GRID = [(0.0, 0.0), (0.7, 0.0), (0.0, 0.7), (0.7, 0.7)]

# The three tactic configs from fudge_controlled_generation.py that
# actually shipped in the results CSVs.
DEFAULT_TACTIC_CONFIGS = [
    ([0, 9], "Curiosity Gap + Provocative Questions"),
    ([2, 3], "Emotional Trigger + Sensationalism"),
    ([3, 9], "Sensationalism + Question"),
]


def create_prompt(neutral_headline: str, tactics: list[str]) -> str:
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


def create_prompt_neutral(neutral_headline: str) -> str:
    """Tactic-AGNOSTIC rewrite prompt.

    Unlike create_prompt(), this never names the target tactics and never asks
    for any stylistic influence, so ALL tactic/clickbait signal must come from
    the FUDGE guides. It makes the (0,0) cell a genuine floor (a plain
    paraphrase) and attributes any tactic lift in the (alpha,beta) cells to the
    decoding-time guides alone -- the clean controllable-decoding test. The
    fidelity rules are kept identical so faithfulness is comparable across arms.
    """
    return f"""

  Rewrite the following news headline in different words.

STRICT RULES:
- Do NOT add any new facts
- Do NOT add any new people, places, or events
- Do NOT change the meaning
- Do NOT introduce claims not present in the original headline
- Only rephrase the wording

Output:
One headline only. No explanations.

Headline:

{neutral_headline}
  """


def _load_models(device: str):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BertForSequenceClassification,
        BertTokenizer,
    )

    llm_tok = AutoTokenizer.from_pretrained(BASE_LLM_ID)
    llm = AutoModelForCausalLM.from_pretrained(
        BASE_LLM_ID,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).eval()

    cb_tok = BertTokenizer.from_pretrained(str(CLICKBAIT_MODEL_DIR))
    cb_m = BertForSequenceClassification.from_pretrained(str(CLICKBAIT_MODEL_DIR)).to(device).eval()

    tc_tok = BertTokenizer.from_pretrained(str(TACTICS_MODEL_DIR))
    tc_m = BertForSequenceClassification.from_pretrained(str(TACTICS_MODEL_DIR)).to(device).eval()

    return llm_tok, llm, cb_tok, cb_m, tc_tok, tc_m


def _fudge_generate(
    llm_tok,
    llm,
    cb_tok,
    cb_m,
    tc_tok,
    tc_m,
    device: str,
    neutral: str,
    tactic_ids: list[int],
    alpha: float,
    beta: float,
    top_k: int,
    max_tokens: int,
    objective: str = "log",
    sample: bool = False,
    temperature: float = 1.0,
    name_tactics: bool = True,
    fluency_top_p: float | None = None,
    latin_only: bool = False,
) -> str:
    """(alpha, beta) FUDGE decoding.

    objective="log" (default) uses the canonical FUDGE log-domain factorization
        score = log p_llm + alpha*log(tac) + beta*log(1 - cb)
    which corresponds to the Bayesian product P(text)*P(attr|text)*P(not-cb|text).
    objective="prob" reproduces the paper's additive-probability heuristic
        score = p_llm + alpha*tac - beta*cb
    kept for the reproduction / ablation.

    Greedy argmax by default (deterministic). With sample=True the next token is
    drawn from the softmax over the re-scored top-k candidates, which makes
    multiple seeds produce genuinely different rows.
    """
    import math

    import torch

    weights = [1 if i in tactic_ids else 0 for i in range(len(TACTIC_NAMES))]
    weights_t = torch.tensor(weights, device=device, dtype=torch.float32)
    pos_mask = weights_t > 0
    tactic_names = [TACTIC_NAMES[i] for i in tactic_ids]

    # Batched guide scoring: one padded forward over ALL top-k candidate texts
    # instead of top_k separate batch-1 forwards. Padding is attention-masked by
    # BERT, so per-example logits are identical to the unbatched version. This
    # is the dominant speedup (top_k x fewer kernel launches per token step).
    @torch.no_grad()
    def score_cb_batch(texts: list[str]) -> list[float]:
        enc = cb_tok(texts, return_tensors="pt", truncation=True,
                     max_length=32, padding=True).to(device)
        return torch.sigmoid(cb_m(**enc).logits[:, 1]).tolist()

    @torch.no_grad()
    def score_tac_batch(texts: list[str]) -> list[float]:
        if pos_mask.sum().item() == 0:
            return [0.0] * len(texts)
        enc = tc_tok(texts, return_tensors="pt", truncation=True,
                     max_length=32, padding=True).to(device)
        probs = torch.sigmoid(tc_m(**enc).logits)      # [B, num_tactics]
        return probs[:, pos_mask].mean(dim=1).tolist()  # [B]

    # name_tactics=False -> tactic-agnostic prompt; the guides carry all the
    # tactic/clickbait signal (isolates FUDGE's contribution).
    prompt = (create_prompt(neutral, tactic_names) if name_tactics
              else create_prompt_neutral(neutral))
    messages = [
        {"role": "system", "content": "You are an experienced news editor."},
        {"role": "user", "content": prompt},
    ]
    input_ids = llm_tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(llm.device)

    have_pos = pos_mask.sum().item() > 0
    generated: list[int] = []
    past = None
    step_input = input_ids  # first step feeds the whole prompt; then one token
    with torch.no_grad():
        for _ in range(max_tokens):
            # KV cache: after the first step we only feed the newly committed
            # token, so the LLM stops recomputing the full prefix (O(n) instead
            # of O(n^2) over the generation).
            out = llm(input_ids=step_input, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]
            probs = torch.softmax(next_logits, dim=-1)[0]
            logprobs = torch.log_softmax(next_logits, dim=-1)[0]
            top_vals, top_idx = torch.topk(next_logits, k=top_k, dim=-1)
            top_idx = top_idx[0].tolist()

            step = len(generated)
            alpha_dyn = alpha * min(step / 6, 1.0)

            # Build all candidate texts once, then score the whole batch through
            # each guide in a single forward (vs top_k separate batch-1 calls).
            cand_texts: list[str] = []
            length_bonuses: list[float] = []
            for tok_id in top_idx:
                text = llm_tok.decode(generated + [tok_id],
                                      clean_up_tokenization_spaces=False)
                if tok_id == llm_tok.eos_token_id:
                    text = text.rstrip()
                    if not text.endswith("."):
                        text += "."
                    L = len(text.split())
                    lb = 0.0
                    if L < 6:
                        lb = -0.4
                    elif 8 <= L <= 30:
                        lb = 0.25
                    elif L > 32:
                        lb = -0.2
                    length_bonuses.append(lb)
                else:
                    length_bonuses.append(0.0)
                cand_texts.append(text)

            cb_probs = score_cb_batch(cand_texts)
            tac_scores = score_tac_batch(cand_texts)

            # lognorm: divide the base-LLM log-prob term by its per-step spread
            # over the top-k set, so alpha/beta weight the guides against a
            # unit-variance LLM signal and become base-model-independent. A more
            # confident model (wider log-prob spread) is rescaled to match.
            llm_tau = 1.0
            if objective == "lognorm":
                _ll = [logprobs[t].item() for t in top_idx]
                _mu = sum(_ll) / len(_ll)
                _var = sum((x - _mu) ** 2 for x in _ll) / max(len(_ll) - 1, 1)
                llm_tau = math.sqrt(_var) or 1.0

            cand_scores: list[float] = []
            for j, tok_id in enumerate(top_idx):
                p_cb = cb_probs[j]
                tac = tac_scores[j]
                length_bonus = length_bonuses[j]
                if objective in ("log", "lognorm"):
                    # Canonical FUDGE: additive log-probabilities. Guard the
                    # off cells (alpha_dyn==0 or beta==0) so we never evaluate
                    # 0 * log(0) = nan, and clamp every log argument.
                    s = logprobs[tok_id].item() / llm_tau
                    if alpha_dyn > 0 and have_pos:
                        s += alpha_dyn * math.log(max(tac, LOG_EPS))
                    if beta > 0:
                        s += beta * math.log(max(1.0 - p_cb, LOG_EPS))
                    s += length_bonus * LENGTH_BONUS_SCALE_LOG
                else:
                    # Paper's additive-probability heuristic.
                    p_llm = probs[tok_id].item()
                    s = p_llm + alpha_dyn * tac - beta * p_cb + length_bonus
                cand_scores.append(s)

            # Language-consistency mask: on a multilingual base model the
            # clickbait brake can reward-hack the English-only guide by
            # code-switching into a script the guide scores as non-clickbait.
            # Drop any candidate that introduces a non-Latin character so the
            # guidance stays within the source language.
            if latin_only:
                for j in range(len(cand_scores)):
                    if _NONLATIN_RE.search(cand_texts[j]):
                        cand_scores[j] = float("-inf")

            # Fluency floor: restrict FUDGE reranking to the LLM's nucleus.
            # topk returns candidates sorted by logit desc, so accumulate their
            # true LLM probability and mask out everything past the smallest
            # prefix that reaches fluency_top_p. This stops the guide from
            # selecting grammatically-broken low-probability tokens (the biggest
            # source of disfluent losses) while still steering within the tokens
            # the LLM finds plausible. The top-1 token is always kept.
            if fluency_top_p is not None and 0.0 < fluency_top_p < 1.0:
                csum = 0.0
                reached = False
                for j, tok_id in enumerate(top_idx):
                    if reached:
                        cand_scores[j] = float("-inf")
                        continue
                    csum += probs[tok_id].item()
                    if csum >= fluency_top_p:
                        reached = True  # include this one, mask the rest

            if sample:
                # Draw from the softmax over the re-scored candidates so that
                # seeds (torch.manual_seed) produce genuinely different rows.
                scores_t = torch.tensor(cand_scores, device=device, dtype=torch.float32)
                cand_probs = torch.softmax(scores_t / max(temperature, 1e-6), dim=-1)
                pick = torch.multinomial(cand_probs, num_samples=1).item()
                best_id = top_idx[pick]
            else:
                best_id = top_idx[int(np.argmax(cand_scores))]

            if best_id == llm_tok.eos_token_id:
                break
            generated.append(best_id)
            # Only the new token goes into the next forward; the cache holds
            # the rest.
            step_input = torch.tensor([[best_id]], device=llm.device)

    text = llm_tok.decode(generated).strip()
    if text and not text.endswith((".", "?", "!")):
        text += "."
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-csv", default=str(find_repo_test_csv()),
                    help="held-out neutral headlines CSV (needs 'title' or 'neutral' column)")
    ap.add_argument("--n-items", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="default is a SINGLE seed: greedy FUDGE decoding is "
                         "deterministic, so extra seeds only replicate identical "
                         "rows. Pass multiple seeds only together with --sample.")
    ap.add_argument("--sample", action="store_true",
                    help="switch decoding from greedy argmax to temperature "
                         "sampling over the re-scored top-k candidates, so that "
                         "multiple --seeds become meaningful. Off by default to "
                         "stay faithful to the paper's greedy decoding.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="softmax temperature used only when --sample is set.")
    ap.add_argument("--objective", choices=["log", "prob", "lognorm"], default="log",
                    help="log (default) = canonical FUDGE log-domain "
                         "factorization log p_llm + a*log(tac) + b*log(1-cb); "
                         "prob = the paper's additive-probability heuristic; "
                         "lognorm = log but the base-LLM log-prob term is divided "
                         "by its per-step std over the top-k candidate set, making "
                         "alpha/beta scale-invariant across base models. "
                         "The on-weight scale differs between objectives; re-tune "
                         "with --sweep after changing this.")
    ap.add_argument("--alpha-on", type=float, default=None,
                    help="active positive-guidance weight for the 4-cell "
                         "ablation. Default depends on --objective "
                         "(log=2.0, prob=0.7).")
    ap.add_argument("--beta-on", type=float, default=None,
                    help="active negative-guidance weight for the 4-cell "
                         "ablation. Default depends on --objective.")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated weight grid (e.g. '0,0.5,1,2,4') to "
                         "run the FULL cross product alpha x beta instead of "
                         "the 4-cell ablation. Use this to re-tune the "
                         "on-weight for a new objective, then set --alpha-on / "
                         "--beta-on to the winner for the reported ablation.")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--neutral-prompt", action="store_true",
                    help="use a tactic-AGNOSTIC rewrite prompt so the FUDGE "
                         "guides are the sole tactic/clickbait signal. Makes "
                         "(0,0) a genuine paraphrase floor and isolates the "
                         "decoding-time contribution of alpha/beta.")
    ap.add_argument("--fluency-top-p", type=float, default=None,
                    help="restrict FUDGE reranking to the LLM's top-p nucleus "
                         "(e.g. 0.9) so the guide cannot pick grammar-breaking "
                         "low-probability tokens. Off by default.")
    ap.add_argument("--latin-only", action="store_true",
                    help="mask top-k candidates that introduce non-Latin "
                         "characters. Prevents the clickbait brake from "
                         "reward-hacking the English-only guide by code-switching "
                         "into another script on a multilingual base model.")
    ap.add_argument("--cells", default=None,
                    help="explicit (alpha:beta) cells to generate, comma "
                         "separated, e.g. '0:0,4:2'. Overrides the default "
                         "4-cell ablation / --sweep grid.")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR / "b2"))
    ap.add_argument("--tactic-configs", default=None,
                    help="optional JSON path overriding tactic list")
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

    # Crash-safe per-datapoint provenance sidecar: one flushed JSONL line per
    # rewrite (including per-item failures), opened once for the whole run and
    # appended across seeds/cells so it survives a crash and is resume-safe.
    gen_writer = JsonlWriter(out_dir / "generations.jsonl")
    print(f"[b2] raw generations -> {out_dir / 'generations.jsonl'}")

    # Build the (alpha, beta) grid. Either the 4-cell ablation from the
    # objective-appropriate on-weight, or a full sweep for re-tuning.
    on = DEFAULT_ON.get(args.objective, DEFAULT_ON["log"])  # lognorm shares log's on-weight
    a_on = args.alpha_on if args.alpha_on is not None else on
    b_on = args.beta_on if args.beta_on is not None else on
    if args.sweep:
        ws = [float(x) for x in args.sweep.split(",")]
        grid = [(a, b) for a in ws for b in ws]
        print(f"[b2] SWEEP mode: {len(grid)} (alpha,beta) cells from {ws}")
    else:
        grid = [(0.0, 0.0), (a_on, 0.0), (0.0, b_on), (a_on, b_on)]
        print(f"[b2] 4-cell ablation with on-weights alpha_on={a_on}, beta_on={b_on}")
    if args.cells:
        grid = []
        for pair in args.cells.split(","):
            a_s, b_s = pair.split(":")
            grid.append((float(a_s), float(b_s)))
        print(f"[b2] explicit cells override: {grid}")
    print(f"[b2] objective = {args.objective}")

    tactic_configs = DEFAULT_TACTIC_CONFIGS
    if args.tactic_configs:
        tactic_configs = json.loads(Path(args.tactic_configs).read_text())

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
    print(f"[b2] test set: {args.test_csv} column={text_col} "
          f"rows={len(df)} (item_id {df.index.min()}..{df.index.max()})")

    if args.sample:
        print(f"[b2] decoding = TEMPERATURE SAMPLING (temperature={args.temperature}); "
              f"multiple seeds are meaningful.")
    else:
        print("[b2] decoding = GREEDY (deterministic, faithful to the paper). "
              "Multiple seeds would only replicate identical rows, so replication "
              "is over ITEMS via bootstrap in score_rewrites.py, NOT over seeds. "
              "Use --sample if you want seed-level variance.")

    total = len(df) * len(grid) * len(args.seeds) * len(tactic_configs)
    print(f"[b2] planned rewrites: {total}")
    print(f"[b2] ETA @ 7 s/rewrite on GPU: {total*7/3600:.1f} h")
    print(f"[b2] ETA @ 30 s/rewrite on CPU: {total*30/3600:.1f} h")
    print("[b2] STRONGLY RECOMMEND running via gpu2modal for anything above N=50.")

    if args.dry_run:
        return 0

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[b2] device = {device}")

    llm_tok, llm, cb_tok, cb_m, tc_tok, tc_m = _load_models(device)

    t0 = time.time()
    n_done = 0

    for seed in args.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        for alpha, beta in grid:
            out_csv = out_dir / f"rewrites_a{alpha}_b{beta}_s{seed}{args.shard_tag}.csv"

            # Resume: if this cell's CSV already has rows, skip the
            # (item_id, tactic_label) pairs already generated. Re-running a
            # crashed job then only does the remaining work instead of
            # starting the cell over. Header is written only for a new file.
            done_keys: set = set()
            header_needed = True
            if out_csv.exists():
                try:
                    prev = pd.read_csv(out_csv)
                    done_keys = set(zip(prev["item_id"].tolist(),
                                        prev["tactic_label"].tolist()))
                    header_needed = prev.empty and len(prev.columns) == 0
                except Exception:
                    done_keys, header_needed = set(), False
                header_needed = False
            n_skip = len(done_keys)
            print(f"[b2] -> {out_csv} (resume: {n_skip} rows already present)")

            with open(out_csv, "a", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                if header_needed:
                    w.writerow(
                        ["item_id", "seed", "alpha", "beta", "top_k",
                         "tactic_ids", "tactic_label", "neutral", "edited"]
                    )
                for i, row in df.iterrows():
                    neutral = str(row[text_col])
                    for tactic_ids, label in tactic_configs:
                        if (i, label) in done_keys:
                            continue
                        # Per-item guard: a single bad headline (tokenizer edge
                        # case, transient CUDA error) must not kill the whole
                        # run. Log it, write an empty rewrite (scoring flags it
                        # is_empty and excludes it), and continue.
                        failed = False
                        err = None
                        try:
                            edited = _fudge_generate(
                                llm_tok, llm, cb_tok, cb_m, tc_tok, tc_m,
                                device, neutral, tactic_ids, alpha, beta,
                                args.top_k, args.max_tokens,
                                objective=args.objective,
                                sample=args.sample, temperature=args.temperature,
                                name_tactics=not args.neutral_prompt,
                                fluency_top_p=args.fluency_top_p,
                                latin_only=args.latin_only,
                            )
                        except Exception as e:
                            print(f"[b2] WARN item {i} tactic '{label}' "
                                  f"(a={alpha},b={beta}) FAILED: {type(e).__name__}: {e}",
                                  flush=True)
                            edited = ""
                            failed = True
                            err = f"{type(e).__name__}: {e}"
                        w.writerow(
                            [i, seed, alpha, beta, args.top_k,
                             json.dumps(tactic_ids), label, neutral, edited]
                        )
                        f.flush()  # keep partial progress on disk for resume
                        # Full per-datapoint provenance + raw completion, flushed.
                        gen_writer.write({
                            "item_id": i, "seed": seed, "alpha": alpha,
                            "beta": beta, "top_k": args.top_k,
                            "tactic_ids": tactic_ids, "tactic_label": label,
                            "objective": args.objective, "neutral": neutral,
                            "edited": edited, "failed": failed, "error": err,
                            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        })
                        n_done += 1
                        if n_done % 25 == 0:
                            dt = time.time() - t0
                            rate = n_done / dt
                            eta = (total - n_done) / max(rate, 1e-9)
                            print(f"[b2] {n_done}/{total} rate={rate:.2f}/s "
                                  f"ETA={eta/60:.1f} min", flush=True)
    gen_writer.close()
    print(f"[b2] done in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
