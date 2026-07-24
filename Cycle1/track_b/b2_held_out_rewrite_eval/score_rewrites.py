"""B2: Score every rewrite CSV emitted by run_rewrites.py.

Metrics per (item, condition)
-----------------------------
- BERTScore F1 vs neutral (rescaled with the English baseline)
- Sentence-Transformers all-mpnet-base-v2 cosine STS vs neutral
- NLI entailment probability, both directions, roberta-large-mnli
- Clickbait probability. Two columns, never conflated:
    * clickbait_prob_external: an INDEPENDENT detector supplied via
      --external-clickbait-dir (e.g. the Chakraborty-trained DistilBERT from
      train_external_clickbait.py). Absent if no external dir is given.
    * clickbait_prob_guide_circular: the repo's beta-guide BERT. This is the
      SAME model FUDGE optimises against, so it is CIRCULAR and must NOT be
      reported as the independent clickbait metric.
- Attribute realisation (circular): attr_realised_frac_guide_circular uses the
  same tactics BERT that is the alpha-guide, so it is circular. The independent
  measure of attribute realisation is judge_attr_confirmed_frac (LLM judge,
  produced only with --with-llm-judge).
- names_tactic: leak check, whether a targeted tactic's surface name appears
  verbatim in the rewrite (the prompt forbids naming the tactic).
- Optional LLM-as-judge attribute confirmation (Anthropic or OpenAI).
  Skipped silently when neither env var is set.

Aggregation
-----------
Group by (alpha, beta, tactic_label) for the per-tactic breakdown, plus the
marginal (alpha, beta) aggregate. Report mean and 95% bootstrap CI over the
ITEM axis (n_boot=1000, percentile). Empty rewrites are excluded from the
aggregation; exact-duplicate rows are deduplicated first. Writes:

    results/b2/summary.json
    results/b2/summary.md

ETA (CPU) : ~15 min per condition CSV of 300 items. ETA (GPU): 2 min.
Cost      : $0 if the LLM judge is off. With judge on: ~$0.01 per rewrite
            (300 x 4 x 3 x 3 configs ~ 10800 calls -> ~$100 at GPT-4o).
            Judge is opt-in via --with-llm-judge.
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
from common.bootstrap_ci import bootstrap_mean_ci
from common.paths import RESULTS_DIR, TACTIC_NAMES, ensure_dirs
from common.scorers import ClickbaitScorer, TacticsScorer


def _load_rewrites(in_dir: Path):
    import pandas as pd

    # Match both the b2 FUDGE files (rewrites_a{alpha}_b{beta}_s{seed}.csv)
    # and the b4 prompt-only files (rewrites_prompt_only_s{seed}.csv). The
    # alpha/beta/seed values are read from columns, not the filename, so any
    # rewrites_*.csv is safe to ingest.
    rows = []
    for p in sorted(in_dir.glob("rewrites_*.csv")):
        df = pd.read_csv(p)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"no rewrite CSVs found under {in_dir}")
    return pd.concat(rows, ignore_index=True)


def _sts(texts_a, texts_b, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    ea = m.encode(texts_a, convert_to_tensor=True, show_progress_bar=False)
    eb = m.encode(texts_b, convert_to_tensor=True, show_progress_bar=False)
    # Row-wise cosine: normalise then elementwise-multiply and sum over dim=1.
    # Same numbers as cos_sim(ea, eb).diagonal() but O(N) memory (no N x N matrix).
    ea = torch.nn.functional.normalize(ea, p=2, dim=1)
    eb = torch.nn.functional.normalize(eb, p=2, dim=1)
    return (ea * eb).sum(dim=1).cpu().numpy()


def _bertscore(cands, refs, device: str = None):
    from bert_score import score as bs

    # rescale_with_baseline spreads F1 out of the compressed 0.85-1.0 band.
    # Pass device explicitly so BERTScore runs on the GPU rather than relying
    # on its auto-detect.
    P, R, F1 = bs(cands, refs, lang="en", rescale_with_baseline=True,
                  verbose=False, device=device)
    return F1.cpu().numpy()


def _nli(pairs, device: str, batch_size: int = 16,
         name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"):
    """Return entail_prob(a->b) and entail_prob(b->a) arrays.

    name selects the NLI model. roberta-large-mnli is the legacy default; the
    DeBERTa-v3 multi-dataset checkpoints (e.g.
    MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) are stronger and
    better calibrated -- switched in only after winning the control-set A/B.
    Both use the standard 3-class MNLI label order (entailment = index 2).
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()

    # Resolve the entailment class index from the model config -- roberta-mnli
    # uses index 2, but the DeBERTa-v3 MNLI/ANLI checkpoints use index 0. Never
    # hardcode it or the swapped model reports contradiction as entailment.
    id2label = getattr(model.config, "id2label", None) or {}
    entail_idx = 2
    for i, lab in id2label.items():
        if str(lab).lower().startswith("entail"):
            entail_idx = int(i)
            break

    def run(premises, hypotheses):
        out = np.empty(len(premises), dtype=float)
        for i in range(0, len(premises), batch_size):
            enc = tok(
                premises[i : i + batch_size],
                hypotheses[i : i + batch_size],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out[i : i + probs.shape[0]] = probs[:, entail_idx]
        return out

    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    return run(a, b), run(b, a)


def _judge_api(prompt: str) -> dict:
    """Single JSON-returning LLM call (Anthropic or OpenAI). Shared plumbing."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-3-5-sonnet-latest", max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return json.loads(resp.content[0].text)
        except Exception as e:
            return {"error": str(e)}
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import requests
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                         "Content-Type": "application/json"},
                json={"model": os.environ.get("OPENAI_API_MODEL", "gpt-4o-mini"),
                      "messages": [{"role": "user", "content": prompt}],
                      "response_format": {"type": "json_object"}},
                timeout=60,
            )
            return json.loads(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            return {"error": str(e)}
    return {"skipped": True}


def _paired_judge(neutral: str, text_a: str, text_b: str,
                  tactic_names: list[str]) -> dict:
    """Blind A/B preference: which rewrite better expresses the target tactics
    while staying faithful? Paired on the SAME item, so item difficulty cancels
    -- far more sensitive than absolute per-row rating for detecting whether
    FUDGE beats the prompt-only baseline. Returns {"winner": "A"|"B"|"tie"}."""
    prompt = (
        "Two rewrites (A and B) of the same neutral news headline are shown.\n\n"
        f"Original neutral headline: \"{neutral}\"\n"
        f"Rewrite A: \"{text_a}\"\n"
        f"Rewrite B: \"{text_b}\"\n"
        f"Target rhetorical tactics: {', '.join(tactic_names)}\n\n"
        "Which rewrite expresses the target tactics MORE strongly and "
        "effectively while preserving the original meaning and adding no new "
        "facts? If they are equivalent, answer \"tie\".\n"
        "Return ONLY strict JSON: {\"winner\": \"A\" | \"B\" | \"tie\"}."
    )
    return _judge_api(prompt)


def _perplexity(texts, device: str, batch_size: int = 32):
    """Per-text perplexity under distilgpt2 -- an INDEPENDENT fluency proxy.

    FUDGE over-steering (very high alpha or beta) degrades the language model's
    own likelihood, producing disfluent / degenerate text that no guide score
    reveals. distilgpt2 is a small held-out LM the guides never touch, so a
    spike in its perplexity is a clean, non-circular signal that a cell has
    pushed the decoder past the fluent region. Lower is more fluent.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    out = np.full(len(texts), np.nan, dtype=float)
    for i in range(0, len(texts), batch_size):
        chunk = [t if isinstance(t, str) and t.strip() else "" for t in texts[i:i + batch_size]]
        for j, t in enumerate(chunk):
            if not t:
                continue  # leave NaN for empty rewrites
            enc = tok(t, return_tensors="pt", truncation=True, max_length=64).to(device)
            ids = enc["input_ids"]
            if ids.shape[1] < 2:
                continue  # need >=2 tokens for a next-token loss
            with torch.no_grad():
                loss = model(**enc, labels=ids).loss
            out[i + j] = float(torch.exp(loss).clamp(max=1e6).item())
    return out


def _llm_judge(neutral: str, edited: str, tactic_names: list[str]) -> dict:
    """Optional; only invoked if --with-llm-judge and an API key is set."""
    prompt = (
        "You are auditing a rewritten news headline.\n\n"
        f"Original neutral headline: \"{neutral}\"\n"
        f"Rewritten headline: \"{edited}\"\n"
        f"Target rhetorical tactics: {', '.join(tactic_names)}\n\n"
        "For each target tactic, rate how STRONGLY it is expressed in the "
        "rewrite on a 0-3 scale (0 = absent, 1 = weak/subtle, 2 = moderate, "
        "3 = strong/overt). Separately rate how engaging/click-inducing the "
        "rewrite is overall on the same 0-3 scale. Also state whether the "
        "rewrite introduces any factual claim not in the original.\n"
        "Return ONLY strict JSON with keys: "
        "{\"tactic_intensity\": {tactic_name: 0-3 int}, "
        "\"engaging\": 0-3 int, \"hallucinates\": bool}."
    )
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return json.loads(resp.content[0].text)
        except Exception as e:
            return {"error": str(e)}
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import requests

            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": os.environ.get("OPENAI_API_MODEL", "gpt-4o-mini"),
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
                timeout=60,
            )
            return json.loads(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            return {"error": str(e)}
    return {"skipped": True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default=str(RESULTS_DIR / "b2"))
    ap.add_argument("--out-json", default=str(RESULTS_DIR / "b2" / "summary.json"))
    ap.add_argument("--out-md", default=str(RESULTS_DIR / "b2" / "summary.md"))
    ap.add_argument("--per-item-csv", default=str(RESULTS_DIR / "b2" / "per_item_scores.csv"))
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--with-llm-judge", action="store_true")
    ap.add_argument("--external-clickbait-dir", default=None,
                    help="directory of an INDEPENDENT clickbait detector (e.g. the "
                         "Chakraborty-trained DistilBERT from train_external_clickbait.py). "
                         "When given, its score is written to clickbait_prob_external. "
                         "Without it, no independent clickbait column is produced.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--judge-workers", type=int, default=16,
                    help="concurrent LLM-judge API calls (judge is latency "
                         "bound, so this gives a near-linear speedup)")
    ap.add_argument("--judge-max-per-cell", type=int, default=0,
                    help="if >0, judge at most this many rows per (alpha,beta) "
                         "cell instead of every row. Used for fast, cheap "
                         "on-weight SELECTION on the independent judge signal "
                         "(e.g. 15 rows x 16 cells = 240 calls). 0 = judge all "
                         "(the reported run).")
    ap.add_argument("--paired-judge", action="store_true",
                    help="also run a blind A/B preference judge pairing each "
                         "(alpha,beta) cell against the (0,0) baseline on the "
                         "SAME item; writes paired_summary.json with FUDGE "
                         "win-rate per cell. Highest-power test of whether "
                         "steering beats the prompt-only baseline.")
    args = ap.parse_args()

    ensure_dirs()

    device = "cpu"
    if args.gpu:
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass

    print("[b2-score] ETA: 2-15 min per rewrite CSV depending on device.")

    import pandas as pd

    df = _load_rewrites(Path(args.in_dir))
    df["edited"] = df["edited"].fillna("").astype(str)
    df["neutral"] = df["neutral"].astype(str)
    df["tactic_ids"] = df["tactic_ids"].apply(json.loads)
    # FIX (S-m1): flag empty/whitespace rewrites. They stay in the per-item CSV
    # with is_empty=True and NaN metrics, but are excluded from aggregation.
    df["is_empty"] = df["edited"].str.strip() == ""
    # Question-form flag: rhetorical questions are a target tactic, but they
    # depress NLI entailment and trip the naive hallucination judge regardless
    # of faithfulness. Reporting metrics split by this flag separates the tactic
    # form from genuine infidelity (see the by_question_status summary section).
    df["is_question"] = df["edited"].astype(str).str.contains(r"\?", regex=True)
    n = len(df)
    print(f"[b2-score] rewrites: {n}")
    print(f"[b2-score] empty/whitespace rewrites flagged: {int(df['is_empty'].sum())}")

    # Each metric block is wrapped so a single failing scorer degrades that
    # metric to NaN (missing) for all rows and the run continues, rather than
    # aborting and losing every metric already computed.
    print("[b2-score] STS...")
    try:
        df["sts"] = _sts(df["neutral"].tolist(), df["edited"].tolist(), device)
    except Exception as e:
        print(f"[b2-score] WARN: STS failed: {e}", flush=True)
        df["sts"] = np.nan

    print("[b2-score] BERTScore...")
    try:
        df["bertscore_f1"] = _bertscore(df["edited"].tolist(),
                                        df["neutral"].tolist(), device)
    except Exception as e:
        print(f"[b2-score] WARN: BERTScore failed: {e}", flush=True)
        df["bertscore_f1"] = np.nan

    print("[b2-score] NLI both directions...")
    try:
        ent_ne, ent_en = _nli(list(zip(df["neutral"].tolist(), df["edited"].tolist())), device)
        df["nli_neutral_entails_edited"] = ent_ne
        df["nli_edited_entails_neutral"] = ent_en
    except Exception as e:
        print(f"[b2-score] WARN: NLI failed: {e}", flush=True)
        df["nli_neutral_entails_edited"] = np.nan
        df["nli_edited_entails_neutral"] = np.nan

    # Fluency (INDEPENDENT): distilgpt2 perplexity. A held-out LM the guides
    # never optimise, so it detects over-steering (disfluent/degenerate text)
    # that the circular guide scores hide. Lower = more fluent.
    print("[b2-score] fluency perplexity (distilgpt2, INDEPENDENT)...")
    try:
        df["fluency_ppl"] = _perplexity(df["edited"].tolist(), device)
    except Exception as e:
        print(f"[b2-score] WARN: fluency_ppl failed: {e}", flush=True)
        df["fluency_ppl"] = np.nan

    # FIX (S-C1): the guide model is the SAME BERT FUDGE optimises against, so
    # its clickbait score is circular. Keep it, but name it honestly.
    print("[b2-score] clickbait probability (GUIDE model, CIRCULAR)...")
    try:
        cb = ClickbaitScorer(device=device)
        df["clickbait_prob_guide_circular"] = cb.prob(df["edited"].tolist())
        print("[b2-score] WARNING: clickbait_prob_guide_circular is the FUDGE "
              "beta-guide itself. It is CIRCULAR and must NOT be reported as the "
              "independent clickbait metric.")
    except Exception as e:
        print(f"[b2-score] WARN: clickbait_prob_guide_circular failed: {e}", flush=True)
        df["clickbait_prob_guide_circular"] = np.nan

    if args.external_clickbait_dir:
        print(f"[b2-score] clickbait probability (INDEPENDENT detector: "
              f"{args.external_clickbait_dir})...")
        try:
            cb_ext = ClickbaitScorer(model_dir=args.external_clickbait_dir, device=device)
            df["clickbait_prob_external"] = cb_ext.prob(df["edited"].tolist())
        except Exception as e:
            print(f"[b2-score] WARN: clickbait_prob_external failed: {e}", flush=True)
            df["clickbait_prob_external"] = np.nan
    else:
        print("[b2-score] WARNING: no --external-clickbait-dir supplied. No "
              "independent clickbait detector is available; clickbait_prob_external "
              "will be ABSENT. The guide-model column is NOT a substitute.")

    # FIX (S-C3): attribute realisation from the tactics BERT is circular (that
    # BERT is the alpha-guide). The independent measure of attribute
    # realisation is judge_attr_confirmed_frac (produced with --with-llm-judge).
    print("[b2-score] tactics probs...")
    try:
        tc = TacticsScorer(device=device)
        tac_probs = tc.probs(df["edited"].tolist())
        for k, name in enumerate(TACTIC_NAMES):
            df[f"tac_{k}"] = tac_probs[:, k]

        def realised(row):
            got = 0
            for tid in row["tactic_ids"]:
                if row[f"tac_{tid}"] >= 0.5:
                    got += 1
            return got / max(len(row["tactic_ids"]), 1)

        df["attr_realised_frac_guide_circular"] = df.apply(realised, axis=1)
    except Exception as e:
        print(f"[b2-score] WARN: attr_realised_frac_guide_circular failed: {e}", flush=True)
        df["attr_realised_frac_guide_circular"] = np.nan

    # FIX (S-m5): tactic-name leak check. The prompt forbids naming the tactic;
    # flag rewrites where a targeted tactic's surface name appears verbatim.
    def names_tactic(row):
        ed = row["edited"].lower()
        for tid in row["tactic_ids"]:
            if TACTIC_NAMES[tid].lower() in ed:
                return True
        return False

    df["names_tactic"] = df.apply(names_tactic, axis=1)

    if args.with_llm_judge:
        print("[b2-score] LLM judge (this can cost real money)...")
        # Crash-safe raw-judge sidecar: one flushed JSONL line per judged row,
        # persisting the RAW judge dict (which is otherwise parsed then
        # discarded) so judge outputs are inspectable and survive a crash.
        judge_raw_path = Path(args.in_dir) / "judge_raw.jsonl"
        print(f"[b2-score] raw judge outputs -> {judge_raw_path}")
        try:
            from concurrent.futures import ThreadPoolExecutor

            # For fast on-weight SELECTION, judge only a subsample per
            # (alpha,beta) cell instead of every row (keeps the independent
            # judge signal cheap: 15 x 16 cells = 240 calls, not 1920). Rows not
            # judged keep NaN and are excluded from the judge aggregates. The
            # reported run uses --judge-max-per-cell 0 (judge all).
            if args.judge_max_per_cell and args.judge_max_per_cell > 0 \
                    and {"alpha", "beta"}.issubset(df.columns):
                keep_idx = []
                # Judge non-empty rows only; subsample within each cell.
                cand = df[~df["is_empty"]]
                for _, g in cand.groupby(["alpha", "beta"], sort=False):
                    keep_idx.extend(g.index[:args.judge_max_per_cell].tolist())
                judge_df = df.loc[keep_idx]
                print(f"[b2-score] judge SUBSAMPLE: {len(judge_df)} rows "
                      f"(<= {args.judge_max_per_cell}/cell) for selection")
            else:
                judge_df = df
            rows_list = list(judge_df.iterrows())

            def _judge_one(item):
                idx, row = item
                names = [TACTIC_NAMES[i] for i in row["tactic_ids"]]
                j = _llm_judge(row["neutral"], row["edited"], names)
                return idx, row, names, j

            # The judge is API-latency bound, so fan the calls out over a thread
            # pool. ex.map yields results in INPUT order, and the main thread
            # consumes them serially, so the JSONL writes and the confirmed /
            # halluc lists stay aligned with df rows without a lock.
            # Results keyed by df index so a per-cell subsample maps back to the
            # right rows; unjudged rows stay NaN.
            conf_map: dict = {}
            halluc_map: dict = {}
            intens_map: dict = {}
            engag_map: dict = {}

            def _parse_graded(j, names):
                """(confirmed_frac, intensity_mean_0_1, engaging_0_1) from a
                graded judge dict; falls back to the old boolean schema."""
                ti = j.get("tactic_intensity")
                if isinstance(ti, dict):
                    vals = []
                    for n in names:
                        try:
                            vals.append(max(0.0, min(3.0, float(ti.get(n, 0)))))
                        except (TypeError, ValueError):
                            vals.append(0.0)
                    conf = sum(1 for v in vals if v >= 1) / max(len(names), 1)
                    inten = (sum(vals) / max(len(vals), 1)) / 3.0
                else:  # legacy boolean schema
                    tp = j.get("tactics_present", {})
                    got = sum(bool(tp.get(n, False)) for n in names)
                    conf = got / max(len(names), 1)
                    inten = conf
                try:
                    eng = max(0.0, min(3.0, float(j.get("engaging", "nan")))) / 3.0
                except (TypeError, ValueError):
                    eng = float("nan")
                return conf, inten, eng
            workers = max(int(args.judge_workers), 1)
            print(f"[b2-score] judging {len(rows_list)} rows with {workers} workers")
            with JsonlWriter(judge_raw_path) as judge_writer, \
                    ThreadPoolExecutor(max_workers=workers) as ex:
                for idx, row, names, j in ex.map(_judge_one, rows_list):
                    judge_writer.write({
                        "item_id": row.get("item_id"),
                        "alpha": row.get("alpha"),
                        "beta": row.get("beta"),
                        "tactic_label": row.get("tactic_label"),
                        "neutral": row["neutral"],
                        "edited": row["edited"],
                        "judge_response": j,
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    if j.get("skipped") or "error" in j:
                        conf_map[idx] = float("nan")
                        halluc_map[idx] = float("nan")
                        intens_map[idx] = float("nan")
                        engag_map[idx] = float("nan")
                        continue
                    conf, inten, eng = _parse_graded(j, names)
                    conf_map[idx] = conf
                    intens_map[idx] = inten
                    engag_map[idx] = eng
                    halluc_map[idx] = 1.0 if j.get("hallucinates") else 0.0
            df["judge_attr_confirmed_frac"] = df.index.map(
                lambda i: conf_map.get(i, np.nan))
            df["judge_attr_intensity"] = df.index.map(
                lambda i: intens_map.get(i, np.nan))
            df["judge_engaging"] = df.index.map(
                lambda i: engag_map.get(i, np.nan))
            df["judge_hallucinates"] = df.index.map(
                lambda i: halluc_map.get(i, np.nan))
        except Exception as e:
            print(f"[b2-score] WARN: LLM judge failed: {e}", flush=True)
            df["judge_attr_confirmed_frac"] = np.nan
            df["judge_hallucinates"] = np.nan

    # Assemble the list of metric columns actually present.
    metrics_cols = [
        "sts", "bertscore_f1", "nli_neutral_entails_edited",
        "nli_edited_entails_neutral", "clickbait_prob_guide_circular",
    ]
    if args.external_clickbait_dir:
        metrics_cols.append("clickbait_prob_external")
    metrics_cols += ["attr_realised_frac_guide_circular", "names_tactic", "fluency_ppl"]
    if args.with_llm_judge:
        metrics_cols += ["judge_attr_confirmed_frac", "judge_attr_intensity",
                         "judge_engaging", "judge_hallucinates"]
    # A metric that failed before its column was ever created is dropped here so
    # every downstream consumer (float-coercion, aggregation, summary/md) only
    # ever references columns that exist.
    metrics_cols = [m for m in metrics_cols if m in df.columns]

    # FIX (S-m1): empty rewrites keep NaN metrics in the per-item CSV so they
    # never contribute a real data point to any aggregate. Coerce each metric
    # column to float first: boolean metrics (names_tactic, judge_hallucinates)
    # become 1.0/0.0, which is both how they are reported as fractions AND
    # required so pandas 2.x can store NaN in the column (it refuses NaN in a
    # bool dtype, even when the mask selects zero rows).
    for m in metrics_cols:
        if m in df.columns:
            df[m] = df[m].astype(float)
            df.loc[df["is_empty"], m] = np.nan

    df.to_csv(args.per_item_csv, index=False, encoding="utf-8-sig")
    print(f"[b2-score] per-item scores: {args.per_item_csv}")

    # ---- Paired A/B preference: FUDGE cell vs (0,0) baseline, same item ----
    if args.paired_judge:
        import random
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor

        print("[b2-score] paired A/B preference judge (FUDGE vs baseline)...")
        paired_path = Path(args.in_dir) / "paired_preference.jsonl"

        def _isnum(x):
            try:
                float(x); return True
            except (TypeError, ValueError):
                return False

        d = df[~df["is_empty"]].copy()
        d = d[d["alpha"].map(_isnum) & d["beta"].map(_isnum)]
        d["alpha"] = d["alpha"].astype(float)
        d["beta"] = d["beta"].astype(float)

        rng = random.Random(0)
        percell: dict = {}
        pairs = []
        for (item_id, tlabel), g in d.groupby(["item_id", "tactic_label"], sort=False):
            base = g[(g["alpha"] == 0.0) & (g["beta"] == 0.0)]
            if base.empty:
                continue
            base_text = str(base.iloc[0]["edited"])
            for _, row in g.iterrows():
                a, b = float(row["alpha"]), float(row["beta"])
                if a == 0.0 and b == 0.0:
                    continue
                cell = (a, b)
                if args.judge_max_per_cell and args.judge_max_per_cell > 0:
                    if percell.get(cell, 0) >= args.judge_max_per_cell:
                        continue
                    percell[cell] = percell.get(cell, 0) + 1
                names = [TACTIC_NAMES[i] for i in row["tactic_ids"]]
                flip = rng.random() < 0.5  # randomise which side is shown as A
                pairs.append((item_id, tlabel, cell, row["neutral"], base_text,
                              str(row["edited"]), names, flip))

        def _pj(p):
            item_id, tlabel, cell, neutral, base_text, fudge_text, names, flip = p
            a_text, b_text = ((base_text, fudge_text) if flip
                              else (fudge_text, base_text))
            j = _paired_judge(neutral, a_text, b_text, names)
            w = j.get("winner")
            if w == "tie":
                score = 0.5
            elif w in ("A", "B"):
                fudge_side = "B" if flip else "A"
                score = 1.0 if w == fudge_side else 0.0
            else:
                score = float("nan")
            return cell, score, {"item_id": item_id, "tactic_label": tlabel,
                                 "alpha": cell[0], "beta": cell[1],
                                 "winner": w, "fudge_shown_as": ("B" if flip else "A"),
                                 "judge_response": j,
                                 "ts": time.strftime("%Y-%m-%d %H:%M:%S")}

        wins = defaultdict(list)
        workers = max(int(args.judge_workers), 1)
        print(f"[b2-score] paired judging {len(pairs)} pairs with {workers} workers")
        with JsonlWriter(paired_path) as pw, \
                ThreadPoolExecutor(max_workers=workers) as ex:
            for cell, score, rec in ex.map(_pj, pairs):
                pw.write(rec)
                if score == score:  # exclude NaN (skipped/errored calls)
                    wins[cell].append(score)

        paired_summary = {}
        for cell, scores in sorted(wins.items()):
            nsc = len(scores)
            paired_summary[f"a{cell[0]}_b{cell[1]}"] = {
                "fudge_win_rate": (sum(scores) / nsc) if nsc else float("nan"),
                "n": nsc,
                "wins": sum(1 for s in scores if s == 1.0),
                "ties": sum(1 for s in scores if s == 0.5),
                "losses": sum(1 for s in scores if s == 0.0),
            }
        Path(args.in_dir, "paired_summary.json").write_text(
            json.dumps(paired_summary, indent=2), encoding="utf-8")
        print(f"[b2-score] paired_summary.json: {len(paired_summary)} cells")

    def _num_or_str(x):
        # FUDGE cells carry numeric alpha/beta; the prompt-only baseline uses
        # the string "prompt_only". Keep whichever is meaningful.
        try:
            return float(x)
        except (TypeError, ValueError):
            return str(x)

    # FIX (S-m1): exclude empty rewrites from aggregation.
    n_empty = int(df["is_empty"].sum())
    agg = df[~df["is_empty"]].copy()
    if n_empty:
        print(f"[b2-score] excluded {n_empty} empty/whitespace rewrites from aggregation.")

    # FIX (S-C2): FUDGE decoding is deterministic (greedy), so extra seeds
    # produce identical rows. Bootstrapping over concatenated identical seeds
    # fabricates variance. Report seed count, and dedup exact-duplicate rows so
    # the bootstrap resamples the ITEM axis, not replicated identical rows.
    distinct_seeds = int(agg["seed"].nunique()) if "seed" in agg.columns else 1
    print(f"[b2-score] distinct seeds present: {distinct_seeds}")
    dedup_keys = ["item_id", "alpha", "beta", "tactic_label", "edited"]
    dedup_keys = [k for k in dedup_keys if k in agg.columns]
    before = len(agg)
    agg = agg.drop_duplicates(subset=dedup_keys)
    n_dupes = before - len(agg)
    if n_dupes:
        print(f"[b2-score] removed {n_dupes} exact-duplicate rows "
              f"(subset={dedup_keys}); bootstrapping on the {len(agg)} deduped rows.")
        if distinct_seeds > 1:
            print("[b2-score] WARNING: >1 seed present AND exact-duplicate rows "
                  "detected. Greedy decoding is deterministic, so multiple seeds "
                  "do not add real variance. Replication is over ITEMS via "
                  "bootstrap, not over seeds.")

    def _agg_condition(alpha, beta, tactic_label, grp):
        cond = {"alpha": _num_or_str(alpha), "beta": _num_or_str(beta),
                "tactic_label": tactic_label, "n": int(len(grp)), "metrics": {}}
        for m in metrics_cols:
            mean, lo, hi = bootstrap_mean_ci(grp[m].values, n_boot=args.n_boot)
            cond["metrics"][m] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
        return cond

    summary = {
        "n_empty_excluded": n_empty,
        "n_duplicate_rows_removed": n_dupes,
        "distinct_seeds": distinct_seeds,
        "names_tactic_leak_fraction": float(agg["names_tactic"].mean()) if len(agg) else float("nan"),
        "conditions": [],
    }
    # FIX (S-M4): per-tactic breakdown, grouped by (alpha, beta, tactic_label).
    for (alpha, beta, tactic_label), grp in agg.groupby(
        ["alpha", "beta", "tactic_label"], sort=True
    ):
        summary["conditions"].append(_agg_condition(alpha, beta, tactic_label, grp))
    # ...plus the marginal (alpha, beta) aggregate across all tactics.
    for (alpha, beta), grp in agg.groupby(["alpha", "beta"], sort=True):
        summary["conditions"].append(_agg_condition(alpha, beta, "__ALL_TACTICS__", grp))

    # Significance testing (addresses the journal's "no significance testing").
    # Every (alpha, beta) cell is compared to the no-guidance baseline (0,0) on
    # ITEM-and-tactic-matched pairs via the Wilcoxon signed-rank test, so we
    # report p-values on the ablation deltas, not just overlapping CIs.
    summary["significance_vs_baseline"] = {}
    if {"alpha", "beta", "item_id", "tactic_label"}.issubset(agg.columns):
        from scipy.stats import wilcoxon

        cells = sorted(set(zip(agg["alpha"], agg["beta"])))
        base = (0.0, 0.0) if (0.0, 0.0) in cells else (cells[0] if cells else None)
        if base is not None:
            base_df = agg[(agg["alpha"] == base[0]) & (agg["beta"] == base[1])] \
                .set_index(["item_id", "tactic_label"])
            for (a, b) in cells:
                if (a, b) == base:
                    continue
                cell_df = agg[(agg["alpha"] == a) & (agg["beta"] == b)] \
                    .set_index(["item_id", "tactic_label"])
                common = base_df.index.intersection(cell_df.index)
                for m in metrics_cols:
                    try:
                        x = base_df.loc[common, m].to_numpy(dtype=float)
                        y = cell_df.loc[common, m].to_numpy(dtype=float)
                        ok = ~(np.isnan(x) | np.isnan(y))
                        x, y = x[ok], y[ok]
                        if len(x) >= 6 and np.any(x != y):
                            st, p = wilcoxon(x, y)
                            summary["significance_vs_baseline"][
                                f"a{a}_b{b} vs a{base[0]}_b{base[1]} [{m}]"] = {
                                "p": round(float(p), 5), "n": int(len(x)),
                                "delta_mean": round(float(np.mean(y - x)), 4)}
                    except Exception:
                        pass

    # Clickbait score DISTRIBUTION per marginal cell (reviewer R2-M2b asked for
    # the distribution under different (lambda_pos, lambda_neg), not just a mean).
    summary["clickbait_distribution"] = {}
    dist_col = "clickbait_prob_external" if "clickbait_prob_external" in agg.columns \
        else "clickbait_prob_guide_circular"
    if dist_col in agg.columns:
        for (alpha, beta), grp in agg.groupby(["alpha", "beta"], sort=True):
            v = grp[dist_col].dropna().to_numpy(dtype=float)
            if len(v):
                summary["clickbait_distribution"][f"a{alpha}_b{beta}"] = {
                    "column": dist_col,
                    "deciles": [round(float(q), 4)
                                for q in np.quantile(v, np.linspace(0.1, 0.9, 9))],
                    "frac_ge_0.5": round(float((v >= 0.5).mean()), 4),
                }

    # Metrics split by QUESTION vs NON-QUESTION rewrite. Rhetorical questions
    # are a target tactic but they depress NLI and trip the naive hallucination
    # judge by form, not by infidelity; splitting shows FUDGE's real fidelity
    # cost is concentrated in (and inflated by) the question form.
    if "is_question" in agg.columns:
        summary["by_question_status"] = {}
        qmetrics = [m for m in ("nli_neutral_entails_edited", "nli_edited_entails_neutral",
                                "judge_hallucinates", "judge_attr_intensity",
                                "sts", "bertscore_f1") if m in agg.columns]
        for (alpha, beta, isq), grp in agg.groupby(["alpha", "beta", "is_question"], sort=True):
            key = f"a{alpha}_b{beta}"
            summary["by_question_status"].setdefault(key, {})
            summary["by_question_status"][key]["question" if isq else "non_question"] = {
                "n": int(len(grp)),
                "frac_of_cell": round(float(len(grp) / max((agg["alpha"] == alpha).sum(), 1)), 3),
                **{m: round(float(grp[m].mean()), 4) for m in qmetrics},
            }

    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[b2-score] tactic-name leak fraction (overall): "
          f"{summary['names_tactic_leak_fraction']:.3f}")

    header = ["alpha", "beta", "tactic", "n"] + metrics_cols
    lines = [
        "# B2 held-out rewrite evaluation summary",
        "",
        f"- Empty rewrites excluded: {n_empty}",
        f"- Exact-duplicate rows removed: {n_dupes}",
        f"- Distinct seeds present: {distinct_seeds}",
        f"- Tactic-name leak fraction (overall): {summary['names_tactic_leak_fraction']:.3f}",
        "",
        "Note: clickbait_prob_guide_circular and attr_realised_frac_guide_circular",
        "are CIRCULAR (guide models). Independent metrics are clickbait_prob_external",
        "and judge_attr_confirmed_frac.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for c in summary["conditions"]:
        cells = [f"{c['alpha']}", f"{c['beta']}", f"{c['tactic_label']}", f"{c['n']}"]
        for m in metrics_cols:
            v = c["metrics"][m]
            cells.append(f"{v['mean']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")
        lines.append("| " + " | ".join(cells) + " |")
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[b2-score] wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
