# Lean-mode runbook — experiments #3 (second base model) and #4 (second domain)

Prepared and ready to launch. **Blocker: Modal is not authenticated** in this
environment (`modal token current` is empty). Launching needs a one-time
interactive step from you:

```bash
modal setup        # opens a browser OAuth; run once
```

After that, the commands below run the two lean experiments (~150 items each,
independent-metric scoring only, A10G). Estimated ~$8–12 combined, ~2–3 h cloud.

## Shared prerequisites (verify once, after `modal setup`)
- The trained guides (`bert_clickbait_prefix_finetuned`, `bert_tactics_prefix_finetuned`)
  and the external DistilBERT detector must be on the Modal volume used by the
  b2 app (they were, from the main run; re-upload from `models/` if the volume
  was cleared).
- Runner: `code/revision_experiments/b2_held_out_rewrite_eval/run_rewrites.py`
  (reads a CSV with a `title` column; `--n-items`, `--alpha-on/--beta-on`,
  `--neutral-prompt`, `--top-k`, `--objective log`).

## #4 — Second domain (human-authored, non-Reuters neutrals)
Source already built: `data/processed/source_neutrals_seconddomain150.csv`
(150 Chakraborty non-clickbait headlines: disasters, culture, markets — a
different register from Reuters political newswire).

```bash
# dual operating point (4,2) and the no-guidance baseline (0,0), 150 items
python run_rewrites.py --test-csv data/processed/source_neutrals_seconddomain150.csv \
    --n-items 150 --objective log --alpha-on 4 --beta-on 2 --top-k 50 \
    --out results/secondomain/
# score with INDEPENDENT metrics only (local/cheap: external detector + BERTScore + NLI + ppl)
python score_rewrites.py --in results/secondomain/ --skip-judge   # drop LLM judge in lean mode
```
Expected artifact: `results/secondomain/per_item_scores.csv` → new Table 9
"Cross-domain rewriting" + a §4.12 paragraph. Read: does clickbait stay low and
fidelity stay high on out-of-domain neutrals? (If it transfers weaker, that is an
honest, quantified generality limit.)

## #3 — Second base generator (model-agnosticism)
Guides are reused unchanged; only the base LLM changes via one env var:

```bash
export BASE_LLM_ID=mistralai/Mistral-7B-Instruct-v0.3     # fallback: Qwen/Qwen2.5-7B-Instruct
# same 150 held-out Reuters neutrals, dual point + baseline
python run_rewrites.py --n-items 150 --objective log --alpha-on 4 --beta-on 2 \
    --top-k 50 --out results/secondbase_mistral/
python score_rewrites.py --in results/secondbase_mistral/ --skip-judge
```
Note: re-tuning is not required for a generality check, but if the (4,2) point
looks off on the new base model, run a small `--sweep "0,2,4 x 0,2"` first and
pick the winner on independent metrics before reporting.
Expected artifact → Table 10 (or added rows to Table 6) "FUDGE on a second base
generator".

## Lean-mode levers applied here
- `--n-items 150` (half of 300): ~2× cheaper, CIs still adequate for a
  "pattern holds" claim.
- `--skip-judge`: the LLM tactic-judge is the API cost; independent detector +
  BERTScore + NLI + perplexity are enough for these checks and run locally.
- A10G (7B fits FP16); cap Modal concurrency ~6–8; runs are idempotent
  (skip-if-present) so a throttled run resumes cheaply.

## After each run (standing integration checklist)
1. Fetch `per_item_scores.csv` locally.
2. Recompute bootstrap CIs + effect sizes with the existing audit script.
3. Rebuild the highlighted HTML (`build_paper.py` → `build_paper_body.py`), add
   the new table, re-run the 5-point audit, push.
4. Update the Zenodo deposit manifest.
