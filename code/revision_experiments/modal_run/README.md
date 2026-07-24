# Modal run guide

Two files:
- [app.py](app.py) — one Modal app, one persistent Volume, one function per experiment.
- [upload_artifacts.py](upload_artifacts.py) — local helper to push guide-model weights and the held-out neutrals CSV into the Volume.

Wall-clock and cost estimates are Modal L40S (b2/b4) / T4 (b1/b7); halve b2 wall-clock on a single H100.

## One-time setup

```bash
pip install modal
modal token new                       # sign into Modal
modal secret create huggingface HF_TOKEN=hf_...           # for gated Llama-3-8B
modal secret create llm-judge-keys ANTHROPIC_API_KEY=... OPENAI_API_KEY=...  # optional
modal deploy app.py                   # publishes the app + creates the Volume
```

Then upload the finetuned guide models and the held-out neutrals CSV from your local machine:

```bash
python upload_artifacts.py \
  --clickbait-dir  ~/…/bert_clickbait_prefix_finetuned \
  --tactics-dir    ~/…/bert_tactics_prefix_finetuned \
  --source-neutrals ~/…/combined_news_test.csv \
  --isot-true      ~/…/True.csv          # only if you want b7
```

Sanity: `modal run app.py::check_env` prints the volume contents from inside the image.

## Reproducing the guide models from scratch

Only needed if you have NOT already uploaded pre-trained guides via `upload_artifacts.py`. Every step is idempotent: it detects existing artifacts on the Volume and skips itself, so re-runs after a partial failure are cheap.

```bash
# 0) Register the OpenAI key used to synthesize the training data
modal secret create openai-key OPENAI_API_KEY=sk-...

# 1) Derive data/source_neutrals.csv from data/isot/True.csv
#    (skipped automatically if source_neutrals.csv already present)
modal run app.py::extract_source_neutrals    # ~30 s, CPU

# 2) Generate ~12,600 synthetic clickbait rewrites of source_neutrals.csv
#    (writes /workspace/store/data/synthetic_clickbait.csv)
#    Automatically triggers step 1 if source_neutrals.csv is missing.
modal run app.py::regenerate_synthetic       # ~1 h wall-clock, ~$5 GPT-4o-mini

# 3) Build the prefix splits and fine-tune both BERT guides
#    (writes /workspace/store/models/bert_{clickbait,tactics}_prefix_finetuned/)
modal run app.py::train_guides               # ~2 h wall-clock on T4, ~$1

# Peek at what's on the volume after any step:
modal run app.py::list_artifacts
```

`extract_source_neutrals` accepts `--max-rows` (default 12,600, stratified by ISOT `subject`) and `--seed`. `regenerate_synthetic` accepts `--max-rows`, `--batch-size` and honours `OPENAI_API_URL` / `OPENAI_API_MODEL` for proxies or alternate models. `train_guides` accepts `--force` (and `regenerate_synthetic --force`) to rebuild prefix splits or retrain even when outputs already exist.

## Run order

Cheap external validation first, expensive rewrite generation last.

```bash
# 1) Fetch Webis-17 + Chakraborty-16 into the Volume  (~2 min, CPU)
modal run app.py::fetch_external_datasets

# 2) Guide-model external evaluation                  (~5 min each, T4)
modal run app.py::run_b1_clickbait
modal run app.py::run_b1_tactics

# 3) Reuters neutrality spot-check                    (~2 min, T4)
modal run app.py::run_b7

# 4) Prompt-only baseline generation                  (~30 min, L40S)
modal run app.py::run_b4 --n-items 300

# 5) FUDGE 4-cell grid generation (the big one)       (~90 min, L40S)
modal run app.py::run_b2 --n-items 300 --top-k 50

# 6) Score both generation runs                       (~15 min each, L4)
modal run app.py::run_score --in-subdir b4
modal run app.py::run_score --in-subdir b2
# For the optional LLM-as-judge pass (needs the llm-judge-keys secret):
#   modal run app.py::run_score_with_judge --in-subdir b2

# 7) Prepare Track C human-labeling packages (~30 s, CPU)
#    Emits results/human_labeling/{c1_rubric_validation,c2_rewrite_quality}/
#    Each subdir contains oracle.csv (analyst side, full provenance +
#    auto-scores + intended tactics + generation params + method), N
#    rater_{k}.csv (rater side, blinded to method AND condition,
#    position-shuffled), and codebook.md.
#    c2-methods lists which scored result dirs to include; each source
#    headline is rated once per method so raters compare methods on the
#    same content (blinded).
modal run app.py::prepare_human_labeling \
    --c1-n-items 150 --c2-n-items 100 --n-raters 3 --c2-methods b2,b4

# 8) Pull the results back to your laptop
modal run app.py::pull_results   # prints the modal-volume-get command
modal volume get newsrewrite-track-b-store results ./cycle1_results
```

## Track C human-labeling data

`prepare_human_labeling` produces two study packages:

**C1 tactic-label validation** (reviewer R2-M6).
Rater sees a synthetic clickbait rewrite and marks which of the 10 engagement tactics are realized. Compare against `intended_tactic_vector` in `oracle.csv` and report per-tactic Cohen's κ. Requires `data/synthetic_clickbait.csv` (produced by `regenerate_synthetic`).

**C2 rewrite quality** (reviewer R2-M2d, R1-2).
Rater sees a source + rewrite pair and scores engagement / faithfulness / perceived clickbait on 1–5 Likert scales. Rewrites come from every method listed in `--c2-methods` (default `b2,b4`) and, within FUDGE, from all four (α, β) conditions of `run_b2`; the rater is blinded to both the method and the cell. Each sampled source headline is rated once per method-condition, so the same content is compared across methods, giving within-source method comparisons. Requires `results/<method>/per_item_scores.csv` for each method (produced by `run_score --in-subdir <method>`); missing methods are skipped with a warning.

Every package is self-contained: `oracle.csv` (analyst-side, keeps `method`, every generation parameter, source, intended-tactic vector, and auto-score for cross-checking), `rater_{k:02d}.csv` (rater-side, no method/condition/ground-truth and no auto-scores, presentation order shuffled per rater), and `codebook.md` (rating instructions and tactic definitions).

## Rough total cost

| step | GPU | wall-clock | ≈ USD |
|------|-----|-----------|-------|
| datasets fetch | none | 2 min | $0 |
| b1 clickbait | T4 | 5 min | $0.05 |
| b1 tactics   | T4 | 5 min | $0.05 |
| b7           | T4 | 2 min | $0.02 |
| b4 generate  | L40S | 30 min | $1 |
| b2 generate  | L40S | 90 min | $3 |
| score b4+b2  | L4 | 30 min | $0.30 |
| prepare_human_labeling | none | 30 s | $0 |
| **total**    |    | **~2.5 h** | **~$5** |

Add ~$50–100 if you enable `--with-llm-judge` at GPT-4o rates on 300 items × 4 conditions × 3 seeds.

## What lives where

- Volume `newsrewrite-track-b-store`:
    - `models/` — user-uploaded finetuned guide models (persistent)
    - `data/` — datasets fetched or uploaded (persistent)
    - `results/` — every JSON, MD, CSV, and PNG produced (persistent)
- Image `newsrewrite-track-b`: reproduces `requirements-track-b.txt` plus torch/transformers pinned to the versions matching the repo.

## Fallbacks

- **Missing guide models.** If you never trained them: they can be reproduced by running the repo's `models/bert_train_binary_prefix.py` and `bert_train_tactics_prefix.py` on Modal too. Add a `train_guides` function to `app.py` when you need it; the wiring is trivial (same image, same volume, gpu=`T4`, ~30 min each). Not included here to keep the initial surface small.
- **Missing ISOT True.csv.** `fetch_external_datasets` tries HuggingFace and UVic; if both 404, print a warning and skip. Upload manually via `upload_artifacts.py --isot-true …`.
- **HF gated-model refusal for Llama-3.** Make sure your HF account has accepted the Meta-Llama license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct before running b2/b4.
