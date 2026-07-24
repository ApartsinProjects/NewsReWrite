# Track B: reviewer-response experiment package

This directory is a self-contained runbook for the empirical work that answers
reviewer concerns on "LLM-guided headline rewriting for clickability
enhancement without clickbait". Nothing here has been executed; every script
is prepared, argparse-driven, and prints an ETA before starting.

The runbook is ordered by ROI. Do B1 and B7 first; they are cheap and answer
the largest concerns. B2 is the expensive one and should go to Modal.

Common setup:

```bash
pip install -r requirements-track-b.txt
export NEWSREWRITE_REPO=E:/Projects/Submitted/NewsReWrite/NewsReWrite
python data/download_webis.py
python data/download_chakraborty.py
python data/download_isot_reuters.py
```

All output goes under `results/`. The bootstrap CI utility, path registry,
and BERT scorer loaders live in `common/`.

---

## B1 External-benchmark validation (start here)

Answers reviewer R1 ("only tested on your own synthetic data"). Two scripts:

```bash
# Clickbait scorer on Webis + Chakraborty
python b1_external_benchmarks/eval_clickbait_scorer.py --gpu

# Tactics scorer on Webis (graded correlation + clickbait vs neutral diff)
python b1_external_benchmarks/eval_tactics_scorer.py --gpu
```

- Inputs      : `data/raw/webis17/`, `data/raw/chakraborty16/`
- Outputs     : `results/b1_clickbait_external.{json,md}`, `results/b1_tactics_external.{json,md}`
- Wall-clock  : 30 s each on GPU, ~5 min each on CPU
- Cost        : $0 (local)

## B7 Reuters neutrality spot-check

Answers reviewer R2 ("your neutral label is a training artefact"). Runs the
clickbait detector over all ISOT True.csv Reuters headlines.

```bash
python b7_reuters_neutrality/spot_check.py --gpu
```

- Input       : `data/raw/isot/True.csv`
- Output      : `results/b7/reuters_neutrality.md`
- Wall-clock  : 10 s GPU, 2 min CPU
- Cost        : $0

## B3 Sweep aggregation

Turns the three pre-existing sweep CSVs at the repo parent into one tidy CSV
and a Pareto plot. If those CSVs do not carry the needed metrics it tells
you so and points at B2.

```bash
python b3_alpha_beta_sweep/aggregate_existing.py
```

- Wall-clock  : seconds
- Cost        : $0

## B4 Prompt-only baseline

Answers "is FUDGE actually doing anything beyond the prompt?" Generates the
same held-out set with plain Llama-3-8B-Instruct + a minimal rewrite prompt.

```bash
python b4_prompt_only_baseline/run.py --n-items 300 --seeds 42 43 44
python b2_held_out_rewrite_eval/score_rewrites.py --in-dir results/b4 \
       --out-json results/b4/summary.json --out-md results/b4/summary.md --gpu
```

- Wall-clock  : ~30 min on 4090 for generation, ~5 min for scoring
- Cost        : ~$1 on Modal A10G

## B2 Held-out quantitative FUDGE evaluation

Answers R1's "no CIs" and produces the numbers for the main results table.
Runs the four-cell (alpha, beta) grid on N=300 held-out neutrals x 3 seeds
x 3 tactic configs.

```bash
# Generate (send this to Modal via gpu2modal; local 4090 takes ~7 h)
python b2_held_out_rewrite_eval/run_rewrites.py --n-items 300 --seeds 42 43 44

# Score
python b2_held_out_rewrite_eval/score_rewrites.py --gpu
# Add --with-llm-judge only if you accept the cost (~$100 at GPT-4o).
```

- Wall-clock  : ~7 h on a single 4090, ~1.5 h on 4x A10G Modal
- Cost        : $3-5 on Modal, plus optional judge cost

## B5 Guide ablation

Falls out of B2 for free. `report.py` just reformats the summary.

```bash
python b5_ablation/report.py
```

## B6 DExperts / GeDi comparators (deferred)

Stubs only. See `b6_dexperts_gedi/README.md` for the design sketch and the
1-to-2-day implementation estimate.

---

## Reviewer-concern crosswalk

| Concern | Answered by |
| --- | --- |
| Generalisation of clickbait scorer beyond synthetic data | B1 |
| Do the 10 tactic attributes carry real signal? | B1 tactics |
| Effect size and confidence intervals for the main claim | B2 + B5 |
| Is the effect just prompt engineering? | B4 |
| Pareto trade-off engagement vs fidelity | B3 |
| "Neutral" is a training artefact | B7 |
| Comparison with other decoding-time controllers | B6 (deferred) |
