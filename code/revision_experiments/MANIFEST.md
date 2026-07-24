# Track B manifest

Every file in this package, with the reviewer concern it maps to.

## Root

| File | Purpose |
| --- | --- |
| README.md | Runbook, ROI-ordered, per-step commands and ETAs |
| MANIFEST.md | This file |
| requirements-track-b.txt | Pip additions on top of NewsReWrite/requirements.txt |

## common/

| File | Purpose |
| --- | --- |
| common/__init__.py | package marker |
| common/paths.py | central path/model registry, env-overridable |
| common/scorers.py | shared ClickbaitScorer and TacticsScorer loaders |
| common/bootstrap_ci.py | 95% percentile bootstrap CI utility |

## data/

| File | Reviewer concern | Purpose |
| --- | --- | --- |
| data/README.md | provenance, licenses |
| data/download_webis.py | R1 generalisation | fetches Webis-Clickbait-17 |
| data/download_chakraborty.py | R1 generalisation | fetches Chakraborty 2016 |
| data/download_isot_reuters.py | R2 neutrality | fetches ISOT True.csv Reuters headlines |

## b1_external_benchmarks/

| File | Reviewer concern | Purpose |
| --- | --- | --- |
| b1_external_benchmarks/__init__.py | package marker |
| b1_external_benchmarks/eval_clickbait_scorer.py | R1 | AUROC/F1/P/R + CIs on Webis + Chakraborty |
| b1_external_benchmarks/eval_tactics_scorer.py | R1 | per-attribute cb-vs-neutral + Spearman vs graded |

## b2_held_out_rewrite_eval/

| File | Reviewer concern | Purpose |
| --- | --- | --- |
| b2_held_out_rewrite_eval/run_rewrites.py | R1 CIs, R1 ablation | FUDGE decoding across 4-cell grid x 3 seeds |
| b2_held_out_rewrite_eval/score_rewrites.py | R1 CIs | STS, BERTScore, NLI both dirs, clickbait, attribute realisation, optional LLM judge |

## b3_alpha_beta_sweep/

| File | Reviewer concern | Purpose |
| --- | --- | --- |
| b3_alpha_beta_sweep/aggregate_existing.py | R1 Pareto | merges pre-existing sweep CSVs, draws Pareto plot |

## b4_prompt_only_baseline/

| File | Reviewer concern | Purpose |
| --- | --- | --- |
| b4_prompt_only_baseline/run.py | R1 baseline separation | plain Llama-3 + minimal rewrite prompt |

## b5_ablation/

| File | Reviewer concern | Purpose |
| --- | --- | --- |
| b5_ablation/README.md | R1 ablation | explains overlap with b2 grid |
| b5_ablation/report.py | R1 ablation | reformats b2 summary as ablation table |

## b6_dexperts_gedi/

| File | Reviewer concern | Purpose |
| --- | --- | --- |
| b6_dexperts_gedi/README.md | R1 comparator | design sketch, references, cost estimate |
| b6_dexperts_gedi/run_dexperts.py.stub | R1 comparator | CLI scaffold, not runnable |
| b6_dexperts_gedi/run_gedi.py.stub | R1 comparator | CLI scaffold, not runnable |

## b7_reuters_neutrality/

| File | Reviewer concern | Purpose |
| --- | --- | --- |
| b7_reuters_neutrality/spot_check.py | R2 neutrality artefact | scores ISOT True.csv, reports fraction above 0.5 |
