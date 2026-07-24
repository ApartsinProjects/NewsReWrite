# Revision Cycle 1 Plan

**Manuscript:** *LLM-guided headline rewriting for clickability enhancement without clickbait*
**Baseline:** `GuidedRewriteClickbait_v8.docx` (2026-03-11)
**Cycle 1 target:** `GuidedRewriteClickbait_v9_cycle1.docx` + a companion quantitative supplementary + a human-eval companion, addressing every point raised by Reviewer 1 and Reviewer 2.

---

## 0. Reviewer-concern crosswalk

Both reviewers converge on three demands; every concern maps to exactly one track.

| Concern | Reviewer | Track | Cycle 1 status |
|---------|----------|-------|----------------|
| R1-1 real-world clickbait dataset validation      | R1    | B   | Scaffolded; runs on Modal |
| R1-2 quantitative + human evaluation              | R1    | B+C | Scaffolded (B); rater packages generated (C) |
| R1-3 ablations + baseline rewriting methods       | R1    | B   | 4-cell (α,β) ablation + prompt-only baseline scaffolded |
| R2-M1 intro assertions uncited; rubric ungrounded | R2    | A   | **DONE** (edit A1) |
| R2-M2 rewrite claims only qualitative             | R2    | A+B+C | Softened in A8; quantified in B; human-checked in C |
| R2-M3 circularity of guide-model eval             | R2    | A+B | Acknowledged in A9; measured in B1 |
| R2-M4 no baselines, ablations, sweep              | R2    | B   | Ablation, sweep, prompt baseline in Track B; DExperts/GeDi deferred |
| R2-M5 missing reproducibility details             | R2    | A   | **DONE** (edits A2, A3, A6) |
| R2-M6 Table 2 tactic-label mismatch               | R2    | A+C | A7 owns it; C1 validates labels |
| R2-M7 weak attribute model, no error propagation  | R2    | A+B | A5 caption fix; B2 four-cell ablation quantifies contribution |
| R2-M8 Table 3 outputs contradict framework goals  | R2    | A   | **DONE** (edit A7 failure-modes section) |
| R2-m1 cross-reference errors                      | R2    | A   | Already resolved in v8 |
| R2-m2 Reuters-neutrality assumption               | R2    | A+B | A9 states it; B7 measures it |
| R2-m3 split stratification not stated             | R2    | A   | **DONE** (edit A4) |
| R2-m4 Figure 1 label vs caption mismatch          | R2    | A   | **DONE** (edit A5) |
| Stats rigour (single runs, no CIs, no tests)      | R2    | B   | 3 seeds + 95% bootstrap CIs on every reported number |

---

## Track A — Paper edits (COMPLETE)

**Goal.** Fix every reviewer complaint that can be answered by rewriting alone, plus preview the Track B and Track C deliverables in a Limitations subsection so the current text stops making unbacked claims.

**Deliverables (all shipped).**

| File | Purpose |
|------|---------|
| `Cycle1/paper_v9/GuidedRewriteClickbait_v9_cycle1.docx` | 9 Track A edits applied, every change highlighted in yellow (21 highlighted runs across 20 paragraphs). |
| `Cycle1/paper_v9/GuidedRewriteClickbait_v9_cycle1.html` | Self-contained HTML mirror with `<mark>` on the same passages. Light/dark aware. |
| `Cycle1/paper_v9/apply_track_a_edits.py` | Reproducible editor script (python-docx). |
| `Cycle1/paper_v9/docx_to_html.py`      | DOCX → HTML renderer. |
| `Cycle1/response_letter_cycle1.md`     | Point-by-point response, every concern tagged [A]/[B]/[C]. |
| `Cycle1/CHANGELOG_cycle1.md`           | 9-edit changelog with traceability matrix (edit → concern → paragraph → anchor phrase). |

**Edits applied.**

| ID | Reviewer concern | Section touched |
|----|------------------|-----------------|
| A1 | R2-M1            | Introduction paragraphs 1, 2, and rubric paragraph — added citations [1,2,3,9] and positioning of Table 1 as extension of [11]. |
| A2 | R2-M5            | Introduction paragraph 4 — named base LLM (Meta-Llama-3-8B-Instruct). |
| A3 | R2-M5, R2-M6     | Section 3.1.2 — appended synthetic-generation reproducibility details. |
| A4 | R2-m3            | Sections 3.1.3 and 3.1.4 — explicit split-stratification statement (random, source-headline level, not stratified). |
| A5 | R2-m4, R2-M7     | Figure 1 caption — reconciled with "F1 Score" y-axis label. |
| A6 | R2-M5            | New Section 3.3 "Implementation Details and Reproducibility" — base model, decoding config, BERT training hyperparameters, synthetic pipeline. |
| A7 | R2-M8            | New Section 4.1 "Observed Failure Modes" — three failure classes named explicitly. |
| A8 | R2-M2, R1-2      | Conclusion — softened "measurable gains in engagement salience"; forward-reference to Track B numbers. |
| A9 | R2-M3, R2-m2, R1-1 | New Section 5.1 "Limitations and Scope of the Guide Models" — Reuters domain, neutrality assumption, distribution-shift measurement forward reference. |

**No further Track A work required for this cycle.**

---

## Track B — New experiments without human labelers

**Goal.** Produce the quantitative numbers the paper currently lacks: external-benchmark AUROC/F1 for both guide models, held-out fidelity + engagement + independent-clickbait metrics for FUDGE rewrites across a full (α, β) ablation, plus a prompt-only baseline. Every number reported with 3 seeds and 95% bootstrap CIs.

### B.1 Deliverables (scaffolded)

Working directory: `Cycle1/track_b/`.

| Path | Purpose |
|------|---------|
| `README.md`, `MANIFEST.md`, `requirements-track-b.txt` | Runbook + inventory + pip pins. |
| `common/paths.py`, `scorers.py`, `bootstrap_ci.py`    | Shared path registry, BERT scorer loaders, 95% CI utility. |
| `data/download_{webis,chakraborty,isot_reuters}.py`   | Argparse-driven fetchers with `--dry-run` and provenance headers. |
| `b1_external_benchmarks/eval_clickbait_scorer.py`     | AUROC/F1/P/R on Webis-17 + Chakraborty-16 + graded Spearman on Webis. |
| `b1_external_benchmarks/eval_tactics_scorer.py`       | Per-attribute cb-vs-neutral bootstrap p-values + Spearman vs Webis truthMean. |
| `b2_held_out_rewrite_eval/run_rewrites.py`            | FUDGE 4-cell ablation grid × 3 seeds × 3 tactic configs. |
| `b2_held_out_rewrite_eval/score_rewrites.py`          | BERTScore, STS, NLI (bi-directional), independent clickbait, attribute-realization, optional LLM judge. |
| `b3_alpha_beta_sweep/aggregate_existing.py`           | Aggregates pre-existing sweep CSVs + Pareto plot. |
| `b4_prompt_only_baseline/run.py`                      | Plain Llama-3-8B rewrite with minimal prompt. |
| `b5_ablation/report.py`                               | Reformats b2 summary into ablation table (grid == ablation). |
| `b6_dexperts_gedi/{README.md,*.stub}`                 | Stubs + design sketch; deferred to Cycle 2. |
| `b7_reuters_neutrality/spot_check.py`                 | External clickbait detector over full ISOT True.csv. |

Local sanity: all 10 argparse `--help` outputs work; b3 aggregator runs end-to-end on real inputs; b1 pipeline (data → BERT → bootstrap CI → JSON + Markdown) verified with a stub BERT in 5m26s CPU on the real Webis + Chakraborty data.

### B.2 Datasets (cached locally, mirror-verified)

| Dataset | Cached at | Rows | Verified mirror |
|---------|-----------|------|-----------------|
| Webis-Clickbait-17 (train, annotated) | `data/raw/webis17/` | 2,459 tweets with graded scores | Zenodo record 5530410, file `clickbait17-train-170331.zip` |
| Chakraborty 2016 "Stop Clickbait"     | `data/raw/chakraborty16/` | 32,000 clickbait + 26,000 non-clickbait | GitHub `bhargaviparanjape/clickbait` (.gz + gunzip) |
| ISOT True.csv (Reuters half)          | `data/raw/isot/True.csv` | 21,417 headlines (politicsNews + worldnews) | GitHub `marwaa123/fake_real_news` mirror |

Downloaders repaired to use the verified URLs; Modal `fetch_external_datasets` succeeds unattended.

### B.3 Modal package

Working directory: `Cycle1/track_b/modal_run/`.

| Path | Purpose |
|------|---------|
| `app.py`                                | Single Modal app; one persistent Volume `newsrewrite-track-b-store`; 12 functions. |
| `upload_artifacts.py`                   | Local uploader for pre-trained guides, source neutrals, ISOT True.csv (optional if using in-Modal training). |
| `train_guides/extract_source_neutrals.py` | Derives `source_neutrals.csv` (stratified 12,600 rows) from ISOT True.csv. |
| `train_guides/generate_synthetic.py`    | Argparse port of the repo's GPT-4o-mini generator; resume-by-default. |
| `train_guides/build_prefix_datasets.py` | One-pass builder for both binary and tactics prefix splits, no leakage. |
| `train_guides/train_clickbait.py`, `train_tactics.py` | BERT-base fine-tuners with length-aware weighted loss. |
| `README.md`                             | Deploy + run guide, wall-clock and cost table. |

**Modal functions.** `check_env`, `list_artifacts`, `fetch_external_datasets`, `extract_source_neutrals`, `regenerate_synthetic`, `train_guides`, `run_b1_clickbait`, `run_b1_tactics`, `run_b7`, `run_b4`, `run_b2`, `run_score`, `prepare_human_labeling`, `pull_results`.

**Run order (self-contained, no local uploads needed after HF + OpenAI secrets are set):**

```
modal secret create huggingface   HF_TOKEN=hf_...
modal secret create openai-key    OPENAI_API_KEY=sk-...
modal secret create llm-judge-keys ANTHROPIC_API_KEY=...   # optional
modal deploy app.py

modal run app.py::fetch_external_datasets      # ~2 min  CPU
modal run app.py::regenerate_synthetic         # ~1 h    auto-extracts neutrals
modal run app.py::train_guides                 # ~2 h    T4
modal run app.py::run_b1_clickbait             # ~5 min  T4
modal run app.py::run_b1_tactics               # ~5 min  T4
modal run app.py::run_b7                       # ~2 min  T4
modal run app.py::run_b4  --n-items 300        # ~30 min L40S
modal run app.py::run_b2  --n-items 300        # ~90 min L40S
modal run app.py::run_score --in-subdir b4     # ~15 min L4
modal run app.py::run_score --in-subdir b2     # ~15 min L4
modal run app.py::prepare_human_labeling       # ~30 s   CPU (feeds Track C)
modal run app.py::pull_results
```

### B.4 Cost & wall-clock

| Step | GPU | Wall-clock | ~USD |
|------|-----|-----------|------|
| Data + synthesis + training | CPU + T4 | ~3 h | ~$6 (mostly GPT-4o-mini) |
| B1 clickbait + tactics + B7 | T4 × 3 | ~12 min | ~$0.15 |
| B4 generation | L40S | ~30 min | ~$1 |
| B2 generation | L40S | ~90 min | ~$3 |
| B4 + B2 scoring | L4 × 2 | ~30 min | ~$0.30 |
| **Total Cycle 1 Track B** | | **~5.5 h** | **~$10** |
| Optional LLM-as-judge (GPT-4o over 300 × 4 × 3) | — | +2 h API | +$50–100 |
| Optional DExperts + GeDi (Cycle 2) | — | +2 days coding + ~$5 | — |

### B.5 Remaining work before pressing "go"

1. **Apply the five code-review fixes** identified below (Modal code-review section). ~20 min.
2. **Extend `prepare_human_labeling` to accept multiple methods** so C2 covers FUDGE and prompt-only on the same source items (see Track C). ~15 min.
3. **Optional: freeze HuggingFace cache on the Volume** (`HF_HOME=/workspace/store/.cache/hf`) so re-runs skip the ~16 GB Llama-3 download.
4. Then execute the Modal run order above.

### B.6 Deferred to Cycle 2

- **DExperts** and **GeDi** comparators. See `Cycle1/track_b/b6_dexperts_gedi/README.md`. 1–2 agent-days of implementation each (mostly training the extra expert / anti-expert / class-conditional LMs; the decoding-loop diff is ~30 lines against `b2/run_rewrites.py`). Same Modal app, same volume, same scoring pipeline. Include if reviewers push back after seeing the FUDGE-vs-prompt-only numbers.

---

## Track C — Human labeling

**Goal.** Give reviewers the two things automatic metrics cannot: human judgment on which of the paper's 10 tactics are actually realized in the synthetic training data (R2-M6), and human ratings of engagement / faithfulness / perceived clickbait on FUDGE rewrites (R2-M2d, R1-2).

### C.1 Deliverables

| Path | Purpose |
|------|---------|
| `Cycle1/track_b/human_labeling_prep/prepare.py` | Argparse tool that consumes `synthetic_clickbait.csv` and/or `per_item_scores.csv` and emits both study packages. Locally smoke-tested. |
| Modal function `prepare_human_labeling`         | Runs the tool on the Volume once b2 and regenerate_synthetic have completed. |
| Volume output `results/human_labeling/c1_rubric_validation/` | oracle.csv/jsonl + rater_{01..N}.csv + codebook.md. |
| Volume output `results/human_labeling/c2_rewrite_quality/`   | oracle.csv/jsonl + rater_{01..N}.csv + codebook.md. |

### C.2 Study C1 — Tactic-label validation (R2-M6)

- **Task per item.** Rater sees `(source, rewrite)` and marks which of the 10 engagement tactics are realized in the rewrite (0/1 per tactic).
- **Sample.** 150 synthetic clickbait rows from `data/synthetic_clickbait.csv`, stratified by the number of activated tactics (1, 2, or 3).
- **Design.** Full-overlap rater assignment (every rater sees every item). Order shuffled per rater. `task_id` deterministic SHA-1 of `(study, index, source, rewrite)`.
- **Ground truth.** Oracle keeps `intended_tactic_vector` and `intended_tactic_names`; rater does not see these.
- **Analysis.** Per-tactic Cohen's κ, overall % agreement, per-tactic recall of intended-tactic-vector.
- **Cost.** ~$100–200 crowdsourced (or ~4 h per rater in-house). 2–3 raters.

### C.3 Study C2 — Rewrite quality (R2-M2d, R1-2)

- **Task per item.** Rater sees `(source, rewrite)` blind to condition and rates rewrite on three 1–5 Likert scales: engagement, faithfulness, perceived clickbait.
- **Sample.** 100 source headlines × up to K methods each (see multi-method extension below); target ~400 total rewrites per rater at K = 4 conditions.
- **Design.** Full-overlap rater assignment. Order shuffled per rater. Condition is hidden from the rater; oracle keeps `condition_alpha`, `condition_beta`, `condition_label`, `gen_top_k`, `gen_seed`, `gen_tactic_ids`, plus every automatic scorer output prefixed `auto_*` for cross-checking.
- **Analysis.** Per-condition means + 95% CI, paired Wilcoxon between conditions, ICC(2,k) for reliability.
- **Cost.** ~$200–400 crowdsourced. 3 raters.

### C.4 Multi-method extension (planned; see Track B remaining work)

Extend `prepare_human_labeling` and `prepare.py` so C2 samples *source headlines* once and then includes rewrites from every available method (b2 FUDGE cells, b4 prompt-only, and eventually b6 DExperts / GeDi) for each. This gives within-source method comparisons and multiplies statistical power roughly by 1/(1−ρ) where ρ is the within-source correlation. Rater format stays flat (independent 1–5 scores per rewrite); an optional within-source ranking task can be added as C3 in Cycle 2.

### C.5 Rater flow

1. Track B produces `synthetic_clickbait.csv` (for C1) and `results/b2/per_item_scores.csv` (for C2).
2. `modal run app.py::prepare_human_labeling` emits both packages into the Volume.
3. `modal volume get newsrewrite-track-b-store results/human_labeling ./cycle1_hlp` pulls them locally.
4. Distribute `rater_{01..N}.csv` + `codebook.md` to raters (crowdsourced via `toloka-labeling` or `human-labeling` skill, or in-house via any spreadsheet).
5. Rater returns completed CSV; analyst joins on `task_id` with oracle to compute agreement statistics.

### C.6 Remaining work

1. Multi-method extension to `prepare.py` and `app.py::prepare_human_labeling` (planned, ~15 min).
2. Choose crowd-sourcing platform (Toloka via `toloka-labeling` skill vs manual in-house). Toloka is faster but $300–600; in-house is free but slower.
3. Recruit and pay raters. 1 week including QA and adjudication of disagreements.
4. Analysis notebook (κ / ICC / paired Wilcoxon). Small; ~half day.

---

## Modal code review (findings that need to land before the run)

Full details in the code-review conversation; these are the bugs and inefficiencies that matter.

### Bugs

**B1. `prepare_human_labeling` sentinel is wrong.**
`if len(cmd) == 8` never fires (base cmd has 12 items). Empty study still spawns the subprocess and errors non-friendly. Fix: check `if not synth.exists() and not per_item.exists()` before building the command.

**B2. `run_score` fails without `JUDGE_SECRET` even when the judge is off.**
Secret is mounted unconditionally at function-start time; if the user never created `llm-judge-keys`, Modal raises `NotFoundError` before doing any work. Fix: split into `run_score` (no secret) and `run_score_with_judge` (with secret) or pass an empty `Secret.from_dict({})` when the flag is False.

**B3. Auto-invocation of extractor ignores caller's seed.**
`regenerate_synthetic`'s fallback that calls `extract_source_neutrals.py` hardcodes `--seed "42"` instead of accepting the caller's seed. Add a `seed: int = 42` param and thread it through.

### Inefficiencies

**E1. `add_local_dir` uploads the entire tree, including cached raw datasets.**
Copies ~70 MB of Webis + Chakraborty + ISOT + local results into every image rebuild. Fix: `ignore=["data/raw/**", "results/**", "modal_run/**", "__pycache__/**"]`.

**E2. HuggingFace cache is not on the Volume.**
Every cold GPU invocation re-downloads Llama-3-8B (~16 GB) and the scoring models (~4 GB). Fix: set `HF_HOME=/workspace/store/.cache/hf` in `_env()`; the first run pays the cost, every subsequent run reads from the Volume.

**E3. Redundant `STORE.commit()` calls.**
Modal Volumes auto-commit on function exit; explicit `STORE.commit()` at the end of every function is a no-op. Cosmetic.

### Inconsistencies

**I1. Mixed argument styles.** `seeds: str = "42,43,44"` in `run_b2/b4` vs `n_boot: int` in `run_b1_*` vs individual ints elsewhere. Modal CLI supports `list[int]` natively; unify.

**I2. Misleading comment on `NEWSREWRITE_REPO`.** The variable IS used by `common/paths.py`; the "not used, harmless" comment is wrong.

**I3. Timeouts vary without a pattern.** Cosmetic.

**Recommendation.** Apply B1 + B2 + B3 + E1 + E2 in one 20-minute pass. Skip inconsistencies unless polishing.

### Round 1 fixes: APPLIED

B1, B2, B3, E1, E2 all applied. `run_score` split into `run_score` (no secret) and `run_score_with_judge`. `add_local_dir` now ignores raw data / results / modal_run. HF cache pinned to the Volume via `HF_HOME`. `regenerate_synthetic` threads `seed`. `prepare_human_labeling` sentinel fixed and extended to multi-method.

### Round 2 inspection: additional bugs found and fixed

A second pass cross-checking the actual column contracts between `run_rewrites.py`, `score_rewrites.py`, `b4/run.py`, and `prepare.py` surfaced five more real defects, all now fixed:

1. **Schema mismatch, C2 would crash.** `prepare.py` C2 required `source_id` and `source`, but the pipeline produces `item_id` and `neutral`; every auto-score column name also differed (`sts` vs `sts_cos`, `clickbait_prob` vs `clickbait_prob_external`, etc.). Added `_normalize_c2_frame` alias layer in `prepare.py`; C2 now ingests the real schema and carries all six auto-scores through.
2. **Prompt-only baseline breaks numeric casts.** `b4/run.py` writes `alpha=beta="prompt_only"` (strings). `prepare.py` `float(alpha)` and `score_rewrites.py` `float(alpha)` both crashed. Added `_as_float` / `_num_or_str` guards; the string is surfaced as the condition label instead.
3. **b4 rewrite files never matched the scorer glob.** `score_rewrites.py` globbed `rewrites_a*_b*_s*.csv` but b4 writes `rewrites_prompt_only_s*.csv`, so `run_score --in-subdir b4` raised FileNotFoundError. Broadened glob to `rewrites_*.csv` (alpha/beta read from columns, so safe).
4. **C1 top-up could duplicate items.** After `reset_index`, `df.drop(sample.index)` dropped the wrong rows and the top-up re-added sampled rows. Rewrote the stratified sampler to track the shuffled-df index; verified 60 requested -> 60 unique, balanced 20/20/20 across 1/2/3-tactic buckets.
5. **pandas FutureWarning** on the groupby-apply in the C1 sampler; silenced by explicit column selection.

All 17 Track B Python files parse; C1 and multi-method C2 verified end-to-end against fixtures built to the real pipeline schema.

---

## Order of operations

```
Day 1  (in this session, mostly done)
  [x] Track A: 9 edits applied, DOCX + HTML shipped, response letter + changelog written.
  [x] Track B: scaffolding complete, datasets cached locally, Modal package deployed-ready.
  [x] Track C: rater package generator scaffolded and smoke-tested.

Day 2  (before pressing "go" on Modal)
  [ ] Apply the 5 code-review fixes above.
  [ ] Extend prepare_human_labeling to accept multiple methods.
  [ ] modal secret create huggingface + openai-key.
  [ ] modal deploy app.py.
  [ ] modal run app.py::fetch_external_datasets.

Day 2-3  (Modal execution, mostly unattended)
  [ ] regenerate_synthetic  (~1 h)
  [ ] train_guides           (~2 h)
  [ ] run_b1_clickbait, run_b1_tactics, run_b7  (~15 min)
  [ ] run_b4, run_b2                            (~2 h)
  [ ] run_score for b4 and b2                   (~30 min)
  [ ] prepare_human_labeling                    (~30 s)
  [ ] pull_results

Day 3-4  (analysis + paper update)
  [ ] Aggregate B1, B2, B3, B4, B5, B7 into the Cycle 2 quantitative supplementary.
  [ ] Update Section 4 in the paper with the Track B numbers; retire the qualitative-only Table 4 in favour of a quantitative main-results table.
  [ ] Fill in the Section 5.1 Limitations block with the actual B1 external-benchmark numbers.

Week 2  (Track C human labeling)
  [ ] Choose platform (Toloka vs in-house).
  [ ] Recruit raters, distribute packages.
  [ ] Collect responses, compute κ and ICC.
  [ ] Fold C1 numbers into the response letter's R2-M6 reply.
  [ ] Fold C2 numbers into the response letter's R2-M2d and R1-2 replies.

Cycle 2 submission
  [ ] Update GuidedRewriteClickbait_v10 with quantitative + human-eval results.
  [ ] Consider adding DExperts + GeDi if reviewers ask for it post-Cycle 1.
```

---

## Success criteria for Cycle 1 resubmission

The paper is ready to resubmit when all of the following hold. Track A is complete when the first two land; Track B is complete when items 3–7 land; Track C is complete when items 8–9 land.

1. Every yellow-highlighted passage in `v9_cycle1.docx` addresses at least one concern in the crosswalk. ✅
2. Every concern in the crosswalk has at least one edit or planned experiment answering it. ✅
3. Both guide models report AUROC/F1/P/R with 95% CIs on Webis-Clickbait-17 and Chakraborty-16.
4. At least one non-FUDGE baseline (prompt-only) is reported on the same held-out set.
5. The four-cell (α, β) ablation reports BERTScore, STS, NLI, independent clickbait, attribute realization, each with 95% CIs across 3 seeds.
6. A Pareto plot showing engagement vs fidelity vs clickbait across the (α, β) sweep.
7. Reuters neutrality spot-check reports fraction of ISOT True.csv predicted as clickbait by the external detector.
8. C1 reports per-tactic Cohen's κ over 150 items × 2+ raters.
9. C2 reports per-condition Likert means with 95% CI and ICC(2,k) over 100 items × 3 raters.
