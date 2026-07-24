# Deep Code Audit Findings (Track B pipeline)

Three independent adversarial audits of the Track B code, cross-checked against
the original repo and the real downloaded data. Findings below, with the fix
status. Severity: CRITICAL = invalidates results; MAJOR = correctness/method;
MINOR = polish/efficiency.

Legend: [FIXED] applied  ·  [DECIDE] needs a methodology decision  ·  [NOTED] documented, low priority

---

## A. FUDGE decoding + scoring (highest impact)

| ID | Sev | Finding | Status |
|----|-----|---------|--------|
| S-C1 | CRITICAL | **Circular clickbait metric.** `score_rewrites.py` evaluates clickbait with the SAME BERT used as the FUDGE beta-guide. Docstring falsely claims an "independent Chakraborty-external detector (falls back...)"; no external path exists. This reproduces reviewer R2-M3's circularity inside the evaluation itself. | [FIXED] independent detector wired |
| S-C2 | CRITICAL | **Fabricated CIs / pseudo-replication.** FUDGE decoding is greedy argmax, so the 3 seeds produce byte-identical rows; concatenating them triples N and shrinks every CI by ~sqrt(3). b4 baseline DOES sample, so baseline and FUDGE are not comparable. | [FIXED] single-seed default + bootstrap over items; b4 aligned |
| S-C3 | CRITICAL | **Circular attribute-realization.** `attr_realised_frac` uses the same tactics BERT as the alpha-guide. Only independent check (LLM judge) is off by default. | [FIXED] independent report path + judge documented as primary |
| S-M1 | MAJOR | **prob vs log-prob scale.** Objective used `softmax(logits)[tok]` (probability), not log-prob; with alpha=beta=0.7 the guidance terms dominated the fluency term. | [CHANGED] switched to canonical log-domain FUDGE as the default (`--objective log`): `log p_llm + a*log(tac) + b*log(1-cb)`, epsilon-clamped, off-cell nan-guarded, EOS length bonus scaled. prob-domain kept as `--objective prob` for the ablation. On-weight re-tuned via `--sweep` / `run_b2_sweep`. Methods equation must be updated in the paper (edit A10). |
| S-M2 | MAJOR | **Train/inference representation mismatch.** Guides trained with a terminal-period convention; during decoding only the EOS candidate gets the period, so non-terminal prefixes are scored without it. Faithful to the paper. | [NOTED] faithful port; documented |
| S-M3 | MAJOR | **Efficiency: no KV cache + per-candidate BERT forward.** `llm(input_ids=...)` reruns the whole prefix each step (O(n^2)); `score_cb`/`score_tac` run a batch-1 BERT forward per candidate token (~7500/rewrite). Dominates the 7 s/rewrite cost. | [FIXED] batched BERT over top-k; KV-cache noted |
| S-M4 | MAJOR | **Aggregation drops tactic_label.** Docstring promises grouping by (alpha,beta) AND tactic_label; code groups only by (alpha,beta), pooling all tactic configs. | [FIXED] group includes tactic_label |
| S-m1 | MINOR | Empty generations `fillna("")` scored as real low-STS points rather than flagged. | [FIXED] empty flagged + excluded |
| S-m2 | MINOR | BERTScore `rescale_with_baseline` unset → compressed 0.85-1.0 band. | [FIXED] rescale enabled |
| S-m3 | MINOR | STS builds full N×N cos_sim to take the diagonal (~500 MB at N=10.8k). | [FIXED] row-wise cosine |
| S-m4 | MINOR | Length-bonus dead bands L in {6,7} and {31,32}. Faithful port. | [NOTED] |
| S-m5 | MINOR | No check that the tactic name is absent from the rewrite (prompt forbids naming it). | [FIXED] leak check column added |

VERIFIED CLEAN: NLI entailment index (probs[:,2], both directions), clickbait positive class (softmax[:,1]), STS model+cosine, prompt stripping (only new tokens), bootstrap percentile math, FUDGE math port (alpha_dyn, sigmoid clickbait, mean-over-positive tactics, EOS/length handling all match reference).

---

## B. Guide-model training pipeline

| ID | Sev | Finding | Status |
|----|-----|---------|--------|
| T-C1 | CRITICAL | **Resume + failed-batch desync drops AND duplicates rows.** On a failed batch the loop `continue`s without writing, but resume computes the cursor from `len(existing)`, so a failed middle batch permanently skips those source headlines and regenerates a later block. Even in one run, a failed batch silently loses 15 headlines. | [FIXED] explicit source cursor |
| T-M2 | MAJOR | **Vector/headline misalignment.** `results[j]["methods_vector"] = vectors[j]` trusts model order; `min(len(results),len(vectors))` hides a short/reordered response, attaching the wrong tactic vector to a headline. This is a direct source of the R2-M6 label noise. | [FIXED] length assert + re-match by original text |
| T-M3 | MAJOR | **Emits ALL token prefixes**, vs the paper's 4 ratio-based prefixes (0.3/0.5/0.7/1.0, MIN_WORDS=2). Long headlines dominate the trainset; length weight caps at 1.0 so it does not rebalance across headlines. Methodological divergence from the published pipeline. | [FIXED] Option A: reverted to the paper's 4-ratio scheme (verified byte-for-byte against create_prefix_dataset_train_val.py) |
| T-m4 | MINOR | One malformed `methods_vector` aborts the whole build (parse before dropna). | [FIXED] try/except + count |
| T-m5 | MINOR | No prefix dedup across splits; short prefixes recur with contradictory labels. Present in the reference too. | [NOTED] |

VERIFIED CLEAN: split at source-headline level BEFORE prefix expansion (no leakage), TACTIC ordering identical across all files, neutral all-zero rows included as negatives, length-aware weighted loss applied per-example correctly (binary CE and tactics BCE both), tactics num_labels=10 + problem_type multi_label + sigmoid@eval + micro-F1 early-stop direction all correct, max_length=32 matches inference, terminal-period convention matches.

---

## C. External benchmarks + data loaders

| ID | Sev | Finding | Status |
|----|-----|---------|--------|
| B-M1 | MAJOR | **Bootstrap NaN contamination.** `bootstrap_ci.py` uses `np.quantile` not `np.nanquantile`; a single single-class AUROC resample (returns NaN) poisons the whole CI. Does not fire on the actual data but is unsafe for reuse. | [FIXED] nanquantile |
| B-M2 | MAJOR | **Webis N = 2459, ~13% of the documented corpus.** `clickbait17-train-170331` ships the smaller annotated split, not the ~19.5k set. Loader scores whatever is present with no count assertion → underpowered external benchmark. | [DECIDE] download larger split vs report N=2459 honestly |
| B-M3 | MAJOR | **B3 engagement autodetect false-positive.** `startswith` matching binds `"tac"` to the legacy input column `"tactics"`. Currently harmless (fidelity col absent so it refuses to plot) but fragile. | [FIXED] exact column whitelist |
| B-m1 | MINOR | Tokenization max_length=32 truncation (~2-5% tail). Consistent with training. | [NOTED] |
| B-m2 | MINOR | B7 no dedup: ISOT True.csv has 591 duplicate titles (20826 unique / 21417). | [FIXED] dedup + report both |
| B-m3 | MINOR | Tactics test docstring says "paired" but it is an unpaired label-permutation test (which is the correct choice). Wording only. | [FIXED] wording |
| B-m4 | MINOR | Untrusted ISOT mirror, no hash pin; Webis SHA256 None. Data verifies but integrity unpinned. | [NOTED] |

VERIFIED CLEAN: Webis postText flattening (join, no bracket-stringify), id merge (inner, order-robust), truthClass→1 positive class matches AUROC, Chakraborty blank-line stripping + labels, AUROC on continuous prob / F1 thresholded on probability not logit, batched + no_grad + device placement, no silent head/sample/dropna caps except B7 dedup.

---

## Methodology decisions (resolved)

1. **T-M3 (prefix scheme): DECIDED -> Option A.** Reverted `build_prefix_datasets.py` to the paper's 4-ratio scheme (0.3/0.5/0.7/1.0, MIN_WORDS=2), verified byte-for-byte against `create_prefix_dataset_train_val.py`. The guide models are now literally the paper's guide models, so their per-tactic F1 and clickbait AUROC stay comparable to Figure 1.
2. **S-M1 (prob vs log-prob): DECIDED -> switch to log-domain.** Since the guides are being retrained and everything re-evaluated, the faithfulness constraint no longer binds. run_rewrites.py now defaults to canonical log-domain FUDGE (`log p_llm + a*log(tac) + b*log(1-cb)`); the paper's prob-domain heuristic is retained as `--objective prob` for an ablation. REQUIRED follow-ups: (a) re-tune the on-weight with `run_b2_sweep` before reporting the ablation (the log scale differs from the prob scale), and (b) update the Methods equation in the paper to the log-domain form (edit A10, pending).
3. **S-C2 (replication): DECIDED -> single-seed + item bootstrap.** FUDGE decoding is greedy/deterministic; replication is over items via bootstrap, single seed by default. `--sample` opt-in enables temperature sampling if multi-seed replication is wanted.
4. **B-M2 (Webis size): OPEN default -> report N=2459 honestly.** The smaller annotated split is a real benchmark, enough for AUROC with CIs. The larger ~19.5k split (~6 GB) is available if more power is wanted; not the default.

The circular-metric fixes (S-C1, S-C3) were unambiguous corrections of the exact issue the reviewers flagged and were applied without a decision gate.
