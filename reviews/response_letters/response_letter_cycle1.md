# Response to Reviewers, Revision Cycle 1

**Manuscript:** *LLM-guided headline rewriting for clickability enhancement without clickbait*
**Version submitted with this response:** `GuidedRewriteClickbait_v9_cycle1.docx` (yellow highlights mark every passage changed in this cycle)

We thank both reviewers for their careful and constructive reading. The critiques converge on three central issues: (i) the closed-loop nature of our evaluation, (ii) the qualitative rather than quantitative support of our rewrite claims, and (iii) the absence of baselines and ablations. This revision addresses these in three parallel tracks:

- **Track A: paper edits.** All applied in this submission and highlighted in yellow. Add missing citations, name the base LLM, document reproducibility, reconcile Figure 1's label and caption, add explicit failure-mode discussion, add a limitations subsection, and soften unbacked adjectives in the Conclusion.
- **Track B: new quantitative experiments without human labelers.** External-benchmark validation on Webis-Clickbait-17 and Chakraborty 2016; held-out rewrite evaluation with BERTScore + NLI + independent clickbait scoring + attribute-realization rate over the full (α, β) sweep; prompt-only baseline; four-cell ablation of the dual-guidance design; Reuters neutrality spot-check. Scaffolding for these runs is committed to `Cycle1/track_b/`; results will accompany the next revision.
- **Track C: human evaluation and label validation.** Human validation of synthetic-data tactic labels (Cohen's κ) and human rating of rewrite engagement / faithfulness / perceived clickbait. Protocol is drafted; execution is planned for the interval between this and the next revision.

Below, every concern raised by each reviewer is reproduced in italics and answered individually. Track labels ([A], [B], [C]) indicate which track carries the fix; edit numbers (A1–A9) point to the marked passages listed in `CHANGELOG_cycle1.md`.

---

## Reviewer 1

### R1-1. Real-world clickbait dataset validation

> *The proposed framework is evaluated primarily on synthetically generated clickbait headlines. Additional validation on publicly available real-world clickbait datasets would considerably strengthen the claims.*

**Response.** We agree fully and treat this as the single most important addition. [B] Both guide models will be evaluated on the Webis Clickbait Corpus 2017 [3] and the Chakraborty et al. 2016 corpus [2] using AUROC, F1, precision, and recall with 95% bootstrap CIs. The rewrite pipeline's output will additionally be scored by the Chakraborty-trained detector so that the negative-guidance validity check no longer relies on our own scorer. In this submission the limitation is now stated explicitly in the new subsection 5.1 (edit A9), and the forward reference to this external evaluation appears in the Conclusion (edit A8).

### R1-2. Quantitative and human evaluation

> *The manuscript claims that the proposed framework improves reader engagement while preserving semantic fidelity, yet these central claims are supported only by qualitative examples.*

**Response.** [B] We will report, over a held-out set of 300 neutral headlines, per-condition means with 95% CIs across three seeds for: (a) BERTScore-F1 and Sentence-Transformer STS fidelity, (b) NLI entailment in both directions using roberta-large-mnli, (c) attribute-realization rate under both our engagement model and an LLM-as-judge, and (d) independent clickbait probability under the Chakraborty detector. [C] A separate human evaluation on 100 headline triples with three raters per item will report engagement / faithfulness / perceived-clickbait ratings and inter-rater reliability (ICC(2,k)). The Conclusion has been softened in this submission (edit A8) so that the current text no longer asserts "measurable gains in engagement salience" or "reducing uncontrolled stylistic drift" as established results; these claims will be reinstated only if the quantitative evidence supports them.

### R1-3. Ablations and baselines for the dual-guidance mechanism

> *The controllability of the proposed dual-guidance mechanism is insufficiently validated experimentally … the manuscript would benefit from ablation studies and comparisons with representative headline rewriting baselines.*

**Response.** [B] The dual-guidance ablation is realized as the four-cell (α, β) grid over {0, 0.7} × {0, 0.7} in Track B: (0, 0) drops both guides, (0.7, 0) uses only positive engagement guidance, (0, 0.7) uses only clickbait suppression, and (0.7, 0.7) uses both. The plain-LLM baseline (Meta-Llama-3-8B-Instruct prompted directly to "rewrite this to be more engaging but not clickbait") is included as an additional cell. The related decoding-time methods DExperts and GeDi are scaffolded in `Cycle1/track_b/b6_dexperts_gedi/`; the full comparison against these two is a longer implementation and, if it does not complete within this revision cycle, will be added in Cycle 2. The (α, β) sweep beyond the two settings previously reported in Table 3 (now Table 4) is already covered by the existing code (`fudge_controlled_generation.py`, `ALPHAS = [0.0, 0.3, 0.7]`, `BETAS = [0.0, 0.3, 0.7]`) and will be aggregated into a Pareto plot.

---

## Reviewer 2

### R2-M1. Uncited intro assertions; Table 1 rubric not grounded

> *The first two paragraphs of the introduction make several assertions without citing supporting references. Moreover, the authors introduce a guideline (Table 1) that is not grounded in any previous work; this guideline is a novel contribution but has not been validated.*

**Response.** [A, edit A1] Citations have been added in the first two introduction paragraphs: [9] on the reader-frustration and trust-erosion claim, [1,2,3] on the "lexical, syntactic, or pragmatic markers" claim. The rubric paragraph has been rewritten to explicitly position Table 1 as an *extension* of the attribute-attribution direction opened by Nofar et al. 2025 [11], rather than as a stand-alone novel contribution. [C] The rubric itself will be validated by two independent raters on a stratified sample of 150 synthetic headlines, with Cohen's κ reported per attribute; the protocol is registered under `Cycle1/track_c/c1_rubric_validation/`.

### R2-M2. Central claims not supported by quantitative evidence

> *The Conclusion states that controlled rewriting yields headlines that are "more restrained, more semantically faithful," with "measurable gains in engagement salience" … None of these quantities is actually measured.*

**Response.** [A, edit A8] The offending Conclusion text has been softened in this submission and now defers the quantitative claim to the companion Section 5.1 and its supplementary materials. [B, C] The four subparts (a)–(d) requested by the reviewer are covered by Track B experiments (a–c) and Track C human evaluation (d), as detailed in the R1-2 response above. Every metric asked for by the reviewer, including QA-based factual consistency, will appear in the Cycle 2 submission with confidence intervals.

### R2-M3. Circularity of the evaluation loop

> *The clickbait scorer's AUROC > 0.99 is reported on synthetic clickbait generated by the same LLM and rubric that define the positive class … the scorer may be detecting GPT-4o-mini stylistic artifacts rather than clickbait per se.*

**Response.** We concur completely and consider this the sharpest single objection. [A, edit A9] The new subsection 5.1 explicitly identifies this circularity as a limitation and previews the external-benchmark evaluation. [B] Both guide models will be re-evaluated on Webis-Clickbait-17 [3] (graded strength) and Chakraborty et al. 2016 [2] (binary), which are entirely human-authored. The full metric suite will be reported with 95% bootstrap CIs. We expect and are prepared to report a substantial drop from the current in-distribution performance; that drop is itself informative and will be discussed rather than concealed.

### R2-M4. No baselines or ablations

> *The obvious baseline (directly prompting a strong LLM…) is absent. So are comparisons to the related decoding-time methods discussed in Section 2.2 (DExperts, GeDi), and any sweep over guidance weights beyond the two settings in Table 3.*

**Response.** [B] Answered in R1-3 above. In summary: (i) plain-prompt baseline is scheduled in Track B; (ii) full (α, β) × top-k sweep is scheduled and will be aggregated into a Pareto figure; (iii) DExperts and GeDi wrappers are scaffolded; if either does not complete this cycle it will appear in Cycle 2 with an explicit note in the Response letter.

### R2-M5. Missing implementation details; reproducibility

> *The base generator is described only as "a state-of-the-art LLM." … the identity of the base model, the candidate set size (top-k) at each step, the decoding hyperparameters, and the computational cost must be specified. Similarly, the exact prompt templates … should be documented.*

**Response.** [A, edits A2, A3, A6] The manuscript now names the base generator (**Meta-Llama-3-8B-Instruct**) in the Introduction; a new **subsection 3.3 "Implementation Details and Reproducibility"** documents FP16 inference on a single RTX 2060, top-k ∈ {50, 500}, α ∈ {0.0, 0.3, 0.7}, β ∈ {0.0, 0.3, 0.7}, the linear warm-up on α, the EOS length bonus, max generation length of 150 tokens, BERT guide-model fine-tuning hyperparameters (AdamW, lr 2×10⁻⁵, 2–3 epochs, early stopping), and synthetic-generation details (12,600 source Reuters headlines, GPT-4o-mini at temperature 0.7, batch size 15, JSON self-repair retry policy up to 3 attempts). Full prompt templates for both synthetic generation and the FUDGE rewrite instruction are released with the public code repository. Wall-clock per rewrite on the reference configuration is stated.

### R2-M6. Table 2 inconsistencies; synthetic-data quality

> *Several generated examples do not match their stated tactic … If these examples are representative … the attribute labels in the training data are noisy … Please clarify, and report any validation of tactic-label fidelity in the synthetic corpus.*

**Response.** [A, edit A7] The new subsection 4.1 "Observed Failure Modes" acknowledges this class of mismatch explicitly. [C] Track C task C1 will re-label 150 synthetic samples with two raters against the ten-attribute rubric, blinded to the intended tactic vector. We will report per-attribute Cohen's κ and, importantly, the fraction of samples where the reader-recovered tactic vector agrees with the generation-time intent. Where disagreement is systematic (e.g. Curiosity Gap vs. Ambiguous References, which the reviewer flagged as commonly conflated), we will either retrain with cleaned labels or narrow the attribute set in the next revision.

### R2-M7. Attribute model may be too weak

> *Per-tactic F1 scores range from 0.31 to 0.52 (Figure 3), and the confusion matrix shows heavy off-diagonal mass. The paper does not analyze how this classifier noise propagates into decoding.*

**Response.** [A, edit A5] Figure 1's y-axis label and caption have been reconciled: the caption now correctly identifies the left panel as **per-tactic F1** and the right panel as the single-tactic confusion matrix. [B] The four-cell ablation (see R1-3) isolates the contribution of the positive guide directly: comparing (0, 0.7) to (0.7, 0.7) quantifies how much attribute realization the noisy positive guide actually gains. Attribute-realization rates will be computed both by our own model and by an independent LLM-as-judge, so that the propagation of classifier noise into the final output is measurable rather than assumed. If the ablation shows that the positive guide contributes negligibly on top of the negative one, we will report that finding openly and revise the framing.

### R2-M8. Table 3 (now Table 4) outputs contradict framework goals

> *"Lockheed Martin Lands Lucrative Pentagon Deal: But at What Cost to Taxpayers?" introduces an insinuation absent from the source … Likewise, "Breed a New Era of Success" and "Encryption Secrets" alter or embellish the semantics.*

**Response.** [A, edit A7] Section **4.1 "Observed Failure Modes"** now names each of these three failure types explicitly: (i) provocative-question outputs in which the manipulative content lives in the presupposition rather than the surface lexicon, so the prefix-level scorer is under-sensitive to it; (ii) semantic embellishment via novel noun phrases introduced under positive guidance targeting referential underspecification; (iii) under-flagging of human-authored patterns by a scorer trained on GPT-4o-mini generations, whose empirical size will be quantified by Track B on the external corpora. Table 4 is no longer presented as uniformly successful.

### R2-m1. Cross-reference errors ("Section 3.1.1", "Section 3.1.2" absent)

**Response.** Already addressed between v7 and v8: subsections 3.1.1, 3.1.2, 3.1.3, and 3.1.4 now exist and the in-text references resolve correctly.

### R2-m2. Reuters-neutrality assumption; domain restriction

**Response.** [A, edit A9] The new subsection 5.1 now states the Reuters-only domain restriction explicitly and identifies the neutrality assumption as unverified in the current version. [B] Task B7 in `Cycle1/track_b/b7_reuters_neutrality/` runs the Chakraborty detector over the full ISOT True.csv pool and reports the distribution of predicted clickbait probability; the outcome will be summarized in Cycle 2.

### R2-m3. Split stratification

**Response.** [A, edit A4] Both training-data-split paragraphs (3.1.3 for the clickbait scorer, 3.1.4 for the attribute model) now state explicitly that the 80/20 split is uniformly random at the source-headline level and is not stratified by tactic combination. The natural class balance is preserved by construction because each source headline contributes one neutral and one clickbait sample.

### R2-m4. Figure 1 label vs caption mismatch

**Response.** [A, edit A5] The caption of Figure 1 has been rewritten to name the two panels correctly ("Per-tactic F1 score" on the left, "confusion matrix over single-tactic examples" on the right), matching the axis label on the figure itself.

---

## Statistical rigor

> *No statistical analysis is presented (single runs, no variance, no significance testing).*

**Response.** [B] Every quantitative number introduced in Cycle 2, from external-benchmark AUROC to per-condition rewrite metrics, will be reported with 95% bootstrap CIs (n = 1000) computed from three seeds. Paired bootstrap significance tests will accompany every pairwise condition comparison in the ablation table.

---

## Summary of what is different in the manuscript submitted with this letter

Every yellow-highlighted passage in `GuidedRewriteClickbait_v9_cycle1.docx` corresponds to one of the nine Track A edits enumerated in `CHANGELOG_cycle1.md`. No unmarked change has been made. The results table (Table 4) and its qualitative examples remain identical to v8; the Cycle 2 submission will replace them with the quantitative Track B results.

We thank both reviewers again for pushing the paper toward a more rigorous empirical footing and are grateful for the constructive tone of both reports.

Sincerely,
The authors
