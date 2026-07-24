# Response to the Editor and Reviewers — Revision 2

**Manuscript:** *LLM-guided headline rewriting for clickability enhancement without clickbait*
**File submitted with this response:** `GuidedRewriteClickbait_v9_cycle1.docx` — every passage changed in this revision is highlighted in **yellow**. A rendered HTML mirror and three data appendices (regenerated from the authoritative run) accompany the submission.

We thank the editor and both reviewers. The previous revision (Track A) added the requested citations, named the base model, documented reproducibility, reconciled the figure caption, and added explicit failure-mode and limitations text, but it deferred the quantitative program to "Cycle 2." **This revision delivers that program in full.** A new **Section 4 (Results)** now reports, with independent metrics and item-matched significance testing:

- external validation of both guide models on human-authored corpora (4.1);
- prefix-level validation on real headlines (4.2);
- a quantified Reuters-neutrality check (4.3);
- a four-cell dual-guidance ablation on 300 held-out headlines (4.4);
- a comparison against the DExperts and GeDi decoding-time baselines and a prompt-only baseline (4.5);
- a human evaluation, reported as an **LLM-simulated proxy** pending the physical human study, whose protocol is implemented and validated (4.6).

Below, each concern is answered individually with the section and numbers that now address it. A second part lists changes we made that were **not** requested by a reviewer.

---

## Part 1 — Point-by-point responses

### Reviewer 1

**R1-1 — Validation on real-world clickbait datasets.** *Done (Section 4.1–4.2).* Both guides are evaluated on the entirely human-authored Chakraborty 2016 (32,000 headlines) and Webis-17 (2,459 graded) corpora, with 95% bootstrap CIs. The clickbait guide reaches **AUROC 0.955** on Chakraborty and **0.740** on the noisier graded Webis; the attribute guide's ten tactics all correlate significantly with human clickbait (Spearman 0.34–0.45, p < 1e-60). A new prefix-level test (4.2) shows the guide predicts the real label from an early prefix (AUROC 0.891 at 30% of the headline, rising monotonically to 0.954). All clickbait numbers in the rewrite evaluation use a **separate DistilBERT detector trained only on human data**, never our own guide.

**R1-2 — Quantitative and human evaluation.** *Done (Section 4.4, 4.6).* Over 300 held-out headlines we report BERTScore, DeBERTa-v3 NLI, independent attribute-realization (LLM judge), independent clickbait probability, and fluency (perplexity), each per condition with significance. The human protocol (100+ items, three raters, ICC/κ) is implemented; in this revision it is run as an LLM-simulated proxy (clearly labeled) that reproduces the automatic findings, with the physical study to follow.

**R1-3 — Ablations and baselines.** *Done (Section 4.4, 4.5).* The dual-guidance mechanism is dissected by the four-cell (α,β) ablation; the prompt-only baseline and the DExperts and GeDi decoding-time methods are all included, with a full (α,β) sweep behind the operating-point selection.

### Reviewer 2

**R2-M1 — Uncited intro claims; ungrounded rubric.** Citations added in Rev. 1; the rubric is now positioned as an extension of Nofar et al. 2025 and is **validated** (Section 4.6, mean Fleiss κ = 0.75 over 150 headlines, proxy).

**R2-M2 — Central claims unquantified.** *Done.* The Conclusion no longer asserts unmeasured gains; every claim is now backed by Section 4 (paired-preference win 0.597, p ≈ 3e-9; Pareto dominance over baselines).

**R2-M3 — Circularity of the evaluation loop.** *Done (Section 4.1).* The guides are re-evaluated on human-authored data and, as anticipated, drop from the near-perfect synthetic score (AUROC 0.9998) to 0.955 / 0.740 — reported openly, not concealed. The independent human-trained detector breaks the loop for the rewrite metrics.

**R2-M4 — No baselines or ablations.** *Done (Section 4.5).* At matched independent tactic level, FUDGE is **Pareto-dominant**: comparable tactic control (statistically indistinguishable from DExperts/GeDi) at 4–5× lower perplexity and far higher BERTScore; DExperts/GeDi reach the same tactic level only by collapsing fidelity (BERTScore −0.32 / 0.06).

**R2-M5 — Implementation details / reproducibility.** *Done (Section 3.3).* Base model, k = 50, log-domain weights, sweep grid, cloud hardware, guide-training and synthetic-generation settings, wall-clock, and released prompts/artifacts are all specified.

**R2-M6 — Table 2 inconsistencies; label fidelity.** *Done (Section 4.6, C1).* A blinded rubric re-labeling yields substantial inter-rater agreement (κ = 0.75) and a mean intended-vs-consensus tactic F1 of 0.43 — honestly reflecting that fine-grained tactic recovery is hard.

**R2-M7 — Attribute model may be too weak.** *Done (Section 4.4, 4.6).* The (0,β)→(α,β) comparison isolates the positive guide's contribution directly, and attribute realization is measured by an **independent** judge, so classifier noise propagation is quantified rather than assumed.

**R2-M8 — Table 4 outputs contradict goals.** *Done (Section 4.7).* Each failure type is named and, for the embellishment case, **quantified**: a stricter new-fact judge (validated on faithful/fact-injected controls) shows the genuine new-fact rate at the dual operating point is 0.20, statistically indistinguishable from the 0.18 baseline — the apparent fidelity cost is dominated by the rhetorical-question form, not fabrication.

**R2-m1 — Cross-reference errors.** Resolved (subsections exist and resolve).

**R2-m2 — Reuters-neutrality assumption.** *Done (Section 4.3).* The independent detector flags only 0.61% of the 20,825-headline source pool.

**R2-m3 — Split stratification.** Stated explicitly (Rev. 1, Sections 3.1.3–3.1.4).

**R2-m4 — Figure 1 label vs caption.** Reconciled (Rev. 1).

**Statistical rigor.** *Done.* Every reported number carries a 95% bootstrap CI or an item-matched significance test (Wilcoxon signed-rank; binomial sign test for paired preferences).

---

## Part 2 — Changes made that were not directly requested

These improve correctness and rigor beyond the specific asks; all are highlighted in the manuscript.

1. **FUDGE objective moved to the log domain.** The combined objective is now the canonical additive-log-probability factorization `log p_LLM + α·log(tac) + β·log(1−cb)`, replacing the earlier additive-probability heuristic (Section 3.2.3). This is numerically cleaner and is the standard FUDGE formulation; on-weight magnitudes were re-tuned accordingly (hence the operating point α=4, β=2 rather than the earlier 0–0.7 range).

2. **Decoding candidate set fixed at top-k = 50** (the earlier draft reported k ∈ {50, 500}); an exploratory k = 500 arm was dropped because it increased semantic drift without a fidelity/engagement benefit.

3. **Compute moved to cloud GPUs (A10G/L40S via Modal)** for the full-scale, statistically-powered run; the earlier single-GPU (RTX 2060) figures are superseded.

4. **Independent evaluators throughout.** A DistilBERT clickbait detector trained only on human data, an LLM-as-judge for tactic realization, DeBERTa-v3 NLI (upgraded from roberta-large-mnli after a labeled control-set A/B), and distilGPT-2 perplexity are all independent of the guides, to avoid any residual circularity.

5. **Two measurement-artifact investigations** were run and folded in honestly: (a) the "hallucination" metric was shown to be dominated by the rhetorical-question tactic rather than fabrication (Section 4.7), validated with faithful/fact-injected control rephrasings; (b) NLI entailment is reported split by question vs non-question form, since questions depress entailment regardless of faithfulness.

6. **Honest scoping of the central claim.** The paper now states plainly that FUDGE does **not** out-realize the DExperts/GeDi baselines on tactic strength; its contribution is Pareto (same control, far better fidelity/fluency), and that the decoding-time guides act as an amplifier of prompted tactics rather than a from-scratch injector. These bounds are in the Limitations.

7. **Three data appendices regenerated** from the authoritative run (α,β ∈ {0,1,2,4}, top-k = 50) with per-cell independent metrics, superseding the earlier exploratory sweep.

8. **Full reproducibility artifacts** (per-item scores, raw model outputs, run manifests, trained checkpoints) are retained and released.

We are grateful to the editor and reviewers; the manuscript is now on the empirical footing both reports asked for.

Sincerely,
The authors
