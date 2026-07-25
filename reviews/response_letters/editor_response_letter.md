# Response to the Editor and Reviewers

**Manuscript:** *LLM-guided headline rewriting for clickability enhancement without clickbait*

We thank the editor and both reviewers for their careful and constructive reports.
We have **significantly improved and expanded the experimental program**, converting a
manuscript whose central claims rested on qualitative examples into one in which every
claim is backed by quantitative evidence with independent metrics, item-matched
significance testing, and confidence intervals. In the revised manuscript, every passage
changed in this revision is highlighted in **yellow**; a white-background HTML mirror and a
public data/code/model deposit accompany the submission.

The revised **Section 4 (Experiments and Results)** now reports, all on held-out data and
with independent evaluators wherever possible:

- external validation of both guide models on entirely human-authored corpora (4.5);
- prefix-level validation of the guide on real headlines (4.6, Figure 7);
- a quantified Reuters-neutrality check of the source pool (4.7);
- a four-cell dual-guidance ablation on 300 held-out headlines (4.8, Table 5, Figure 5);
- a comparison against the DExperts and GeDi decoding-time baselines showing that dual
  control decouples engagement from clickbait (4.9, Table 6, Figure 6, Table 11);
- a rater-based evaluation with three independent annotators (4.10, Table 7);
- a robustness and factuality analysis (4.11, Table 8);
- cross-domain generalization to non-Reuters headlines (4.12, Table 9);
- transfer to a second base-model family, Qwen2.5-7B (4.13, Table 10).

Each reviewer point is answered individually below, with the section and numbers that now
address it. Part 2 lists improvements we made that were not explicitly requested.

---

## Part 1 — Point-by-point responses

### Reviewer 1

**R1-1 — Generalization to naturally occurring clickbait / validation on public real-world
datasets.** *Addressed (Sections 4.5–4.6).* Both guides are now evaluated on two entirely
human-authored, publicly available corpora never seen in training: Chakraborty et al. 2016
(32,000 headlines) and the Webis Clickbait Corpus 2017 (2,459 graded), each with 95%
bootstrap CIs. The clickbait guide reaches **AUROC 0.955** on Chakraborty and **0.740** on
the noisier graded Webis; the ten attribute dimensions all correlate significantly with human
clickbait (Spearman 0.34–0.45, every *p* < 1×10⁻⁶⁰). A new prefix-level test (Section 4.6,
Figure 7) shows the guide predicts the human label from an early prefix, with AUROC rising
monotonically from **0.891** at 30% of the headline to **0.954** at full length. Every
clickbait number in the rewrite evaluation is produced by a **separate DistilBERT detector
trained only on the human-authored Chakraborty data**, never by our own guide.

**R1-2 — Central claims supported only by qualitative examples; add quantitative and human
evaluation.** *Addressed (Sections 4.8, 4.10).* Over 300 held-out headlines we now report
BERTScore-F1, DeBERTa-v3 NLI entailment, independent attribute realization (an LLM judge),
independent clickbait probability, and fluency (perplexity) for each configuration, with
item-matched significance. We additionally conducted a **rater-based evaluation** (Section
4.10) with three independent human annotators over 1,500 rewrites, obtaining good-to-excellent
reliability (ICC(2,k) 0.82–0.94); annotator faithfulness correlates with automatic BERTScore
(Spearman *ρ* = 0.61, *n* = 1500). With item-matched significance the human ratings confirm the
two designed effects: the brake lowers perceived clickbait (dual 2.03 vs no-guidance 2.61 on a
1–5 scale, *p* = 1×10⁻⁷) and the dual guidance is rated more faithful than the no-guidance
baseline (3.58 vs 2.89, *p* = 1×10⁻⁷) and than the DExperts and GeDi baselines (both
*p* < 1×10⁻¹⁴). The full annotation workbooks and per-item ratings are released in the deposit.

**R1-3 — Dual-guidance controllability insufficiently validated; add ablations and baselines.**
*Addressed (Sections 4.8–4.9).* The dual-guidance mechanism is dissected by a four-cell
(α, β) ablation, showing the two guides exert the designed, independent, opposite effects
(the engagement guide raises independent tactic realization from 0.48 to 0.51; the brake
lowers the independent clickbait score from 0.079 to 0.057). We compare against the two representative
decoding-time methods **DExperts and GeDi**, each with its own strength sweep. The dual guide
adds an **independent clickbait brake** (Table 6, Figure 6): a lever that lowers induced
clickbait separately from the engagement weight, which the single-axis methods lack. At low
steering the baselines are competitive, but because their single control couples engagement and
clickbait, pushing it past the operating point drives their induced clickbait up steeply
(DExperts 0.06 to 0.73 across its strength sweep; GeDi to 0.59) with fidelity falling in step.
For a fair comparison DExperts is decoded with a repetition penalty (1.3) and a headline-length
cap, since unconstrained greedy product-of-experts decoding degenerates into repetition at high
steering strength.

### Reviewer 2

**R2-M1 — Unsupported intro claims; ungrounded, unvalidated rubric.** *Addressed.* The
opening paragraphs now carry supporting citations, and the rubric is positioned as an
extension of the attribute-attribution work of Nofar et al. 2025. The rubric is **validated
empirically**: a blinded re-labeling study (Section 4.10, C1) over 150 headlines yields
substantial inter-rater agreement (mean Fleiss κ = 0.70).

**R2-M2 — Central claims not supported by quantitative evidence.** *Addressed.* The
Conclusion no longer asserts unmeasured gains. Every claim is backed by Section 4: the brake
measurably lowers both the independent detector score and human-rated clickbait (dual 2.03 vs
2.61, *p* = 1×10⁻⁷), independent human annotators rate the dual guidance more faithful than the
no-guidance baseline (3.58 vs 2.89, *p* = 1×10⁻⁷), and against the
baselines the dual control holds induced clickbait low at matched tactic realization while
the single-axis methods' clickbait rises steeply with steering.

**R2-M3 — Circularity of the evaluation loop.** *Addressed (Section 4.5).* Both guides are
re-evaluated on entirely human-authored corpora, scoring **AUROC 0.955 / 0.740** there
alongside the in-distribution synthetic score (0.9998). Critically, every clickbait number in
the rewrite evaluation uses an **independent DistilBERT detector trained only on human data**,
which breaks the loop the reviewer identified.

**R2-M4 — No baselines or ablations.** *Addressed (Sections 4.8–4.9, Tables 5, 6, 11;
Figures 5, 6).* Both are now present: the full four-cell ablation and the
DExperts and GeDi baselines, with a decoupling analysis (Table 6, Figure 6) and a
qualitative side-by-side (Table 11) showing that when the single-axis baselines are pushed to
high steering their induced clickbait rises steeply, whereas the dual guide's independent brake
keeps it low.
DExperts is decoded with a repetition penalty and a headline-length cap for a fair comparison.

**R2-M5 — Missing implementation details / reproducibility.** *Addressed (Section 3.6).* The
base model (Meta-Llama-3-8B-Instruct), top-k = 50, the log-domain objective and its swept
weights, the tuning grid, cloud hardware, guide-training settings, the full synthetic-data
generation procedure (batch size, tactic sampling, JSON self-repair), and per-headline
wall-clock are all specified. All prompt templates, hyperparameters, data, code, and the
trained guide checkpoints are released in a public deposit.

**R2-M6 — Apparent Table 2 inconsistencies; validate tactic-label fidelity.** *Addressed
(Section 4.10, C1).* A blinded rubric re-labeling yields κ = 0.70 and a mean
intended-vs-consensus tactic F1 of **0.44**, matching the attribute model's own recoverability
and reflecting the graded, fine-grained nature of the tactics. The
synthetic examples in Table 2 were also refreshed from the authoritative generation run.

**R2-M7 — Attribute model may be too weak to guide reliably.** *Addressed (Sections 4.8, 3.7).*
The (0,0)→(4,0) contrast isolates the positive guide's contribution directly, and attribute
realization is measured by an **independent** LLM judge rather than the guide itself, so
classifier-noise propagation is quantified rather than assumed. The judge's own reliability
is cross-checked against the rater study's recoverability (Section 3.7).

**R2-M8 — Some controlled outputs contradict the framework's goals.** *Addressed (Section
4.11, Table 8).* Each boundary case is discussed explicitly, and the embellishment concern
is **quantified**: a stricter new-fact judge (validated on faithful-paraphrase controls that
score ≈ 0.02 and fact-injected controls that score 1.0, AUROC ≈ 1.0) puts the genuine
new-fact rate at the dual operating point at **0.20, statistically indistinguishable from the
0.18 no-guidance baseline**. The residual fidelity difference is attributable to the
rhetorical question form rather than to fabricated content.

**R2 (review question) — "No statistical analysis (single runs, no variance, no significance
testing)."** *Addressed throughout.* Every reported comparison now uses **item-matched
Wilcoxon signed-rank tests**, **95% percentile-bootstrap confidence intervals** on all table
estimates, **matched-pairs rank-biserial effect sizes**, and **Holm correction** across each
family of tests.

**R2-m1 — Cross-reference errors.** *Fixed.* The methodology subsections referenced in the
introduction now exist and resolve correctly.

**R2-m2 — Reuters-neutrality assumption unverified.** *Addressed (Section 4.7).* The
independent detector was run over the full 20,825-headline source pool: only **0.61%** exceed
the clickbait threshold, confirming the neutral baseline rather than assuming it. The domain
scope is stated explicitly in the manuscript.

**R2-m3 — Is the dataset split stratified?** *Clarified (Sections 4.2–4.3).* The split is
performed uniformly at random at the **source-headline level** (not stratified by tactic);
because every source contributes both a neutral and a clickbait sample, class balance is
preserved by construction.

**R2-m4 — Figure y-axis / caption mismatch.** *Fixed.* Figure 3 now labels the axis and
caption consistently; the per-tactic figure was regenerated from the current guide.

---

## Part 2 — Additional improvements (not explicitly requested)

Beyond the reviewers' requests, we strengthened the manuscript as follows; all are
highlighted in the manuscript.

1. **Full statistical protocol.** Bootstrap CIs, effect sizes, and multiple-comparison
   correction are applied uniformly across Section 4, not only where variance was questioned.
2. **Cross-domain generalization (Section 4.12, Table 9).** The same guides, at the same
   operating point, reproduce their designed effects on 150 human-authored **non-Reuters**
   headlines (a different register): the brake lowers independent clickbait (0.097 → 0.061)
   and fidelity is preserved (BERTScore 0.22–0.28), extending the demonstrated scope to a
   second register.
3. **Transfer to a second base generator (Section 4.13, Table 10).** The framework transfers
   to **Qwen2.5-7B-Instruct** (a different model family) with the same guides and operating
   point, maintaining high fidelity and fluency; for multilingual base generators we add a
   language-consistency constraint to the decoder (Section 3.6).
4. **Factuality calibration (Table 8).** The new-fact judge and NLI entailment are calibrated
   on labelled controls, giving the fidelity analysis a verified basis.
5. **Qualitative depth.** A worked "engagement continuum" for one headline across the α sweep
   (Section 4.4), a full four-cell grid on three headlines (Table 12), and a qualitative
   cross-method comparison (Table 11) make the control behaviour concrete.
6. **Objective and evaluators.** The combined objective uses the canonical log-domain FUDGE
   factorization; all evaluators (independent DistilBERT detector, LLM tactic judge,
   DeBERTa-v3 NLI, distilGPT-2 perplexity) are independent of the guides.
7. **Complete citation coverage.** All datasets, methods, models, and evaluation metrics are
   now cited, including the base generators (Llama-3, Qwen2.5, GPT-4o) and the evaluation
   models/metrics (BERT, DistilBERT, DeBERTaV3, BERTScore, GPT-2).
8. **Reproducibility deposit.** Data, code, and the trained guide checkpoints are released
   publicly, with per-item score files behind every table.
9. **Presentation.** Consistent Scientific Reports reference style, a white-background HTML
   mirror, and full yellow highlighting of all revisions.

We believe the manuscript now meets the standard of claims being fully supported by the data,
and we thank the reviewers again for feedback that materially improved the work.

Sincerely,
The authors
