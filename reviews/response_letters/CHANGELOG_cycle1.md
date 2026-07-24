# Change Log — Revision Cycle 1

**Baseline:** `GuidedRewriteClickbait_v8.docx` (Mar 11, 2026).
**Revised:** `Cycle1/paper_v9/GuidedRewriteClickbait_v9_cycle1.docx` and matching `GuidedRewriteClickbait_v9_cycle1.html`.

All edits below are **highlighted in yellow** in both artifacts. No unmarked change has been made to the manuscript.
The editor script that applied them is version-controlled at `Cycle1/paper_v9/apply_track_a_edits.py`.

Reviewer-concern codes:
- **R1-1 / R1-2 / R1-3** — Reviewer 1 concerns 1–3
- **R2-M1..M8** — Reviewer 2 major concerns 1–8
- **R2-m1..m4** — Reviewer 2 minor concerns 1–4

---

## A1. Introduction citations
**Addresses:** R2-M1
**Location:** Introduction paragraphs 1, 2, and 3 (paragraphs 16, 17, 18 in the docx paragraph index).
**Change:**
- Added citation **[9]** (Aubin Le Quéré & Matias 2025) after the "long-term erosion of trust in news outlets" claim in intro paragraph 1.
- Added citations **[1,2,3]** (Biyani, Chakraborty, Potthast) after the "correlated with exaggeration or deception" claim in intro paragraph 2.
- Appended a sentence to the rubric paragraph explicitly positioning Table 1 as an extension of Nofar et al. 2025 **[11]** and forward-referencing the human validation planned in Section 5.1.

## A2. Base LLM named
**Addresses:** R2-M5
**Location:** Introduction paragraph 4 (paragraph 19).
**Change:** Replaced "a state-of-the-art LLM as the base generator" with **"Meta-Llama-3-8B-Instruct as the base generator"**.

## A3. Synthetic-generation details
**Addresses:** R2-M5, R2-M6
**Location:** End of Section 3.1.2 (paragraph 49).
**Change:** Appended a sentence documenting the batch size (15), the up-to-three-tactic sampling rule, the JSON self-repair retry policy (up to 3 attempts), and the release of full prompt templates via the public code repository, with a forward reference to Section 3.3.

## A4. Split stratification statement
**Addresses:** R2-m3
**Location:** Section 3.1.3 (paragraph 63) and Section 3.1.4 (paragraph 69).
**Change:** In both split descriptions, added the explicit statement that the 80/20 split is uniformly random at the source-headline level and is not stratified by tactic combination, with a note that class balance is preserved by construction because each source headline contributes both a neutral sample and a clickbait variant.

## A5. Figure 1 caption reconciled with y-axis label
**Addresses:** R2-m4, R2-M7 (in part)
**Location:** Figure 1 caption (paragraph 71).
**Change:** Rewrote the caption to identify the two panels correctly:
"(Left) Per-tactic F1 score of the engagement attribute model on single-tactic test examples. (Right) Confusion matrix of predicted vs. targeted tactic on the same subset."
This matches the "F1 Score" y-axis label already present on the figure.

## A6. New Section 3.3 — Implementation Details and Reproducibility
**Addresses:** R2-M5
**Location:** New Heading 2 inserted between Section 3.2.3 and Section 4 (paragraphs 97–101 in the revised docx).
**Change:** New four-paragraph subsection listing:
- Base generator (Meta-Llama-3-8B-Instruct, FP16, RTX 2060 6 GB).
- Decoding config: top-k ∈ {50, 500}; α ∈ {0.0, 0.3, 0.7}; β ∈ {0.0, 0.3, 0.7}; α warm-up = min(step/6, 1.0); EOS length bonus schedule; max 150 tokens.
- Guide-model training config: BERT-base, 2–3 epochs with early stopping, AdamW at 2×10⁻⁵ with linear decay, length-aware weighted loss with down-weighting for prefixes < 4 tokens.
- Synthetic-data pipeline: 12,600 Reuters source headlines from ISOT True.csv; GPT-4o-mini at temperature 0.7; batch size 15; JSON self-repair up to 3 attempts; failing batches logged and skipped; resume state persisted.
- Public repository release of full prompt templates and hyperparameters, with wall-clock cost per rewrite.

## A7. New Section 4.1 — Observed Failure Modes
**Addresses:** R2-M8
**Location:** New Heading 2 inserted between Table 4 (paragraph 101 in v8) and Section 5 (paragraphs 107–108 in the revised docx).
**Change:** New subsection acknowledging three recurring failure modes evident in Table 4:
1. Provocative-question outputs in which the manipulative content lives in the presupposition rather than in surface lexicon, which the prefix-level clickbait scorer is not explicitly trained to detect (example: "Lockheed Martin Lands Lucrative Pentagon Deal: But at What Cost to Taxpayers?").
2. Semantic embellishment via novel noun phrases under positive guidance targeting referential underspecification (examples: "Secrets", "A New Era").
3. Under-flagging of human-authored clickbait patterns by a scorer trained on GPT-4o-mini generations, with the residual gap to be quantified by the external evaluation in Section 5.1.

## A8. Softened Conclusion
**Addresses:** R2-M2, R1-2
**Location:** First Conclusion paragraph (paragraph 111 in the revised docx; originally paragraph 104 in v8).
**Change:** Replaced the unbacked "measurable gains in engagement salience" sentence with a version that:
- Explicitly scopes the claim to the qualitative examples of Table 4.
- Forward-references the quantitative BERTScore / NLI / attribute-realization / independent-clickbait evaluation to Section 5.1 and companion supplementary materials.

## A9. New Section 5.1 — Limitations and Scope of the Guide Models
**Addresses:** R2-M3, R2-m2, R1-1
**Location:** New Heading 2 inserted inside Section 5, between the second Conclusion paragraph and the "Several directions naturally extend this study" future-work paragraph (paragraphs 113–114 in the revised docx).
**Change:** New subsection stating three explicit limitations:
1. Domain restriction to Reuters political / world news; extension to tabloid, entertainment, and non-English registers is open.
2. Neutrality of Reuters source headlines is assumed, not externally verified in the present version; an independent-detector spot-check is scheduled.
3. Because both guide models are trained on prefixes from the same GPT-4o-mini generation pipeline that defines the positive class, the synthetic-test performance overstates generalization; the Webis Clickbait Corpus 2017 [3] and Chakraborty et al. 2016 [2] evaluations provide the distribution-shift measurement.

---

## Traceability matrix

| Edit | Reviewer concern(s) | Docx paragraph(s) after revision | Anchor phrase to locate |
|------|--------------------|-------------------------------|-------------------------|
| A1   | R2-M1              | 16, 17, 18                    | "erosion of trust", "exaggeration or deception", "extends prior attribute-based" |
| A2   | R2-M5              | 19                            | "Meta-Llama-3-8B-Instruct as the base" |
| A3   | R2-M5, R2-M6       | 49                            | "The full generation prompt" |
| A4   | R2-m3              | 63, 69                        | "not stratified by tactic" |
| A5   | R2-m4, R2-M7       | 71                            | "(Left) Per-tactic F1 score" |
| A6   | R2-M5              | 97–101                        | "3.3 Implementation Details and Reproducibility" |
| A7   | R2-M8              | 107–108                       | "4.1 Observed Failure Modes" |
| A8   | R2-M2, R1-2        | 111                           | "in the qualitative examples of Table 4" |
| A9   | R2-M3, R2-m2, R1-1 | 113–114                       | "5.1 Limitations and Scope of the Guide Models" |

---

## What Cycle 1 does NOT change

The following are intentionally deferred to Cycle 2 because they require data or human labor beyond what Track A can produce:

- Quantitative rewrite evaluation (Track B, scaffolded in `Cycle1/track_b/`).
- External-benchmark validation of both guide models (Track B).
- Plain-prompt baseline; DExperts and GeDi comparisons (Track B; DExperts/GeDi may spill into Cycle 3).
- Human validation of synthetic-tactic labels and of rewrite quality (Track C, protocol drafted).
- Replacement of Table 4 qualitative examples with a quantitative main-results table.

The Conclusion (edit A8) and Section 5.1 (edit A9) forward-reference these deferred items so that no unsupported quantitative claim survives in the current text.
