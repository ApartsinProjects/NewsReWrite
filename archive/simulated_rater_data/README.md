# Archived: LLM-simulated rater data (superseded)

These files hold the **LLM-based (GPT-simulated) rater responses** from the
annotation study reported in Section 4.10. Every response here carries
`rater_source = gpt_sim`. They are retained for provenance only and are
**superseded by the real human-rater study** to be collected with the blank
workbook at `results/human_labeling/rater_annotation_TEMPLATE.xlsx`.

Do **not** use these numbers as human data. They were disclosed in the
manuscript explicitly as an LLM-based annotation study.

## Contents
- `c1_rubric_validation/` — tactic-labelling study (150 items):
  `rater_01/02/03.csv` (gpt_sim responses), `oracle.csv/.jsonl` (item set +
  intended labels), `analysis.json` (agreement / Fleiss kappa / F1).
- `c2_rewrite_quality/` — rewrite-quality study (1,500 items):
  same file layout (engagement / faithfulness / clickbait 1-5).
- `rater_evaluation_data.xlsx` — the same simulated results organized as a
  7-sheet workbook.

## Reusable infrastructure (kept live, NOT archived)
Under `results/human_labeling/`:
- `rater_annotation_TEMPLATE.xlsx` — blank annotator workbook for the real study.
- `c1_rubric_validation/codebook.md`, `c2_rewrite_quality/codebook.md` — the rubrics.

## To run the real study
Give three annotators a copy of the template + the matching codebook, collect
their filled sheets as `rater_01/02/03.csv` in fresh
`results/human_labeling/<study>/` folders, re-run the analysis, and swap the
human numbers into Section 4.10.
