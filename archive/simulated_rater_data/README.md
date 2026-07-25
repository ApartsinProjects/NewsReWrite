# Archived: LLM-simulated rater data (superseded)

These files hold the **LLM-based (GPT-simulated) rater responses** from the
annotation study reported in Section 4.10. Every response here carries
`rater_source = gpt_sim`. They are retained for provenance only and are
**superseded by the real human-rater study** to be collected with the blank
workbook at `results/human_labeling/rater_annotation_TEMPLATE.xlsx`.

Do **not** use these numbers as human data. They were disclosed in the
manuscript explicitly as an LLM-based annotation study.

## Contents (archived — simulated rater RESPONSES only)
- `c1_rubric_validation/rater_01/02/03.csv` — gpt_sim tactic labels (150 items).
- `c2_rewrite_quality/rater_01/02/03.csv` — gpt_sim 1-5 ratings (1,500 items).
- `c1_/c2_ .../analysis.json` — agreement/Fleiss-kappa/F1 and ICC computed from
  the simulated responses above.
- `rater_evaluation_data.xlsx` — the same simulated results as a 7-sheet workbook.

## Reusable infrastructure (kept LIVE, NOT archived)
Under `results/human_labeling/`:
- `rater_annotation_TEMPLATE.xlsx` — blank annotator workbook for the real study.
- `<study>/codebook.md` — the rubrics.
- `<study>/oracle.csv` + `oracle.jsonl` — the item set + intended labels + auto
  metrics. This is **not** rater output; the real human study rates the same
  items, and `analyze.py` merges rater responses against it. It must stay live.

## To run the real study
Give three annotators a copy of the template + the matching codebook, collect
their filled sheets as `rater_01/02/03.csv` into the live
`results/human_labeling/<study>/` folders (which already hold `oracle.csv` +
`codebook.md`), then run, per study:

```
python code/revision_experiments/human_labeling_prep/analyze.py \
    --study-dir results/human_labeling/c1_rubric_validation
python code/revision_experiments/human_labeling_prep/analyze.py \
    --study-dir results/human_labeling/c2_rewrite_quality
```

This writes a fresh `analysis.json` in each study dir; swap those human numbers
into Section 4.10.
