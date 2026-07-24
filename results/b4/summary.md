# B2 held-out rewrite evaluation summary

- Empty rewrites excluded: 0
- Exact-duplicate rows removed: 11
- Distinct seeds present: 3
- Tactic-name leak fraction (overall): 0.000

Note: clickbait_prob_guide_circular and attr_realised_frac_guide_circular
are CIRCULAR (guide models). Independent metrics are clickbait_prob_external
and judge_attr_confirmed_frac.

| alpha | beta | tactic | n | sts | bertscore_f1 | nli_neutral_entails_edited | nli_edited_entails_neutral | clickbait_prob_guide_circular | clickbait_prob_external | attr_realised_frac_guide_circular | names_tactic | fluency_ppl | judge_attr_confirmed_frac | judge_attr_intensity | judge_engaging | judge_hallucinates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt_only | prompt_only | prompt_only | 889 | 0.763 [0.758, 0.767] | 0.366 [0.358, 0.372] | 0.173 [0.155, 0.191] | 0.672 [0.642, 0.700] | 0.110 [0.091, 0.130] | 0.066 [0.053, 0.081] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 174.116 [166.501, 182.572] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.823 [0.790, 0.857] | 0.090 [0.040, 0.140] |
| prompt_only | prompt_only | __ALL_TACTICS__ | 889 | 0.763 [0.758, 0.767] | 0.366 [0.358, 0.372] | 0.173 [0.155, 0.191] | 0.672 [0.642, 0.700] | 0.110 [0.091, 0.130] | 0.066 [0.053, 0.081] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 174.116 [166.501, 182.572] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.823 [0.790, 0.857] | 0.090 [0.040, 0.140] |