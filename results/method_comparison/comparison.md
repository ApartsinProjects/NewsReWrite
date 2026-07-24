# Method comparison (R2-M4)

Reference: fudge. Metrics: bertscore_f1, sts, clickbait_prob_external, attr_realised_frac_guide_circular, judge_attr_confirmed_frac.

| method | setting | n | bertscore_f1 | sts | clickbait_prob_external | attr_realised_frac_guide_circular | judge_attr_confirmed_frac |
|---|---|---|---|---|---|---|---|
| dexperts | a0.5_b0.0 | 900 | 0.264 | 0.621 | 0.073 | 0.159 | 0.450 |
| dexperts | a1.0_b0.0 | 900 | 0.220 | 0.571 | 0.154 | 0.213 | 0.455 |
| dexperts | a2.0_b0.0 | 900 | -0.090 | 0.480 | 0.546 | 0.242 | 0.475 |
| dexperts | a4.0_b0.0 | 900 | -0.319 | 0.401 | 0.735 | 0.237 | 0.525 |
| fudge | a0.0_b0.0 | 900 | 0.282 | 0.646 | 0.045 | 0.125 | 0.476 |
| fudge | a0.0_b2.0 | 900 | 0.280 | 0.654 | 0.028 | 0.008 | 0.446 |
| fudge | a4.0_b0.0 | 900 | 0.280 | 0.641 | 0.079 | 0.183 | 0.509 |
| fudge | a4.0_b2.0 | 900 | 0.251 | 0.633 | 0.057 | 0.074 | 0.500 |
| gedi | a10.0_b0.0 | 900 | 0.176 | 0.489 | 0.510 | 0.303 | 0.465 |
| gedi | a20.0_b0.0 | 900 | 0.103 | 0.412 | 0.552 | 0.259 | 0.480 |
| gedi | a30.0_b0.0 | 900 | 0.055 | 0.365 | 0.588 | 0.256 | 0.500 |
| gedi | a5.0_b0.0 | 900 | 0.224 | 0.562 | 0.203 | 0.297 | 0.465 |
| prompt_only | prompt_only | 900 | 0.366 | 0.763 | 0.066 | 0.000 | 0.000 |

## Significance vs reference (Wilcoxon signed-rank, item-matched)

- fudge(a4.0_b2.0) vs dexperts(a4.0_b0.0) [bertscore_f1]: p=0.0, n=900, delta=0.5699
- fudge(a4.0_b2.0) vs dexperts(a4.0_b0.0) [sts]: p=0.0, n=900, delta=0.2314
- fudge(a4.0_b2.0) vs dexperts(a4.0_b0.0) [clickbait_prob_external]: p=0.0, n=900, delta=-0.678
- fudge(a4.0_b2.0) vs dexperts(a4.0_b0.0) [attr_realised_frac_guide_circular]: p=0.0, n=900, delta=-0.1633
- fudge(a4.0_b2.0) vs dexperts(a4.0_b0.0) [judge_attr_confirmed_frac]: p=0.93265, n=100, delta=0.01
- fudge(a4.0_b2.0) vs gedi(a30.0_b0.0) [bertscore_f1]: p=0.0, n=900, delta=0.1965
- fudge(a4.0_b2.0) vs gedi(a30.0_b0.0) [sts]: p=0.0, n=900, delta=0.2674
- fudge(a4.0_b2.0) vs gedi(a30.0_b0.0) [clickbait_prob_external]: p=0.0, n=900, delta=-0.5312
- fudge(a4.0_b2.0) vs gedi(a30.0_b0.0) [attr_realised_frac_guide_circular]: p=0.0, n=900, delta=-0.1817
- fudge(a4.0_b2.0) vs gedi(a30.0_b0.0) [judge_attr_confirmed_frac]: p=0.38487, n=100, delta=0.035
