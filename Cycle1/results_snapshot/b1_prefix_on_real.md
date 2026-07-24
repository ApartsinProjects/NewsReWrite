# B1-prefix: prefix clickbait guide on REAL data at partial lengths

AUROC = prefix score vs the real full-headline label. sep = mean(prob | clickbait) - mean(prob | non-clickbait).

## webis17
- n = 2459, positive rate = 0.310
- AUROC monotone as prefix grows: True

| ratio | AUROC | 95% CI | mean_prob cb | mean_prob non | sep |
|---|---|---|---|---|---|
| 0.3 | 0.6947 | 0.6725-0.7171 | 0.3885 | 0.1281 | +0.2605 |
| 0.5 | 0.7271 | 0.7061-0.7468 | 0.4253 | 0.1267 | +0.2986 |
| 0.7 | 0.7331 | 0.7130-0.7537 | 0.4406 | 0.1354 | +0.3053 |
| 1.0 | 0.7386 | 0.7178-0.7589 | 0.4100 | 0.1129 | +0.2971 |

## chakraborty16
- n = 8000, positive rate = 0.497
- AUROC monotone as prefix grows: True

| ratio | AUROC | 95% CI | mean_prob cb | mean_prob non | sep |
|---|---|---|---|---|---|
| 0.3 | 0.8910 | 0.8839-0.8982 | 0.6702 | 0.0518 | +0.6184 |
| 0.5 | 0.9150 | 0.9088-0.9211 | 0.7068 | 0.0422 | +0.6646 |
| 0.7 | 0.9363 | 0.9310-0.9415 | 0.7351 | 0.0376 | +0.6974 |
| 1.0 | 0.9543 | 0.9499-0.9586 | 0.7779 | 0.0324 | +0.7455 |
