# Obtaining the external corpora

Three human-authored corpora are used for evaluation (Section 4.1–4.3 of the
paper) but are **not** redistributed in this deposit, because each carries its
own redistribution terms. Download them from the original sources and place
them under `data/external/` to reproduce the external-validation results.

| Corpus | Used for | Source |
|---|---|---|
| Chakraborty et al. 2016 | External clickbait AUROC; trains the independent DistilBERT detector | https://github.com/bhargaviparanjape/clickbait (clickbait_data / non_clickbait_data) |
| ISOT Fake News Dataset (True.csv) | Neutral Reuters source pool; Reuters-neutrality check | https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php |
| Webis Clickbait Corpus 2017 | Graded-strength external validation | https://webis.de/data/webis-clickbait-17.html (Zenodo: 10.5281/zenodo.5530410) |

Expected layout after download:

```
data/external/
  chakraborty16/   clickbait_data, non_clickbait_data
  isot/            True.csv, Fake.csv
  webis17/         instances.jsonl, truth.jsonl (or the released split files)
```

The processed/derived artifacts we generated from these (prefix splits, the
synthetic corpus, per-item scores) **are** included in `newsrewrite_data.zip`
and `newsrewrite_code_results.zip`, so most tables can be regenerated without
re-downloading the raw corpora.
