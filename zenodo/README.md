# NewsReWrite — Zenodo deposit

Reproducibility package for *LLM-guided headline rewriting for clickability
enhancement without clickbait*. Paper (HTML): https://apartsinprojects.github.io/NewsReWrite/
· Code: https://github.com/ApartsinProjects/NewsReWrite

This folder is the **recipe** for the deposit. The heavy archives are built
locally into `../zenodo_build/` (not committed) and uploaded to Zenodo with the
included script. Scope: **core models** (BERT guides + independent DistilBERT
detector); Llama-derived baselines and third-party corpora are referenced, not
rehosted (see `LICENSES.md`, `EXTERNAL_DATA.md`).

## Deposit contents

| File | ~Size | What |
|---|---|---|
| `newsrewrite_data.zip` | 23 MB | Synthetic clickbait corpus, source neutrals, prefix splits, prefix-generation CSVs |
| `newsrewrite_code_results.zip` | 55 MB | `code/` (generation, guide training, FUDGE decoding, evaluation) + `results/` (per-item scores behind the tables) + `requirements.txt` + `LICENSES.md` |
| `bert_clickbait_prefix_guide.tar.gz` | 415 MB | Prefix clickbait guide (negative guidance), `bert-base-uncased` fine-tune, inference files only |
| `bert_tactics_prefix_guide.tar.gz` | 415 MB | Engagement-attribute guide (positive guidance), `bert-base-uncased` fine-tune |
| `external_clickbait_distilbert.tar.gz` | 255 MB | Independent DistilBERT detector used for unbiased clickbait scoring in Section 4 |
| `SHA256SUMS.txt`, `MANIFEST.txt` | — | Integrity + inventory |

Model tarballs contain only inference files (`config.json`, `model.safetensors`,
tokenizer). Optimizer/trainer state is stripped.

## How to build and deposit

```bash
# 1. Build archives into ../zenodo_build/
/c/Python314/python zenodo/make_package.py

# 2. Get a Zenodo token (deposit:write, deposit:actions) and export it
export ZENODO_TOKEN=xxxxxxxx

# 3. Dry run on the sandbox first (recommended)
python zenodo/upload_to_zenodo.py --sandbox

# 4. Create the real draft (does NOT publish)
python zenodo/upload_to_zenodo.py
```

The script creates a **draft** and attaches metadata from `.zenodo.json`. It
never publishes: review the draft in the Zenodo web UI and click **Publish**
yourself to mint the DOI (publishing is irreversible). After publishing, add the
DOI back into the paper's Data Availability section and `.zenodo.json`.

## Reproducing the baselines and external evaluation

- DExperts / GeDi checkpoints: train from `meta-llama/Llama-3.2-1B` with the
  scripts in `code/revision_experiments/` (not redistributed; Llama license).
- External corpora (Chakraborty16 / ISOT / Webis-17): download per
  `EXTERNAL_DATA.md`, place under `data/external/`, then run the Section 4.1–4.3
  evaluation scripts.
