# Licenses — NewsReWrite Zenodo deposit

This deposit bundles several artifact types under different licenses.

| Component | Files | License |
|---|---|---|
| Source code | `newsrewrite_code_results.zip` → `code/` | MIT |
| Generated data | `newsrewrite_data.zip` → `data/processed`, `data/prefix_generation` | CC-BY-4.0 |
| Clickbait guide model | `bert_clickbait_prefix_guide.tar.gz` | Apache-2.0 (fine-tuned from `bert-base-uncased`) |
| Engagement-attribute guide model | `bert_tactics_prefix_guide.tar.gz` | Apache-2.0 (fine-tuned from `bert-base-uncased`) |
| Independent clickbait detector | `external_clickbait_distilbert.tar.gz` | Apache-2.0 (fine-tuned from `distilbert-base-uncased`) |

## Not included in this deposit

- **Third-party corpora.** The Chakraborty et al. 2016 corpus, the ISOT Fake
  News Dataset, and the Webis Clickbait Corpus 2017 are redistributed under
  their own terms by their original authors and are **not** rehosted here.
  See `EXTERNAL_DATA.md` for retrieval instructions.
- **DExperts / GeDi baseline checkpoints.** These are derived from
  `meta-llama/Llama-3.2-1B` and are therefore governed by the Llama 3.2
  Community License. They are **not** redistributed here; the training and
  evaluation scripts to reproduce them from the base model are provided in
  `code/revision_experiments/`.

## Synthetic-data provenance

The synthetic clickbait corpus was produced by rewriting neutral Reuters
headlines (ISOT True.csv) with GPT-4o-mini. Users must comply with the source
corpus terms and the OpenAI usage policies when reusing the generated text.
