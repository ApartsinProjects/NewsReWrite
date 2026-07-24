# Track B external datasets

This directory hosts the fetch scripts for the three external corpora we use
to answer reviewer concerns about generalisation and dataset choice.

| Corpus | Purpose | Reviewer concern | Script |
| --- | --- | --- | --- |
| Webis-Clickbait-17 | Cross-corpus validation of the clickbait scorer and the tactics scorer against a graded human-annotated benchmark | R1: "only tested on your own synthetic data" | `download_webis.py` |
| Chakraborty 2016 (Stop Clickbait) | Second independent clickbait benchmark, binary labels | R1: robustness across labelling conventions | `download_chakraborty.py` |
| ISOT Fake News (Reuters True.csv) | Neutral-headline baseline for the b7 spot check: does the clickbait detector fire on well-formed Reuters headlines? | R2: "your `neutral` label is a training artefact" | `download_isot_reuters.py` |

All scripts write to `data/raw/{corpus}/`. Every script accepts `--dry-run`
which prints the URLs and target paths without downloading. Provenance URLs
and license notes are at the top of each script. Datasets are NOT redistributed
inside this repository; the human running the pipeline is responsible for
respecting each dataset's license.

Total disk footprint after all three downloads is under 200 MB.
