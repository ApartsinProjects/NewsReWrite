"""Central path registry for track_b scripts.

All other scripts import from here so no absolute Windows paths leak into
individual runners. Overridable via environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

# ------------------------------------------------------------------
# Roots
# ------------------------------------------------------------------
# TRACK_B_ROOT points at .../Cycle1/track_b (this file's grandparent).
TRACK_B_ROOT: Path = Path(__file__).resolve().parent.parent

# REPO_ROOT points at .../NewsReWrite (the code repository the paper describes).
# Overridable via NEWSREWRITE_REPO env var.
REPO_ROOT: Path = Path(
    os.environ.get(
        "NEWSREWRITE_REPO",
        str(TRACK_B_ROOT.parent.parent / "NewsReWrite"),
    )
).resolve()

# ------------------------------------------------------------------
# Model directories (produced by the repo's training scripts)
# ------------------------------------------------------------------
CLICKBAIT_MODEL_DIR: Path = Path(
    os.environ.get(
        "CLICKBAIT_MODEL_DIR",
        str(REPO_ROOT / "bert_clickbait_prefix_finetuned"),
    )
)
TACTICS_MODEL_DIR: Path = Path(
    os.environ.get(
        "TACTICS_MODEL_DIR",
        str(REPO_ROOT / "bert_tactics_prefix_finetuned"),
    )
)

# ------------------------------------------------------------------
# Data directories
# ------------------------------------------------------------------
DATA_DIR: Path = TRACK_B_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
WEBIS_DIR: Path = RAW_DIR / "webis17"
CHAKRABORTY_DIR: Path = RAW_DIR / "chakraborty16"
ISOT_DIR: Path = RAW_DIR / "isot"

# ------------------------------------------------------------------
# Repo dataset paths (in-repo synthetic data used as source neutrals)
# ------------------------------------------------------------------
REPO_TEST_CSV_CANDIDATES = [
    REPO_ROOT / "Dataset_generation" / "combined_news_test.csv",
    REPO_ROOT / "Dataset_generation" / "clickbait_generated_test.csv",
    REPO_ROOT / "Dataset_generation" / "clickbait_generated_train_val.csv",
]


def find_repo_test_csv() -> Path:
    """Return the first existing repo test CSV, or the first candidate."""
    for p in REPO_TEST_CSV_CANDIDATES:
        if p.exists():
            return p
    return REPO_TEST_CSV_CANDIDATES[0]


# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
RESULTS_DIR: Path = TRACK_B_ROOT / "results"

# ------------------------------------------------------------------
# Base LLM (kept in sync with fudge_controlled_generation.py)
# ------------------------------------------------------------------
BASE_LLM_ID: str = os.environ.get(
    "BASE_LLM_ID", "meta-llama/Meta-Llama-3-8B-Instruct"
)

# Tactic vocabulary (must match the training-time order)
TACTIC_NAMES = [
    "curiosity gap",
    "exaggeration",
    "emotional trigger",
    "sensationalism",
    "lists or superlatives",
    "ambiguous references",
    "direct appeals",
    "unfinished narratives",
    "unexpected associations",
    "provocative questions",
]


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, WEBIS_DIR, CHAKRABORTY_DIR, ISOT_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("TRACK_B_ROOT       :", TRACK_B_ROOT)
    print("REPO_ROOT          :", REPO_ROOT)
    print("CLICKBAIT_MODEL_DIR:", CLICKBAIT_MODEL_DIR)
    print("TACTICS_MODEL_DIR  :", TACTICS_MODEL_DIR)
    print("Repo test CSV      :", find_repo_test_csv())
