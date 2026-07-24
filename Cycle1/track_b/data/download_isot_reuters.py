"""Download the ISOT Fake News dataset (True.csv is the Reuters half).

Provenance
----------
Homepage : https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/
Kaggle   : https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
Paper    : Ahmed, Traore, Saad, 2017/2018 (ISOT).

Files
-----
True.csv  : Reuters news, 21417 rows, columns [title, text, subject, date].
Fake.csv  : mixed unreliable sources, 23481 rows.

We only need True.csv for the b7 spot-check.

License
-------
Distributed for research use. Cite the ISOT team.

Because the primary hosts (Kaggle, UVic) require login or occasionally change
URLs, this script tries a small ordered list of known public mirrors. If all
fail, it prints instructions for a manual download to the target directory.

Usage
-----
    python download_isot_reuters.py
    python download_isot_reuters.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import ISOT_DIR, ensure_dirs

# Best-effort mirrors. Add or reorder as needed. The first entry is a public
# GitHub mirror that has been verified to serve the exact 21,417-row True.csv;
# the later entries are the original sources for reference and future proofing.
CANDIDATE_URLS = [
    # GitHub mirror (verified 2026-07: 53 MB, 21,417 rows, columns
    # [title, text, subject, date], subjects politicsNews + worldnews).
    "https://raw.githubusercontent.com/marwaa123/fake_real_news/master/True.csv",
    # HuggingFace mirror (may 401 without auth).
    "https://huggingface.co/datasets/mvarma/isot_fake_news/resolve/main/True.csv",
    # UVic host (may 403 without a browser UA).
    "https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2022/11/True.csv",
]

MANUAL_INSTRUCTIONS = """
Automatic download failed. Please fetch True.csv manually:

  1. Go to https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
     (or the UVic ISOT page) and download True.csv.
  2. Place it at:
       {target}

Then re-run this script; it will detect the file and exit cleanly.
"""


def _download(url: str, dst: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
            f.write(r.read())
        return True
    except Exception as e:
        print(f"[isot] failed {url}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ISOT_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    dst = out / "True.csv"

    if args.dry_run:
        print("[dry-run] candidate URLs:")
        for u in CANDIDATE_URLS:
            print("  ", u)
        print(f"[dry-run] target: {dst}")
        return 0

    ensure_dirs()
    out.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        print(f"[isot] already present: {dst}")
        return 0

    for url in CANDIDATE_URLS:
        print(f"[isot] trying {url}")
        if _download(url, dst) and dst.stat().st_size > 100_000:
            print(f"[isot] downloaded to {dst}")
            return 0
        if dst.exists() and dst.stat().st_size <= 100_000:
            dst.unlink()

    print(MANUAL_INSTRUCTIONS.format(target=dst), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
