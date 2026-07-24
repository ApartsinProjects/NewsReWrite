"""Download the Webis-Clickbait-17 corpus.

Provenance
----------
Homepage : https://webis.de/data/webis-clickbait-17.html
Zenodo   : https://zenodo.org/records/5530410
Paper    : Potthast et al., "Crowdsourcing a Large Corpus of Clickbait on Twitter",
           COLING 2018.

Files of interest
-----------------
- instances.jsonl  : one JSON per tweet with the "postText" field (the headline).
- truth.jsonl      : one JSON per tweet with "truthMean" (graded clickbait score
                     in [0,1]) and "truthClass" ("clickbait" | "no-clickbait").
The two files are joined on "id".

License
-------
CC BY 4.0. Cite the paper above if you use it.

Usage
-----
    python download_webis.py                # download to data/raw/webis17/
    python download_webis.py --dry-run      # print URLs, no download
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import WEBIS_DIR, ensure_dirs

# The Webis-Clickbait-17 record on Zenodo (5530410) publishes several archives.
# For guide-model evaluation we only need the fully annotated training subset
# clickbait17-train-170331.zip (~141 MB), which ships instances.jsonl and
# truth.jsonl. The larger 170630 release adds unlabelled media assets we do not
# need. If you want more data, switch to clickbait17-train-170630.zip.
WEBIS_ZIP_URL = (
    "https://zenodo.org/api/records/5530410/files/"
    "clickbait17-train-170331.zip/content"
)
WEBIS_ZIP_SHA256 = None  # not pinned; the human can pin after first download.


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[webis] GET {url}")
    print(f"[webis]  -> {dst}")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"  ...{total/1e6:.1f} MB", end="\r")
        print()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(WEBIS_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    zip_path = out / "webis-clickbait-17.zip"

    if args.dry_run:
        print(f"[dry-run] would download {WEBIS_ZIP_URL} to {zip_path}")
        print("[dry-run] would unzip into", out)
        return 0

    ensure_dirs()
    out.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        _download(WEBIS_ZIP_URL, zip_path)
    else:
        print(f"[webis] already present: {zip_path}")

    if WEBIS_ZIP_SHA256:
        got = _sha256(zip_path)
        if got != WEBIS_ZIP_SHA256:
            print(f"[webis] SHA256 MISMATCH: {got} != {WEBIS_ZIP_SHA256}",
                  file=sys.stderr)
            return 2

    import zipfile

    print(f"[webis] unpacking to {out}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)

    # The zip nests the two jsonl files under clickbait17-train-170331/.
    # Flatten so downstream scripts can find them at the top level.
    nested = out / "clickbait17-train-170331"
    if nested.is_dir():
        for name in ("instances.jsonl", "truth.jsonl"):
            src = nested / name
            if src.exists():
                src.replace(out / name)
        # Leave the media/ subdirectory in place if the user wants it, but
        # drop the empty parent if we cleaned everything out.
        try:
            nested.rmdir()
        except OSError:
            pass

    # Drop the archive to save disk once the files are extracted.
    zip_path.unlink(missing_ok=True)

    print("[webis] done. Expect instances.jsonl and truth.jsonl under", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
