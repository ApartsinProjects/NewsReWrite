"""Download the Chakraborty et al. 2016 "Stop Clickbait" dataset.

Provenance
----------
Paper : Chakraborty, Paranjape, Kakarla, Ganguly, "Stop Clickbait: Detecting
        and Preventing Clickbaits in Online News Media", ASONAM 2016.
Repo  : https://github.com/bhargaviparanjape/clickbait

The repo ships two plain-text files:
    clickbait_data     : ~16k clickbait headlines
    non_clickbait_data : ~16k non-clickbait headlines

License
-------
The repo does not state an explicit license. Treat as research-use only and
cite the ASONAM 2016 paper.

Usage
-----
    python download_chakraborty.py
    python download_chakraborty.py --dry-run
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import CHAKRABORTY_DIR, ensure_dirs

RAW_BASE = "https://raw.githubusercontent.com/bhargaviparanjape/clickbait/master/dataset"
FILES = {
    "clickbait_data.gz": f"{RAW_BASE}/clickbait_data.gz",
    "non_clickbait_data.gz": f"{RAW_BASE}/non_clickbait_data.gz",
}


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[chakraborty] GET {url}")
    print(f"[chakraborty]  -> {dst}")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        f.write(r.read())


def _gunzip(src: Path, dst: Path) -> None:
    print(f"[chakraborty] gunzip {src.name} -> {dst.name}")
    with gzip.open(src, "rb") as gz, open(dst, "wb") as out:
        shutil.copyfileobj(gz, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(CHAKRABORTY_DIR))
    args = ap.parse_args()

    out = Path(args.out)

    if args.dry_run:
        for name, url in FILES.items():
            print(f"[dry-run] would download {url} to {out / name}")
        return 0

    ensure_dirs()
    out.mkdir(parents=True, exist_ok=True)

    for name, url in FILES.items():
        gz_dst = out / name
        plain_dst = out / name.removesuffix(".gz")
        if plain_dst.exists():
            print(f"[chakraborty] already present: {plain_dst}")
            continue
        if not gz_dst.exists():
            _download(url, gz_dst)
        _gunzip(gz_dst, plain_dst)
        gz_dst.unlink(missing_ok=True)

    print("[chakraborty] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
