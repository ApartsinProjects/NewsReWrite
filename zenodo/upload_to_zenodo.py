#!/usr/bin/env python
"""Upload the built NewsReWrite deposit to Zenodo via the REST API.

This creates a DRAFT deposition and uploads every artifact in ../zenodo_build/.
It does NOT publish — you review and click Publish in the Zenodo web UI, which
mints the DOI. Publishing is irreversible.

Prerequisites:
  1. Run  zenodo/make_package.py  first (populates ../zenodo_build/).
  2. Create a personal access token with 'deposit:write' + 'deposit:actions'
     scopes at https://zenodo.org/account/settings/applications/tokens/new/
     (or the sandbox at https://sandbox.zenodo.org for a dry run).
  3. Export it:   export ZENODO_TOKEN=xxxxxxxx

Usage:
  python zenodo/upload_to_zenodo.py --sandbox     # dry run on sandbox.zenodo.org
  python zenodo/upload_to_zenodo.py               # real zenodo.org draft
"""
import argparse, json, os, sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO.parent / "zenodo_build"
META = json.loads((REPO / "zenodo" / ".zenodo.json").read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Set ZENODO_TOKEN (see header).")
    base = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"

    artifacts = sorted(p for p in BUILD.glob("*")
                       if p.suffix in {".zip", ".gz"} or p.name.endswith(".tar.gz"))
    if not artifacts:
        sys.exit(f"No artifacts in {BUILD}. Run make_package.py first.")
    print(f"Target: {base}  |  {len(artifacts)} files")

    s = requests.Session()
    s.params = {"access_token": token}

    r = s.post(f"{base}/api/deposit/depositions", json={})
    r.raise_for_status()
    dep = r.json()
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    print("Created draft deposition", dep_id)

    for a in artifacts:
        print(f"  uploading {a.name} ({a.stat().st_size/1e6:.1f} MB) ...", flush=True)
        with open(a, "rb") as fh:
            up = s.put(f"{bucket}/{a.name}", data=fh)
        up.raise_for_status()

    # also attach the human-readable manifest + checksums if present
    for extra in ["MANIFEST.txt", "SHA256SUMS.txt"]:
        p = BUILD / extra
        if p.exists():
            with open(p, "rb") as fh:
                s.put(f"{bucket}/{extra}", data=fh).raise_for_status()

    meta = {k: v for k, v in META.items()}
    r = s.put(f"{base}/api/deposit/depositions/{dep_id}",
              json={"metadata": meta})
    r.raise_for_status()
    print("Metadata attached.")
    print(f"\nDraft ready (NOT published):\n  {base}/deposit/{dep_id}")
    print("Review it in the web UI, then click Publish to mint the DOI.")

if __name__ == "__main__":
    main()
