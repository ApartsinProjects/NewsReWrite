"""One-shot uploader that pushes the finetuned guide models and the
held-out source-neutrals CSV into the newsrewrite-track-b-store Volume.

Run locally (not on Modal) after `modal deploy app.py`:

    python upload_artifacts.py \
        --clickbait-dir  PATH_TO/bert_clickbait_prefix_finetuned \
        --tactics-dir    PATH_TO/bert_tactics_prefix_finetuned \
        --source-neutrals PATH_TO/combined_news_test.csv \
        [--isot-true PATH_TO/True.csv]

If any --* argument is omitted, that artefact is skipped. Re-runs are safe:
Modal Volumes overwrite by path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import modal

VOL_NAME = "newsrewrite-track-b-store"


def push_dir(vol: modal.Volume, local: Path, remote: str) -> None:
    print(f"[upload] {local} -> {remote}/  (dir)")
    with vol.batch_upload() as batch:
        batch.put_directory(str(local), remote)


def push_file(vol: modal.Volume, local: Path, remote: str) -> None:
    print(f"[upload] {local} -> {remote}  (file)")
    with vol.batch_upload() as batch:
        batch.put_file(str(local), remote)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clickbait-dir")
    ap.add_argument("--tactics-dir")
    ap.add_argument("--source-neutrals")
    ap.add_argument("--isot-true")
    args = ap.parse_args()

    vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

    if args.clickbait_dir:
        push_dir(
            vol,
            Path(args.clickbait_dir),
            "models/bert_clickbait_prefix_finetuned",
        )
    if args.tactics_dir:
        push_dir(
            vol,
            Path(args.tactics_dir),
            "models/bert_tactics_prefix_finetuned",
        )
    if args.source_neutrals:
        push_file(vol, Path(args.source_neutrals), "data/source_neutrals.csv")
    if args.isot_true:
        push_file(vol, Path(args.isot_true), "data/isot/True.csv")

    print("[upload] done. Verify with:  modal run app.py::check_env")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
