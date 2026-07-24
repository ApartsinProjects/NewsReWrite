"""Append-only, crash-safe artifact capture for the pipeline.

Every generation and scoring step writes its RAW per-datapoint records to a
JSONL sidecar via JsonlWriter, in addition to the final CSV/summary. Because
each record is one line written with an immediate flush, a crash preserves
every record produced so far. The same file then serves three purposes:

  1. Analysis / bug hunting: the raw model input and output for every data
     point (raw GPT responses, full rewrite prompts + completions, raw LLM
     judge outputs), not just the parsed final value.
  2. Resume: `load_done_keys` reads an existing JSONL and returns the set of
     already-processed keys so a re-run skips them.
  3. Provenance: `save_manifest` records how each artifact was produced
     (args, models, seeds, counts, timestamps).

Durability model: flush-per-line keeps records in the OS buffer immediately;
the Modal step's periodic STORE.commit() (see _run in app.py) pushes them to
the volume backend, so a container kill loses at most one commit interval.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class JsonlWriter:
    """Append-only JSONL writer, one flushed line per record."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered append; existing content is preserved (resume-friendly)
        self._f = open(self.path, "a", encoding="utf-8", buffering=1)

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_done_keys(path, key_fields) -> set:
    """Return the set of key tuples already present in an existing JSONL.

    key_fields: list of record field names whose tuple identifies a datapoint.
    Malformed lines are skipped so a half-written trailing line never breaks
    resume.
    """
    keys: set = set()
    p = Path(path)
    if not p.exists():
        return keys
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        keys.add(tuple(r.get(k) for k in key_fields))
    return keys


def save_manifest(path, **fields) -> None:
    """Write a provenance manifest (how this artifact was produced)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields.setdefault("saved_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    p.write_text(
        json.dumps(fields, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
