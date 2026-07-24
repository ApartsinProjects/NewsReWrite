"""Shared loaders for the repo's two BERT guide models.

Kept in one place so b1, b2, b4, b7 all use identical tokenization
and truncation behaviour (max_length=32, same as fudge_controlled_generation.py).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .paths import CLICKBAIT_MODEL_DIR, TACTICS_MODEL_DIR, TACTIC_NAMES


def _lazy_torch():
    import torch  # local import so --help works without torch
    return torch


class ClickbaitScorer:
    """Binary clickbait probability from the repo's fine-tuned BERT."""

    def __init__(self, model_dir=None, device: str = "cpu"):
        torch = _lazy_torch()
        # Auto classes so the same loader handles the BERT guide models AND
        # the DistilBERT external detector (train_external_clickbait.py).
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        model_dir = str(model_dir or CLICKBAIT_MODEL_DIR)
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
        )

    def prob(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        torch = _lazy_torch()
        out: List[float] = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i : i + batch_size])
            enc = self.tok(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=32,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            out.extend(p.tolist())
        return np.asarray(out, dtype=float)


class TacticsScorer:
    """Multi-label engagement-attribute probabilities."""

    def __init__(self, model_dir=None, device: str = "cpu"):
        torch = _lazy_torch()
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        model_dir = str(model_dir or TACTICS_MODEL_DIR)
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
        )

    def probs(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Return shape (N, num_tactics) sigmoid probabilities."""
        torch = _lazy_torch()
        rows: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i : i + batch_size])
            enc = self.tok(
                chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=32,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
                p = torch.sigmoid(logits).cpu().numpy()
            rows.append(p)
        return np.concatenate(rows, axis=0) if rows else np.zeros((0, len(TACTIC_NAMES)))
