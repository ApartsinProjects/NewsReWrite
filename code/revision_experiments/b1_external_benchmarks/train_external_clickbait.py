"""B1: train an INDEPENDENT clickbait detector on the Chakraborty corpus.

This DistilBERT is fine-tuned ONLY on human-authored Chakraborty headlines
(clickbait_data -> label 1, non_clickbait_data -> label 0). Because it never
sees the paper's synthetic pipeline or the FUDGE guide models, its score is an
independent clickbait metric: point score_rewrites.py at its --out-dir via
--external-clickbait-dir to fill the clickbait_prob_external column.

Usage
-----
    python train_external_clickbait.py [--data-dir ...] [--out-dir ...]

ETA  : ~2-5 min on a GPU, ~20-40 min on CPU for the two epochs.
Cost : $0 (local). Prints validation AUROC and F1 at the end.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.paths import CHAKRABORTY_DIR, DATA_DIR


def _read_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(CHAKRABORTY_DIR),
                    help="dir holding clickbait_data and non_clickbait_data")
    ap.add_argument("--out-dir",
                    default=str(DATA_DIR / "models" / "external_clickbait_distilbert"))
    ap.add_argument("--model-name", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Heavy imports AFTER argparse so `--help` works without torch/transformers.
    import numpy as np
    import torch
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    pos = _read_lines(data_dir / "clickbait_data")
    neg = _read_lines(data_dir / "non_clickbait_data")
    texts = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    print(f"[b1-cb] clickbait={len(pos)} non_clickbait={len(neg)} total={len(texts)}")

    tr_x, va_x, tr_y, va_y = train_test_split(
        texts, labels, test_size=args.val_frac, random_state=args.seed,
        stratify=labels,
    )
    print(f"[b1-cb] train={len(tr_x)} val={len(va_x)} (85/15 stratified)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)

    class HeadlineDS(Dataset):
        def __init__(self, x, y):
            self.x, self.y = x, y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            enc = tok(self.x[i], truncation=True, max_length=args.max_length,
                      padding="max_length", return_tensors="pt")
            return (enc["input_ids"][0], enc["attention_mask"][0],
                    torch.tensor(self.y[i], dtype=torch.long))

    tr_dl = DataLoader(HeadlineDS(tr_x, tr_y), batch_size=args.batch_size, shuffle=True)
    va_dl = DataLoader(HeadlineDS(va_x, va_y), batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for ids, mask, y in tr_dl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=y)
            out.loss.backward()
            opt.step()
            running += out.loss.item()
        print(f"[b1-cb] epoch {epoch + 1}/{args.epochs} mean_loss={running / len(tr_dl):.4f}")

    model.eval()
    probs, preds = [], []
    with torch.no_grad():
        for ids, mask, _ in va_dl:
            ids, mask = ids.to(device), mask.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            preds.extend((p >= 0.5).astype(int).tolist())
    auroc = roc_auc_score(va_y, probs)
    f1 = f1_score(va_y, preds)
    print(f"[b1-cb] validation AUROC={auroc:.4f} F1={f1:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[b1-cb] saved independent detector to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
