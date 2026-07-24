"""Fine-tune the multi-label tactics BERT guide on prefix data.

Reproduces NewsReWrite/models/bert_train_tactics_prefix.py, with the same
CLI + per-example length-aware weight adaptations as train_clickbait.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TACTIC_NAMES = [
    "Curiosity Gap",
    "Exaggeration",
    "Emotional Triggers",
    "Sensationalism",
    "Lists/Superlatives",
    "Ambiguous References",
    "Direct Appeals",
    "Unfinished Narratives",
    "Unexpected Associations",
    "Provocative Questions",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trainval-csv", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-name", default="bert-base-uncased")
    p.add_argument("--max-len", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset
    from transformers import (
        BertForSequenceClassification, BertTokenizer,
        EarlyStoppingCallback, Trainer, TrainingArguments,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.trainval_csv)
    df["text"] = df["text"].astype(str)
    if "weight" not in df.columns:
        df["weight"] = 1.0

    def parse_vec(cell):
        s = str(cell).strip()
        try:
            return [int(x) for x in json.loads(s)]
        except Exception:
            return [int(x) for x in json.loads(s.replace("'", '"'))]

    df["tactics_vector"] = df["tactics_vector"].apply(parse_vec)
    y = np.array(df["tactics_vector"].tolist(), dtype=np.float32)
    num_labels = y.shape[1]
    print(f"[tac] trainval size {len(df)} labels={num_labels} "
          f"pos_per_label={y.sum(axis=0).tolist()}", flush=True)

    X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
        df["text"].values, y, df["weight"].values,
        test_size=args.val_frac, random_state=args.seed, shuffle=True,
    )
    print(f"[tac] internal split train={len(X_tr)} val={len(X_va)}", flush=True)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    def tok(texts):
        return tokenizer(
            list(texts), padding=True, truncation=True, max_length=args.max_len,
        )

    train_enc = tok(X_tr)
    val_enc = tok(X_va)

    class TacDS(Dataset):
        def __init__(self, enc, y, w):
            self.enc = enc
            self.y = y
            self.w = list(w)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.y[i], dtype=torch.float)
            item["weight"] = torch.tensor(float(self.w[i]), dtype=torch.float)
            return item

    train_ds = TacDS(train_enc, y_tr, w_tr)
    val_ds = TacDS(val_enc, y_va, w_va)

    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    def compute_metrics(pred):
        logits, labels = pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs > 0.5).astype(int)
        return {
            "tactics_precision_macro": precision_score(
                labels, preds, average="macro", zero_division=0),
            "tactics_recall_macro": recall_score(
                labels, preds, average="macro", zero_division=0),
            "tactics_f1_macro": f1_score(
                labels, preds, average="macro", zero_division=0),
            "micro_f1": f1_score(
                labels, preds, average="micro", zero_division=0),
        }

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("weight")
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            per = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="none",
            ).mean(dim=1)
            loss = (per * weights).sum() / weights.sum().clamp(min=1e-6)
            return (loss, outputs) if return_outputs else loss

    import torch as _torch
    _use_fp16 = _torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(Path(args.out_dir) / "_trainer"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        # dataloader_num_workers=0 on purpose: the pre-tokenized dataset's
        # __getitem__ is a trivial in-process tensor index, so workers add no
        # throughput but exhaust Modal's small /dev/shm (SIGBUS -> RemoteError).
        # The speedup comes from fp16 + batch size, not worker processes.
        fp16=_use_fp16,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    val_pred = trainer.predict(val_ds)
    probs = torch.sigmoid(torch.tensor(val_pred.predictions)).numpy()
    preds = (probs > 0.5).astype(int)
    macro = f1_score(y_va, preds, average="macro", zero_division=0)
    per_label = f1_score(y_va, preds, average=None, zero_division=0)
    print(f"[tac] final val macro-F1 = {macro:.4f}")
    for name, s in zip(TACTIC_NAMES, per_label):
        print(f"  {name:28s} F1 = {s:.4f}")

    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"[tac] saved model to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
