"""Fine-tune the binary clickbait BERT guide on prefix data.

Reproduces NewsReWrite/models/bert_train_binary_prefix.py with three
changes suited to Modal training:
  - CLI paths (no hard-coded D:\\...)
  - per-example length-aware weight (weight column from
    build_prefix_datasets.py)
  - explicit validation AUROC + F1 print at the end
"""
from __future__ import annotations

import argparse
from pathlib import Path


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
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Dataset
    from transformers import (
        BertForSequenceClassification, BertTokenizer,
        EarlyStoppingCallback, Trainer, TrainingArguments,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.trainval_csv)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    if "weight" not in df.columns:
        df["weight"] = 1.0
    print(f"[bin] trainval size {len(df)} pos={int(df['label'].sum())} "
          f"neg={int((df['label'] == 0).sum())}", flush=True)

    X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
        df["text"].values, df["label"].values, df["weight"].values,
        test_size=args.val_frac, random_state=args.seed, stratify=df["label"].values,
    )
    print(f"[bin] internal split train={len(X_tr)} val={len(X_va)}", flush=True)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    def tok(texts):
        return tokenizer(
            list(texts), padding=True, truncation=True, max_length=args.max_len,
        )

    train_enc = tok(X_tr)
    val_enc = tok(X_va)

    class BinDS(Dataset):
        def __init__(self, enc, y, w):
            self.enc = enc
            self.y = list(y)
            self.w = list(w)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(int(self.y[i]))
            item["weight"] = torch.tensor(float(self.w[i]), dtype=torch.float)
            return item

    train_ds = BinDS(train_enc, y_tr, w_tr)
    val_ds = BinDS(val_enc, y_va, w_va)

    model = BertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2,
    )

    def compute_metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0,
        )
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("weight")
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            per = torch.nn.functional.cross_entropy(
                logits, labels, reduction="none",
            )
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
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        # Feed the GPU properly: fp16 tensor cores on T4/A10G plus pinned
        # memory give a several-x throughput increase over the fp32 default.
        # Keep dataloader_num_workers=0: the dataset is pre-tokenized and its
        # __getitem__ is a trivial in-process tensor index, so worker
        # processes add no throughput but DO exhaust the container's small
        # /dev/shm (64 MB on Modal), crashing with SIGBUS -> RemoteError.
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

    print("[bin] per-epoch history:")
    for log in trainer.state.log_history:
        if "loss" in log and "eval_loss" not in log:
            print(f"  epoch {log.get('epoch')} train_loss={log['loss']:.4f}")
        if "eval_loss" in log:
            print(f"  epoch {log.get('epoch')} val_loss={log['eval_loss']:.4f} "
                  f"f1={log.get('eval_f1'):.4f}")

    # Final validation AUROC + F1.
    val_pred = trainer.predict(val_ds)
    probs = torch.softmax(torch.tensor(val_pred.predictions), dim=1)[:, 1].numpy()
    val_preds = (probs > 0.5).astype(int)
    print(f"[bin] final val AUROC = {roc_auc_score(y_va, probs):.4f}")
    print(f"[bin] final val F1    = {f1_score(y_va, val_preds):.4f}")

    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"[bin] saved model to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
