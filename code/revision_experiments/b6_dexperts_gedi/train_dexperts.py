"""B6 DExperts: fine-tune the expert and anti-expert small LMs.

DExperts (Liu et al. 2021) steers a large base LM at decoding time with a
product-of-experts over next-token logits:

    final_logits = base_logits + alpha * (expert_logits - antiexpert_logits)

The expert and anti-expert must SHARE the base model's tokenizer/vocabulary.
The base generator in this project is meta-llama/Meta-Llama-3-8B-Instruct, so
the small experts are fine-tuned from meta-llama/Llama-3.2-1B, which carries the
same Llama-3 tokenizer.

This script fine-tunes that small LM twice with plain causal-LM (next-token)
training on raw headline strings:

    expert     -> CLICKBAIT headlines   (the engagement-bearing class)
    antiexpert -> NEUTRAL source headlines

At decode time alpha then traces the fidelity-vs-engagement tradeoff, directly
comparable to FUDGE's alpha in b2/run_rewrites.py.

Outputs: two save_pretrained directories (--expert-out, --antiexpert-out).
Heavy torch/transformers imports happen AFTER argparse so --help stays fast.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic-csv", required=True,
                    help="synthetic dataset CSV (columns: original, clickbait, "
                         "methods_vector). The 'clickbait' column trains the expert.")
    ap.add_argument("--source-csv", required=True,
                    help="source_neutrals.csv with a 'title' column; trains the antiexpert.")
    ap.add_argument("--expert-out", required=True, help="output dir for the clickbait expert LM")
    ap.add_argument("--antiexpert-out", required=True, help="output dir for the neutral antiexpert LM")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import random

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device} model={args.model_name}")

    def load_headlines(csv_path: str, column: str) -> list[str]:
        df = pd.read_csv(csv_path)
        if column not in df.columns:
            raise SystemExit(f"[train] ERROR: column '{column}' not in {csv_path} "
                             f"(has: {list(df.columns)})")
        texts = [str(t).strip() for t in df[column].tolist() if str(t).strip()]
        print(f"[train] {csv_path} column='{column}': {len(texts)} headlines")
        return texts

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print("[train] ERROR loading tokenizer/model. If this is a gated-model "
              "license error, accept the license at "
              "https://huggingface.co/meta-llama/Llama-3.2-1B then re-run.")
        print(f"[train] {type(e).__name__}: {e}")
        return 2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def finetune(texts: list[str], out_dir: str, tag: str) -> None:
        print(f"[train] === fine-tuning {tag} -> {out_dir} ===")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
        ).to(device)
        model.train()

        def collate(batch_texts: list[str]):
            enc = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                            max_length=args.max_length, padding=True)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            enc["labels"] = labels
            return enc

        loader = DataLoader(texts, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            running = 0.0
            n = 0
            for step, enc in enumerate(loader):
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                loss = out.loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item()
                n += 1
                if step % 20 == 0:
                    print(f"[train] {tag} epoch {epoch} step {step} "
                          f"loss={loss.item():.4f}", flush=True)
            avg = running / max(n, 1)
            print(f"[train] {tag} epoch {epoch} mean train loss={avg:.4f}", flush=True)

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"[train] saved {tag} to {out_dir}")

    expert_texts = load_headlines(args.synthetic_csv, "clickbait")
    antiexpert_texts = load_headlines(args.source_csv, "title")

    finetune(expert_texts, args.expert_out, "expert(clickbait)")
    finetune(antiexpert_texts, args.antiexpert_out, "antiexpert(neutral)")

    print("[train] done. Both expert and antiexpert saved and ready for run_dexperts.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
