"""B6: Train the class-conditional LM for the GeDi controlled-generation baseline.

GeDi (Krause et al. 2021) steers a large base LM with a much smaller
class-conditional LM (the "discriminator"). This script fine-tunes that small
LM: a single causal LM that models P(headline | class) for the two classes
{clickbait, neutral}. We do NOT do any tokenizer surgery. Instead we simply
prepend a literal control-code STRING to every training headline:

    "<clickbait> "  + clickbait_headline
    "<neutral> "    + neutral_headline

and fine-tune the model to predict the whole prefixed string. At decode time
(run_gedi.py) the same two control-code strings are prepended to the running
rewrite, and Bayes' rule over the two next-token distributions gives the class
posterior for each candidate token.

The small LM MUST share the base tokenizer so its next-token log-probs align
with the base LM's vocabulary. meta-llama/Llama-3.2-1B shares the Llama-3 128k
tokenizer with the base meta-llama/Meta-Llama-3-8B-Instruct.

Outputs
-------
--out-dir/                     the fine-tuned class-conditional LM (save_pretrained)
--out-dir/gedi_control_codes.json   the two literal control-code strings

Heavy torch/transformers imports happen AFTER argparse so `--help` is cheap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Literal control-code strings prepended to headlines. Kept here (and mirrored
# into gedi_control_codes.json) so run_gedi.py reads the exact same strings.
CLICKBAIT_CODE = "<clickbait> "
NEUTRAL_CODE = "<neutral> "


def build_texts(synthetic_csv: str, source_csv: str) -> list[str]:
    """Assemble the prefixed training strings from both sources.

    synthetic_csv: columns original, clickbait, methods_vector. The 'clickbait'
        column supplies clickbait-class headlines; 'original' supplies neutral.
    source_csv: source_neutrals.csv with a 'title' column, extra neutral text.
    """
    import pandas as pd

    texts: list[str] = []

    syn = pd.read_csv(synthetic_csv)
    for col, code, kind in (
        ("clickbait", CLICKBAIT_CODE, "clickbait"),
        ("original", NEUTRAL_CODE, "neutral"),
    ):
        if col in syn.columns:
            for v in syn[col].dropna().astype(str):
                v = v.strip()
                if v:
                    texts.append(code + v)

    if source_csv and Path(source_csv).exists():
        src = pd.read_csv(source_csv)
        tcol = "title" if "title" in src.columns else src.columns[0]
        for v in src[tcol].dropna().astype(str):
            v = v.strip()
            if v:
                texts.append(NEUTRAL_CODE + v)

    return texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic-csv", required=True,
                    help="CSV with columns original, clickbait, methods_vector")
    ap.add_argument("--source-csv", default="",
                    help="source_neutrals.csv with a 'title' column (extra neutrals)")
    ap.add_argument("--out-dir", required=True,
                    help="output dir for the fine-tuned class-conditional LM")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=56)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    texts = build_texts(args.synthetic_csv, args.source_csv)
    if not texts:
        print("[gedi-train] no training texts assembled; check input CSVs", flush=True)
        return 1
    n_cb = sum(t.startswith(CLICKBAIT_CODE) for t in texts)
    n_ne = len(texts) - n_cb
    print(f"[gedi-train] training strings: {len(texts)} "
          f"(clickbait={n_cb}, neutral={n_ne})", flush=True)

    # Heavy imports deferred so `--help` runs under a bare interpreter.
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gedi-train] device={device} model={args.model_name}", flush=True)

    try:
        tok = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    except Exception as e:
        print(f"[gedi-train] FAILED to load '{args.model_name}': "
              f"{type(e).__name__}: {e}", flush=True)
        print("[gedi-train] this model is gated; request access and log in "
              "(huggingface-cli login). See "
              "https://huggingface.co/meta-llama/Llama-3.2-1B", flush=True)
        return 2
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.to(device).train()

    class HeadlineDS(Dataset):
        def __init__(self, rows: list[str]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int) -> str:
            return self.rows[i]

    def collate(batch: list[str]):
        enc = tok(batch, return_tensors="pt", truncation=True,
                  max_length=args.max_length, padding=True)
        # Causal LM labels = input_ids with pad positions masked out (-100).
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc

    loader = DataLoader(HeadlineDS(texts), batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        running = 0.0
        n_batches = 0
        for enc in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            opt.zero_grad()
            out = model(**enc)
            out.loss.backward()
            opt.step()
            running += out.loss.item()
            n_batches += 1
            if n_batches % 25 == 0:
                print(f"[gedi-train] epoch {epoch} batch {n_batches} "
                      f"loss={out.loss.item():.4f}", flush=True)
        avg = running / max(n_batches, 1)
        print(f"[gedi-train] epoch {epoch} mean train loss = {avg:.4f}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "gedi_control_codes.json").write_text(
        json.dumps({"clickbait": CLICKBAIT_CODE, "neutral": NEUTRAL_CODE,
                    "base_model": args.model_name}, indent=2),
        encoding="utf-8",
    )
    print(f"[gedi-train] saved class-conditional LM -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
