#!/usr/bin/env python
"""Build the Zenodo deposit archives for NewsReWrite (core-models scope).

Produces, in ../zenodo_build/ (relative to the repo root, NOT committed):
  newsrewrite_data.zip            our generated data (processed + prefix_generation)
  newsrewrite_code_results.zip    code/ + results/ + requirements.txt + LICENSES
  bert_clickbait_prefix_guide.tar.gz     clickbait guide, inference files only
  bert_tactics_prefix_guide.tar.gz       engagement-attribute guide, inference files only
  external_clickbait_distilbert.tar.gz   independent DistilBERT detector
  SHA256SUMS.txt, MANIFEST.txt

Third-party corpora (data/external/*) and Llama-derived baselines are NOT
redistributed here; see zenodo/EXTERNAL_DATA.md and code/revision_experiments.

Run:  /c/Python314/python zenodo/make_package.py
"""
import hashlib, os, tarfile, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO.parent / "zenodo_build"
OUT.mkdir(exist_ok=True)

# inference files to keep for a model checkpoint (drop _trainer/ optimizer state)
KEEP = {"config.json", "model.safetensors", "pytorch_model.bin",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.txt", "merges.txt", "vocab.json"}

def add_zip(zf, base, arc_root):
    for p in sorted(base.rglob("*")):
        if p.is_file():
            zf.write(p, Path(arc_root) / p.relative_to(base.parent))

def build_data_zip():
    dst = OUT / "newsrewrite_data.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for sub in ["data/processed", "data/prefix_generation"]:
            add_zip(zf, REPO / sub, "newsrewrite_data")
        zf.writestr("newsrewrite_data/README.txt",
            "Generated data for NewsReWrite.\n"
            "processed/       synthetic clickbait corpus, source neutrals, prefix splits\n"
            "prefix_generation/ raw prefix-generation CSVs\n"
            "Third-party corpora (Chakraborty16, ISOT, Webis-17) are NOT included; "
            "see EXTERNAL_DATA.md for retrieval.\n")
    return dst

def build_code_zip():
    dst = OUT / "newsrewrite_code_results.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for sub in ["code", "results"]:
            add_zip(zf, REPO / sub, "newsrewrite_code_results")
        zf.write(REPO / "requirements.txt", "newsrewrite_code_results/requirements.txt")
        zf.write(REPO / "zenodo" / "LICENSES.md", "newsrewrite_code_results/LICENSES.md")
    return dst

def build_model_tar(model_dir: Path, out_name: str):
    dst = OUT / out_name
    kept = []
    with tarfile.open(dst, "w:gz", compresslevel=1) as tf:
        for p in sorted(model_dir.iterdir()):
            if p.is_file() and p.name in KEEP:
                tf.add(p, arcname=f"{model_dir.name}/{p.name}")
                kept.append(p.name)
    assert "model.safetensors" in kept or "pytorch_model.bin" in kept, \
        f"no weights packed for {model_dir}"
    return dst, kept

def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    artifacts = []
    print("[1/5] data zip ...");        artifacts.append(build_data_zip())
    print("[2/5] code+results zip ..."); artifacts.append(build_code_zip())
    models = [
        (REPO / "models/guides/bert_clickbait_prefix_finetuned", "bert_clickbait_prefix_guide.tar.gz"),
        (REPO / "models/guides/bert_tactics_prefix_finetuned",   "bert_tactics_prefix_guide.tar.gz"),
        (REPO / "models/baselines/external_clickbait_distilbert","external_clickbait_distilbert.tar.gz"),
    ]
    for i, (md, name) in enumerate(models, start=3):
        print(f"[{i}/5] model {md.name} -> {name} ...")
        dst, kept = build_model_tar(md, name)
        print("       packed:", ", ".join(kept))
        artifacts.append(dst)

    sums, manifest = [], []
    for a in artifacts:
        sz = a.stat().st_size
        digest = sha256(a)
        sums.append(f"{digest}  {a.name}")
        manifest.append(f"{a.name:42s} {sz/1e6:9.1f} MB")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    (OUT / "MANIFEST.txt").write_text(
        "NewsReWrite Zenodo deposit — file manifest\n" + "=" * 48 + "\n" +
        "\n".join(manifest) + "\n")
    print("\nBuilt", len(artifacts), "artifacts in", OUT)
    print("\n".join(manifest))

if __name__ == "__main__":
    main()
