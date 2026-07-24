"""Modal app for Track B experiments.

One image, one persistent Volume, and one function per experiment. Every
function calls back into the plain-Python scripts under Cycle1/track_b/
via subprocess, so the local runbook and the Modal runbook stay in sync.

Volume layout (mounted at /workspace/store):
    /workspace/store/
        models/
            bert_clickbait_prefix_finetuned/   <- YOU upload once
            bert_tactics_prefix_finetuned/     <- YOU upload once
        data/
            webis17/       instances.jsonl, truth.jsonl
            chakraborty16/ clickbait_data, non_clickbait_data
            isot/          True.csv                  <- YOU upload once if needed
            source_neutrals.csv                      <- YOU upload once for b2/b4
        results/           b1_*, b2/, b4/, b7/

Deploy once, then invoke the functions individually:
    modal deploy app.py
    modal run app.py::fetch_external_datasets
    modal run app.py::run_b1_clickbait
    modal run app.py::run_b1_tactics
    modal run app.py::run_b7
    modal run app.py::run_b4 --n-items 300
    modal run app.py::run_b2 --n-items 300
    modal run app.py::run_score --in-subdir b2       # auto metrics only
    modal run app.py::run_score_with_judge --in-subdir b2   # + LLM judge
    modal run app.py::prepare_human_labeling
    modal run app.py::pull_results
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# ------------------------------------------------------------------
# Image: reuse requirements-track-b.txt plus the repo's requirements.
# We pin bert-score / sentence-transformers here explicitly.
# ------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "unzip")
    .pip_install(
        "torch==2.5.1",
        # 4.45+ required to parse Llama-3.2's llama3 rope_scaling config
        # (DExperts/GeDi 1B controllers); 4.41 raised ValueError on it.
        "transformers==4.45.2",
        "accelerate>=1.12.0",
        "numpy>=1.26",
        "pandas>=2.2",
        "scipy>=1.11",
        "scikit-learn>=1.4",
        "seaborn>=0.13",
        "matplotlib>=3.8",
        "tqdm>=4.66",
        "bert-score==0.3.13",
        "sentence-transformers==5.1.2",
        "sentencepiece>=0.1.99",  # DeBERTa-v3 NLI tokenizer
        "huggingface_hub>=0.24",
        "hf_transfer>=0.1.6",
        "python-dotenv>=1.0",
        "requests>=2.32",
    )
    # Copy the whole track_b tree into the image so scripts + common/ are
    # on-path. Exclude only cached raw datasets, results, caches and venvs.
    # Do NOT exclude modal_run/: the train_guides/ helper scripts and
    # extract_source_neutrals.py live under it and are invoked at runtime.
    .add_local_dir(
        local_path=str(Path(__file__).resolve().parent.parent),
        remote_path="/workspace/track_b",
        ignore=[
            "data/raw/**",
            "results/**",
            "modal_export/**",   # local snapshot of the volume; never mount it
            "__pycache__/**",
            "**/__pycache__/**",
            "*.pyc",
            ".venv/**",
            ".idea/**",
        ],
    )
)

app = modal.App("newsrewrite-track-b", image=image)

# ------------------------------------------------------------------
# Persistent volume. Created on first deploy; survives across runs.
# ------------------------------------------------------------------
STORE = modal.Volume.from_name(
    "newsrewrite-track-b-store", create_if_missing=True
)
STORE_MOUNT = "/workspace/store"

# HuggingFace secret is needed for Llama-3-8B-Instruct (gated).
HF_SECRET = modal.Secret.from_name(
    "huggingface", required_keys=["HF_TOKEN"]
)
# OpenAI (or compatible) key for synthetic data generation.
OPENAI_SECRET = modal.Secret.from_name(
    "openai-key", required_keys=["OPENAI_API_KEY"]
)
# Optional API keys for the LLM-as-judge in score_rewrites.py. Only mounted
# on run_score_with_judge so run_score works even if the secret does not
# exist. If you enable the judge, create the secret with any subset of
# ANTHROPIC_API_KEY / OPENAI_API_KEY set.
JUDGE_SECRET = modal.Secret.from_name(
    "llm-judge-keys",
    required_keys=[],
)


# ------------------------------------------------------------------
# Shared helpers -- all functions call this so paths stay consistent.
# ------------------------------------------------------------------
def _env(extra: dict | None = None) -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Unbuffered stdout/stderr so every print, tqdm bar and traceback reaches
    # Modal's log server immediately. This is what makes `modal app logs <id>`
    # observable in REAL TIME on a still-running container.
    env["PYTHONUNBUFFERED"] = "1"
    # Reduce CUDA allocator fragmentation (helps the 1B fp32 trainers avoid OOM
    # on tight cards; harmless elsewhere).
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CLICKBAIT_MODEL_DIR"] = f"{STORE_MOUNT}/models/bert_clickbait_prefix_finetuned"
    env["TACTICS_MODEL_DIR"] = f"{STORE_MOUNT}/models/bert_tactics_prefix_finetuned"
    # Persist HuggingFace + Sentence-Transformers + bert-score caches on the
    # Volume so cold GPU invocations do not re-download Llama-3-8B (~16 GB)
    # and the scoring models (~4 GB) every time.
    hf_cache = f"{STORE_MOUNT}/.cache/hf"
    env["HF_HOME"] = hf_cache
    env["HUGGINGFACE_HUB_CACHE"] = f"{hf_cache}/hub"
    env["TRANSFORMERS_CACHE"] = f"{hf_cache}/hub"
    env["SENTENCE_TRANSFORMERS_HOME"] = f"{STORE_MOUNT}/.cache/sentence-transformers"
    # Parallel Rust downloader: 5-10x faster HF Hub fetches than the default
    # single-stream Python downloader. Applies to every model download.
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if extra:
        env.update(extra)
    return env


def _verify():
    """Import the step-level sanity checkers from the copied track_b tree."""
    if "/workspace/track_b" not in sys.path:
        sys.path.insert(0, "/workspace/track_b")
    from common import verify  # noqa: E402
    return verify


def _run(cmd: list[str], env_extra: dict | None = None,
         log_name: str | None = None,
         periodic_commit_s: int | None = None) -> None:
    """Run a subprocess, streaming its combined stdout+stderr line by line to
    BOTH Modal's log server (real-time, observable via `modal app logs`) AND a
    durable per-step log file on the volume under results/logs/. The volume is
    committed after the step whether it succeeds or fails, so a crashed step
    still leaves an inspectable log. stderr is merged into stdout so tracebacks
    and tqdm bars are captured too.

    periodic_commit_s: if set, a daemon thread commits the volume every N
    seconds WHILE the subprocess runs. For multi-hour generation this means a
    container preemption or timeout loses at most ~N seconds of output rather
    than the whole run; combined with the resumable run_rewrites.py, a re-run
    then continues from the last committed rows.
    """
    import threading
    import time

    logs_dir = Path(f"{STORE_MOUNT}/results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    if log_name is None:
        log_name = Path(cmd[1]).stem if len(cmd) > 1 else "run"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{log_name}_{stamp}.log"

    header = "[modal] $ " + " ".join(cmd)
    print(header, flush=True)

    # Provenance manifest: how this artifact was produced (cmd, timing, exit).
    manifest_dir = Path(f"{STORE_MOUNT}/results/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{log_name}_{stamp}.json"
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t_start = time.time()

    stop = {"flag": False}
    committer = None
    if periodic_commit_s:
        def _periodic():
            while not stop["flag"]:
                time.sleep(periodic_commit_s)
                if stop["flag"]:
                    break
                try:
                    STORE.commit()
                    print(f"[modal] periodic volume commit ({log_name})", flush=True)
                except Exception:
                    pass
        committer = threading.Thread(target=_periodic, daemon=True)
        committer.start()

    rc = None
    try:
        with open(log_path, "w", encoding="utf-8", buffering=1) as lf:
            lf.write(header + "\n")
            proc = subprocess.Popen(
                cmd, env=_env(env_extra),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                print(line, end="", flush=True)   # -> Modal log server (live)
                lf.write(line)                    # -> durable volume log
            proc.wait()
            rc = proc.returncode
            lf.write(f"\n[modal] exit code: {rc}\n")
    finally:
        stop["flag"] = True

    # Write the provenance manifest for this step (always, success or fail).
    try:
        import json as _json
        manifest_path.write_text(_json.dumps({
            "step": log_name,
            "cmd": cmd,
            "started": started,
            "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": round(time.time() - t_start, 1),
            "exit_code": rc,
            "log": str(log_path),
        }, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Persist the log (and any final output) even when the step failed.
    try:
        STORE.commit()
    except Exception:
        pass
    print(f"[modal] log saved to volume: {log_path}", flush=True)
    if rc != 0:
        raise SystemExit(
            f"command failed with {rc}: {cmd[0]} (full log: {log_path})"
        )


def _ensure_dirs() -> None:
    for d in (
        "models",
        "data/webis17",
        "data/chakraborty16",
        "data/isot",
        "results",
        ".cache/hf/hub",
        ".cache/sentence-transformers",
    ):
        Path(f"{STORE_MOUNT}/{d}").mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# 0. Prefetch large models to the volume cache on CPU.
#    Downloads never need a GPU; doing them on a cheap CPU container (with
#    hf_transfer for 5-10x parallel throughput) means the GPU generation and
#    scoring containers start instantly instead of paying GPU-rate to wait on
#    a ~16 GB network fetch. Run once per fresh volume before run_b2/b4/score.
# ------------------------------------------------------------------
@app.function(
    volumes={STORE_MOUNT: STORE},
    secrets=[HF_SECRET],
    timeout=60 * 60,
    retries=2,
    cpu=4.0,
)
def prefetch_models(llm: bool = True, scoring: bool = True) -> None:
    import os as _os
    import time

    _ensure_dirs()
    hub = f"{STORE_MOUNT}/.cache/hf/hub"
    _os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    _os.environ["HF_HOME"] = f"{STORE_MOUNT}/.cache/hf"
    _os.environ["HUGGINGFACE_HUB_CACHE"] = hub

    from huggingface_hub import snapshot_download

    # Each target is (repo, ignore_patterns). transformers loads Llama-3 from
    # the sharded HF safetensors, so we skip the ~16 GB original PyTorch
    # consolidated checkpoint (original/consolidated.*.pth) and any .pth/.gguf.
    # That roughly halves the Llama footprint on the volume.
    targets = []
    if llm:
        targets.append((
            "meta-llama/Meta-Llama-3-8B-Instruct",
            ["original/*", "*.pth", "*.gguf"],
        ))
    if scoring:
        # Models pulled by score_rewrites.py: NLI, STS, and the BERTScore
        # English backbone. Prefetching these makes run_score start instantly.
        # No ignore list: these are small and may ship only .bin weights.
        targets += [
            ("roberta-large-mnli", None),
            ("sentence-transformers/all-mpnet-base-v2", None),
            ("roberta-large", None),
        ]

    for repo, ignore in targets:
        t0 = time.time()
        print(f"[prefetch] downloading {repo} "
              f"(ignore={ignore}) ...", flush=True)
        try:
            snapshot_download(repo_id=repo, cache_dir=hub,
                              ignore_patterns=ignore)
            mb = sum(p.stat().st_size for p in Path(hub).rglob("*") if p.is_file())
            print(f"[prefetch] {repo} done in {time.time()-t0:.0f}s "
                  f"(cache now {mb/1e9:.1f} GB)", flush=True)
        except Exception as e:
            print(f"[prefetch] WARN: {repo} failed: {e}", flush=True)
    STORE.commit()
    print("[prefetch] models cached to volume", flush=True)


# ------------------------------------------------------------------
# 1. Fetch public external datasets into the volume.
#    Runs on CPU; ~2 min end-to-end.
# ------------------------------------------------------------------
@app.function(
    volumes={STORE_MOUNT: STORE},
    timeout=15 * 60,
    retries=2,
)
def fetch_external_datasets() -> None:
    _ensure_dirs()
    scripts_dir = "/workspace/track_b/data"
    _run([sys.executable, f"{scripts_dir}/download_webis.py",
          "--out", f"{STORE_MOUNT}/data/webis17"])
    _run([sys.executable, f"{scripts_dir}/download_chakraborty.py",
          "--out", f"{STORE_MOUNT}/data/chakraborty16"])
    # ISOT is often gated; if the automatic fetch fails, user must upload.
    try:
        _run([sys.executable, f"{scripts_dir}/download_isot_reuters.py",
              "--out", f"{STORE_MOUNT}/data/isot"])
    except SystemExit as e:
        print(f"[modal] WARN: ISOT auto-fetch failed ({e}). Upload True.csv "
              f"manually to {STORE_MOUNT}/data/isot/True.csv and re-run b7.")
    # Gate: the two external corpora must be present and non-trivial before
    # any downstream benchmark trusts them.
    v = _verify()
    v.require_file(f"{STORE_MOUNT}/data/webis17/instances.jsonl", min_bytes=10_000)
    v.require_file(f"{STORE_MOUNT}/data/webis17/truth.jsonl", min_bytes=10_000)
    v.require_file(f"{STORE_MOUNT}/data/chakraborty16/clickbait_data", min_bytes=10_000)
    v.require_file(f"{STORE_MOUNT}/data/chakraborty16/non_clickbait_data", min_bytes=10_000)
    STORE.commit()


# ------------------------------------------------------------------
# 2. b1 external benchmark, clickbait scorer.
#    Small model; CPU is fine (~5 min) but a T4 is cheap and 10x faster.
# ------------------------------------------------------------------
@app.function(
    volumes={STORE_MOUNT: STORE},
    gpu="T4",
    timeout=30 * 60,
    retries=2,
)
def run_b1_clickbait(n_boot: int = 1000) -> None:
    _ensure_dirs()
    _run([
        sys.executable,
        "/workspace/track_b/b1_external_benchmarks/eval_clickbait_scorer.py",
        "--gpu",
        "--n-boot", str(n_boot),
        "--webis-dir", f"{STORE_MOUNT}/data/webis17",
        "--chakraborty-dir", f"{STORE_MOUNT}/data/chakraborty16",
        "--out-json", f"{STORE_MOUNT}/results/b1_clickbait_external.json",
        "--out-md", f"{STORE_MOUNT}/results/b1_clickbait_external.md",
    ])
    _verify().check_json_report(
        f"{STORE_MOUNT}/results/b1_clickbait_external.json")
    STORE.commit()


@app.function(
    volumes={STORE_MOUNT: STORE},
    gpu="T4",
    timeout=30 * 60,
    retries=2,
)
def run_b1_prefix_on_real(n_boot: int = 1000, max_n: int = 8000) -> None:
    """Validate the PREFIX clickbait guide on REAL data at partial prefix
    lengths (the FUDGE decode regime), closing the full-vs-partial gap."""
    _ensure_dirs()
    _verify().check_model_dir(
        f"{STORE_MOUNT}/models/bert_clickbait_prefix_finetuned", expect_labels=2)
    _run([
        sys.executable,
        "/workspace/track_b/b1_external_benchmarks/eval_prefix_on_real.py",
        "--gpu",
        "--n-boot", str(n_boot),
        "--max-n", str(max_n),
        "--webis-dir", f"{STORE_MOUNT}/data/webis17",
        "--chakraborty-dir", f"{STORE_MOUNT}/data/chakraborty16",
        "--out-json", f"{STORE_MOUNT}/results/b1_prefix_on_real.json",
        "--out-md", f"{STORE_MOUNT}/results/b1_prefix_on_real.md",
    ])
    _verify().check_json_report(f"{STORE_MOUNT}/results/b1_prefix_on_real.json")
    STORE.commit()


@app.function(
    volumes={STORE_MOUNT: STORE},
    gpu="T4",
    timeout=30 * 60,
    retries=2,
)
def run_b1_tactics(n_boot: int = 1000) -> None:
    _ensure_dirs()
    _run([
        sys.executable,
        "/workspace/track_b/b1_external_benchmarks/eval_tactics_scorer.py",
        "--gpu",
        "--n-boot", str(n_boot),
        "--webis-dir", f"{STORE_MOUNT}/data/webis17",
        "--out-json", f"{STORE_MOUNT}/results/b1_tactics_external.json",
        "--out-md", f"{STORE_MOUNT}/results/b1_tactics_external.md",
    ])
    STORE.commit()


# ------------------------------------------------------------------
# 3. b7 Reuters neutrality spot check.
# ------------------------------------------------------------------
@app.function(
    volumes={STORE_MOUNT: STORE},
    gpu="T4",
    timeout=30 * 60,
    retries=2,
)
def run_b7(use_external: bool = True) -> None:
    # The reviewer (R2-m2) asks whether the "neutral" Reuters label holds up
    # under an INDEPENDENT detector. Using the guide (trained to treat Reuters
    # as neutral) would be circular, so by default we override the detector to
    # the Chakraborty-trained external model. Pass use_external=False to also
    # get the (circular) guide number for comparison.
    _ensure_dirs()
    isot = Path(f"{STORE_MOUNT}/data/isot/True.csv")
    if not isot.exists():
        raise SystemExit(
            f"missing {isot}. Upload ISOT True.csv into the volume first."
        )
    env_extra = None
    tag = "guide"
    if use_external:
        ext = Path(f"{STORE_MOUNT}/models/external_clickbait_distilbert")
        if not (ext / "config.json").exists():
            raise SystemExit(
                f"missing {ext}. Run train_external_detector first, or pass "
                "use_external=False to fall back to the (circular) guide."
            )
        env_extra = {"CLICKBAIT_MODEL_DIR": str(ext)}
        tag = "external"
    _run([
        sys.executable,
        "/workspace/track_b/b7_reuters_neutrality/spot_check.py",
        "--gpu",
        "--isot-csv", str(isot),
        "--out-md", f"{STORE_MOUNT}/results/b7/reuters_neutrality_{tag}.md",
        "--out-csv", f"{STORE_MOUNT}/results/b7/reuters_probs_{tag}.csv",
    ], env_extra=env_extra)
    _verify().require_file(
        f"{STORE_MOUNT}/results/b7/reuters_neutrality_{tag}.md", min_bytes=50)
    STORE.commit()


# ------------------------------------------------------------------
# 4. b4 prompt-only baseline generation. Uses Llama-3-8B in FP16, so
#    needs a GPU with >=24 GB VRAM. L40S (48 GB) is the cheap sweet spot;
#    A100-40GB works too.
# ------------------------------------------------------------------
@app.function(
    volumes={STORE_MOUNT: STORE},
    secrets=[HF_SECRET],
    gpu=["A10G", "L40S", "A100-40GB"],
    timeout=6 * 60 * 60,
)
def run_b4(n_items: int = 300, seeds: str = "42,43,44") -> None:
    _ensure_dirs()
    src = Path(f"{STORE_MOUNT}/data/source_neutrals.csv")
    if not src.exists():
        raise SystemExit(
            f"missing {src}. Upload your held-out neutrals CSV (with a "
            "'title' column) into the volume first."
        )
    v = _verify()
    v.check_source_neutrals(src, min_rows=n_items)
    seed_list = [s.strip() for s in seeds.split(",")]
    _run([
        sys.executable,
        "/workspace/track_b/b4_prompt_only_baseline/run.py",
        "--test-csv", str(src),
        "--n-items", str(n_items),
        "--seeds", *seed_list,
        "--out-dir", f"{STORE_MOUNT}/results/b4",
    ], periodic_commit_s=180)
    v.check_rewrites(f"{STORE_MOUNT}/results/b4", expect_min_files=1)
    STORE.commit()


# ------------------------------------------------------------------
# 5. b2 FUDGE-guided generation (the big one).
# ------------------------------------------------------------------
@app.function(
    volumes={STORE_MOUNT: STORE},
    secrets=[HF_SECRET],
    gpu=["A10G", "L40S", "A100-40GB"],
    timeout=8 * 60 * 60,
)
def run_b2(n_items: int = 300, seeds: str = "42", top_k: int = 50,
           objective: str = "log", alpha_on: float = 0.0, beta_on: float = 0.0,
           sweep: str = "", item_start: int = 0, item_end: int = 0,
           shard_tag: str = "",
           src_name: str = "source_neutrals.csv", out_subdir: str = "b2") -> None:
    # objective="log" (default) is canonical log-domain FUDGE. The on-weight
    # scale differs from the paper's prob-domain heuristic, so run the tuning
    # sweep (run_b2_sweep) once and pass the winning alpha_on/beta_on here.
    # Single seed: greedy decoding is deterministic, replication is over items
    # via bootstrap; multiple seeds only matter with the --sample flag.
    # item_start/item_end/shard_tag are set by run_b2_parallel for fan-out.
    _ensure_dirs()
    v = _verify()
    src = Path(f"{STORE_MOUNT}/data/{src_name}")
    if not src.exists():
        raise SystemExit(f"missing {src}")
    v.check_source_neutrals(src, min_rows=(item_end or n_items))
    v.check_model_dir(f"{STORE_MOUNT}/models/bert_clickbait_prefix_finetuned",
                      expect_labels=2)
    v.check_model_dir(f"{STORE_MOUNT}/models/bert_tactics_prefix_finetuned",
                      expect_labels=10)
    seed_list = [s.strip() for s in seeds.split(",")]
    cmd = [
        sys.executable,
        "/workspace/track_b/b2_held_out_rewrite_eval/run_rewrites.py",
        "--test-csv", str(src),
        "--n-items", str(n_items),
        "--seeds", *seed_list,
        "--top-k", str(top_k),
        "--objective", objective,
        "--item-start", str(item_start),
        "--item-end", str(item_end),
        "--shard-tag", shard_tag,
        "--out-dir", f"{STORE_MOUNT}/results/{out_subdir}",
    ]
    if alpha_on > 0:
        cmd += ["--alpha-on", str(alpha_on)]
    if beta_on > 0:
        cmd += ["--beta-on", str(beta_on)]
    if sweep:
        cmd += ["--sweep", sweep]
    _run(cmd, periodic_commit_s=180)
    v.check_rewrites(f"{STORE_MOUNT}/results/{out_subdir}", expect_min_files=1)
    STORE.commit()


@app.function(volumes={STORE_MOUNT: STORE}, timeout=10 * 60 * 60)
def run_b2_parallel(n_items: int = 300, n_shards: int = 4, seeds: str = "42",
                    top_k: int = 50, objective: str = "log",
                    alpha_on: float = 0.0, beta_on: float = 0.0) -> None:
    """Fan out run_b2 across n_shards GPU containers over disjoint item ranges,
    each writing shard-tagged CSVs to results/b2. Wall-clock drops ~n_shards x
    (same total compute). item_ids stay global so score_rewrites reads every
    shard's rewrites_*.csv and the items align across methods."""
    _ensure_dirs()
    per = (n_items + n_shards - 1) // n_shards
    handles = []
    for k in range(n_shards):
        s = k * per
        e = min((k + 1) * per, n_items)
        if s >= e:
            break
        print(f"[modal] fan-out shard {k}: items [{s},{e})", flush=True)
        handles.append(run_b2.spawn(
            n_items=n_items, seeds=seeds, top_k=top_k, objective=objective,
            alpha_on=alpha_on, beta_on=beta_on,
            item_start=s, item_end=e, shard_tag=f"_sh{k}"))
    print(f"[modal] {len(handles)} shards running concurrently; waiting...",
          flush=True)
    for h in handles:
        h.get()  # re-raises any shard failure
    # The shards wrote+committed from their own containers; the parent must
    # reload the volume or it still sees the pre-spawn (empty) view and the
    # check spuriously fails with "found 0".
    STORE.reload()
    _verify().check_rewrites(f"{STORE_MOUNT}/results/b2", expect_min_files=len(handles))
    print(f"[modal] all {len(handles)} b2 shards complete.", flush=True)


@app.function(
    volumes={STORE_MOUNT: STORE},
    secrets=[HF_SECRET],
    gpu=["A10G", "L40S", "A100-40GB"],
    timeout=8 * 60 * 60,
)
def run_b2_sweep(n_items: int = 40, objective: str = "log",
                 sweep: str = "0,1,2,4", top_k: int = 50,
                 neutral_prompt: bool = False, out_subdir: str = "",
                 cells: str = "", fluency_top_p: float = 0.0,
                 item_start: int = 0, item_end: int = 0) -> None:
    """Re-tune the FUDGE on-weight for the chosen objective on a small subset.

    Default is a 4x4 grid on 40 items x 3 tactic configs = 1920 rewrites
    (~1 h on A10G, ~$1), which is enough to locate the best on-weight. Widen
    (larger n_items or a finer sweep) only if the trade-off surface looks flat.

    neutral_prompt=True uses the tactic-AGNOSTIC prompt so the FUDGE guides are
    the sole tactic signal (isolates the decoding contribution); it defaults to
    a separate out dir (results/b2_sweep_neutral) so it never collides with the
    tactic-prompt sweep. Override with out_subdir.

    Writes rewrites for the full alpha x beta cross product; score them, pick
    the (alpha,beta) with the best fidelity/engagement trade, then run run_b2
    with that alpha_on/beta_on for the reported ablation.
    """
    _ensure_dirs()
    v = _verify()
    src = Path(f"{STORE_MOUNT}/data/source_neutrals.csv")
    v.check_source_neutrals(src, min_rows=n_items)
    v.check_model_dir(f"{STORE_MOUNT}/models/bert_clickbait_prefix_finetuned", expect_labels=2)
    v.check_model_dir(f"{STORE_MOUNT}/models/bert_tactics_prefix_finetuned", expect_labels=10)
    sub = out_subdir or ("b2_sweep_neutral" if neutral_prompt else "b2_sweep")
    out_dir = f"{STORE_MOUNT}/results/{sub}"
    cmd = [
        sys.executable,
        "/workspace/track_b/b2_held_out_rewrite_eval/run_rewrites.py",
        "--test-csv", str(src),
        "--n-items", str(n_items),
        "--seeds", "42",
        "--top-k", str(top_k),
        "--objective", objective,
        "--item-start", str(item_start),
        "--item-end", str(item_end),
        "--out-dir", out_dir,
    ]
    # explicit cells override the sweep cross-product
    if cells:
        cmd += ["--cells", cells]
    else:
        cmd += ["--sweep", sweep]
    if neutral_prompt:
        cmd.append("--neutral-prompt")
    if fluency_top_p and fluency_top_p > 0:
        cmd += ["--fluency-top-p", str(fluency_top_p)]
    _run(cmd, periodic_commit_s=180)
    v.check_rewrites(out_dir, expect_min_files=1)
    STORE.commit()


# ------------------------------------------------------------------
# 6. Score any rewrite dir (b2 or b4) with the full metric suite.
#    Two entrypoints: `run_score` (default, no judge, no secret required)
#    and `run_score_with_judge` (requires the llm-judge-keys secret to
#    exist). Splitting is necessary because Modal secrets referenced in
#    the function decorator must exist at invocation time.
# ------------------------------------------------------------------
def _score_cmd(in_dir: str, with_llm_judge: bool,
               judge_max_per_cell: int = 0, paired: bool = False) -> list[str]:
    cmd = [
        sys.executable,
        "/workspace/track_b/b2_held_out_rewrite_eval/score_rewrites.py",
        "--gpu",
        "--in-dir", in_dir,
        "--out-json", f"{in_dir}/summary.json",
        "--out-md", f"{in_dir}/summary.md",
        "--per-item-csv", f"{in_dir}/per_item_scores.csv",
    ]
    # Pass the INDEPENDENT clickbait detector (trained only on human-authored
    # Chakraborty data) if it exists, so the reported clickbait metric is not
    # the circular guide-model score. Absent -> score_rewrites warns loudly.
    ext = Path(f"{STORE_MOUNT}/models/external_clickbait_distilbert")
    if (ext / "config.json").exists():
        cmd += ["--external-clickbait-dir", str(ext)]
    if with_llm_judge:
        cmd.append("--with-llm-judge")
        if judge_max_per_cell and judge_max_per_cell > 0:
            cmd += ["--judge-max-per-cell", str(judge_max_per_cell)]
    if paired:
        cmd.append("--paired-judge")
        if judge_max_per_cell and judge_max_per_cell > 0 and not with_llm_judge:
            cmd += ["--judge-max-per-cell", str(judge_max_per_cell)]
    return cmd


@app.function(
    volumes={STORE_MOUNT: STORE},
    secrets=[HF_SECRET],
    gpu="L4",
    timeout=6 * 60 * 60,
)
def run_score(in_subdir: str = "b2") -> None:
    """Score a rewrite dir with automatic metrics only (no LLM judge)."""
    _ensure_dirs()
    in_dir = f"{STORE_MOUNT}/results/{in_subdir}"
    _verify().check_rewrites(in_dir, expect_min_files=1)
    _run(_score_cmd(in_dir, with_llm_judge=False))
    _verify().check_per_item_scores(f"{in_dir}/per_item_scores.csv", min_rows=1)
    STORE.commit()


@app.function(
    volumes={STORE_MOUNT: STORE},
    secrets=[HF_SECRET, JUDGE_SECRET],
    gpu="T4",
    timeout=60 * 60,
)
def run_control_validation(n: int = 40) -> None:
    """Negative-control validation (c) + NLI model A/B (b) in one pass.

    Generates known-faithful (paraphrase, rhetorical-question) and
    known-unfaithful (fact-injected) rephrasings, then: runs the refined
    new-fact judge on them (expect ~0 for faithful, ~1 for unfaithful) and
    scores both roberta-mnli and DeBERTa-v3 NLI to see which separates
    faithful from unfaithful better (esp. on the question form)."""
    _ensure_dirs()
    src = f"{STORE_MOUNT}/data/source_neutrals.csv"
    _run([
        sys.executable,
        "/workspace/track_b/b2_held_out_rewrite_eval/control_validation.py",
        "--src-csv", src,
        "--out-dir", f"{STORE_MOUNT}/results/control",
        "--n", str(n),
        "--gpu",
    ])
    _verify().check_json_report(f"{STORE_MOUNT}/results/control/control_validation.json")
    STORE.commit()


@app.function(
    volumes={STORE_MOUNT: STORE},
    secrets=[JUDGE_SECRET],
    timeout=60 * 60,
)
def run_rejudge_factuality(in_subdir: str = "b2", max_per_cell: int = 150) -> None:
    """Root-cause the hallucination metric: re-judge with a rubric that
    separates a genuine NEW VERIFIABLE FACT from a rhetorical question/framing
    (the target tactics themselves). CPU-only (pure API calls)."""
    _ensure_dirs()
    in_dir = f"{STORE_MOUNT}/results/{in_subdir}"
    _run([
        sys.executable,
        "/workspace/track_b/b2_held_out_rewrite_eval/rejudge_factuality.py",
        "--in-dir", in_dir,
        "--max-per-cell", str(max_per_cell),
    ])
    _verify().check_json_report(f"{in_dir}/factuality_refined.json")
    STORE.commit()


@app.function(
    volumes={STORE_MOUNT: STORE},
    secrets=[HF_SECRET, JUDGE_SECRET],
    gpu="L4",
    timeout=6 * 60 * 60,
)
def run_score_with_judge(in_subdir: str = "b2",
                         judge_max_per_cell: int = 0,
                         paired: bool = False) -> None:
    """Score a rewrite dir WITH the LLM-as-judge attribute confirmation.

    Requires `modal secret create llm-judge-keys ANTHROPIC_API_KEY=...` (or
    OPENAI_API_KEY, or both) beforehand.

    judge_max_per_cell > 0 judges only that many rows per (alpha,beta) cell --
    the cheap independent signal for on-weight SELECTION. 0 judges all rows
    (the reported run).
    """
    _ensure_dirs()
    in_dir = f"{STORE_MOUNT}/results/{in_subdir}"
    _verify().check_rewrites(in_dir, expect_min_files=1)
    _run(_score_cmd(in_dir, with_llm_judge=True,
                    judge_max_per_cell=judge_max_per_cell, paired=paired))
    _verify().check_per_item_scores(f"{in_dir}/per_item_scores.csv", min_rows=1)
    STORE.commit()


# ------------------------------------------------------------------
# 6b. Train the INDEPENDENT clickbait detector on Chakraborty (human-authored),
#     used by run_score to break the circular-evaluation problem the reviewers
#     flagged (the FUDGE beta-guide must not also be the clickbait evaluator).
# ------------------------------------------------------------------
@app.function(
    volumes={STORE_MOUNT: STORE},
    gpu="T4",
    timeout=60 * 60,
    retries=2,
)
def train_external_detector(force: bool = False) -> None:
    _ensure_dirs()
    out = Path(f"{STORE_MOUNT}/models/external_clickbait_distilbert")
    if (out / "config.json").exists() and not force:
        print(f"[modal] {out} exists; skip. Pass force=True to retrain.")
        return
    v = _verify()
    v.require_file(f"{STORE_MOUNT}/data/chakraborty16/clickbait_data", min_bytes=10_000)
    v.require_file(f"{STORE_MOUNT}/data/chakraborty16/non_clickbait_data", min_bytes=10_000)
    _run([
        sys.executable,
        "/workspace/track_b/b1_external_benchmarks/train_external_clickbait.py",
        "--data-dir", f"{STORE_MOUNT}/data/chakraborty16",
        "--out-dir", str(out),
    ])
    v.check_model_dir(out, expect_labels=2)
    STORE.commit()


# ------------------------------------------------------------------
# 6c. DExperts and GeDi decoding-time baselines (reviewer R2-M4).
#     Each trains a small Llama-3.2-1B controller (shares the base tokenizer),
#     generates rewrites in the b2 schema, then run_score scores them and
#     compare_methods contrasts all methods.
# ------------------------------------------------------------------
B6_DIR = "/workspace/track_b/b6_dexperts_gedi"
SMALL_LLM = "meta-llama/Llama-3.2-1B"  # gated; the HF account must accept it


@app.function(volumes={STORE_MOUNT: STORE}, secrets=[HF_SECRET],
              gpu=["L40S", "A100-40GB", "A10G"], timeout=3 * 60 * 60, retries=1)
def train_dexperts(force: bool = False) -> None:
    _ensure_dirs()
    v = _verify()
    synth = Path(f"{STORE_MOUNT}/data/synthetic_clickbait.csv")
    src = Path(f"{STORE_MOUNT}/data/source_neutrals.csv")
    v.check_synthetic(synth, min_rows=100)
    v.check_source_neutrals(src, min_rows=100)
    exp = Path(f"{STORE_MOUNT}/models/dexperts_expert")
    anti = Path(f"{STORE_MOUNT}/models/dexperts_antiexpert")
    if not force and (exp / "config.json").exists() and (anti / "config.json").exists():
        print("[modal] dexperts models exist; skip (force=True to retrain).")
        return
    _run([sys.executable, f"{B6_DIR}/train_dexperts.py",
          "--synthetic-csv", str(synth), "--source-csv", str(src),
          "--expert-out", str(exp), "--antiexpert-out", str(anti),
          "--model-name", SMALL_LLM])
    v.check_model_dir(exp)
    v.check_model_dir(anti)
    STORE.commit()


@app.function(volumes={STORE_MOUNT: STORE}, secrets=[HF_SECRET],
              gpu=["L40S", "A100-40GB", "A10G"], timeout=8 * 60 * 60)
def run_dexperts(n_items: int = 300, alphas: str = "0.5,1,2,4",
                 top_k: int = 50, item_start: int = 0, item_end: int = 0,
                 shard_tag: str = "") -> None:
    _ensure_dirs()
    v = _verify()
    src = Path(f"{STORE_MOUNT}/data/source_neutrals.csv")
    v.check_source_neutrals(src, min_rows=(item_end or n_items))
    exp = f"{STORE_MOUNT}/models/dexperts_expert"
    anti = f"{STORE_MOUNT}/models/dexperts_antiexpert"
    v.check_model_dir(exp)
    v.check_model_dir(anti)
    _run([sys.executable, f"{B6_DIR}/run_dexperts.py",
          "--test-csv", str(src), "--n-items", str(n_items),
          "--expert-dir", exp, "--antiexpert-dir", anti,
          "--alphas", alphas, "--top-k", str(top_k),
          "--item-start", str(item_start), "--item-end", str(item_end),
          "--shard-tag", shard_tag,
          "--out-dir", f"{STORE_MOUNT}/results/b6_dexperts"],
         periodic_commit_s=180)
    v.check_rewrites(f"{STORE_MOUNT}/results/b6_dexperts", expect_min_files=1)
    STORE.commit()


@app.function(volumes={STORE_MOUNT: STORE}, timeout=10 * 60 * 60)
def run_dexperts_parallel(n_items: int = 300, n_shards: int = 4,
                          alphas: str = "0.5,1,2,4", top_k: int = 50) -> None:
    """Fan out run_dexperts across n_shards containers over disjoint item ranges."""
    _ensure_dirs()
    per = (n_items + n_shards - 1) // n_shards
    handles = []
    for k in range(n_shards):
        s, e = k * per, min((k + 1) * per, n_items)
        if s >= e:
            break
        print(f"[modal] dexperts shard {k}: items [{s},{e})", flush=True)
        handles.append(run_dexperts.spawn(
            n_items=n_items, alphas=alphas, top_k=top_k,
            item_start=s, item_end=e, shard_tag=f"_sh{k}"))
    for h in handles:
        h.get()
    STORE.reload()  # see the shards' committed writes (avoid stale "found 0")
    _verify().check_rewrites(f"{STORE_MOUNT}/results/b6_dexperts",
                             expect_min_files=len(handles))


@app.function(volumes={STORE_MOUNT: STORE}, secrets=[HF_SECRET],
              gpu=["L40S", "A100-40GB", "A10G"], timeout=3 * 60 * 60, retries=1)
def train_gedi(force: bool = False) -> None:
    _ensure_dirs()
    v = _verify()
    synth = Path(f"{STORE_MOUNT}/data/synthetic_clickbait.csv")
    src = Path(f"{STORE_MOUNT}/data/source_neutrals.csv")
    v.check_synthetic(synth, min_rows=100)
    v.check_source_neutrals(src, min_rows=100)
    out = Path(f"{STORE_MOUNT}/models/gedi_conditional")
    if not force and (out / "config.json").exists():
        print("[modal] gedi model exists; skip (force=True to retrain).")
        return
    _run([sys.executable, f"{B6_DIR}/train_gedi.py",
          "--synthetic-csv", str(synth), "--source-csv", str(src),
          "--out-dir", str(out), "--model-name", SMALL_LLM])
    v.check_model_dir(out)
    STORE.commit()


@app.function(volumes={STORE_MOUNT: STORE}, secrets=[HF_SECRET],
              gpu=["A10G", "L40S", "A100-40GB"], timeout=8 * 60 * 60)
def run_gedi(n_items: int = 300, omegas: str = "5,10,20,30",
             top_k: int = 50, item_start: int = 0, item_end: int = 0,
             shard_tag: str = "") -> None:
    _ensure_dirs()
    v = _verify()
    src = Path(f"{STORE_MOUNT}/data/source_neutrals.csv")
    v.check_source_neutrals(src, min_rows=(item_end or n_items))
    gedi = f"{STORE_MOUNT}/models/gedi_conditional"
    v.check_model_dir(gedi)
    _run([sys.executable, f"{B6_DIR}/run_gedi.py",
          "--test-csv", str(src), "--n-items", str(n_items),
          "--gedi-dir", gedi, "--omegas", omegas, "--top-k", str(top_k),
          "--item-start", str(item_start), "--item-end", str(item_end),
          "--shard-tag", shard_tag,
          "--out-dir", f"{STORE_MOUNT}/results/b6_gedi"],
         periodic_commit_s=180)
    v.check_rewrites(f"{STORE_MOUNT}/results/b6_gedi", expect_min_files=1)
    STORE.commit()


@app.function(volumes={STORE_MOUNT: STORE}, timeout=10 * 60 * 60)
def run_gedi_parallel(n_items: int = 300, n_shards: int = 4,
                      omegas: str = "5,10,20,30", top_k: int = 50) -> None:
    """Fan out run_gedi across n_shards containers over disjoint item ranges."""
    _ensure_dirs()
    per = (n_items + n_shards - 1) // n_shards
    handles = []
    for k in range(n_shards):
        s, e = k * per, min((k + 1) * per, n_items)
        if s >= e:
            break
        print(f"[modal] gedi shard {k}: items [{s},{e})", flush=True)
        handles.append(run_gedi.spawn(
            n_items=n_items, omegas=omegas, top_k=top_k,
            item_start=s, item_end=e, shard_tag=f"_sh{k}"))
    for h in handles:
        h.get()
    STORE.reload()  # see the shards' committed writes (avoid stale "found 0")
    _verify().check_rewrites(f"{STORE_MOUNT}/results/b6_gedi",
                             expect_min_files=len(handles))


# ------------------------------------------------------------------
# 6d. Cross-method comparison (R2-M4) and human-label analysis (Track C).
# ------------------------------------------------------------------
@app.function(volumes={STORE_MOUNT: STORE}, timeout=15 * 60)
def compare_methods(methods: str = "b2,b4,b6_dexperts,b6_gedi",
                    labels: str = "fudge,prompt_only,dexperts,gedi") -> None:
    """Contrast every method's scored per_item results. Methods whose
    per_item_scores.csv is missing are dropped with a warning."""
    _ensure_dirs()
    ms = [m.strip() for m in methods.split(",")]
    ls = [x.strip() for x in labels.split(",")]
    inputs, keep = [], []
    for m, lab in zip(ms, ls):
        p = Path(f"{STORE_MOUNT}/results/{m}/per_item_scores.csv")
        if p.exists():
            inputs.append(str(p))
            keep.append(lab)
        else:
            print(f"[modal] WARN: {p} missing; method '{lab}' excluded")
    if len(inputs) < 2:
        raise SystemExit("need >=2 scored methods to compare")
    _run([sys.executable,
          "/workspace/track_b/b8_method_comparison/compare_methods.py",
          "--inputs", *inputs, "--labels", *keep,
          "--out-dir", f"{STORE_MOUNT}/results/method_comparison"])
    STORE.commit()


@app.function(volumes={STORE_MOUNT: STORE}, secrets=[JUDGE_SECRET],
              timeout=60 * 60)
def run_gpt_raters(study: str = "c2_rewrite_quality",
                   temperature: float = 0.9) -> None:
    """Fill a study's rater packages with GPT-simulated ratings (PROXY for the
    human study) so the Track C analysis + paper can be completed now. Writes
    rater_source=gpt_sim and a PROXY_NOTICE.md. NOT human data."""
    _ensure_dirs()
    study_dir = f"{STORE_MOUNT}/results/human_labeling/{study}"
    if not Path(f"{study_dir}/oracle.csv").exists():
        raise SystemExit(f"missing {study_dir}; run prepare_human_labeling first")
    _run([sys.executable,
          "/workspace/track_b/human_labeling_prep/gpt_rater_fill.py",
          "--study-dir", study_dir, "--temperature", str(temperature)])
    STORE.commit()


@app.function(volumes={STORE_MOUNT: STORE}, timeout=15 * 60)
def run_human_analysis(study: str = "c1_rubric_validation",
                       simulate: bool = False) -> None:
    """Analyze a Track C study (kappa/ICC/Wilcoxon/diagnostics). With
    simulate=True it fills rater files from the oracle first, so the whole
    analysis pipeline can be validated before any human rates."""
    _ensure_dirs()
    study_dir = f"{STORE_MOUNT}/results/human_labeling/{study}"
    if not Path(f"{study_dir}/oracle.csv").exists():
        raise SystemExit(f"missing {study_dir}/oracle.csv; run prepare_human_labeling first")
    cmd = [sys.executable,
           "/workspace/track_b/human_labeling_prep/analyze.py",
           "--study-dir", study_dir]
    if simulate:
        cmd.append("--simulate")
    _run(cmd)
    STORE.commit()


# ------------------------------------------------------------------
# 7. Pull results down. `modal volume get` is the CLI-friendly way; this
#    helper just prints the exact command the user should run locally.
# ------------------------------------------------------------------
@app.function(volumes={STORE_MOUNT: STORE})
def pull_results() -> None:
    print("Run this from your local machine to download everything:\n")
    print("  modal volume get newsrewrite-track-b-store results ./cycle1_results")
    print("\nCurrent contents of results/ on the volume:")
    for p in sorted(Path(f"{STORE_MOUNT}/results").rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"  {size_kb:8.1f} KB  {p.relative_to(STORE_MOUNT)}")


# ------------------------------------------------------------------
# 8. Reproduce the two BERT guide models from scratch.
#    regenerate_synthetic: CPU, GPT-4o-mini, ~1 h, ~$5.
#    train_guides:         T4, ~2-3 h, ~$1.
#    Both are idempotent so re-runs after a partial failure are cheap.
# ------------------------------------------------------------------
GUIDE_DIR = "/workspace/track_b/modal_run/train_guides"


@app.function(
    volumes={STORE_MOUNT: STORE},
    timeout=10 * 60,
)
def extract_source_neutrals(max_rows: int = 12600, seed: int = 42,
                            force: bool = False) -> None:
    """Derive data/source_neutrals.csv from data/isot/True.csv.

    Called automatically by regenerate_synthetic if source_neutrals.csv
    is missing. Safe to call directly too: idempotent unless force=True.
    """
    _ensure_dirs()
    isot = Path(f"{STORE_MOUNT}/data/isot/True.csv")
    if not isot.exists():
        raise SystemExit(
            f"missing {isot}. Run `modal run app.py::fetch_external_datasets` "
            "first, or upload True.csv via upload_artifacts.py."
        )
    out = Path(f"{STORE_MOUNT}/data/source_neutrals.csv")
    if out.exists() and not force:
        print(f"[modal] {out} exists; skip. Pass force=True to rebuild.")
        return
    _run([
        sys.executable,
        f"{GUIDE_DIR}/extract_source_neutrals.py",
        "--isot-csv", str(isot),
        "--out-csv", str(out),
        "--max-rows", str(max_rows),
        "--seed", str(seed),
    ])
    _verify().check_source_neutrals(out, min_rows=max(100, max_rows // 4))
    STORE.commit()


@app.function(
    volumes={STORE_MOUNT: STORE},
    secrets=[OPENAI_SECRET],
    timeout=2 * 60 * 60,
    cpu=2.0,
)
def regenerate_synthetic(
    max_rows: int = 12600,
    batch_size: int = 15,
    seed: int = 42,
    workers: int = 16,
    force: bool = False,
) -> None:
    """Ask GPT-4o-mini for a synthetic clickbait dataset. Resumes by default.

    Auto-derives source_neutrals.csv from ISOT True.csv if the former is
    missing, so a fresh volume only needs `fetch_external_datasets` first.
    """
    _ensure_dirs()
    src = Path(f"{STORE_MOUNT}/data/source_neutrals.csv")
    if not src.exists():
        print(f"[modal] {src} missing; extracting from ISOT True.csv")
        isot = Path(f"{STORE_MOUNT}/data/isot/True.csv")
        if not isot.exists():
            raise SystemExit(
                f"missing {src} AND {isot}. Run "
                "`modal run app.py::fetch_external_datasets` first, or "
                "upload one of the two files via upload_artifacts.py."
            )
        _run([
            sys.executable,
            f"{GUIDE_DIR}/extract_source_neutrals.py",
            "--isot-csv", str(isot),
            "--out-csv", str(src),
            "--max-rows", str(max_rows),
            "--seed", str(seed),
        ])
    out = Path(f"{STORE_MOUNT}/data/synthetic_clickbait.csv")
    if out.exists() and not force:
        try:
            import pandas as pd
            n = len(pd.read_csv(out))
        except Exception:
            n = -1
        print(f"[modal] {out} exists ({n} rows). Resuming; pass force=True to restart.")
    cmd = [
        sys.executable,
        f"{GUIDE_DIR}/generate_synthetic.py",
        "--source-csv", str(src),
        "--out-csv", str(out),
        "--batch-size", str(batch_size),
        "--max-rows", str(max_rows),
        "--seed", str(seed),
        "--workers", str(workers),
    ]
    if force:
        cmd.append("--no-resume")
    # periodic commit so a container kill loses at most ~2 min of the parallel
    # generation instead of the whole run (resume-by-content picks up the rest).
    _run(cmd, periodic_commit_s=120)
    _verify().check_synthetic(out, source_path=src,
                              min_rows=max(100, max_rows // 2))
    STORE.commit()


@app.function(
    volumes={STORE_MOUNT: STORE},
    gpu="T4",
    timeout=4 * 60 * 60,
)
def train_guides(force: bool = False) -> None:
    """Build prefix datasets then train both BERT guides. Idempotent."""
    _ensure_dirs()
    synth = Path(f"{STORE_MOUNT}/data/synthetic_clickbait.csv")
    if not synth.exists():
        raise SystemExit(
            f"missing {synth}. Run `modal run app.py::regenerate_synthetic` "
            "first."
        )
    # Gate on synthetic-data quality before spending GPU time on training.
    _verify().check_synthetic(synth, min_rows=100)

    prefix_dir = Path(f"{STORE_MOUNT}/data/prefix_splits")
    prefix_dir.mkdir(parents=True, exist_ok=True)
    tv_bin = prefix_dir / "trainval_binary.csv"
    te_bin = prefix_dir / "test_binary.csv"
    tv_tac = prefix_dir / "trainval_tactics.csv"
    te_tac = prefix_dir / "test_tactics.csv"

    all_prefix_files = [tv_bin, te_bin, tv_tac, te_tac]
    if force or not all(p.exists() for p in all_prefix_files):
        _run([
            sys.executable, f"{GUIDE_DIR}/build_prefix_datasets.py",
            "--headlines-csv", str(synth),
            "--out-trainval-binary", str(tv_bin),
            "--out-test-binary", str(te_bin),
            "--out-trainval-tactics", str(tv_tac),
            "--out-test-tactics", str(te_tac),
        ])
    else:
        print("[modal] prefix split CSVs already present, skipping build step")

    # Gate: verify the prefix splits have no full-headline leakage and sane
    # labels before spending GPU minutes on training.
    v = _verify()
    v.check_prefix_split(tv_bin, te_bin, kind="binary")
    v.check_prefix_split(tv_tac, te_tac, kind="tactics")

    bin_out = Path(f"{STORE_MOUNT}/models/bert_clickbait_prefix_finetuned")
    if force or not (bin_out / "config.json").exists():
        _run([
            sys.executable, f"{GUIDE_DIR}/train_clickbait.py",
            "--trainval-csv", str(tv_bin),
            "--test-csv", str(te_bin),
            "--out-dir", str(bin_out),
        ])
    else:
        print(f"[modal] {bin_out}/config.json exists, skipping binary training")

    tac_out = Path(f"{STORE_MOUNT}/models/bert_tactics_prefix_finetuned")
    if force or not (tac_out / "config.json").exists():
        _run([
            sys.executable, f"{GUIDE_DIR}/train_tactics.py",
            "--trainval-csv", str(tv_tac),
            "--test-csv", str(te_tac),
            "--out-dir", str(tac_out),
        ])
    else:
        print(f"[modal] {tac_out}/config.json exists, skipping tactics training")

    # Gate: both trained models must exist with the expected label counts.
    v.check_model_dir(bin_out, expect_labels=2)
    v.check_model_dir(tac_out, expect_labels=10)
    STORE.commit()


@app.function(volumes={STORE_MOUNT: STORE}, gpu="T4",
              timeout=4 * 60 * 60, retries=1)
def train_one_guide(which: str, force: bool = False) -> None:
    """Train a single guide (which in {clickbait, tactics}) from prebuilt
    prefix splits. Used by train_guides_parallel so the two independent
    trainings run on two containers concurrently."""
    _ensure_dirs()
    v = _verify()
    pd_dir = f"{STORE_MOUNT}/data/prefix_splits"
    if which == "clickbait":
        v.check_prefix_split(f"{pd_dir}/trainval_binary.csv",
                             f"{pd_dir}/test_binary.csv", kind="binary")
        out = f"{STORE_MOUNT}/models/bert_clickbait_prefix_finetuned"
        if not force and Path(f"{out}/config.json").exists():
            print(f"[modal] {out} exists; skip.")
            return
        _run([sys.executable, f"{GUIDE_DIR}/train_clickbait.py",
              "--trainval-csv", f"{pd_dir}/trainval_binary.csv",
              "--test-csv", f"{pd_dir}/test_binary.csv", "--out-dir", out])
        v.check_model_dir(out, expect_labels=2)
    else:
        v.check_prefix_split(f"{pd_dir}/trainval_tactics.csv",
                             f"{pd_dir}/test_tactics.csv", kind="tactics")
        out = f"{STORE_MOUNT}/models/bert_tactics_prefix_finetuned"
        if not force and Path(f"{out}/config.json").exists():
            print(f"[modal] {out} exists; skip.")
            return
        _run([sys.executable, f"{GUIDE_DIR}/train_tactics.py",
              "--trainval-csv", f"{pd_dir}/trainval_tactics.csv",
              "--test-csv", f"{pd_dir}/test_tactics.csv", "--out-dir", out])
        v.check_model_dir(out, expect_labels=10)
    STORE.commit()


@app.function(volumes={STORE_MOUNT: STORE}, timeout=6 * 60 * 60)
def train_guides_parallel(force: bool = False) -> None:
    """Build the prefix splits once (CPU), then train the two BERT guides on
    two concurrent containers. Halves the training wall-clock vs train_guides."""
    _ensure_dirs()
    v = _verify()
    synth = Path(f"{STORE_MOUNT}/data/synthetic_clickbait.csv")
    v.check_synthetic(synth, min_rows=100)
    pd_dir = Path(f"{STORE_MOUNT}/data/prefix_splits")
    pd_dir.mkdir(parents=True, exist_ok=True)
    tv_bin, te_bin = pd_dir / "trainval_binary.csv", pd_dir / "test_binary.csv"
    tv_tac, te_tac = pd_dir / "trainval_tactics.csv", pd_dir / "test_tactics.csv"
    if force or not all(p.exists() for p in (tv_bin, te_bin, tv_tac, te_tac)):
        _run([sys.executable, f"{GUIDE_DIR}/build_prefix_datasets.py",
              "--headlines-csv", str(synth),
              "--out-trainval-binary", str(tv_bin), "--out-test-binary", str(te_bin),
              "--out-trainval-tactics", str(tv_tac), "--out-test-tactics", str(te_tac)])
    v.check_prefix_split(tv_bin, te_bin, kind="binary")
    v.check_prefix_split(tv_tac, te_tac, kind="tactics")
    STORE.commit()  # make the prefix CSVs visible to the spawned containers
    h1 = train_one_guide.spawn("clickbait", force=force)
    h2 = train_one_guide.spawn("tactics", force=force)
    h1.get()
    h2.get()
    v.check_model_dir(f"{STORE_MOUNT}/models/bert_clickbait_prefix_finetuned", expect_labels=2)
    v.check_model_dir(f"{STORE_MOUNT}/models/bert_tactics_prefix_finetuned", expect_labels=10)
    print("[modal] both guides trained (parallel).", flush=True)


@app.function(volumes={STORE_MOUNT: STORE}, timeout=60 * 60)
def run_validation_parallel() -> None:
    """Run b1_clickbait, b1_tactics and b7 concurrently (they are independent)."""
    _ensure_dirs()
    handles = [run_b1_clickbait.spawn(), run_b1_tactics.spawn(), run_b7.spawn()]
    for h in handles:
        h.get()
    print("[modal] b1 (clickbait + tactics) + b7 complete.", flush=True)


@app.function(volumes={STORE_MOUNT: STORE}, timeout=5 * 60)
def list_artifacts() -> None:
    """Print a size-annotated tree of models/ and data/ on the volume."""
    for root in ("models", "data"):
        base = Path(f"{STORE_MOUNT}/{root}")
        print(f"\n[{root}/]")
        if not base.exists():
            print("  (missing)")
            continue
        for p in sorted(base.rglob("*")):
            if p.is_dir():
                print(f"  {p.relative_to(STORE_MOUNT)}/")
            else:
                mb = p.stat().st_size / (1024 * 1024)
                print(f"  {mb:8.2f} MB  {p.relative_to(STORE_MOUNT)}")


@app.function(volumes={STORE_MOUNT: STORE}, timeout=5 * 60)
def show_logs(name: str = "", tail: int = 60) -> None:
    """Inspect the durable per-step logs saved on the volume.

    No `name`: list every step log with size + mtime (newest last).
    With `name` (a substring, e.g. 'run_rewrites' or 'train_clickbait'):
    print the last `tail` lines of the most recent matching log. This is the
    post-hoc failure-inspection path: even a crashed step left its full log
    here (committed by _run), so you can read exactly where it died.
    """
    logs_dir = Path(f"{STORE_MOUNT}/results/logs")
    if not logs_dir.exists():
        print("(no logs yet)")
        return
    files = sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not files:
        print("(no logs yet)")
        return
    if not name:
        print(f"[logs] {len(files)} step logs on volume (oldest first):")
        for p in files:
            kb = p.stat().st_size / 1024
            print(f"  {kb:8.1f} KB  {p.name}")
        print("\nRun show_logs --name <substring> to read the latest match.")
        return
    matches = [p for p in files if name in p.name]
    if not matches:
        print(f"[logs] no log matching '{name}'. Available:")
        for p in files[-15:]:
            print(f"  {p.name}")
        return
    target = matches[-1]
    print(f"[logs] {target.name} (last {tail} lines):\n")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    for ln in lines[-tail:]:
        print(ln)


# ------------------------------------------------------------------
# 9. Prepare Track C human-labeling packages (C1 and C2).
#    Writes /workspace/store/results/human_labeling/{c1_*,c2_*}/
#    with oracle.csv (analyst side), rater_{k}.csv (rater side), and
#    codebook.md. Requires:
#        - data/synthetic_clickbait.csv (for C1; from regenerate_synthetic)
#        - results/b2/per_item_scores.csv (for C2; from run_score --in-subdir b2)
#    Either input may be omitted; that study is then skipped.
# ------------------------------------------------------------------
@app.function(
    volumes={STORE_MOUNT: STORE},
    timeout=15 * 60,
)
def prepare_human_labeling(
    c1_n_items: int = 150,
    c2_n_items: int = 100,
    n_raters: int = 3,
    seed: int = 20260723,
    c2_methods: str = "b2,b4",
) -> None:
    """Build the Track C rater packages.

    C1 (tactic-label validation) reads data/synthetic_clickbait.csv.
    C2 (rewrite quality) reads results/<m>/per_item_scores.csv for every
    method m in `c2_methods` (comma-separated). Each method whose
    per_item_scores.csv is missing is skipped with a warning; the study
    runs on whatever methods are present. Either study is skipped entirely
    if none of its inputs exist.
    """
    _ensure_dirs()
    synth = Path(f"{STORE_MOUNT}/data/synthetic_clickbait.csv")
    out_dir = Path(f"{STORE_MOUNT}/results/human_labeling")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve C2 method inputs.
    methods = [m.strip() for m in c2_methods.split(",") if m.strip()]
    per_item_paths: list[str] = []
    method_labels: list[str] = []
    for m in methods:
        p = Path(f"{STORE_MOUNT}/results/{m}/per_item_scores.csv")
        if p.exists():
            per_item_paths.append(str(p))
            method_labels.append(m)
        else:
            print(f"[modal] WARN: {p} missing; method '{m}' excluded from C2")

    have_c1 = synth.exists()
    have_c2 = len(per_item_paths) > 0
    if not have_c1 and not have_c2:
        raise SystemExit(
            "No inputs found. Run regenerate_synthetic (for C1) and/or "
            "run_score --in-subdir <method> (for C2) first."
        )
    if not have_c1:
        print(f"[modal] WARN: {synth} missing; C1 (tactic-label validation) skipped")

    cmd = [
        sys.executable,
        "/workspace/track_b/human_labeling_prep/prepare.py",
        "--out-dir", str(out_dir),
        "--c1-n-items", str(c1_n_items),
        "--c2-n-items", str(c2_n_items),
        "--n-raters", str(n_raters),
        "--seed", str(seed),
    ]
    if have_c1:
        cmd += ["--synthetic-csv", str(synth)]
    if have_c2:
        cmd += ["--per-item-csv", *per_item_paths]
        cmd += ["--method-labels", *method_labels]

    _run(cmd)
    # Gate: verify the rater packages are complete and do NOT leak ground truth.
    _verify().check_human_labeling(out_dir)
    STORE.commit()


# ------------------------------------------------------------------
# Local sanity: `modal run app.py::check_env` prints what the image sees.
# ------------------------------------------------------------------
@app.function(volumes={STORE_MOUNT: STORE}, timeout=5 * 60)
def check_env() -> None:
    import torch
    print("[modal] torch", torch.__version__, "cuda?", torch.cuda.is_available())
    print("[modal] volume mount:")
    for p in sorted(Path(STORE_MOUNT).rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(STORE_MOUNT)}  ({p.stat().st_size} B)")
        elif p.is_dir():
            print(f"  {p.relative_to(STORE_MOUNT)}/")
