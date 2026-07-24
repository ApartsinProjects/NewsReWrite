"""Generate synthetic clickbait headlines from real neutral headlines.

Port of NewsReWrite/Dataset_generation/generate_clickbait.py, cleaned up so
it can run inside the Modal container without .env files, no hard-coded
paths, and full argparse. All configuration (source CSV, output CSV, batch
size, cap, resume flag) is CLI-driven; API endpoint and model come from
env vars.

Output schema: CSV with columns
    original         : the source neutral headline
    clickbait        : GPT-generated clickbait rewrite
    methods_vector   : JSON string, list of 10 ints (0/1), one flag per
                       tactic in the fixed order below
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Reach the track_b package so the crash-safe JSONL sidecar helper is importable
# (mirrors the common.paths access pattern used by the other track_b scripts).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.artifacts import JsonlWriter


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


PROMPT_TEMPLATE = """
You are generating a controlled synthetic dataset for academic research on explainable clickbait detection.

Your task:
Given a factual, non-clickbait news headline, generate EXACTLY ONE clickbait headline and a binary methods vector indicating which clickbait tactics were used.

CLICKBAIT TACTICS (FIXED ORDER, DO NOT CHANGE)

Index 0: Curiosity Gap: create explicit information withholding by signaling that a specific but unnamed piece of knowledge is missing. Cues such as "what you don't see", "behind the scenes", "this detail", "what remains hidden". Do NOT use continuation cues.
Index 1: Exaggeration: amplify importance, scale, or impact using intensity modifiers ("major", "dramatic", "significant", "unprecedented") without new facts.
Index 2: Emotional Triggers: evoke a specific emotion via explicit emotional wording ("fear", "outrage", "concern", "anger", "hope"). Do NOT rely on dramatic tone alone.
Index 3: Sensationalism: create dramatic impact through heightened framing or spectacle without emotional wording.
Index 4: Lists/Superlatives: use list-based structures or extreme ranking language ("top", "first", "largest", "most").
Index 5: Ambiguous References: use deliberately vague or non-specific references ("this", "they", "something", "a certain move"). Do NOT imply hidden information.
Index 6: Direct Appeals: address the reader or a specific audience directly ("you", "voters", "investors", "parents"). Do NOT phrase as a question.
Index 7: Unfinished Narratives: present the event as ongoing or unresolved ("what happens next", "the outcome remains unclear"). Do NOT frame as hidden information.
Index 8: Unexpected Associations: explicitly link two concepts or domains that are not commonly connected, both stated in the headline.
Index 9: Provocative Questions: question-like syntactic structure, even without a question mark. Do NOT combine with Direct Appeals.

If multiple tactics are used (max 3), each must have a distinct, identifiable linguistic cue in the final headline. If the cue is not present, the flag MUST be 0.

STRICT CONSTRAINTS
- Use ONLY the tactics listed above
- Do NOT introduce any new factual information
- Do NOT invent names, numbers, locations, or events
- Maintain the same topic and factual scope as the original headline

STYLE AND FORMATTING RULES
- The clickbait headline MUST be written entirely in lowercase
- The clickbait headline MUST NOT contain exclamation marks or question marks
- The clickbait headline MUST NOT be longer (in words) than the original headline
- Sound natural and journalistic, not exaggerated beyond realistic media standards
- Avoid deceptive or false claims

INPUT HEADLINES (JSON ARRAY):
Each element has:
- "original": the original headline
- "allowed_tactics": the subset of tactics allowed for this headline

You must generate ONE clickbait headline per input item.

{batch_json}

OUTPUT FORMAT (STRICT)
Return a VALID JSON ARRAY of length N (N = number of input headlines).
Each element MUST be an object with EXACTLY these three keys:

{{
  "original": "<original headline>",
  "clickbait": "<generated clickbait headline>",
  "methods_vector": [0,1,0,0,0,0,1,0,0,0]
}}

Rules:
- Return JSON ONLY (no text before or after, no markdown)
- The array length MUST equal the number of input headlines
- Preserve the order of the input headlines
- methods_vector must be a list of 10 integers (0/1) matching the fixed order
- methods_vector must contain between 1 and 3 ones
"""


REPAIR_PROMPT = """The following output was intended to be valid JSON, but it is malformed.

Return ONLY a valid JSON array. Do NOT add or remove items. Preserve the
original meaning. Do NOT include any text outside the JSON.

Broken output:
{broken}
"""


def extract_json(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("no JSON array in response")
    return text[start:end]


def make_ask_gpt(api_key: str, api_url: str, model: str, pool: int = 32):
    import requests
    from requests.adapters import HTTPAdapter

    session = requests.Session()
    # Enlarge the connection pool so many worker threads can hold concurrent
    # connections to the API without serializing on a small default pool.
    adapter = HTTPAdapter(pool_connections=pool, pool_maxsize=pool)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Transient API failures (rate limits, 5xx, connection resets, timeouts)
    # are retried with exponential backoff plus a small deterministic jitter
    # (index-derived, not time/random-derived, so runs stay reproducible).
    # Non-retryable errors (400 bad request, 401 auth) raise immediately.
    max_attempts = 6
    retryable_status = {429, 500, 502, 503, 504}

    def _is_retryable_body_error(err) -> bool:
        # OpenAI-compatible bodies may carry {"error": {"type": ..., "code": ...}}
        # even on a 200. Treat rate-limit / server-error markers as retryable.
        if not isinstance(err, dict):
            return False
        markers = str(err.get("type", "")) + " " + str(err.get("code", ""))
        markers = markers.lower()
        return ("rate_limit" in markers or "rate-limit" in markers
                or "server_error" in markers or "server-error" in markers
                or "overloaded" in markers)

    def ask(prompt: str, json_object: bool = False) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        # json_object mode is honored by OpenAI-compatible endpoints and
        # sharply reduces malformed-JSON round-trips. Harmless if ignored.
        if json_object:
            payload["response_format"] = {"type": "json_object"}

        last_reason = "unknown error"
        for attempt in range(max_attempts):
            retry_reason = None
            try:
                r = session.post(f"{api_url}/chat/completions", json=payload, timeout=120)
            except requests.exceptions.RequestException as e:
                retry_reason = f"request exception: {e}"
            else:
                if r.status_code in retryable_status:
                    retry_reason = f"HTTP {r.status_code}"
                else:
                    try:
                        data = r.json()
                    except ValueError as e:
                        # A non-retryable HTTP failure with an unparseable body
                        # (e.g. a 400/401 HTML page) must surface immediately.
                        r.raise_for_status()
                        raise RuntimeError(f"unparseable API response: {e}")
                    if "error" in data and _is_retryable_body_error(data["error"]):
                        retry_reason = f"body error: {data['error']}"
                    elif "error" in data:
                        raise RuntimeError(f"API error: {data['error']}")
                    elif "choices" not in data:
                        raise RuntimeError(f"unexpected API response: {data}")
                    else:
                        return data["choices"][0]["message"]["content"]

            # Reaching here means the attempt was retryable.
            last_reason = retry_reason
            if attempt < max_attempts - 1:
                delay = (2 ** (attempt + 1)) + 0.1 * attempt
                print(f"[gen] API retry {attempt + 1}/{max_attempts} "
                      f"after {delay}s: {retry_reason}", flush=True)
                time.sleep(delay)

        raise RuntimeError(f"API failed after {max_attempts} attempts: {last_reason}")

    return ask


def generate_batch(ask, batch_items):
    """Return (parsed_results, raw_response).

    raw_response is the FIRST raw GPT string, captured before any JSON parsing
    so the sidecar preserves the model's real output even when a repair round
    follows. On unrecoverable decode failure the RuntimeError carries the raw
    response as a .raw_response attribute so the caller can still log it.
    """
    prompt = PROMPT_TEMPLATE.format(batch_json=json.dumps(batch_items))
    response = ask(prompt)
    raw_response = response
    for _ in range(3):
        try:
            cleaned = response.replace("```json", "").replace("```", "")
            return json.loads(extract_json(cleaned)), raw_response
        except Exception:
            print("[gen] JSON decode failed, asking GPT to repair", flush=True)
            response = ask(REPAIR_PROMPT.format(broken=response))
    exc = RuntimeError("failed to decode JSON after 3 repair attempts")
    exc.raw_response = raw_response
    raise exc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv", required=True,
                   help="input CSV with a 'title' column of neutral headlines")
    p.add_argument("--out-csv", required=True,
                   help="output CSV (original, clickbait, methods_vector)")
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--max-rows", type=int, default=12600,
                   help="cap total generated rows (paper default 12600)")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore existing out-csv and regenerate from scratch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=12,
                   help="concurrent API requests. The generation is pure API "
                        "latency, so parallelism gives a near-linear speedup up "
                        "to the account rate limit (10k rpm / 10M tpm here).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Heavy imports after argparse so --help works with a bare interpreter.
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY is not set")
    api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")
    model = os.environ.get("OPENAI_API_MODEL", "gpt-4o-mini")
    ask = make_ask_gpt(api_key, api_url, model)

    src = pd.read_csv(args.source_csv)
    if "title" not in src.columns:
        sys.exit(f"source CSV {args.source_csv} has no 'title' column")
    titles = src["title"].dropna().astype(str).tolist()
    if args.max_rows and len(titles) > args.max_rows:
        titles = titles[: args.max_rows]
    print(f"[gen] loaded {len(titles)} titles (cap={args.max_rows})", flush=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Crash-safe raw-generation sidecar: one flushed JSONL line per batch,
    # capturing the RAW GPT response (incl. malformed/repaired/failed batches)
    # so synthetic-label quality is inspectable later. Append-only, so it is
    # resume-safe alongside the CSV.
    raw_path = out_path.with_name(out_path.stem + ".raw.jsonl")
    # --no-resume means a fresh run: TRUNCATE any existing output first. The
    # CSV is written in append mode (for thread-safe incremental writes), so
    # without this a rerun would pile new rows on top of the old ones and
    # produce duplicate originals. Delete before any writer opens the files.
    if args.no_resume:
        out_path.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)
    raw_writer = JsonlWriter(raw_path)
    print(f"[gen] raw generations -> {raw_path}", flush=True)

    # Resume by CONTENT, not row count. Tracking which source titles are
    # already present makes the process idempotent and gap-free: a failed
    # middle batch is retried on the next run instead of permanently
    # skipping those titles (and later titles are never duplicated).
    if out_path.exists() and not args.no_resume:
        existing = pd.read_csv(out_path)
        done_titles = set(existing["original"].astype(str))
        print(f"[gen] resuming: {len(existing)} rows, "
              f"{len(done_titles)} source titles already done", flush=True)
    else:
        existing = pd.DataFrame(columns=["original", "clickbait", "methods_vector"])
        done_titles = set()

    todo = [t for t in titles if t not in done_titles]

    # Pre-sample the intended tactic vector for every title UP FRONT, single
    # threaded, because numpy's Generator is not thread-safe. Workers then only
    # read this map. title -> vector lets us re-match results by content rather
    # than trusting the model to preserve order/length.
    rng = np.random.default_rng(args.seed)
    title_to_vec = {}
    for t in todo:
        k = int(rng.integers(1, 4))
        idx = rng.choice(len(TACTIC_NAMES), size=k, replace=False)
        title_to_vec[t] = [1 if m in idx.tolist() else 0
                           for m in range(len(TACTIC_NAMES))]

    batches = [todo[i:i + args.batch_size]
               for i in range(0, len(todo), args.batch_size)]
    eta_min = len(todo) / (200.0 * max(args.workers, 1))
    print(f"[gen] {len(todo)} titles in {len(batches)} batches, "
          f"{args.workers} concurrent workers, wall-clock ETA ~{eta_min:.1f} min",
          flush=True)

    import csv as _csv
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process(batch_titles):
        """Runs in a worker thread: one API batch -> matched rows + raw record.
        Returns a dict; does NO shared-state writes (the main thread does those
        under a lock), so this is thread-safe.

        Matching is scoped to THIS batch's titles only (local_vec), so a
        returned "original" can never collide with a title assigned to a
        different batch. The positional fallback uses the enumerate index (not
        results.index, which is buggy on duplicate dicts), and a within-batch
        seen-set drops any title the model echoes more than once. Together these
        guarantee at most one row per source title.
        """
        local_vec = {t: title_to_vec[t] for t in batch_titles}
        batch_items = [
            {"original": t,
             "allowed_tactics": [TACTIC_NAMES[j]
                                 for j, v in enumerate(local_vec[t]) if v]}
            for t in batch_titles
        ]
        allowed = [it["allowed_tactics"] for it in batch_items]
        try:
            results, raw_response = generate_batch(ask, batch_items)
        except Exception as e:
            return {"rows": [], "matched": set(), "dropped": 0, "raw": {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "titles": batch_titles,
                "allowed_tactics": allowed,
                "raw_response": getattr(e, "raw_response", None),
                "parsed_ok": False, "n_parsed": 0, "n_dropped": 0,
                "error": f"{type(e).__name__}: {e}"}}
        rows, matched, dropped = [], set(), 0
        for idx, r in enumerate(results):
            orig = str(r.get("original", "")).strip()
            cb = str(r.get("clickbait", "")).strip()
            vec = local_vec.get(orig)  # LOCAL to this batch, never cross-batch
            if vec is None and len(results) == len(batch_titles):
                orig = batch_titles[idx]  # positional fallback by index
                vec = local_vec[orig]
            if vec is None or not cb or orig in matched:
                dropped += 1
                continue
            rows.append((orig, cb, json.dumps(vec)))
            matched.add(orig)
        return {"rows": rows, "matched": matched, "dropped": dropped, "raw": {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "titles": batch_titles,
            "allowed_tactics": allowed, "raw_response": raw_response,
            "parsed_ok": True, "n_parsed": len(rows), "n_dropped": dropped,
            "error": None}}

    lock = threading.Lock()
    header_needed = (not out_path.exists()) or out_path.stat().st_size == 0
    csv_f = open(out_path, "a", newline="", encoding="utf-8")
    writer = _csv.writer(csv_f)
    if header_needed:
        writer.writerow(["original", "clickbait", "methods_vector"])
        csv_f.flush()

    total_rows = 0 if header_needed else len(existing)
    n_dropped = 0
    n_batches_done = 0

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as ex:
        futures = [ex.submit(process, b) for b in batches]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            res = fut.result()
            with lock:  # serialize all writes -> thread-safe + crash-safe flush
                raw_writer.write(res["raw"])
                for (orig, cb, vecjson) in res["rows"]:
                    writer.writerow([orig, cb, vecjson])
                csv_f.flush()
                total_rows += len(res["rows"])
                n_dropped += res["dropped"]
                done_titles |= res["matched"]
                n_batches_done += 1
                if n_batches_done % 20 == 0:
                    print(f"[gen] {total_rows} rows "
                          f"({len(done_titles)}/{len(titles)} titles, "
                          f"{n_dropped} dropped, "
                          f"{n_batches_done}/{len(batches)} batches)", flush=True)

    csv_f.close()
    raw_writer.close()
    print(f"[gen] done. {total_rows} rows, {n_dropped} dropped for "
          f"misalignment/empty across the run.", flush=True)


if __name__ == "__main__":
    main()
