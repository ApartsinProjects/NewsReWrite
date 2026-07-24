"""Fill rater packages with GPT-simulated ratings (a PROXY for the human study).

This is an INTERIM stand-in so the Track C analysis pipeline and the paper can
be completed before the real human experiment runs. Three GPT "raters" with
distinct annotator personas at raised temperature produce genuine inter-rater
disagreement (unlike the oracle-copy --simulate path), so Fleiss kappa,
ICC(2,k), human-vs-automatic Spearman, and per-condition means are all real,
non-degenerate numbers.

Every output is stamped `rater_source=gpt_sim` and the study writes a
PROXY_NOTICE.md. These are NOT human labels and must be reported as
"LLM-simulated proxy ratings, pending the human study" -- never as human
results.

The GPT raters see only (source_headline, rewrite) -- the rater files are
already blind (method / condition / intended tactics live only in oracle.csv),
so the simulation is faithful to what a human annotator would see.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

TACTICS = [
    ("curiosity_gap", "withholds key information to make you click"),
    ("exaggeration", "overstates or inflates"),
    ("emotional_trigger", "strong emotional language"),
    ("sensationalism", "shocking or dramatic framing"),
    ("lists_or_superlatives", "numbers/lists or best/worst superlatives"),
    ("ambiguous_references", "vague 'this/these/that' references"),
    ("direct_appeals", "directly addresses the reader ('you')"),
    ("unfinished_narratives", "cliffhanger / leaves the story open"),
    ("unexpected_associations", "surprising or odd juxtaposition"),
    ("provocative_questions", "a rhetorical or provocative question"),
]

PERSONAS = {
    "rater_01": "You are a fairly STRICT annotator; you only mark a tactic present when it is clearly and strongly there.",
    "rater_02": "You are a fairly LENIENT annotator; you mark a tactic present even when it is mildly or subtly there.",
    "rater_03": "You are a BALANCED, careful annotator.",
}


def _openai(prompt: str, temperature: float) -> dict:
    import requests
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"},
        json={"model": os.environ.get("OPENAI_API_MODEL", "gpt-4o-mini"),
              "messages": [{"role": "user", "content": prompt}],
              "temperature": temperature,
              "response_format": {"type": "json_object"}},
        timeout=60,
    )
    return json.loads(r.json()["choices"][0]["message"]["content"])


def _c1_prompt(persona, source, rewrite):
    lines = "\n".join(f"- {name}: {desc}" for name, desc in TACTICS)
    return (
        f"You are a human annotator labeling clickbait tactics. {persona}\n\n"
        f"Original neutral headline: \"{source}\"\n"
        f"Rewritten headline: \"{rewrite}\"\n\n"
        "For each tactic decide if it is present in the rewrite (true/false):\n"
        f"{lines}\n\n"
        "Return ONLY strict JSON with all ten keys mapping to booleans, e.g. "
        "{\"curiosity_gap\": true, \"exaggeration\": false, ...}."
    )


def _c2_prompt(persona, source, rewrite):
    return (
        f"You are a human annotator rating a rewritten news headline. {persona}\n\n"
        f"Original neutral headline: \"{source}\"\n"
        f"Rewritten headline: \"{rewrite}\"\n\n"
        "Rate the rewrite with integers 1-5:\n"
        "- engagement: how attention-grabbing / clickable (1 dull ... 5 very engaging)\n"
        "- faithfulness: preserves original meaning, adds no new facts (1 distorted ... 5 fully faithful)\n"
        "- clickbait: how clickbait-y it feels (1 not at all ... 5 extremely)\n\n"
        "Return ONLY strict JSON: {\"engagement\": int, \"faithfulness\": int, \"clickbait\": int}."
    )


def _clamp15(x):
    try:
        return int(max(1, min(5, round(float(x)))))
    except (TypeError, ValueError):
        return 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-dir", required=True)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    sd = Path(args.study_dir)
    rater_files = sorted(sd.glob("rater_*.csv"))
    if not rater_files:
        raise SystemExit(f"no rater_*.csv under {sd}")

    for rf in rater_files:
        df = pd.read_csv(rf)
        rid = rf.stem  # rater_01
        persona = PERSONAS.get(rid, PERSONAS["rater_03"])
        is_c1 = any(c.startswith("tactic__") for c in df.columns)
        kind = "C1-tactics" if is_c1 else "C2-quality"
        print(f"[gpt-raters] {rf.name} [{kind}] persona={rid} rows={len(df)}",
              flush=True)

        def _one(item):
            i, row = item
            src = str(row["source_headline"]); rw = str(row["rewrite"])
            try:
                if is_c1:
                    j = _openai(_c1_prompt(persona, src, rw), args.temperature)
                    return i, {f"tactic__{n}": (1 if bool(j.get(n)) else 0)
                               for n, _ in TACTICS}
                else:
                    j = _openai(_c2_prompt(persona, src, rw), args.temperature)
                    return i, {"engagement_1_to_5": _clamp15(j.get("engagement")),
                               "faithfulness_1_to_5": _clamp15(j.get("faithfulness")),
                               "clickbait_1_to_5": _clamp15(j.get("clickbait"))}
            except Exception as e:
                return i, {"__error__": str(e)}

        n_err = 0
        with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as ex:
            for i, vals in ex.map(_one, list(df.iterrows())):
                if "__error__" in vals:
                    n_err += 1
                    continue
                for k, v in vals.items():
                    df.at[i, k] = v
        df["rater_source"] = "gpt_sim"
        df["rater_notes"] = f"LLM-simulated ({rid}); pending human study"
        df.to_csv(rf, index=False)
        print(f"[gpt-raters] wrote {rf.name} ({n_err} errored rows kept as-is)",
              flush=True)

    (sd / "PROXY_NOTICE.md").write_text(
        "# PROXY ratings -- NOT human data\n\n"
        "The rater_*.csv files in this directory were filled by GPT "
        f"({os.environ.get('OPENAI_API_MODEL', 'gpt-4o-mini')}) at temperature "
        f"{args.temperature} with three distinct annotator personas, as an "
        "INTERIM stand-in for the human study. Every downstream number "
        "(Fleiss kappa, ICC, Spearman, per-condition means) must be reported "
        "as 'LLM-simulated proxy ratings, pending the human experiment', never "
        "as human results.\n", encoding="utf-8")
    print(f"[gpt-raters] wrote {sd/'PROXY_NOTICE.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
