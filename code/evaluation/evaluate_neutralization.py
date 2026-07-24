import os
import json
import pandas as pd
from pathlib import Path
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ---- SIMILARITY MODELS ----
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bertscore

# ======================================================
# CONFIG
# ======================================================
CHUNK_SIZE = 2000
OUTPUT_FILE = "neutralization_evaluation.csv"

# ======================================================
# 1. Load environment (.env)
# ======================================================
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")
OPENAI_MODEL = os.getenv("OPENAI_API_MODEL")

# ======================================================
# OpenAI Client
# ======================================================
session = requests.Session()
session.headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

def ask_gpt(prompt):
    """Safe GPT call with fallback (no crash on quota / API errors)."""
    try:
        response = session.post(
            url=f"{OPENAI_API_URL}/chat/completions",
            json={
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=60
        )

        data = response.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        else:
            # API error / quota / rate limit
            return None

    except Exception:
        return None


# ======================================================
# 2. Load dataset (resume if exists)
# ======================================================
if os.path.exists(OUTPUT_FILE):
    print("Resuming from existing results file...")
    df = pd.read_csv(OUTPUT_FILE)
else:
    print("Starting from scratch...")
    df = pd.read_csv(
        r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\Dataset_generation\clickbait_generated_train_val.csv"
    )

# Ensure required columns exist
for col in [
    "neutralized",
    "STS_similarity",
    "BERTScore_P",
    "BERTScore_R",
    "BERTScore_F1",
    "LLM_Judge_Level",
    "LLM_Judge_Explanation",
]:
    if col not in df.columns:
        df[col] = None


# ======================================================
# 3. GPT Neutralization (chunked + resume)
# ======================================================
to_neutralize = df[df["neutralized"].isna()].head(CHUNK_SIZE)

print(f"Neutralizing {len(to_neutralize)} headlines...")

for idx, row in tqdm(to_neutralize.iterrows(), total=len(to_neutralize)):
    prompt = f"""
    Rewrite the following clickbait headline into a neutral, factual headline.
    Do NOT add new information. Preserve the original meaning.
    Use lowercase letters only.
    Do NOT use quotation marks

    Clickbait: "{row['clickbait']}"
    Neutral headline:
    """

    neutral = ask_gpt(prompt)

    if neutral is not None:
        neutral = neutral.strip()
        neutral = neutral.strip('"').strip("'")  # remove quotation marks
        neutral = neutral.lower()  # force lowercase
        df.at[idx, "neutralized"] = neutral
    else:
        # leave as None -> will retry next run
        break

# Save checkpoint
df.to_csv(OUTPUT_FILE, index=False)
print("Checkpoint saved after neutralization.")


# ======================================================
# 4. STS + BERTScore (ONLY for rows with neutralized text)
# ======================================================
ready_for_metrics = df[
    df["neutralized"].notna() &
    df["STS_similarity"].isna()
]

if len(ready_for_metrics) > 0:
    print(f"Computing STS & BERTScore for {len(ready_for_metrics)} rows...")

    sts_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    originals = ready_for_metrics["original"].tolist()
    neutralized = ready_for_metrics["neutralized"].tolist()

    emb1 = sts_model.encode(originals, convert_to_tensor=True, show_progress_bar=True)
    emb2 = sts_model.encode(neutralized, convert_to_tensor=True, show_progress_bar=True)

    sts_scores = util.cos_sim(emb1, emb2).diagonal().cpu().numpy().tolist()

    P, R, F1 = bertscore(neutralized, originals, lang="en")

    for i, idx in enumerate(ready_for_metrics.index):
        df.at[idx, "STS_similarity"] = sts_scores[i]
        df.at[idx, "BERTScore_P"] = P[i].item()
        df.at[idx, "BERTScore_R"] = R[i].item()
        df.at[idx, "BERTScore_F1"] = F1[i].item()

    df.to_csv(OUTPUT_FILE, index=False)
    print("Checkpoint saved after STS & BERTScore.")


# ======================================================
# 5. GPT LLM Judge (chunked + resume)
# ======================================================
to_judge = df[
    df["neutralized"].notna() &
    df["LLM_Judge_Level"].isna()
].head(CHUNK_SIZE)

print(f"Judging {len(to_judge)} headlines...")

for idx, row in tqdm(to_judge.iterrows(), total=len(to_judge)):
    judge_prompt = f"""
    Rate how well the neutralized headline preserves the meaning of the original headline.

    Original: "{row['original']}"
    Neutralized: "{row['neutralized']}"

    Classify the semantic preservation level into ONE of the following categories:
    - "high": meaning is fully or almost fully preserved
    - "medium": meaning is partially preserved, with noticeable changes
    - "low": meaning is largely lost or significantly altered

    Then provide a 1-sentence explanation.

    Return ONLY valid JSON:
    {{
        "level": "low" | "medium" | "high",
        "explanation": "<text>"
    }}
    """

    result = ask_gpt(judge_prompt)

    if result is None:
        break

    try:
        parsed = json.loads(result)
        level = parsed.get("level", "").lower()

        if level not in ["low", "medium", "high"]:
            raise ValueError

        df.at[idx, "LLM_Judge_Level"] = level
        df.at[idx, "LLM_Judge_Explanation"] = parsed["explanation"]

    except Exception:
        df.at[idx, "LLM_Judge_Level"] = None
        df.at[idx, "LLM_Judge_Explanation"] = "Parse error"

# Final save
df.to_csv(OUTPUT_FILE, index=False)

print("\n====================================")
print("EVALUATION CHUNK COMPLETED")
print("You can safely rerun the script.")
print("====================================")
