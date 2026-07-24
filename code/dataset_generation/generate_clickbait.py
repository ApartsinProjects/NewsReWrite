import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import requests
from dotenv import load_dotenv


def extract_json(text):
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == -1:
        raise ValueError("No JSON array found")
    return text[start:end]

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ------------------------------
# Load Clickbait Methods (10 total)
# ------------------------------
methods_path = Path(__file__).resolve().parent / "clickbait_methods.json"
with open(methods_path, "r") as f:
    METHODS_CATALOG = json.load(f)

print("Loaded methods:", METHODS_CATALOG)

# ------------------------------
# OpenAI client
# ------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")

session = requests.Session()
session.headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

def ask_gpt(prompt):
    response = session.post(
        url=f"{OPENAI_API_URL}/chat/completions",
        json={
            "model": os.getenv("OPENAI_API_MODEL"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        },
        timeout=60
    )

    data = response.json()

    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")

    if "choices" not in data:
        raise RuntimeError(f"Unexpected API response: {data}")

    return data["choices"][0]["message"]["content"]



def repair_json_with_gpt(broken_output):
    repair_prompt = f"""
The following output was intended to be valid JSON, but it is malformed.

Your task:
- Return ONLY a valid JSON array
- Do NOT add or remove items
- Preserve the original meaning
- Do NOT include any text outside the JSON

Broken output:
{broken_output}
"""
    return ask_gpt(repair_prompt)


# ------------------------------
# Batch clickbait generator
# ------------------------------
def generate_clickbait_batch(batch):
    """
    batch = [{'original': "...", 'methods': ["Curiosity Gap", ...]}, ...]
    """

    prompt = f"""
    You are generating a controlled synthetic dataset for academic research on explainable clickbait detection.

Your task:
Given a factual, non-clickbait news headline, generate EXACTLY ONE clickbait headline and a binary methods vector indicating which clickbait tactics were used.

────────────────────────────
CLICKBAIT TACTICS (FIXED ORDER — DO NOT CHANGE)

Index 0: Curiosity Gap:
Create explicit information withholding by signaling that a specific but unnamed piece of knowledge is missing.
Required cues:
- Phrases such as “what you don’t see”, “behind the scenes”, “this detail”, “what remains hidden”.
Constraints:
- Do NOT rely on general ambiguity alone.
- Do NOT use continuation cues (those belong to Unfinished Narratives).

Index 1: Exaggeration:
Amplify importance, scale, or impact using intensity modifiers without adding new facts.
Required cues:
- Intensity modifiers such as “major”, “dramatic”, “significant”, “unprecedented”.
Constraints:
- Do NOT introduce emotional language.
- Do NOT use sensational or shocking framing.

Index 2: Emotional Triggers:
Evoke a specific emotion (fear, concern, outrage, hope) through explicit emotional wording.
Required cues:
- Clear emotional terms such as “fear”, “outrage”, “concern”, “anger”, “hope”.
Constraints:
- Do NOT rely on dramatic tone alone.
- Do NOT overlap with Sensationalism.

Index 3: Sensationalism:
Create dramatic impact through heightened framing or spectacle without emotional wording.
Required cues:
- Strong dramatic framing suggesting shock or spectacle.
Constraints:
- Do NOT use explicit emotional words.
- Do NOT use exaggerative magnitude modifiers.

Index 4: Lists / Superlatives:
Present information using list-based structures or extreme ranking language.
Required cues:
- Explicit list or ranking terms such as “top”, “first”, “largest”, “most”.
Constraints:
- Do NOT introduce arbitrary numbers.
- Do NOT rely on exaggeration without list or ranking structure.

Index 5: Ambiguous References:
Use deliberately vague or non-specific references to create uncertainty.
Required cues:
- Indefinite references such as “this”, “they”, “something”, “a certain move”.
Constraints:
- Do NOT imply hidden information (Curiosity Gap).
- Do NOT suggest narrative continuation (Unfinished Narratives).

Index 6: Direct Appeals:
Address the reader or a specific audience directly to prompt engagement.
Required cues:
- Direct address such as “you”, “voters”, “investors”, “parents”.
Constraints:
- Do NOT phrase as a question.
- Do NOT use emotional or sensational language.

Index 7: Unfinished Narratives:
Present the event as ongoing or unresolved, emphasizing that the outcome is not yet known.
Required cues:
- Explicit continuation signals such as “what happens next”, “the outcome remains unclear”.
Constraints:
- Do NOT frame as hidden information (Curiosity Gap).
- Do NOT rely on vague references alone.

Index 8: Unexpected Associations:
Explicitly link two concepts or domains that are not commonly connected.
Required cues:
- Clear mention of both elements and their surprising linkage.
Constraints:
- Do NOT rely on subtle implication alone.
- Do NOT add emotional or dramatic framing.

Index 9: Provocative Questions:
Frame the headline as an implicit or explicit challenge to an assumption.
Required cues:
- Question-like syntactic structure, even without a question mark.
Constraints:
- Do NOT rely on general curiosity cues.
- Do NOT combine with Direct Appeals.
- Note: Provocative Questions must be expressed through syntax or phrasing,
not through punctuation.


────────────────────────────
IMPORTANT CLARIFICATION ABOUT CLICKBAIT TACTICS

Each clickbait tactic must be expressed through a DISTINCT and IDENTIFIABLE linguistic signal
that can be inferred directly from the generated headline text alone.

If multiple tactics are used in the same headline (maximum 3):
- Each tactic must have its own clear and unique textual cue.
- Do NOT rely on vague phrasing that could ambiguously represent multiple tactics.
- Avoid overlapping or conflated realizations of different tactics.
- If a tactic’s required linguistic cue is not clearly present in the final selected headline,
that tactic MUST NOT be marked in the methods_vector, even if it was intended during generation.
- When multiple tactics are used, avoid reusing the same phrase or construction
to realize more than one tactic.


The goal is that both a human reader and a downstream classification model
can reliably distinguish which specific tactics were applied based solely on the headline text.

────────────────────────────
STRICT CONSTRAINTS

- Use ONLY the tactics listed above
- Do NOT introduce any new factual information
- Do NOT invent names, numbers, locations, or events
- Maintain the same topic and factual scope as the original headline


Multi-Tactic Compatibility Rule:
When using more than one clickbait tactic in a single headline, select ONLY tactics that are linguistically, semantically, and structurally distinct.

- Do NOT combine tactics that rely on similar cues, framing, or discourse mechanisms.
- Avoid pairing overlapping tactics such as Curiosity Gap with Unfinished Narratives, or Ambiguous References with Curiosity Gap.
- Each selected tactic must be clearly identifiable through a different linguistic signal in the headline.



STYLE & FORMATTING RULES (VERY IMPORTANT)

- The clickbait headline MUST be written entirely in lowercase
- The clickbait headline MUST NOT contain:
  - exclamation marks (!)
  - question marks (?)
- The clickbait headline MUST NOT be longer (in number of words) than the original headline
- The clickbait headline should sound natural and journalistic, not exaggerated beyond realistic media standards
- Avoid deceptive or false claims

────────────────────────────
INPUT HEADLINES (JSON ARRAY):

Below is a list of factual, non-clickbait news headlines.
Each element contains:
- "original": the original headline
- "allowed_tactics": the subset of tactics allowed for this headline

You must generate ONE clickbait headline per input item.

{json.dumps(batch)}

Language: English

────────────────────────────
PROCESS (INTERNAL — DO NOT OUTPUT)

1) Generate 3 candidate clickbait headlines using the allowed tactics.
2) Ensure all candidates:
   - comply with lowercase formatting
   - contain no ! or ?
   - do not exceed the original headline length
3) Evaluate the candidates for:
   - clear use of the selected tactics
   - strength of engagement
   - faithfulness to the original headline
   - absence of hallucinated facts
4) Select the best candidate.

────────────────────────────
OUTPUT FORMAT (STRICT)

Return a VALID JSON ARRAY of length N,
where N is the number of input headlines.

Each element in the array MUST be an object with EXACTLY these three keys:

{{
  "original": "<original headline>",
  "clickbait": "<generated clickbait headline>",
  "methods_vector": [0,1,0,0,0,0,1,0,0,0]
}}

Rules:
- Return JSON ONLY (no text before or after)
- Do NOT wrap in markdown
- The array length MUST equal the number of input headlines
- Preserve the order of the input headlines
- methods_vector must be a list of 10 integers (0/1)
- methods_vector must contain between 1 and 3 ones


Rules for methods vector:
- Put 1 at index i if tactic i was used
- Put 0 otherwise
- The vector order MUST match the fixed index order above
- The vector MUST contain between 1 and 3 ones

DO NOT include explanations, headers, labels, or additional text.
Return only the single formatted output line.

    """

    response = ask_gpt(prompt)


    for attempt in range(3):
        try:

            cleaned = response.replace("```json", "").replace("```", "")
            json_text = extract_json(cleaned)

            return json.loads(json_text)
        except Exception:
            print("JSON decode failed — retrying...")
            response = repair_json_with_gpt(response)


    raise Exception("Failed to decode JSON after 3 attempts")

# ------------------------------
# Load titles
# ------------------------------
df = pd.read_csv("combined_news_test.csv")
titles = df["title"].dropna().tolist()
print(f"Loaded {len(titles)} titles.")

# ------------------------------
# Output file
# ------------------------------
OUTPUT_FILE = Path("clickbait_generated_test.csv")

if OUTPUT_FILE.exists():
    existing = pd.read_csv(OUTPUT_FILE)
    generated_count = len(existing)
    print(f"Resuming from {generated_count} rows.")
else:
    existing = pd.DataFrame(columns=["original", "clickbait", "methods_vector"])
    generated_count = 0

# ------------------------------
# Batch loop
# ------------------------------
batch_size = 15

for i in tqdm(range(generated_count, len(titles), batch_size)):
    batch = titles[i: i + batch_size]

    batch_items = []
    vectors = []

    for t in batch:
        k = np.random.randint(1, 4)
        chosen_methods = np.random.choice(METHODS_CATALOG, k, replace=False).tolist()

        # Build vector
        vector = [1 if m in chosen_methods else 0 for m in METHODS_CATALOG]
        vectors.append(vector)

        batch_items.append({
            "original": t,
            "methods": chosen_methods
        })

    # GPT clickbait generation
    try:
        results = generate_clickbait_batch(batch_items)
    except Exception as e:
        print(f"Batch failed at index {i}: {str(e)}")
        continue

    # Insert vectors
    for j in range(len(results)):
        results[j]["methods_vector"] = vectors[j]

    temp_df = pd.DataFrame(results)

    temp_df = pd.DataFrame(results)

    # Keep only desired columns
    temp_df = temp_df[["original", "clickbait", "methods_vector"]]

    # Save
    existing = pd.concat([existing, temp_df], ignore_index=True)
    existing.to_csv(OUTPUT_FILE, index=False)



    print(f"Saved {len(existing)} / {len(titles)} clickbait headlines.")

print("DONE! All clickbait headlines generated.")
