import pandas as pd

# ===============================
# CONFIG
# ===============================
INPUT_CSV = "clickbait_generated_test.csv"
OUTPUT_CSV = "test_prefixes.csv"
PREFIX_RATIOS = [0.3, 0.5, 0.7, 1.0]
MIN_WORDS = 2

# ===============================
# PREFIX FUNCTION
# ===============================
def make_prefixes(text, ratios=PREFIX_RATIOS):
    if not isinstance(text, str):
        return []

    words = text.split()
    prefixes = []

    for r in ratios:
        k = max(MIN_WORDS, int(len(words) * r))
        prefix = " ".join(words[:k])

        if r == 1.0:
            if not prefix.endswith("."):
                prefix = prefix + "."
        prefixes.append(prefix)

    return prefixes


# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(INPUT_CSV)

rows = []

# ===============================
# BUILD PREFIX DATASET
# ===============================
for _, row in df.iterrows():

    real_title = row["original"]
    clickbait_title = row["clickbait"]
    tactics_vector = row["methods_vector"]

    # -------- REAL TITLE (label=0) --------
    real_prefixes = make_prefixes(real_title)
    for p in real_prefixes:
        # REAL TITLE
        rows.append({
            "prefix_text": p,
            "is_clickbait": 0,
            "tactics_vector": "[0,0,0,0,0,0,0,0,0,0]",
            "source": "real"
        })

    # -------- CLICKBAIT TITLE (label=1) --------
    click_prefixes = make_prefixes(clickbait_title)
    for p in click_prefixes:
        rows.append({
            "prefix_text": p,
            "is_clickbait": 1,
            "tactics_vector": tactics_vector,
            "source": "clickbait"
        })

# ===============================
# SAVE
# ===============================
prefix_df = pd.DataFrame(rows)
prefix_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {len(prefix_df)} rows to {OUTPUT_CSV}")


