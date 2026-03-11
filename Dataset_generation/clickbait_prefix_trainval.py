import pandas as pd

df = pd.read_csv("trainval_prefixes.csv")

texts = df["prefix_text"].tolist()
labels = df["is_clickbait"].tolist()

new_texts = []

for i in range(len(texts)):
    current = texts[i].strip()

    # Check if this is the full headline:
    # If next prefix is shorter -> this is the last prefix in the group
    is_last = (
        i == len(texts) - 1 or
        len(texts[i+1]) < len(current)
    )

    if is_last:
        # Add period only if no ending punctuation
        if not current.endswith((".", "!", "?")):
            current = current + "."

    new_texts.append(current)

out_df = pd.DataFrame({
    "text": new_texts,   # ← כאן התיקון
    "label": labels
})

out_df.to_csv("clickbait_prefix_trainval.csv", index=False)
print("Saved clickbait_prefix_trainval.csv")
