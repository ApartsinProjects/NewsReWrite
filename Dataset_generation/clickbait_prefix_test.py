import pandas as pd

# קובץ prefixes של הטסט
df = pd.read_csv("test_prefixes.csv")

texts = df["prefix_text"].tolist()
labels = df["is_clickbait"].tolist()

new_texts = []

for i in range(len(texts)):
    current = texts[i].strip()

    # אם ה-prefix הבא קצר יותר → זה המשפט המלא
    is_last = (
        i == len(texts) - 1 or
        len(texts[i+1]) < len(current)
    )

    if is_last:
        # הוספת נקודה רק למשפט השלם
        if not current.endswith((".", "!", "?")):
            current = current + "."

    new_texts.append(current)

# שמירה בפורמט שמתאים למודל
out_df = pd.DataFrame({
    "text": new_texts,
    "label": labels
})

out_df.to_csv("clickbait_prefix_test.csv", index=False)
print("Saved clickbait_prefix_test.csv")
