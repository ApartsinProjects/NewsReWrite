import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset

# ======================================================
# PATHS
# ======================================================
TEST_FILE_FOR_MODEL = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\Dataset_generation\clickbait_prefix_test.csv"
TEST_PREFIXES_FILE  = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\Dataset_generation\test_prefixes.csv"
MODEL_DIR = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\models\bert_clickbait_prefix_finetuned"

# ======================================================
# 1) LOAD clickbait_prefix_test.csv (for predictions + ROC)
# ======================================================
df_test = pd.read_csv(TEST_FILE_FOR_MODEL)
texts = df_test["text"].astype(str)
labels = df_test["label"].astype(int).to_numpy()

# ======================================================
# TOKENIZER
# ======================================================
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

def tokenize(texts):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )

encodings = tokenize(texts)

# ======================================================
# DATASET
# ======================================================
class ClickbaitDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

test_dataset = ClickbaitDataset(encodings, labels)

# ======================================================
# LOAD MODEL
# ======================================================
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# ======================================================
# PREDICT (on clickbait_prefix_test.csv)
# ======================================================
all_logits = []

with torch.no_grad():
    for item in test_dataset:
        input_ids = item["input_ids"].unsqueeze(0)
        attention_mask = item["attention_mask"].unsqueeze(0)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(outputs.logits.squeeze(0).cpu().numpy())

logits = np.array(all_logits)

preds = np.argmax(logits, axis=1)

# ======================================================
# METRICS (binary)
# ======================================================
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
acc = accuracy_score(labels, preds)

print("Accuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

# ======================================================
# ROC (on clickbait_prefix_test.csv)
# ======================================================
probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Clickbait Scoring Model")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ======================================================
# 2) Accuracy vs Number of Tactics (using test_prefixes.csv)
# ======================================================
df_prefixes = pd.read_csv(TEST_PREFIXES_FILE)

# IMPORTANT: the number of rows must match, and order must match!
if len(df_prefixes) != len(df_test):
    raise ValueError(
        f"Row mismatch: test_prefixes.csv has {len(df_prefixes)} rows, "
        f"but clickbait_prefix_test.csv has {len(df_test)} rows. "
        "They must match exactly in length and order."
    )

def count_tactics(vec):
    if isinstance(vec, str):
        nums = [int(x) for x in vec.replace("[","").replace("]","").split(",")]
        return sum(nums)
    return 0

df_prefixes["num_tactics"] = df_prefixes["tactics_vector"].apply(count_tactics)

# attach predictions + true labels from df_test (same order)
df_prefixes["pred"] = preds
df_prefixes["true"] = df_test["label"].to_numpy()
df_prefixes["correct"] = (df_prefixes["pred"] == df_prefixes["true"]).astype(int)

acc_by_tactics = df_prefixes.groupby("num_tactics")["correct"].mean()

print("\nAccuracy by number of tactics:")
print(acc_by_tactics)

plt.figure(figsize=(6,5))

# make sure bars show 0..3 even if missing
x = [0,1,2,3]
y = [acc_by_tactics.get(i, np.nan) for i in x]

plt.bar(["Neutral","One tactic","Two tactics","Three tactics"], y)
plt.ylim(0,1)
plt.xlabel("Number of Clickbait Tactics")
plt.ylabel("Detection Accuracy")
plt.title("Clickbait Detection Accuracy vs. Number of Tactics")
plt.grid(axis="y")
plt.show()
