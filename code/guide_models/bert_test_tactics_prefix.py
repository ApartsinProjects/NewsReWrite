import pandas as pd
import numpy as np
import torch

from ast import literal_eval
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset

# ======================================================
# PATHS
# ======================================================
TEST_FILE = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\Dataset_generation\test_prefixes.csv"
MODEL_DIR = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\models\bert_tactics_prefix_finetuned"

MAX_LEN = 32

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv(TEST_FILE)

df["prefix_text"] = df["prefix_text"].astype(str)
df["tactics_vector"] = df["tactics_vector"].apply(literal_eval)

texts = df["prefix_text"]
labels = np.array(df["tactics_vector"].tolist())

print("Test size:", len(df))
print("Num tactics:", labels.shape[1])

# ======================================================
# TOKENIZER
# ======================================================
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

def tokenize(texts):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

encodings = tokenize(texts)

# ======================================================
# DATASET
# ======================================================
class PrefixTacticsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

test_dataset = PrefixTacticsDataset(encodings, labels)

# ======================================================
# LOAD MODEL
# ======================================================
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# ======================================================
# PREDICT
# ======================================================
all_logits = []

with torch.no_grad():
    for item in test_dataset:
        input_ids = item["input_ids"].unsqueeze(0)
        attention_mask = item["attention_mask"].unsqueeze(0)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(outputs.logits.squeeze(0).cpu().numpy())

logits = np.array(all_logits)

# ======================================================
# MULTI-LABEL METRICS
# ======================================================
probs = torch.sigmoid(torch.tensor(logits)).numpy()
preds = (probs > 0.5).astype(int)

precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
recall_macro    = recall_score(labels, preds, average="macro", zero_division=0)
f1_macro        = f1_score(labels, preds, average="macro", zero_division=0)
f1_micro        = f1_score(labels, preds, average="micro", zero_division=0)

print("\n=== TACTICS MODEL TEST RESULTS ===")
print("Precision Macro:", precision_macro)
print("Recall Macro:", recall_macro)
print("F1 Macro:", f1_macro)
print("F1 Micro:", f1_micro)



import matplotlib.pyplot as plt

TACTIC_NAMES = [
    "Curiosity Gap",
    "Exaggeration",
    "Emotional Trigger",
    "Sensationalism",
    "Lists/Superlatives",
    "Ambiguous References",
    "Direct Appeals",
    "Unfinished Narratives",
    "Unexpected Associations",
    "Provocative Questions"
]

# F1 לכל טקטיקה בנפרד
f1_per_tactic = []

for i in range(labels.shape[1]):
    f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
    f1_per_tactic.append(f1)

plt.figure(figsize=(8,5))
bars = plt.bar(TACTIC_NAMES, f1_per_tactic)

plt.xticks(rotation=45)
plt.ylabel("F1 Score")
plt.title("Engagement Attribute Model – Prediction Score per Tactic")

# הוספת המספרים מעל העמודות
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height:.2f}",
        ha='center',
        va='bottom'
    )

plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()




import seaborn as sns
from sklearn.metrics import confusion_matrix


single_idx = np.where(labels.sum(axis=1) == 1)[0]

true_single = labels[single_idx]
pred_single = preds[single_idx]

true_class = np.argmax(true_single, axis=1)
pred_class = np.argmax(pred_single, axis=1)

cm = confusion_matrix(true_class, pred_class)

print(cm.sum())


plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=TACTIC_NAMES,
    yticklabels=TACTIC_NAMES
)


plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Engagement Attribute Model – Confusion Matrix")
plt.tight_layout()



plt.show()


