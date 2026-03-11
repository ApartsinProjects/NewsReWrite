import pandas as pd
import numpy as np
import torch

from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from torch.utils.data import Dataset


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

# ======================================================
# CONFIG
# ======================================================
DATA_FILE = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\Dataset_generation\trainval_prefixes.csv"
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 32
RANDOM_SEED = 42

SAVE_DIR = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\models\bert_tactics_prefix_finetuned"

# ======================================================
# 1. Load dataset
# ======================================================
df = pd.read_csv(DATA_FILE)

df["prefix_text"] = df["prefix_text"].astype(str)
df["tactics_vector"] = df["tactics_vector"].apply(literal_eval)

X = df["prefix_text"]
y = np.array(df["tactics_vector"].tolist())

num_labels = y.shape[1]

print("Dataset size:", len(df))
print("Number of tactics:", num_labels)
print("Tactic distribution:", y.sum(axis=0))

# ======================================================
# 2. Train / Validation split
# ======================================================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=RANDOM_SEED,
    shuffle=True
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))

# ======================================================
# 3. Tokenizer (SINGLE TEXT – PREFIX)
# ======================================================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )

train_enc = tokenize(X_train)
val_enc   = tokenize(X_val)

# ======================================================
# 4. Dataset class
# ======================================================
class PrefixTacticsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

train_dataset = PrefixTacticsDataset(train_enc, y_train)
val_dataset   = PrefixTacticsDataset(val_enc, y_val)

# ======================================================
# 5. Model (MULTI-LABEL)
# ======================================================
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# ======================================================
# 6. Metrics
# ======================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)

    return {
        "tactics_precision_macro": precision_score(
            labels, preds, average="macro", zero_division=0
        ),
        "tactics_recall_macro": recall_score(
            labels, preds, average="macro", zero_division=0
        ),
        "tactics_f1_macro": f1_score(
            labels, preds, average="macro", zero_division=0
        ),
        "micro_f1": f1_score(
            labels, preds, average="micro", zero_division=0
        )
    }

# ======================================================
# 7. Training arguments
# ======================================================
training_args = TrainingArguments(
    output_dir="./bert_tactics_prefix_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    logging_steps=50,
    save_total_limit=1,
    report_to="none"
)

# ======================================================
# 8. Trainer
# ======================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ======================================================
# 9. Train
# ======================================================
trainer.train()

# ======================================================
# 10. Save model
# ======================================================
trainer.model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Prefix tactics model saved to: {SAVE_DIR}")
