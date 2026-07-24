import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset

# ======================================================
# CONFIG
# ======================================================
DATA_FILE = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\Dataset_generation\clickbait_prefix_trainval.csv"
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 32
RANDOM_SEED = 42


# ======================================================
# 1. Load dataset
# ======================================================
df = pd.read_csv(DATA_FILE)

df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)

print("Dataset size:", len(df))
print(df["label"].value_counts())





# ======================================================
# TRAIN / VALIDATION SPLIT (INTERNAL)
# ======================================================
X = df["text"]
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.15,      # 15% validation
    random_state=42,
    stratify=y
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))





# ======================================================
# 3. Tokenizer
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
class ClickbaitDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = ClickbaitDataset(train_enc, y_train)
val_dataset   = ClickbaitDataset(val_enc, y_val)


# ======================================================
# 5. Model
# ======================================================
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# ======================================================
# 6. Metrics
# ======================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ======================================================
# 7. Training arguments
# ======================================================
training_args = TrainingArguments(
    output_dir="./bert_clickbait_prefix_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir="./logs",
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
# PLOT TRAIN / VALIDATION LOSS PER EPOCH
# ======================================================

log_history = trainer.state.log_history

train_epochs = []
train_losses = []
eval_epochs = []
eval_losses = []

for log in log_history:
    if "loss" in log and "epoch" in log and "eval_loss" not in log:
        train_epochs.append(log["epoch"])
        train_losses.append(log["loss"])

    if "eval_loss" in log and "epoch" in log:
        eval_epochs.append(log["epoch"])
        eval_losses.append(log["eval_loss"])

plt.figure(figsize=(6, 4))
plt.plot(train_epochs, train_losses, label="Training Loss")
plt.plot(eval_epochs, eval_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

LOSS_PLOT_PATH = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\metrics\BERT\bert_clickbait_loss_curve.png"
plt.tight_layout()
plt.savefig(LOSS_PLOT_PATH)
plt.show()

print(f"Loss curve saved to: {LOSS_PLOT_PATH}")





# ======================================================
# SAVE FINE-TUNED MODEL
# ======================================================
SAVE_DIR = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\models\bert_clickbait_prefix_finetuned"

trainer.model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Fine-tuned model saved to: {SAVE_DIR}")


