import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset (only real news)
df = pd.read_csv("combined_news_dataset.csv")

# 80% train+validation, 20% test
train_val_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Save to new CSV files
train_val_df.to_csv("combined_news_train_val.csv", index=False)
test_df.to_csv("combined_news_test.csv", index=False)

print("Total:", len(df))
print("Train+Val:", len(train_val_df))
print("Test:", len(test_df))
