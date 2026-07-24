import pandas as pd
from ast import literal_eval

# ===============================
# PATHS
# ===============================
TRAINVAL_FILE = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\Dataset_generation\clickbait_generated_train_val.csv"
TEST_FILE     = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\Dataset_generation\clickbait_generated_test.csv"

OUTPUT_CSV = r"D:\MS.c\MS.c\Yehudit\Clickbait_Detection_Project\ClickbaitTacticsDetection\Dataset_generation\table3_dataset_composition.csv"

# ===============================
# LOAD
# ===============================
df_train = pd.read_csv(TRAINVAL_FILE)
df_test  = pd.read_csv(TEST_FILE)

df_all = pd.concat([df_train, df_test], ignore_index=True)

# ===============================
# COUNT ACTIVE TACTICS
# ===============================
def count_tactics(vec):
    if isinstance(vec, str):
        nums = literal_eval(vec)
        return sum(nums)
    return 0

df_all["num_tactics"] = df_all["methods_vector"].apply(count_tactics)

# ===============================
# BUILD TABLE
# ===============================
neutral_count = len(df_all)  # כל original הוא נייטרלי

one_tactic   = (df_all["num_tactics"] == 1).sum()
two_tactics  = (df_all["num_tactics"] == 2).sum()
three_tactics= (df_all["num_tactics"] == 3).sum()

table = pd.DataFrame([
    ["Neutral", "0", neutral_count],
    ["Clickbait", "1 (single tactic)", one_tactic],
    ["Clickbait", "2 (two tactics)", two_tactics],
    ["Clickbait", "3 (three tactics)", three_tactics],
], columns=["Class", "Number of Active Tactics", "Number of Samples"])

print(table)

# ===============================
# SAVE
# ===============================
table.to_csv(OUTPUT_CSV, index=False)
print("\nSaved to:", OUTPUT_CSV)
