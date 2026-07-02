# label_test.py
import pandas as pd
from src.stage1_doctype.label_consolidation import (
    assign_labels, print_label_distribution, get_class_weights
)
from src.stage1_doctype.data_split import make_splits, save_splits

df = pd.read_csv("data/processed/mtsamples_clean.csv", index_col=0)

# Assign labels
df = assign_labels(df)

# Print distributions
print_label_distribution(df)

# Stage 1 splits (all rows)
print("\n\n=== STAGE 1 SPLITS ===")
s1_train, s1_val, s1_test = make_splits(df, label_col="stage1_label")
save_splits(s1_train, s1_val, s1_test, prefix="data/processed/stage1")

# Stage 2 splits (specialty reports only)
print("\n\n=== STAGE 2 SPLITS ===")
specialty_df = df[df["is_specialty_report"]].copy()
s2_train, s2_val, s2_test = make_splits(specialty_df, label_col="stage2_label")
save_splits(s2_train, s2_val, s2_test, prefix="data/processed/stage2")

# Class weights for imbalance
print("\n\n=== STAGE 2 CLASS WEIGHTS ===")
weights = get_class_weights(s2_train["stage2_label"])
for label, w in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {label:25} weight: {w:.3f}")