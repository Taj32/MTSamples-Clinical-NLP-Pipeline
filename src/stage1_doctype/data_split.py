import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test split.
    Drops any class with fewer than 3 samples (can't stratify).
    Returns (train_df, val_df, test_df).
    """
    # Drop classes too small to stratify
    counts = df[label_col].value_counts()
    valid_labels = counts[counts >= 3].index
    dropped = counts[counts < 3]
    if len(dropped) > 0:
        print(f"Dropping {len(dropped)} classes with <3 samples: {list(dropped.index)}")
    df = df[df[label_col].isin(valid_labels)].copy()

    # First split off test set
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state,
    )

    # Then split val from train
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val[label_col],
        random_state=random_state,
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train):4d} ({len(train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val):4d} ({len(val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test):4d} ({len(test)/len(df)*100:.1f}%)")

    return train, val, test


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    prefix: str = "data/processed/stage1",
) -> None:
    """Save splits to CSV."""
    train.to_csv(f"{prefix}_train.csv", index=True)
    val.to_csv(f"{prefix}_val.csv", index=True)
    test.to_csv(f"{prefix}_test.csv", index=True)
    print(f"\nSaved splits to {prefix}_train/val/test.csv")