import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# Load the training data
train_df = pd.read_csv("data/dataNew/trainData.csv")

# Create stratification key
train_df["stratify_key"] = train_df["HLA"].astype(str) + "_" + train_df["Length"].astype(str) + "_" + train_df["label"].astype(str)

# Set up StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratify_labels = train_df["stratify_key"]

for fold_idx, (train_index, val_index) in enumerate(skf.split(train_df, stratify_labels)):
    fold_data = train_df.iloc[val_index].copy()
    fold_data["stratify_key"] = fold_data["HLA"].astype(str) + "_" + fold_data["Length"].astype(str) + "_" + fold_data["label"].astype(str)

    # Count stratum occurrences in this fold
    fold_counts = fold_data["stratify_key"].value_counts()

    if (fold_counts < 2).any():
        print(f"⚠️ Fold {fold_idx}: Not all strata ≥2 samples. Doing random split.")
        fold_train, fold_val = train_test_split(fold_data, test_size=0.2, random_state=fold_idx)
    else:
        fold_train, fold_val = train_test_split(
            fold_data,
            test_size=0.2,
            stratify=fold_data["stratify_key"],
            random_state=fold_idx
        )

    # Drop helper column
    fold_train = fold_train.drop(columns=["stratify_key"])
    fold_val = fold_val.drop(columns=["stratify_key"])

    # Save to CSV
    fold_train.to_csv(f"data/dataNew/train_data_fold_{fold_idx}.csv", index=False)
    fold_val.to_csv(f"data/dataNew/val_data_fold_{fold_idx}.csv", index=False)

    print(f"✅ Fold {fold_idx} saved: train = {len(fold_train)}, val = {len(fold_val)}")
