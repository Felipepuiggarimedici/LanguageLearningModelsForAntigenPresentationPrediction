import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full dataset
df = pd.read_csv("data/scriptsForData/fullDataSetNew.csv")

# Create a composite key for stratification
df["stratify_key"] = df["HLA"].astype(str) + "_" + df["Length"].astype(str) 

# Count how many times each stratum appears
stratum_counts = df["stratify_key"].value_counts()

# Check if all strata have at least 2 entries
if (stratum_counts < 2).any():
    print("⚠️ Warning: Some strata have fewer than 2 samples. Performing random split without stratification.")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
else:
    # Safe to perform stratified split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["stratify_key"],
        random_state=42
    )

# Drop the helper column
train_df = train_df.drop(columns=["stratify_key"])
test_df = test_df.drop(columns=["stratify_key"])

# Save the result
train_df.to_csv("data/trainData/trainData.csv", index=False)
test_df.to_csv("data/testData/testData.csv", index=False)
