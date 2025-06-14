import pandas as pd

# Load your peptide dataset
df = pd.read_csv("data/scriptsForData/fullDataSetNew.csv")  # replace with your file path

# Calculate the length of each peptide
df["peptide_length"] = df["peptide"].apply(len)

# Find the maximum peptide length
max_length = df["peptide_length"].max()

print(f"Maximum peptide length: {max_length}")