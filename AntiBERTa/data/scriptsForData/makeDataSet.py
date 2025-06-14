import pandas as pd
import random

# Load the input files
main_df = pd.read_csv("data/scriptsForData/mhc_ligand_MS(in).csv")
ref_df = pd.read_csv("data/scriptsForData/mhc_ligand_count_1000_MS(in).csv")

# Create a set of valid alleles from the reference file
valid_alleles = set(ref_df["MHC allele"])

# Prepare new rows
output_rows = []

for _, row in main_df.iterrows():
    allele = row["MHC allele"]
    length = int(row["Length"])
    peptide = row["Peptide"]

    # Skip if allele not found in reference
    if allele not in valid_alleles:
        continue

    # Save the original row
    output_rows.append({
        "HLA": allele,
        "Length": length,
        "peptide": peptide,
    })

# Convert to DataFrame and save
output_df = pd.DataFrame(output_rows)
output_df.to_csv("data/scriptsForData/fullDataSetIntermediate.csv", index=False)
