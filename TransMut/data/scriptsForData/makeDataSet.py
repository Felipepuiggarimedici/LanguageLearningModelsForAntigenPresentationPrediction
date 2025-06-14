import pandas as pd
import random

# Load proteome sequence from FASTA
def load_proteome_sequence(fasta_path):
    sequence = ""
    with open(fasta_path, "r") as f:
        for line in f:
            if not line.startswith(">"):
                sequence += line.strip()
    return sequence

proteome = load_proteome_sequence("data/dataNew/UP000005640_9606.fasta")  # Replace with your actual file

# Load the input files
main_df = pd.read_csv("data/dataNew/mhc_ligand_MS(in).csv")
ref_df = pd.read_csv("data/dataNew/mhc_ligand_count_1000_MS(in).csv")

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
        "label": 1
    })

    # Create random peptide from proteome
    if length <= len(proteome):
        start = random.randint(0, len(proteome) - length)
        random_peptide = proteome[start:start + length]
        while ("X" in random_peptide or "x" in random_peptide or "U" in random_peptide):
                start = random.randint(0, len(proteome) - length)
                random_peptide = proteome[start:start + length]
        output_rows.append({
            "HLA": allele,
            "Length": length,
            "peptide": random_peptide,"label": 0
        })

# Convert to DataFrame and save
output_df = pd.DataFrame(output_rows)
output_df.to_csv("data/dataNew/fullDataSetIntermediate.csv", index=False)
