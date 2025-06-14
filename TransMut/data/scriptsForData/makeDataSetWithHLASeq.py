import pandas as pd

# Load peptide data
peptide_df = pd.read_csv("data/dataNew/fullDataSetIntermediate.csv")

# Load pseudo sequence data
pseudo_df = pd.read_csv("data/dataNew/pseudoSequence(ELIM).csv")

# Clean column names (if needed)
pseudo_df.columns = [col.strip() for col in pseudo_df.columns]

# Ensure strings
pseudo_df["HLA"] = pseudo_df["HLA"].astype(str)
pseudo_map = dict(zip(pseudo_df["HLA"], pseudo_df["pseudoSeq"]))

# Function to convert HLA name to pseudo format
def convert_hla_format(hla_name):
    return hla_name.replace("*", "").replace(":", "")

# Convert HLA and map to sequence
peptide_df["HLA_pseudo_name"] = peptide_df["HLA"].apply(convert_hla_format)
peptide_df["HLA_sequence"] = peptide_df["HLA_pseudo_name"].map(pseudo_map)

# Check 1: Missing HLA sequences
missing_hlas = peptide_df[peptide_df["HLA_sequence"].isnull()]["HLA"].unique()
if len(missing_hlas) > 0:
    raise ValueError(f"Error: The following HLA alleles were not found in the pseudo-sequence file:\n{missing_hlas}")

# Check 2: Invalid amino acids in peptide
valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
invalid_peptides = peptide_df[~peptide_df["peptide"].apply(lambda pep: set(pep).issubset(valid_aas))]
if not invalid_peptides.empty:
    bad_rows = invalid_peptides[["HLA", "peptide"]]
    raise ValueError(f"Error: The following peptides contain invalid amino acids:\n{bad_rows.to_string(index=False)}")

# Drop helper column and save
peptide_df.drop(columns=["HLA_pseudo_name"], inplace=True)
peptide_df.to_csv("data/dataNew/fullDataSetNew.csv", index=False)