import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

MASTER = Path(
    "data/external/OpenSplice/downloads/"
    "open_splice_experimental_master_unique_sequence_all_predictor_cols.tsv.gz"
)
METADATA = Path(
    "data/external/OpenSplice/libraries_design/"
    "02_mutagenesis_libraries/exon_list_with_metadata.tsv"
)
OUT = Path("data/processed/opensplice_splicegpt_pairs")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42

print("Loading OpenSplice variant data...")
cols = [
    "gene", "exon_id", "variant_id", "nt_seq",
    "mut_type", "region", "lib_id", "psi",
    "wt_psi", "delta_psi"
]
df = pd.read_csv(MASTER, sep="\t", usecols=cols)

print("Loading wild-type sequence metadata...")
meta = pd.read_csv(
    METADATA,
    sep="\t",
    usecols=["exon_id", "wildtype_sequence"]
).drop_duplicates("exon_id")

df = df.merge(meta, on="exon_id", how="left", validate="many_to_one")

if df["wildtype_sequence"].isna().any():
    raise ValueError("Some variants have no matching WT sequence.")

def reconstruct(row):
    wt = str(row["wildtype_sequence"]).upper()
    observed = str(row["nt_seq"]).upper()
    variant_id = str(row["variant_id"])
    mutation = variant_id.rsplit("_", 1)[-1]

    if "_del" in variant_id:
        match = re.fullmatch(r"del(\d+)to(\d+)", mutation)
        if not match:
            raise ValueError(f"Cannot parse deletion ID: {variant_id}")
        start, end = map(int, match.groups())
        predicted = wt[:start - 1] + wt[end:]
    else:
        match = re.fullmatch(r"([ACGT])(\d+)([ACGT])", mutation)
        if not match:
            raise ValueError(f"Cannot parse substitution ID: {variant_id}")
        ref, pos, alt = match.groups()
        pos = int(pos)

        if pos < 1 or pos > len(wt) or wt[pos - 1] != ref:
            raise ValueError(f"WT allele mismatch: {variant_id}")

        predicted = wt[:pos - 1] + alt + wt[pos:]

    if predicted != observed:
        raise ValueError(f"Reconstruction mismatch: {variant_id}")

    return predicted

print("Validating and reconstructing sequences...")
df["variant_seq"] = [
    reconstruct(row)
    for row in df[
        ["wildtype_sequence", "nt_seq", "variant_id"]
    ].rename(columns={
        "wildtype_sequence": "wildtype_sequence",
        "nt_seq": "nt_seq",
        "variant_id": "variant_id"
    }).to_dict("records")
]

# The target is experimentally measured PSI change in percentage points.
df["delta_psi"] = pd.to_numeric(df["delta_psi"], errors="raise")
df["psi"] = pd.to_numeric(df["psi"], errors="raise")
df["wt_psi"] = pd.to_numeric(df["wt_psi"], errors="raise")

if not np.allclose(
    df["psi"] - df["wt_psi"], df["delta_psi"], atol=0.01
):
    raise ValueError("PSI consistency check failed.")

# Use genes as the unit of splitting to prevent gene overlap.
genes = np.array(sorted(df["gene"].dropna().unique()))
train_genes, remaining_genes = train_test_split(
    genes, test_size=0.30, random_state=SEED
)
val_genes, test_genes = train_test_split(
    remaining_genes, test_size=0.50, random_state=SEED
)

split_map = {g: "train" for g in train_genes}
split_map.update({g: "validation" for g in val_genes})
split_map.update({g: "test" for g in test_genes})
df["split"] = df["gene"].map(split_map)

# Keep only the columns required for modeling and analysis.
result = df[[
    "gene", "exon_id", "variant_id", "mut_type", "region", "lib_id",
    "wildtype_sequence", "variant_seq", "psi", "wt_psi", "delta_psi",
    "split"
]].rename(columns={"wildtype_sequence": "wt_seq"})

# Save separate splits. Existing benchmark files are untouched.
for split in ["train", "validation", "test"]:
    subset = result[result["split"] == split]
    subset.to_csv(OUT / f"{split}.csv", index=False)
    print(f"{split}: {len(subset):,} rows, "
          f"{subset['gene'].nunique():,} genes")

# Save a compact summary for checking the dataset.
summary = result.groupby(["split", "mut_type"]).agg(
    rows=("variant_id", "size"),
    genes=("gene", "nunique"),
    mean_delta_psi=("delta_psi", "mean"),
    sd_delta_psi=("delta_psi", "std"),
).reset_index()
summary.to_csv(OUT / "split_summary.csv", index=False)

assert set(train_genes).isdisjoint(set(val_genes))
assert set(train_genes).isdisjoint(set(test_genes))
assert set(val_genes).isdisjoint(set(test_genes))
assert len(result) == len(df)

print("\nMutation types:")
print(result["mut_type"].value_counts().to_string())
print("\nTarget summary:")
print(result["delta_psi"].describe().to_string())
print("\nGene overlap check: PASS")
print("Reconstruction check: PASS for every row")
print("Output directory:", OUT)
print("Summary file:", OUT / "split_summary.csv")
