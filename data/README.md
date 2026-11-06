# Dataset Documentation

## Dataset Source

This project uses experimental splicing data from OpenSplice.

The original data consists of experimentally measured variants, exon metadata,
wild-type DNA sequences and PSI measurements.

The raw dataset files are not included in this repository.

## Dataset Preparation

The paired-sequence dataset is generated using:

`scripts/multi_gene/25_prepare_splicegpt_pairs.py`

The preparation workflow:

1. Loads experimental variant information.
2. Retrieves corresponding wild-type exon sequences.
3. Reconstructs variant sequences from mutation identifiers.
4. Validates reconstructed sequences against observed sequences.
5. Checks consistency between PSI, wild-type PSI and delta PSI.
6. Splits genes into training, validation and test partitions.

## Dataset Structure

The processed dataset contains the following columns:

| Column | Description |
|---|---|
| gene | Gene identifier |
| exon_id | Exon identifier |
| variant_id | Variant identifier |
| mut_type | Mutation category |
| region | Genomic region |
| lib_id | Experimental library |
| wt_seq | Wild-type DNA sequence |
| variant_seq | Mutated DNA sequence |
| psi | Variant PSI |
| wt_psi | Wild-type PSI |
| delta_psi | Change in PSI |
| split | Dataset partition |

## Mutation Categories

- sub: Nucleotide substitution
- Δ1nt: 1-nucleotide deletion
- Δ3nt: 3-nucleotide deletion
- Δ6nt: 6-nucleotide deletion
- Δ21nt: 21-nucleotide deletion

## Dataset Splits

| Partition | Variants | Genes |
|---|---:|---:|
| Training | 409,084 | 326 |
| Validation | 88,581 | 70 |
| Test | 92,439 | 70 |

Genes are assigned to mutually exclusive partitions using random seed 42.

## Prediction Target

The regression target is:

delta_psi = psi - wt_psi

The target represents the experimentally measured change in PSI.

## Reproducing the Dataset

Obtain the original OpenSplice data and place the required source files in the paths specified in the preparation script.

Then execute:

`python scripts/multi_gene/25_prepare_splicegpt_pairs.py`

The script generates train.csv, validation.csv, test.csv and split_summary.csv.

## Data Availability

Please consult the original OpenSplice data source and its licensing or access conditions before obtaining or redistributing the raw data.
