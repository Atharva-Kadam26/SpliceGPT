# SpliceGPT: Deep Learning-Based Prediction of Splicing Changes

## Overview

SpliceGPT is a comparative deep learning framework for predicting changes in exon inclusion following genetic variants.

The project investigates whether paired wild-type (WT) and variant DNA sequence representations can predict changes in Percent Spliced In (ΔPSI) using pretrained genomic foundation models, neural network architectures, and conventional machine learning approaches.

## Research Objective

To develop and benchmark computational models that predict changes in exon inclusion caused by sequence variants.

The prediction target is:

**ΔPSI = PSI_variant − PSI_wild-type**

This is formulated as a regression task.

## Dataset

The study uses experimentally derived OpenSplice data.

After preprocessing and quality checks, the paired dataset contains:

| Dataset split | Variants | Genes |
|---|---:|---:|
| Training | 409,084 | 326 |
| Validation | 88,581 | 70 |
| Testing | 92,439 | 70 |
| **Total** | **590,104** | **466** |

### Mutation categories

- Substitutions
- 1-nucleotide deletions
- 3-nucleotide deletions
- 6-nucleotide deletions
- 21-nucleotide deletions

### Data splitting

The dataset was split at the gene level using a 70:15:15 train-validation-test strategy with `random_state=42`.

Genes in the test set are not present in the training or validation sets, allowing evaluation on unseen genes.

Raw experimental data and processed datasets are not included in this repository.

See [data/README.md](data/README.md) for preprocessing details.

## Model Architectures

Eight model configurations were evaluated:

1. **Fine-tuned HyenaDNA** – A pretrained genomic foundation model with partial backbone fine-tuning.
2. **Frozen HyenaDNA** – A pretrained HyenaDNA backbone used without updating its parameters.
3. **DNABERT-2** – A pretrained DNA language model used for paired-sequence regression.
4. **SpliceGPT Transformer** – A custom encoder-only Transformer trained from scratch.
5. **CNN** – A convolutional neural network for paired DNA sequence modelling.
6. **LSTM** – A recurrent neural network for paired sequence representation learning.
7. **Ridge Regression** – A conventional regression baseline using nucleotide-frequency features.
8. **Random Forest** – An ensemble learning baseline using sequence-derived features.

For paired models, wild-type and variant sequences are represented jointly to learn their relationship.

See [models/README.md](models/README.md) for architecture and training details.

## Overall Results

Performance on the held-out test set:

| Model | MAE | RMSE | Pearson r |
|---|---:|---:|---:|
| Fine-tuned HyenaDNA | 9.990 | 15.574 | 0.745 |
| LSTM | 10.395 | 16.270 | 0.717 |
| CNN | 12.189 | 19.205 | 0.584 |
| Frozen HyenaDNA | 13.501 | 22.166 | 0.337 |
| Random Forest | 13.673 | 20.633 | 0.462 |
| SpliceGPT Transformer | 13.854 | 22.312 | 0.376 |
| Frozen DNABERT-2 | 13.868 | 23.291 | 0.222 |
| Ridge Regression | 15.895 | 23.267 | 0.015 |

Fine-tuned HyenaDNA achieved the strongest overall performance among the evaluated approaches.

These results suggest that partial fine-tuning of pretrained genomic representations can improve ΔPSI prediction compared with using a frozen backbone. The LSTM also performed competitively.

These findings represent predictive performance on the specified held-out dataset and do not establish causal biological relationships.

## Mutation-Wise Evaluation

The fine-tuned HyenaDNA model was additionally evaluated across mutation categories.

| Mutation type | Variants | MAE | RMSE | Pearson r |
|---|---:|---:|---:|---:|
| Substitution | 47,861 | 8.353 | 13.114 | 0.693 |
| 1-nt deletion | 11,195 | 7.771 | 12.400 | 0.704 |
| 3-nt deletion | 11,418 | 9.595 | 14.814 | 0.726 |
| 6-nt deletion | 11,234 | 11.673 | 17.228 | 0.744 |
| 21-nt deletion | 10,731 | 18.265 | 24.852 | 0.750 |

Prediction error generally increased with deletion length, indicating that larger sequence alterations presented a more challenging regression task.

## Repository Structure

```text
SpliceGPT/
├── data/
│   └── README.md
├── models/
│   └── README.md
├── results/
│   ├── overall_model_metrics.csv
│   ├── mutation_type_metrics.csv
│   ├── hyenadna_finetuning_training_history.csv
│   └── splicegpt_transformer_training_history.csv
├── scripts/
│   └── multi_gene/
│       ├── 25_prepare_splicegpt_pairs.py
│       ├── 26_train_splicegpt_transformer.py
│       ├── 28_train_paired_cnn.py
│       ├── 29_train_paired_hyenadna.py
│       ├── 30_train_paired_lstm.py
│       ├── 31_train_paired_ridge.py
│       ├── 32_train_paired_dnabert2.py
│       ├── 34_finetune_paired_hyenadna.py
│       ├── 35_train_paired_random_forest.py
│       └── 36_evaluate_finetuned_hyenadna_by_mutation_type.py
├── .gitignore
├── requirements.txt
└── README.md
## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/SpliceGPT.git
cd SpliceGPT
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch separately according to your CPU/CUDA environment.

## Reproducibility

The repository includes preprocessing, model training, and evaluation scripts.

To reproduce the experiments, obtain the original OpenSplice data and configure the input paths in the scripts.

Pretrained model weights may also need to be downloaded from their respective model repositories.

The results directory contains evaluation metrics and selected training histories.

## Limitations

- Model performance depends on the quality and coverage of the experimental dataset.
- Evaluation on held-out genes does not guarantee generalisation to every biological context.
- Larger deletions generally produced higher absolute prediction errors.
- Further validation using independent experimental datasets is necessary.

## Technologies

Python | PyTorch | Hugging Face Transformers | HyenaDNA | DNABERT-2 | NumPy | Pandas | Scikit-learn

## Research Focus

Computational Genomics | Alternative Splicing | Deep Learning | Genomic Foundation Models | Sequence-Based Regression
