
import os
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

BASE = "data/processed/opensplice_splicegpt_pairs"
OUT = "results/multi_gene/opensplice_paired_ridge"
os.makedirs(OUT, exist_ok=True)

BASES = "ACGTN"


def sequence_features(seq):
    seq = str(seq).upper()
    length = max(len(seq), 1)
    return [seq.count(base) / length for base in BASES]


def paired_features(df):
    rows = []
    for wt, var in zip(df["wt_seq"], df["variant_seq"]):
        wt_f = sequence_features(wt)
        var_f = sequence_features(var)
        rows.append(wt_f + var_f + [v - w for w, v in zip(wt_f, var_f)])
    return np.asarray(rows, dtype=np.float32)


def metrics(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "pearson_r": float(pearsonr(y, pred).statistic),
        "spearman_r": float(spearmanr(y, pred).statistic),
    }


print("Loading paired OpenSplice splits...", flush=True)
train = pd.read_csv(f"{BASE}/train.csv")
val = pd.read_csv(f"{BASE}/validation.csv")
test = pd.read_csv(f"{BASE}/test.csv")

print(
    f"Rows: train={len(train)}, validation={len(val)}, test={len(test)}",
    flush=True,
)

print("Extracting nucleotide-composition features...", flush=True)
X_train = paired_features(train)
X_val = paired_features(val)
X_test = paired_features(test)

y_train = train["delta_psi"].to_numpy(dtype=np.float32)
y_val = val["delta_psi"].to_numpy(dtype=np.float32)
y_test = test["delta_psi"].to_numpy(dtype=np.float32)

print("Selecting Ridge regularization using validation MAE...", flush=True)
best_model = None
best_alpha = None
best_val_mae = float("inf")

for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    print(f"alpha={alpha:g} | validation MAE={val_mae:.4f}", flush=True)

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_model = model
        best_alpha = alpha

print("Evaluating the selected model on the held-out test set...", flush=True)
test_pred = best_model.predict(X_test)

results = {
    "model": "Ridge Regression",
    "features": "WT and variant A/C/G/T/N nucleotide frequencies plus frequency differences",
    "target": "delta_psi",
    "best_alpha": best_alpha,
    "validation_mae": float(best_val_mae),
    "n_train": len(train),
    "n_validation": len(val),
    "n_test": len(test),
    "n_test_genes": int(test["gene"].nunique()),
    "test_metrics": metrics(y_test, test_pred),
}

pd.DataFrame({
    "gene": test["gene"],
    "variant_id": test["variant_id"],
    "mut_type": test["mut_type"],
    "y_true_delta_psi": y_test,
    "y_pred_delta_psi": test_pred,
}).to_csv(f"{OUT}/test_predictions.csv", index=False)

with open(f"{OUT}/test_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nRIDGE REGRESSION HELD-OUT TEST RESULTS", flush=True)
print(json.dumps(results, indent=2), flush=True)
print(f"Saved to: {OUT}", flush=True)
