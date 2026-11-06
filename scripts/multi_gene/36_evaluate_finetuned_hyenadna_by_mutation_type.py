import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_FILES = {
    "HyenaDNA_finetuned": (
        "results/multi_gene/opensplice_paired_hyenadna_partial_ft/"
        "test_predictions.csv"
    ),
    "HyenaDNA_frozen": (
        "results/multi_gene/opensplice_paired_hyenadna/test_predictions.csv"
    ),
    "LSTM": "results/multi_gene/opensplice_paired_lstm/test_predictions.csv",
    "CNN": "results/multi_gene/opensplice_paired_cnn/test_predictions.csv",
}

OUT_DIR = "results/multi_gene/mutation_type_evaluation_finetuned"
os.makedirs(OUT_DIR, exist_ok=True)

results = []
reference_ids = None

for model_name, path in MODEL_FILES.items():
    if not os.path.exists(path):
        print(f"Skipping {model_name}: file not found: {path}")
        continue

    df = pd.read_csv(path)

    required = [
        "variant_id", "mut_type", "delta_psi", "predicted_delta_psi"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Skipping {model_name}: missing columns {missing}")
        continue

    df = df[required].copy()
    df = df.rename(columns={
        "delta_psi": "y_true",
        "predicted_delta_psi": "y_pred",
    })

    if df["variant_id"].duplicated().any():
        raise ValueError(f"{model_name} has duplicate variant IDs.")

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["variant_id", "mut_type", "y_true", "y_pred"])

    ids = set(df["variant_id"])
    if reference_ids is None:
        reference_ids = ids
    elif ids != reference_ids:
        raise ValueError(
            f"{model_name} does not contain exactly the same variant IDs "
            "as the other models. Check the test splits before comparing."
        )

    for mut_type, group in df.groupby("mut_type"):
        y_true = group["y_true"].to_numpy()
        y_pred = group["y_pred"].to_numpy()

        pearson = (
            pearsonr(y_true, y_pred).statistic
            if len(group) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0
            else np.nan
        )
        spearman = (
            spearmanr(y_true, y_pred).statistic
            if len(group) > 1
            and group["y_true"].nunique() > 1
            and group["y_pred"].nunique() > 1
            else np.nan
        )

        results.append({
            "model": model_name,
            "mutation_type": mut_type,
            "n_variants": len(group),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "Pearson_r": pearson,
            "Spearman_r": spearman,
        })

results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError("No evaluation results were generated.")

results_df = results_df.sort_values(
    ["mutation_type", "MAE"]
).reset_index(drop=True)

metrics_path = os.path.join(OUT_DIR, "mutation_type_metrics.csv")
results_df.to_csv(metrics_path, index=False)

mae_table = results_df.pivot(
    index="mutation_type",
    columns="model",
    values="MAE",
)
mae_path = os.path.join(OUT_DIR, "mutation_type_mae_comparison.csv")
mae_table.to_csv(mae_path)

print("\n=== Mutation-type evaluation ===")
print(results_df.round(4).to_string(index=False))

print("\n=== MAE comparison (lower is better) ===")
print(mae_table.round(3).to_string())

print(f"\nSaved metrics: {metrics_path}")
print(f"Saved MAE table: {mae_path}")
print(f"Number of shared test variants: {len(reference_ids)}")
