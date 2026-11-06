
import json
import time
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

SEED = 42
N_TREES = 100
MAX_DEPTH = 18
MIN_SAMPLES_LEAF = 5
N_JOBS = 2

DATA_DIR = Path("data/processed/opensplice_splicegpt_pairs")
OUT_DIR = Path("results/multi_gene/opensplice_paired_random_forest")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASES = "ACGT"
KMER_LISTS = {
    k: ["".join(x) for x in product(BASES, repeat=k)]
    for k in (1, 2, 3)
}
KMER_INDEX = {
    k: {mer: i for i, mer in enumerate(KMER_LISTS[k])}
    for k in (1, 2, 3)
}
N_KMERS = sum(len(KMER_LISTS[k]) for k in (1, 2, 3))


def sequence_features(seq):
    seq = str(seq).upper()
    features = []

    # Normalized mono-, di-, and trinucleotide frequencies.
    for k in (1, 2, 3):
        counts = Counter(
            seq[i:i + k]
            for i in range(max(0, len(seq) - k + 1))
            if all(base in BASES for base in seq[i:i + k])
        )

        total = sum(counts.values())
        values = np.zeros(len(KMER_LISTS[k]), dtype=np.float32)

        if total:
            for mer, count in counts.items():
                values[KMER_INDEX[k][mer]] = count / total

        features.extend(values)

    return np.asarray(features, dtype=np.float32)


def paired_features(frame):
    n = len(frame)
    X = np.empty((n, N_KMERS * 3 + 3), dtype=np.float32)

    for i, (wt, var) in enumerate(
        zip(frame["wt_seq"], frame["variant_seq"])
    ):
        wt = str(wt).upper()
        var = str(var).upper()

        wt_features = sequence_features(wt)
        var_features = sequence_features(var)

        X[i, :N_KMERS] = wt_features
        X[i, N_KMERS:2 * N_KMERS] = var_features
        X[i, 2 * N_KMERS:3 * N_KMERS] = var_features - wt_features

        X[i, 3 * N_KMERS] = len(wt)
        X[i, 3 * N_KMERS + 1] = len(var)
        X[i, 3 * N_KMERS + 2] = len(var) - len(wt)

        if (i + 1) % 50000 == 0:
            print(f"  Featurized {i + 1:,}/{n:,} sequence pairs", flush=True)

    return X


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson_r": float(pearsonr(y_true, y_pred).statistic)
            if np.std(y_true) > 0 and np.std(y_pred) > 0 else None,
        "spearman_r": float(spearmanr(y_true, y_pred).statistic)
            if np.std(y_true) > 0 and np.std(y_pred) > 0 else None,
    }


def main():
    start = time.time()

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print("Train/validation/test:",
          len(train_df), len(val_df), len(test_df), flush=True)
    print("Feature count:", N_KMERS * 3 + 3, flush=True)

    print("\nBuilding training features...", flush=True)
    X_train = paired_features(train_df)
    y_train = train_df["delta_psi"].to_numpy(dtype=np.float32)

    print("\nBuilding validation features...", flush=True)
    X_val = paired_features(val_df)
    y_val = val_df["delta_psi"].to_numpy(dtype=np.float32)

    print("\nBuilding test features...", flush=True)
    X_test = paired_features(test_df)
    y_test = test_df["delta_psi"].to_numpy(dtype=np.float32)

    print("\nTraining Random Forest...", flush=True)
    model = RandomForestRegressor(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=0.7,
        n_jobs=N_JOBS,
        random_state=SEED,
        verbose=1,
    )
    model.fit(X_train, y_train)

    print("\nValidation metrics:")
    val_pred = model.predict(X_val)
    val_metrics = evaluate(y_val, val_pred)
    print(json.dumps(val_metrics, indent=2))

    # Test set is evaluated after model fitting, with no test-based tuning.
    print("\nHeld-out test metrics:")
    test_pred = model.predict(X_test)
    test_metrics = evaluate(y_test, test_pred)
    print(json.dumps(test_metrics, indent=2))

    pred_df = test_df.copy()
    pred_df["y_true_delta_psi"] = y_test
    pred_df["y_pred_delta_psi"] = test_pred
    pred_df["prediction_error"] = test_pred - y_test
    pred_df["absolute_error"] = np.abs(test_pred - y_test)
    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    metrics = {
        "model": "Paired-sequence Random Forest",
        "features": (
            "WT and variant mono/di/trinucleotide frequencies, "
            "frequency differences, and sequence lengths"
        ),
        "n_features": int(X_train.shape[1]),
        "n_estimators": N_TREES,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "n_jobs": N_JOBS,
        "n_train": len(train_df),
        "n_validation": len(val_df),
        "n_test": len(test_df),
        "n_test_genes": int(test_df["gene"].nunique()),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "elapsed_seconds": round(time.time() - start, 2),
    }

    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved predictions and metrics to:", OUT_DIR)
    print("Elapsed minutes:", round((time.time() - start) / 60, 2))


if __name__ == "__main__":
    main()
