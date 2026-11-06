
import os
import sys
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

# --------------------------------------------------
# Configuration
# --------------------------------------------------
SEED = 42
MAX_LEN = 260
BATCH_SIZE = 16
HEAD_BATCH_SIZE = 1024
EPOCHS = 15
PATIENCE = 4
LEARNING_RATE = 1e-3

DATA_DIR = "data/processed/opensplice_splicegpt_pairs"
OUT_DIR = "results/multi_gene/opensplice_paired_dnabert2"
CACHE_DIR = os.path.join(OUT_DIR, "embedding_cache")

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/"
    "snapshots/7bce263b15377fc15361f52cfab88f8b586abda0"
)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_splits():
    print("Loading data splits...", flush=True)
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    print(
        f"Train/validation/test: {len(train)} / {len(val)} / {len(test)}",
        flush=True,
    )
    return train, val, test


def configure_model():
    print("Loading DNABERT-2 locally...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    # DNABERT-2's bundled Triton attention is incompatible with the
    # installed Triton version. Activate the tested PyTorch fallback
    # before running any forward pass.
    attention_modules = []
    for name, module in list(sys.modules.items()):
        if name.endswith(".bert_layers") and hasattr(
            module, "flash_attn_qkvpacked_func"
        ):
            module.flash_attn_qkvpacked_func = None
            attention_modules.append(name)

    if not attention_modules:
        raise RuntimeError(
            "Could not find DNABERT-2's custom attention module. "
            "Stopping rather than risking the incompatible Triton path."
        )

    print("Using PyTorch attention fallback:", attention_modules, flush=True)

    model = model.to(DEVICE)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    return tokenizer, model


def cache_path(split_name):
    return os.path.join(CACHE_DIR, f"{split_name}_embeddings.npz")


def encode_sequences(sequences, tokenizer, model, cache_file):
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings: {cache_file}", flush=True)
        cached = np.load(cache_file)
        return cached["embeddings"]

    embeddings = []
    total = len(sequences)
    start = time.time()

    print(f"Extracting embeddings for {total:,} sequences...", flush=True)

    for start_idx in range(0, total, BATCH_SIZE):
        batch_sequences = sequences[start_idx:start_idx + BATCH_SIZE]

        tokens = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        tokens = {key: value.to(DEVICE) for key, value in tokens.items()}

        with torch.no_grad():
            output = model(**tokens)
            hidden = output[0] if isinstance(output, tuple) else output.last_hidden_state

            # Mean pool over real tokens only. DNABERT-2 uses a
            # subword tokenizer, so attention_mask is used for masking.
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        embeddings.append(pooled.cpu().numpy().astype(np.float32))

        done = min(start_idx + BATCH_SIZE, total)
        if done % 2048 == 0 or done == total:
            elapsed = time.time() - start
            print(
                f"  {done:,}/{total:,} sequences "
                f"({done / total * 100:.1f}%) | {elapsed / 60:.1f} min",
                flush=True,
            )

    result = np.concatenate(embeddings, axis=0)
    np.savez_compressed(cache_file, embeddings=result)
    print(f"Saved embeddings: {cache_file}", flush=True)
    return result


def paired_embeddings(df, tokenizer, model, split_name):
    wt_cache = os.path.join(CACHE_DIR, f"{split_name}_wt.npz")
    var_cache = os.path.join(CACHE_DIR, f"{split_name}_variant.npz")

    wt_embeddings = encode_sequences(
        df["wt_seq"].astype(str).tolist(),
        tokenizer,
        model,
        wt_cache,
    )
    var_embeddings = encode_sequences(
        df["variant_seq"].astype(str).tolist(),
        tokenizer,
        model,
        var_cache,
    )

    # Preserve the two sequence representations and their difference.
    X = np.concatenate(
        [wt_embeddings, var_embeddings, var_embeddings - wt_embeddings],
        axis=1,
    )
    return X.astype(np.float32)


class RegressionHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson_r": float(pearsonr(y_true, y_pred).statistic),
        "spearman_r": float(spearmanr(y_true, y_pred).statistic),
    }


def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=HEAD_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def predict(head, X):
    head.eval()
    predictions = []

    with torch.no_grad():
        for start_idx in range(0, len(X), HEAD_BATCH_SIZE):
            batch = torch.from_numpy(
                X[start_idx:start_idx + HEAD_BATCH_SIZE]
            ).to(DEVICE)
            predictions.append(head(batch).cpu().numpy())

    return np.concatenate(predictions)


def main():
    print("Device:", DEVICE, flush=True)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(DEVICE), flush=True)

    train, val, test = load_splits()
    tokenizer, encoder = configure_model()

    X_train = paired_embeddings(train, tokenizer, encoder, "train")
    X_val = paired_embeddings(val, tokenizer, encoder, "validation")
    X_test = paired_embeddings(test, tokenizer, encoder, "test")

    y_train = train["delta_psi"].to_numpy(dtype=np.float32)
    y_val = val["delta_psi"].to_numpy(dtype=np.float32)
    y_test = test["delta_psi"].to_numpy(dtype=np.float32)

    print("Paired feature shapes:", X_train.shape, X_val.shape, X_test.shape, flush=True)

    train_loader = make_loader(X_train, y_train, shuffle=True)

    head = RegressionHead(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    loss_fn = nn.SmoothL1Loss()

    best_val_mae = float("inf")
    best_epoch = 0
    patience_count = 0
    history = []

    checkpoint_path = os.path.join(OUT_DIR, "best_regression_head.pt")

    print(
        "Trainable head parameters:",
        sum(p.numel() for p in head.parameters() if p.requires_grad),
        flush=True,
    )

    for epoch in range(1, EPOCHS + 1):
        head.train()
        losses = []
        epoch_start = time.time()

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            pred = head(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_pred = predict(head, X_val)
        val_metrics = evaluate(y_val, val_pred)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_pearson_r": val_metrics["pearson_r"],
            "seconds": time.time() - epoch_start,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"val_MAE={row['val_mae']:.4f} | "
            f"val_RMSE={row['val_rmse']:.4f} | "
            f"val_r={row['val_pearson_r']:.4f}",
            flush=True,
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            patience_count = 0
            torch.save(head.state_dict(), checkpoint_path)
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("Early stopping.", flush=True)
            break

    pd.DataFrame(history).to_csv(
        os.path.join(OUT_DIR, "training_history.csv"), index=False
    )

    head.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    test_pred = predict(head, X_test)
    test_metrics = evaluate(y_test, test_pred)

    results = {
        "model": "DNABERT-2 frozen encoder with paired regression head",
        "best_epoch": best_epoch,
        "target": "delta_psi",
        "n_train": len(train),
        "n_validation": len(val),
        "n_test": len(test),
        "n_test_genes": int(test["gene"].nunique()),
        "embedding_dim_per_sequence": int(X_train.shape[1] // 3),
        "test_metrics": test_metrics,
    }

    pd.DataFrame({
        "gene": test["gene"].values,
        "variant_id": test["variant_id"].values,
        "mut_type": test["mut_type"].values,
        "y_true_delta_psi": y_test,
        "y_pred_delta_psi": test_pred,
    }).to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    with open(os.path.join(OUT_DIR, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nPAIRED DNABERT-2 HELD-OUT TEST RESULTS", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print("Saved to:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
