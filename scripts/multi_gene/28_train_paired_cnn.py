
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

SEED = 42
MAX_LEN = 260
BATCH_SIZE = 512
EPOCHS = 12
PATIENCE = 3
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data/processed/opensplice_splicegpt_pairs")
OUT_DIR = Path("results/multi_gene/opensplice_paired_cnn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

TOKEN_MAP = {"A": 1, "C": 2, "G": 3, "T": 4}
PAD_ID = 0
UNK_ID = 5


def encode_sequence(seq):
    ids = [TOKEN_MAP.get(b, UNK_ID) for b in str(seq).upper()[:MAX_LEN]]
    ids += [PAD_ID] * (MAX_LEN - len(ids))
    return ids


class PairDataset(Dataset):
    def __init__(self, df, mean, std):
        self.wt = torch.tensor(
            np.array([encode_sequence(s) for s in df["wt_seq"]]),
            dtype=torch.long
        )
        self.var = torch.tensor(
            np.array([encode_sequence(s) for s in df["variant_seq"]]),
            dtype=torch.long
        )
        self.y = torch.tensor(
            (df["delta_psi"].to_numpy(dtype=np.float32) - mean) / std,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.wt[idx], self.var[idx], self.y[idx]


class SharedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6, 32, padding_idx=PAD_ID)
        self.conv1 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(64)

    def forward(self, tokens):
        mask = tokens.ne(PAD_ID).unsqueeze(1).float()
        x = self.embedding(tokens).transpose(1, 2)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = x * mask
        pooled = x.sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return self.norm(pooled)


class PairedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedCNN()
        self.head = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, wt, var):
        a = self.encoder(wt)
        b = self.encoder(var)
        return self.head(torch.cat([a, b, b - a], dim=1)).squeeze(-1)


def evaluate(model, loader, mean, std):
    model.eval()
    pred, true = [], []

    with torch.no_grad():
        for wt, var, y in loader:
            out = model(wt.to(DEVICE), var.to(DEVICE))
            pred.extend((out.cpu().numpy() * std + mean).tolist())
            true.extend((y.numpy() * std + mean).tolist())

    pred = np.asarray(pred)
    true = np.asarray(true)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "pearson_r": float(pearsonr(true, pred).statistic)
            if np.std(pred) > 0 else None,
        "spearman_r": float(spearmanr(true, pred).statistic)
            if np.std(pred) > 0 else None
    }


def main():
    start = time.time()
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    mean = float(train_df["delta_psi"].mean())
    std = float(train_df["delta_psi"].std())
    print("Device:", DEVICE)
    print("Train/validation/test:", len(train_df), len(val_df), len(test_df))
    print("Target mean/std:", round(mean, 4), round(std, 4))

    train_ds = PairDataset(train_df, mean, std)
    val_ds = PairDataset(val_df, mean, std)
    test_ds = PairDataset(test_df, mean, std)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    model = PairedCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    best_mae = float("inf")
    best_epoch = 0
    patience = 0
    history = []

    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for wt, var, y in train_loader:
            wt = wt.to(DEVICE)
            var = var.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            pred = model(wt, var)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, mean, std)
        train_loss = float(np.mean(losses))
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_MAE={val_metrics['mae']:.4f} | "
            f"val_RMSE={val_metrics['rmse']:.4f} | "
            f"val_r={val_metrics['pearson_r']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
        else:
            patience += 1

        if patience >= PATIENCE:
            print("Early stopping.")
            break

    model.load_state_dict(torch.load(
        OUT_DIR / "best_model.pt", map_location=DEVICE, weights_only=True
    ))
    test_metrics = evaluate(model, test_loader, mean, std)

    predictions, observed = [], []
    model.eval()
    with torch.no_grad():
        for wt, var, y in test_loader:
            out = model(wt.to(DEVICE), var.to(DEVICE))
            predictions.extend((out.cpu().numpy() * std + mean).tolist())
            observed.extend((y.numpy() * std + mean).tolist())

    result_df = test_df.copy()
    result_df["predicted_delta_psi"] = predictions
    result_df["prediction_error"] = (
        result_df["predicted_delta_psi"] - np.asarray(observed)
    )
    result_df["absolute_error"] = np.abs(result_df["prediction_error"])
    result_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
            "training_seconds": round(time.time() - start, 1),
            "target_mean": mean,
            "target_std": std
        }, f, indent=2)

    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    print("\nCNN HELD-OUT TEST RESULTS")
    print(json.dumps({
        "best_epoch": best_epoch,
        "test_metrics": test_metrics
    }, indent=2))
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
