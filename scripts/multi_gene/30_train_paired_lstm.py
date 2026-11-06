
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
EPOCHS = 10
PATIENCE = 3
LR = 1e-3

DATA_DIR = Path("data/processed/opensplice_splicegpt_pairs")
OUT_DIR = Path("results/multi_gene/opensplice_paired_lstm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKEN_MAP = {"A": 1, "C": 2, "G": 3, "T": 4}
PAD_ID = 0
UNK_ID = 5


def encode_sequence(seq):
    ids = [TOKEN_MAP.get(b, UNK_ID) for b in str(seq).upper()[:MAX_LEN]]
    ids += [PAD_ID] * (MAX_LEN - len(ids))
    return ids


class PairDataset(Dataset):
    def __init__(self, df, mean, std):
        # uint8 keeps CPU memory use low; convert to long in each batch.
        self.wt = torch.tensor(
            np.array([encode_sequence(s) for s in df["wt_seq"]]),
            dtype=torch.uint8
        )
        self.var = torch.tensor(
            np.array([encode_sequence(s) for s in df["variant_seq"]]),
            dtype=torch.uint8
        )
        self.y = torch.tensor(
            (df["delta_psi"].to_numpy(dtype=np.float32) - mean) / std,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.wt[idx], self.var[idx], self.y[idx]


class SharedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6, 32, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(64)

    def forward(self, tokens):
        mask = tokens.ne(PAD_ID).unsqueeze(-1).float()
        x = self.embedding(tokens)
        x, _ = self.lstm(x)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)


class PairedLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedLSTM()
        self.regressor = nn.Sequential(
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
        features = torch.cat([a, b, b - a], dim=1)
        return self.regressor(features).squeeze(-1)


def evaluate(model, loader, mean, std):
    model.eval()
    preds, actual = [], []

    with torch.no_grad():
        for wt, var, y in loader:
            pred = model(
                wt.to(DEVICE, dtype=torch.long),
                var.to(DEVICE, dtype=torch.long)
            )
            preds.extend((pred.cpu().numpy() * std + mean).tolist())
            actual.extend((y.numpy() * std + mean).tolist())

    preds = np.asarray(preds)
    actual = np.asarray(actual)

    return {
        "mae": float(mean_absolute_error(actual, preds)),
        "rmse": float(np.sqrt(mean_squared_error(actual, preds))),
        "pearson_r": float(pearsonr(actual, preds).statistic)
            if np.std(preds) > 0 else None,
        "spearman_r": float(spearmanr(actual, preds).statistic)
            if np.std(preds) > 0 else None
    }


def main():
    start = time.time()
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    mean = float(train_df["delta_psi"].mean())
    std = float(train_df["delta_psi"].std())

    print("Device:", DEVICE, flush=True)
    print("Train/validation/test:",
          len(train_df), len(val_df), len(test_df), flush=True)
    print("Preparing compact sequence arrays...", flush=True)

    train_ds = PairDataset(train_df, mean, std)
    val_ds = PairDataset(val_df, mean, std)
    test_ds = PairDataset(test_df, mean, std)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = PairedLSTM().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    loss_fn = nn.SmoothL1Loss()

    print("Trainable parameters:",
          sum(p.numel() for p in model.parameters()), flush=True)

    best_mae = float("inf")
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for wt, var, y in train_loader:
            wt = wt.to(DEVICE, dtype=torch.long)
            var = var.to(DEVICE, dtype=torch.long)
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
            f"val_r={val_metrics['pearson_r']:.4f}",
            flush=True
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
            print("Early stopping.", flush=True)
            break

    model.load_state_dict(torch.load(
        OUT_DIR / "best_model.pt",
        map_location=DEVICE,
        weights_only=True
    ))

    test_metrics = evaluate(model, test_loader, mean, std)

    preds, actual = [], []
    model.eval()
    with torch.no_grad():
        for wt, var, y in test_loader:
            out = model(
                wt.to(DEVICE, dtype=torch.long),
                var.to(DEVICE, dtype=torch.long)
            )
            preds.extend((out.cpu().numpy() * std + mean).tolist())
            actual.extend((y.numpy() * std + mean).tolist())

    output = test_df.copy()
    output["predicted_delta_psi"] = preds
    output["prediction_error"] = (
        output["predicted_delta_psi"] - np.asarray(actual)
    )
    output["absolute_error"] = np.abs(output["prediction_error"])
    output.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
            "training_seconds": round(time.time() - start, 1),
            "target_mean": mean,
            "target_std": std
        }, f, indent=2)

    pd.DataFrame(history).to_csv(
        OUT_DIR / "training_history.csv", index=False
    )

    print("\nPAIRED LSTM HELD-OUT TEST RESULTS", flush=True)
    print(json.dumps({
        "best_epoch": best_epoch,
        "test_metrics": test_metrics
    }, indent=2), flush=True)
    print("Saved to:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
