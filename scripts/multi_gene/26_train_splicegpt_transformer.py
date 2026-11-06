import random
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SEED = 42
BATCH_SIZE = 256
EPOCHS = 12
LR = 3e-4
MAX_LEN = 260
D_MODEL = 96
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.15

DATA_DIR = Path("data/processed/opensplice_splicegpt_pairs")
OUT_DIR = Path("results/multi_gene/opensplice_splicegpt_transformer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# Token IDs: A=1, C=2, G=3, T=4, unknown=5, PAD=0
TOKEN_MAP = {"A": 1, "C": 2, "G": 3, "T": 4}
PAD_ID = 0
UNK_ID = 5

def encode_sequence(seq):
    seq = str(seq).upper()
    ids = [TOKEN_MAP.get(base, UNK_ID) for base in seq[:MAX_LEN]]
    ids += [PAD_ID] * (MAX_LEN - len(ids))
    return ids

class PairedSequenceDataset(Dataset):
    def __init__(self, frame, target_mean, target_std):
        self.wt = torch.tensor(
            np.array([encode_sequence(s) for s in frame["wt_seq"]]),
            dtype=torch.long
        )
        self.variant = torch.tensor(
            np.array([encode_sequence(s) for s in frame["variant_seq"]]),
            dtype=torch.long
        )
        self.y = torch.tensor(
            ((frame["delta_psi"].to_numpy(dtype=np.float32) - target_mean)
             / target_std),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.wt[idx], self.variant[idx], self.y[idx]

class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(6, D_MODEL, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(MAX_LEN, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=4 * D_MODEL,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=N_LAYERS,
            enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, tokens):
        pad_mask = tokens.eq(PAD_ID)
        positions = torch.arange(
            tokens.shape[1], device=tokens.device
        ).unsqueeze(0)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return pooled

class SpliceGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.head = nn.Sequential(
            nn.Linear(3 * D_MODEL, 192),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1)
        )

    def forward(self, wt, variant):
        wt_emb = self.encoder(wt)
        variant_emb = self.encoder(variant)
        features = torch.cat(
            [wt_emb, variant_emb, variant_emb - wt_emb], dim=1
        )
        return self.head(features).squeeze(-1)

def evaluate(model, loader, target_mean, target_std):
    model.eval()
    pred_norm, true_norm = [], []

    with torch.no_grad():
        for wt, variant, y in loader:
            wt = wt.to(DEVICE)
            variant = variant.to(DEVICE)
            pred = model(wt, variant)
            pred_norm.extend(pred.cpu().numpy())
            true_norm.extend(y.numpy())

    pred = np.asarray(pred_norm) * target_std + target_mean
    true = np.asarray(true_norm) * target_std + target_mean
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    corr = pearsonr(true, pred).statistic if np.std(pred) > 0 else 0.0
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson_r": float(corr)
    }

def main():
    print("Loading train/validation/test data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Statistics from training data only.
    target_mean = float(train_df["delta_psi"].mean())
    target_std = float(train_df["delta_psi"].std())
    if target_std <= 0:
        raise ValueError("Invalid training target standard deviation.")

    train_ds = PairedSequenceDataset(train_df, target_mean, target_std)
    val_ds = PairedSequenceDataset(val_df, target_mean, target_std)
    test_ds = PairedSequenceDataset(test_df, target_mean, target_std)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(DEVICE.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(DEVICE.type == "cuda")
    )

    model = SpliceGPT().to(DEVICE)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Target mean/std:", round(target_mean, 4), round(target_std, 4))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01
    )
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_mae = float("inf")
    best_epoch = 0
    history = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        epoch_start = time.time()

        for wt, variant, y in train_loader:
            wt = wt.to(DEVICE, non_blocking=True)
            variant = variant.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(wt, variant)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, target_mean, target_std)
        scheduler.step(val_metrics["mae"])

        record = {
            "epoch": epoch,
            "train_smooth_l1": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "seconds": round(time.time() - epoch_start, 1)
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={record['train_smooth_l1']:.4f} | "
            f"val_MAE={val_metrics['mae']:.4f} | "
            f"val_RMSE={val_metrics['rmse']:.4f} | "
            f"val_r={val_metrics['pearson_r']:.4f} | "
            f"{record['seconds']}s",
            flush=True
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
            with open(OUT_DIR / "model_config.json", "w") as f:
                json.dump({
                    "seed": SEED,
                    "max_len": MAX_LEN,
                    "d_model": D_MODEL,
                    "n_heads": N_HEADS,
                    "n_layers": N_LAYERS,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "best_epoch": best_epoch
                }, f, indent=2)

        pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)

    print("\nLoading best validation checkpoint from epoch", best_epoch)
    model.load_state_dict(torch.load(
        OUT_DIR / "best_model.pt", map_location=DEVICE, weights_only=True
    ))
    test_metrics = evaluate(model, test_loader, target_mean, target_std)

    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "test_metrics": test_metrics,
            "training_seconds": round(time.time() - start_time, 1)
        }, f, indent=2)

    print("\nFINAL HELD-OUT TEST RESULTS")
    print(json.dumps(test_metrics, indent=2))
    print("Results saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
