
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

SEED = 42
MAX_LEN = 260
BATCH_SIZE = 128
EPOCHS = 15
PATIENCE = 4
LR = 5e-4

MODEL_PATH = "LongSafari/hyenadna-small-32k-seqlen-hf"
DATA_DIR = Path("data/processed/opensplice_splicegpt_pairs")
OUT_DIR = Path("results/multi_gene/opensplice_paired_hyenadna")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True
)
PAD_ID = tokenizer.pad_token_id


def tokenize_sequence(seq):
    encoded = tokenizer(
        str(seq).upper(),
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )
    return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)


class PairDataset(Dataset):
    def __init__(self, frame, mean, std):
        wt_ids, wt_masks, var_ids, var_masks = [], [], [], []

        for wt, var in zip(frame["wt_seq"], frame["variant_seq"]):
            ids, mask = tokenize_sequence(wt)
            wt_ids.append(ids.numpy())
            wt_masks.append(mask.numpy())

            ids, mask = tokenize_sequence(var)
            var_ids.append(ids.numpy())
            var_masks.append(mask.numpy())

        self.wt = torch.tensor(np.array(wt_ids), dtype=torch.long)
        self.wt_mask = torch.tensor(np.array(wt_masks), dtype=torch.long)
        self.var = torch.tensor(np.array(var_ids), dtype=torch.long)
        self.var_mask = torch.tensor(np.array(var_masks), dtype=torch.long)
        self.y = torch.tensor(
            (frame["delta_psi"].to_numpy(dtype=np.float32) - mean) / std,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.wt[idx], self.wt_mask[idx],
            self.var[idx], self.var_mask[idx], self.y[idx]
        )


class FrozenPairedHyenaDNA(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden = self.backbone.config.d_model
        self.regressor = nn.Sequential(
            nn.Linear(hidden * 3, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def encode(self, ids, mask):
        # HyenaDNA's AutoModel forward accepts input_ids, not attention_mask.
        hidden = self.backbone(input_ids=ids).last_hidden_state
        valid = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(self, wt, wt_mask, var, var_mask):
        wt_emb = self.encode(wt, wt_mask)
        var_emb = self.encode(var, var_mask)
        features = torch.cat(
            [wt_emb, var_emb, var_emb - wt_emb], dim=1
        )
        return self.regressor(features).squeeze(-1)


def evaluate(model, loader, mean, std):
    model.eval()
    predictions, observed = [], []

    with torch.no_grad():
        for wt, wt_mask, var, var_mask, y in loader:
            pred = model(
                wt.to(DEVICE), wt_mask.to(DEVICE),
                var.to(DEVICE), var_mask.to(DEVICE)
            )
            predictions.extend((pred.cpu().numpy() * std + mean).tolist())
            observed.extend((y.numpy() * std + mean).tolist())

    pred = np.asarray(predictions)
    true = np.asarray(observed)

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

    print("Train/validation/test:", len(train_df), len(val_df), len(test_df))
    print("Target mean/std:", round(mean, 4), round(std, 4))
    print("Tokenizing sequences...")

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

    model = FrozenPairedHyenaDNA().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.regressor.parameters(), lr=LR, weight_decay=1e-4
    )
    loss_fn = nn.SmoothL1Loss()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total)
    print("Trainable parameters:", trainable)

    best_mae = float("inf")
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.backbone.eval()
        losses = []

        for wt, wt_mask, var, var_mask, y in train_loader:
            wt = wt.to(DEVICE)
            wt_mask = wt_mask.to(DEVICE)
            var = var.to(DEVICE)
            var_mask = var_mask.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            pred = model(wt, wt_mask, var, var_mask)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.regressor.parameters(), 1.0)
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
            torch.save(model.regressor.state_dict(), OUT_DIR / "best_regressor.pt")
        else:
            patience += 1

        if patience >= PATIENCE:
            print("Early stopping.")
            break

    model.regressor.load_state_dict(torch.load(
        OUT_DIR / "best_regressor.pt",
        map_location=DEVICE, weights_only=True
    ))

    test_metrics = evaluate(model, test_loader, mean, std)

    predictions, observed = [], []
    model.eval()
    with torch.no_grad():
        for wt, wt_mask, var, var_mask, y in test_loader:
            pred = model(
                wt.to(DEVICE), wt_mask.to(DEVICE),
                var.to(DEVICE), var_mask.to(DEVICE)
            )
            predictions.extend((pred.cpu().numpy() * std + mean).tolist())
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
            "target_std": std,
            "backbone_frozen": True
        }, f, indent=2)

    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
    print("\nPAIRED HYENADNA HELD-OUT TEST RESULTS")
    print(json.dumps({
        "best_epoch": best_epoch,
        "test_metrics": test_metrics
    }, indent=2))
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
