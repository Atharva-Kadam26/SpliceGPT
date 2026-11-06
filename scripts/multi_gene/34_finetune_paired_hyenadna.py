
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
BATCH_SIZE = 32
EPOCHS = 10
PATIENCE = 3

BACKBONE_LR = 1e-5
HEAD_LR = 1e-3

MODEL_PATH = "LongSafari/hyenadna-small-32k-seqlen-hf"
DATA_DIR = Path("data/processed/opensplice_splicegpt_pairs")
OUT_DIR = Path("results/multi_gene/opensplice_paired_hyenadna_partial_ft")
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


def tokenize_sequence(seq):
    encoded = tokenizer(
        str(seq).upper(),
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    return (
        encoded["input_ids"].squeeze(0),
        encoded["attention_mask"].squeeze(0),
    )


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
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.wt[idx], self.wt_mask[idx],
            self.var[idx], self.var_mask[idx], self.y[idx],
        )


class PartialFTPairedHyenaDNA(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )

        # Freeze the entire backbone first.
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze only the final two HyenaDNA blocks and final norm.
        for layer in self.backbone.backbone.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.backbone.backbone.ln_f.parameters():
            param.requires_grad = True

        hidden = self.backbone.config.d_model
        self.regressor = nn.Sequential(
            nn.Linear(hidden * 3, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def encode(self, ids, mask):
        # Keep the graph enabled for the unfrozen backbone layers.
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
                var.to(DEVICE), var_mask.to(DEVICE),
            )
            predictions.extend((pred.cpu().numpy() * std + mean).tolist())
            observed.extend((y.numpy() * std + mean).tolist())

    pred = np.asarray(predictions)
    true = np.asarray(observed)

    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "pearson_r": float(pearsonr(true, pred).statistic)
            if np.std(pred) > 0 and np.std(true) > 0 else None,
        "spearman_r": float(spearmanr(true, pred).statistic)
            if np.std(pred) > 0 and np.std(true) > 0 else None,
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
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = PartialFTPairedHyenaDNA().to(DEVICE)

    backbone_params = [
        p for p in model.backbone.parameters() if p.requires_grad
    ]
    head_params = list(model.regressor.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=1e-4,
    )
    loss_fn = nn.SmoothL1Loss()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters:", total)
    print("Trainable parameters:", trainable)
    print("Unfrozen backbone blocks: 2 and 3, plus final norm")
    print("Batch size:", BATCH_SIZE)

    best_mae = float("inf")
    best_epoch = 0
    patience = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
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

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, mean, std)
        train_loss = float(np.mean(losses))

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_MAE={val_metrics['mae']:.4f} | "
            f"val_RMSE={val_metrics['rmse']:.4f} | "
            f"val_r={val_metrics['pearson_r']}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), OUT_DIR / "best_model.pt")
        else:
            patience += 1

        pd.DataFrame(history).to_csv(
            OUT_DIR / "training_history.csv", index=False
        )

        if patience >= PATIENCE:
            print("Early stopping.")
            break

    # Load the validation-selected checkpoint before test evaluation.
    model.load_state_dict(torch.load(
        OUT_DIR / "best_model.pt", map_location=DEVICE, weights_only=True
    ))
    test_metrics = evaluate(model, test_loader, mean, std)

    predictions, observed = [], []
    model.eval()
    with torch.no_grad():
        for wt, wt_mask, var, var_mask, y in test_loader:
            pred = model(
                wt.to(DEVICE), wt_mask.to(DEVICE),
                var.to(DEVICE), var_mask.to(DEVICE),
            )
            predictions.extend((pred.cpu().numpy() * std + mean).tolist())
            observed.extend((y.numpy() * std + mean).tolist())

    pred_df = test_df.copy()
    pred_df["predicted_delta_psi"] = predictions
    pred_df["prediction_error"] = pred_df["predicted_delta_psi"] - pred_df["delta_psi"]
    pred_df["absolute_error"] = pred_df["prediction_error"].abs()
    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    summary = {
        "model": "HyenaDNA partial fine-tuning",
        "unfrozen_backbone_blocks": [2, 3],
        "final_norm_unfrozen": True,
        "backbone_learning_rate": BACKBONE_LR,
        "head_learning_rate": HEAD_LR,
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "n_test": len(test_df),
        "n_test_genes": int(test_df["gene"].nunique()),
        "test_metrics": test_metrics,
        "elapsed_seconds": round(time.time() - start, 2),
    }

    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTest metrics:", json.dumps(test_metrics, indent=2))
    print("Best epoch:", best_epoch)
    print("Saved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
