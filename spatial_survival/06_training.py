"""
06_training.py
Patient-level GroupKFold cross-validation for the GraphSAGE survival model.

For each fold:
  - Train set: samples from (N_CV_FOLDS - 1) patient groups
  - Val   set: 10 % of train patients held out for early stopping
  - Test  set: samples from the held-out patient group
  - StandardScaler fit on train node features only (no leakage)
  - Cox partial likelihood loss with early stopping on val C-index
  - Best model checkpoint saved per fold

Results saved to:
  output/results/training_logs/fold_<k>_log.csv
  output/results/checkpoints/fold_<k>_best.pt
  output/results/cv_results.csv

Run:
    python spatial_survival/06_training.py
"""

import sys
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import importlib

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, RESULTS_DIR, N_NODE_FEATURES, HIDDEN_DIM, N_LAYERS,
    DROPOUT, N_INTERACTION_TYPES, INTERACTION_EMBED_DIM, LEARNING_RATE,
    WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR_PATIENCE, SEED,
    N_CV_FOLDS, VAL_FRACTION, N_PROTEINS,
)
from utils import get_logger, ensure_dirs, set_seed, compute_cindex, CoxPHLoss

_model_module = importlib.import_module("05_graphsage_model")
GraphSAGESurvival = _model_module.GraphSAGESurvival

PYG_RAW_DIR  = OUTPUT_DIR / "pyg_dataset" / "raw"
INDEX_PATH   = OUTPUT_DIR / "pyg_dataset" / "dataset_index.csv"
CKPT_DIR     = RESULTS_DIR / "checkpoints"
LOG_DIR      = RESULTS_DIR / "training_logs"

logger = get_logger("training", RESULTS_DIR / "training.log")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_all_data(index_df: pd.DataFrame) -> list[Data]:
    data_list = []
    for _, row in index_df.iterrows():
        pt = PYG_RAW_DIR / f"{row['acquisition_id']}.pt"
        if pt.exists():
            data_list.append(torch.load(pt, weights_only=False))
    return data_list


def scale_node_features(train_data: list[Data],
                         val_data:   list[Data],
                         test_data:  list[Data]):
    """
    Fit StandardScaler on train node features, apply to all splits in-place.
    Only the first N_PROTEINS features are continuous and scaled;
    the 6 spatial features (indices 39–44) are also scaled.
    """
    scaler = StandardScaler()

    # Fit on train
    train_x = np.vstack([d.x.numpy() for d in train_data])
    scaler.fit(train_x)

    def _apply(data_list):
        for d in data_list:
            x_np  = d.x.numpy()
            x_sc  = scaler.transform(x_np)
            d.x   = torch.tensor(x_sc, dtype=torch.float32)

    _apply(train_data)
    _apply(val_data)
    _apply(test_data)
    return scaler


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        risk   = model(batch).squeeze(1)
        times  = batch.y_time.squeeze()
        events = batch.y_event.squeeze()
        loss   = criterion(risk, times, events)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    risks_all  = []
    times_all  = []
    events_all = []
    for batch in loader:
        batch = batch.to(device)
        risk   = model(batch).squeeze(1).cpu().numpy()
        times  = batch.y_time.squeeze().cpu().numpy()
        events = batch.y_event.squeeze().cpu().numpy()
        risks_all.append(risk)
        times_all.append(times)
        events_all.append(events)

    risks  = np.concatenate(risks_all)
    times  = np.concatenate(times_all)
    events = np.concatenate(events_all)
    ci     = compute_cindex(risks, times, events)
    return ci, risks, times, events


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    set_seed(SEED)
    ensure_dirs(CKPT_DIR, LOG_DIR, RESULTS_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    index_df = pd.read_csv(INDEX_PATH)
    logger.info(f"Dataset: {len(index_df)} samples, {index_df['patient_id'].nunique()} patients")

    all_data = load_all_data(index_df)
    groups   = index_df["patient_id"].to_numpy()

    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    criterion = CoxPHLoss()

    fold_results = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(
            gkf.split(index_df, groups=groups)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx+1}/{N_CV_FOLDS}")

        # Split train/val by patient groups
        train_val_patients = np.unique(groups[train_val_idx])
        rng = np.random.default_rng(SEED + fold_idx)
        rng.shuffle(train_val_patients)
        n_val_patients = max(1, int(len(train_val_patients) * VAL_FRACTION))
        val_patients   = set(train_val_patients[:n_val_patients])

        train_mask  = [i for i in train_val_idx if groups[i] not in val_patients]
        val_mask    = [i for i in train_val_idx if groups[i] in val_patients]

        train_data = [all_data[i] for i in train_mask]
        val_data   = [all_data[i] for i in val_mask]
        test_data  = [all_data[i] for i in test_idx]

        logger.info(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

        # Scale features (no leakage)
        scaler = scale_node_features(train_data, val_data, test_data)

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

        # Model
        set_seed(SEED)
        model = GraphSAGESurvival(
            in_channels           = N_NODE_FEATURES,
            hidden_dim            = HIDDEN_DIM,
            n_layers              = N_LAYERS,
            dropout               = DROPOUT,
            n_interaction_types   = N_INTERACTION_TYPES,
            interaction_embed_dim = INTERACTION_EMBED_DIM,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=LEARNING_RATE,
                         weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                      patience=LR_PATIENCE, factor=0.5,
                                      min_lr=1e-6)

        best_val_ci  = -1.0
        best_epoch   = 0
        no_improve   = 0
        best_state   = None
        fold_log     = []

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_ci, _, _, _ = evaluate(model, val_loader, device)
            scheduler.step(val_ci if not np.isnan(val_ci) else 0.0)

            fold_log.append({"epoch": epoch, "train_loss": train_loss, "val_cindex": val_ci})

            if not np.isnan(val_ci) and val_ci > best_val_ci:
                best_val_ci = val_ci
                best_epoch  = epoch
                no_improve  = 0
                best_state  = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | "
                            f"val_C={val_ci:.4f} | best={best_val_ci:.4f}")

            if no_improve >= PATIENCE:
                logger.info(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

        # Save best model
        ckpt_path = CKPT_DIR / f"fold_{fold_idx+1}_best.pt"
        if best_state is not None:
            torch.save({"state_dict": best_state, "scaler": scaler,
                        "fold": fold_idx + 1, "best_epoch": best_epoch},
                       ckpt_path)

        # Test evaluation with best model
        if best_state is not None:
            model.load_state_dict(best_state)
        test_ci, test_risks, test_times, test_events = evaluate(model, test_loader, device)
        logger.info(f"  Fold {fold_idx+1} | best_val_C={best_val_ci:.4f} | "
                    f"test_C={test_ci:.4f} (epoch {best_epoch})")

        # Save fold log
        log_df = pd.DataFrame(fold_log)
        log_df.to_csv(LOG_DIR / f"fold_{fold_idx+1}_log.csv", index=False)

        # Save test predictions for pooled evaluation
        pred_df = pd.DataFrame({
            "acquisition_id": [index_df.iloc[i]["acquisition_id"] for i in test_idx],
            "risk_score":     test_risks,
            "y_time":         test_times,
            "y_event":        test_events,
            "fold":           fold_idx + 1,
        })
        pred_df.to_csv(RESULTS_DIR / f"fold_{fold_idx+1}_predictions.csv", index=False)

        fold_results.append({
            "fold":        fold_idx + 1,
            "n_train":     len(train_data),
            "n_val":       len(val_data),
            "n_test":      len(test_data),
            "best_epoch":  best_epoch,
            "best_val_ci": best_val_ci,
            "test_ci":     test_ci,
        })

        # Restore original node features for next fold (reload from disk)
        all_data = load_all_data(index_df)

    # Summary
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)

    logger.info("\n=== Cross-Validation Summary ===")
    logger.info(f"\n{results_df.to_string(index=False)}")
    logger.info(f"\nMean test C-index: {results_df['test_ci'].mean():.4f} "
                f"± {results_df['test_ci'].std():.4f}")


if __name__ == "__main__":
    train()
