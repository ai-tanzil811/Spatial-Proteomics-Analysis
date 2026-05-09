"""
09_ablation.py
Systematic ablation study and hyperparameter sweep.

1. Feature ablation: zero out one feature group at a time, retrain, measure ΔC-index
   Groups: Expression (0:39), Density (39), Entropy (40), Boundary (41),
           Degree (42), Gradient (43), Edge features (all edge_attr set to 0)

2. Hyperparameter sweeps (trained once each on all data with internal val split):
   - k ∈ {3, 4, 5, 6}   — requires re-running 02_graph_construction.py
   - σ multiplier ∈ {0.5, 1.0, 2.0}  — modifies edge_attr[:, 1] in-memory
   - n_layers ∈ {1, 2, 3, 4}
   - hidden_dim ∈ {32, 64, 128}

Results saved to:
  output/results/ablation/feature_ablation.csv
  output/results/ablation/hyperparam_sweep.csv

Run AFTER 06_training.py (uses the existing PyG dataset and CV splits).

Run:
    python spatial_survival/09_ablation.py
"""

import sys
import copy
import importlib
from pathlib import Path
from typing import Optional

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

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, RESULTS_DIR, N_NODE_FEATURES, HIDDEN_DIM, N_LAYERS,
    DROPOUT, N_INTERACTION_TYPES, INTERACTION_EMBED_DIM, LEARNING_RATE,
    WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS, PATIENCE, LR_PATIENCE,
    SEED, N_CV_FOLDS, VAL_FRACTION, N_PROTEINS, PROTEIN_COLS,
)
from utils import get_logger, ensure_dirs, set_seed, compute_cindex, CoxPHLoss

_model_module = importlib.import_module("05_graphsage_model")
GraphSAGESurvival = _model_module.GraphSAGESurvival

PYG_RAW_DIR  = OUTPUT_DIR / "pyg_dataset" / "raw"
INDEX_PATH   = OUTPUT_DIR / "pyg_dataset" / "dataset_index.csv"
ABLATION_DIR = RESULTS_DIR / "ablation"

logger = get_logger("ablation", RESULTS_DIR / "ablation.log")

# Feature group index ranges in 45-dim node feature vector
FEATURE_GROUPS = {
    "Expression":  (0, N_PROTEINS),      # indices 0:39
    "Density":     (N_PROTEINS,     N_PROTEINS + 1),
    "Entropy":     (N_PROTEINS + 1, N_PROTEINS + 2),
    "Boundary":    (N_PROTEINS + 2, N_PROTEINS + 3),
    "Degree":      (N_PROTEINS + 3, N_PROTEINS + 4),
    "Gradient":    (N_PROTEINS + 4, N_PROTEINS + 5),
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_all_data(index_df: pd.DataFrame) -> list[Data]:
    data_list = []
    for _, row in index_df.iterrows():
        pt = PYG_RAW_DIR / f"{row['acquisition_id']}.pt"
        if pt.exists():
            data_list.append(torch.load(pt, weights_only=False))
    return data_list


def scale_node_features(train_data, val_data, test_data):
    scaler = StandardScaler()
    scaler.fit(np.vstack([d.x.numpy() for d in train_data]))
    for ds in (train_data, val_data, test_data):
        for d in ds:
            d.x = torch.tensor(scaler.transform(d.x.numpy()), dtype=torch.float32)


def apply_feature_mask(data_list: list[Data],
                        group_name: Optional[str],
                        zero_edges: bool = False) -> list[Data]:
    """Return deep-copied data with specified features zeroed out."""
    masked = []
    for d in data_list:
        d2 = copy.deepcopy(d)
        if group_name is not None and group_name in FEATURE_GROUPS:
            start, end = FEATURE_GROUPS[group_name]
            d2.x[:, start:end] = 0.0
        if zero_edges:
            d2.edge_attr = torch.zeros_like(d2.edge_attr)
        masked.append(d2)
    return masked


def apply_sigma_multiplier(data_list: list[Data], multiplier: float) -> list[Data]:
    """Scale distance weight (edge_attr[:, 1]) by recomputing with σ * multiplier."""
    scaled = []
    for d in data_list:
        d2 = copy.deepcopy(d)
        # Recover original dist_weight → back to distance, rescale σ
        # We directly raise dist_weight to (1/multiplier²) as an approximation:
        # exp(-d²/2(σ*m)²) = exp(-d²/2σ² * 1/m²) = w^(1/m²)
        w = d2.edge_attr[:, 1].clamp(1e-10, 1.0)
        d2.edge_attr[:, 1] = w.pow(1.0 / (multiplier ** 2))
        scaled.append(d2)
    return scaled


# ---------------------------------------------------------------------------
# Quick CV run (one full GroupKFold, returns mean test C-index)
# ---------------------------------------------------------------------------

def run_cv(data_source: list[Data], index_df: pd.DataFrame,
           in_channels: int = N_NODE_FEATURES,
           hidden_dim: int = HIDDEN_DIM,
           n_layers: int = N_LAYERS) -> float:
    """Run GroupKFold CV, return mean test C-index."""
    groups    = index_df["patient_id"].to_numpy()
    gkf       = GroupKFold(n_splits=N_CV_FOLDS)
    criterion = CoxPHLoss()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_cis  = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(
            gkf.split(index_df, groups=groups)):

        train_val_patients = np.unique(groups[train_val_idx])
        rng = np.random.default_rng(SEED + fold_idx)
        rng.shuffle(train_val_patients)
        n_val = max(1, int(len(train_val_patients) * VAL_FRACTION))
        val_patients = set(train_val_patients[:n_val])

        t_idxs  = [i for i in train_val_idx if groups[i] not in val_patients]
        v_idxs  = [i for i in train_val_idx if groups[i] in val_patients]

        t_data  = copy.deepcopy([data_source[i] for i in t_idxs])
        v_data  = copy.deepcopy([data_source[i] for i in v_idxs])
        te_data = copy.deepcopy([data_source[i] for i in test_idx])

        scale_node_features(t_data, v_data, te_data)

        set_seed(SEED)
        model = GraphSAGESurvival(
            in_channels=in_channels, hidden_dim=hidden_dim,
            n_layers=n_layers, dropout=DROPOUT,
            n_interaction_types=N_INTERACTION_TYPES,
            interaction_embed_dim=INTERACTION_EMBED_DIM,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=LEARNING_RATE,
                         weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                      patience=LR_PATIENCE, factor=0.5, min_lr=1e-6)

        t_loader  = DataLoader(t_data,  batch_size=BATCH_SIZE, shuffle=True)
        v_loader  = DataLoader(v_data,  batch_size=BATCH_SIZE, shuffle=False)
        te_loader = DataLoader(te_data, batch_size=BATCH_SIZE, shuffle=False)

        best_val_ci = -1.0
        no_improve  = 0
        best_state  = None

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            for batch in t_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                risk  = model(batch).squeeze(1)
                loss  = criterion(risk, batch.y_time.squeeze(), batch.y_event.squeeze())
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            # Validate
            model.eval()
            v_risks, v_times, v_events = [], [], []
            with torch.no_grad():
                for batch in v_loader:
                    batch = batch.to(device)
                    v_risks.append(model(batch).squeeze(1).cpu().numpy())
                    v_times.append(batch.y_time.squeeze().cpu().numpy())
                    v_events.append(batch.y_event.squeeze().cpu().numpy())
            val_ci = compute_cindex(
                np.concatenate(v_risks),
                np.concatenate(v_times),
                np.concatenate(v_events),
            )
            scheduler.step(val_ci if not np.isnan(val_ci) else 0.0)

            if not np.isnan(val_ci) and val_ci > best_val_ci:
                best_val_ci = val_ci
                no_improve  = 0
                best_state  = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1
            if no_improve >= PATIENCE:
                break

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        te_risks, te_times, te_events = [], [], []
        with torch.no_grad():
            for batch in te_loader:
                batch = batch.to(device)
                te_risks.append(model(batch).squeeze(1).cpu().numpy())
                te_times.append(batch.y_time.squeeze().cpu().numpy())
                te_events.append(batch.y_event.squeeze().cpu().numpy())
        test_ci = compute_cindex(
            np.concatenate(te_risks),
            np.concatenate(te_times),
            np.concatenate(te_events),
        )
        fold_cis.append(test_ci)

    return float(np.nanmean(fold_cis))


# ---------------------------------------------------------------------------
# Feature ablation
# ---------------------------------------------------------------------------

def run_feature_ablation(baseline_ci: float, all_data, index_df) -> pd.DataFrame:
    rows = [{"ablation": "Full model", "group": "—", "mean_cindex": round(baseline_ci, 4),
             "delta_cindex": 0.0}]

    groups_to_test = list(FEATURE_GROUPS.keys()) + ["EdgeFeatures"]

    for group in tqdm(groups_to_test, desc="Feature ablation"):
        zero_edges = (group == "EdgeFeatures")
        masked = apply_feature_mask(all_data, None if zero_edges else group,
                                     zero_edges=zero_edges)
        mean_ci = run_cv(masked, index_df)
        delta   = round(mean_ci - baseline_ci, 4)
        logger.info(f"  Ablation [{group}]: C-index={mean_ci:.4f}  Δ={delta:+.4f}")
        rows.append({"ablation": f"No {group}", "group": group,
                     "mean_cindex": round(mean_ci, 4), "delta_cindex": delta})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Hyperparameter sweeps
# ---------------------------------------------------------------------------

def run_sigma_sweep(all_data, index_df, baseline_ci: float) -> pd.DataFrame:
    rows = []
    for mult in [0.5, 1.0, 2.0]:
        modified = apply_sigma_multiplier(all_data, mult)
        mean_ci  = run_cv(modified, index_df)
        rows.append({"sigma_mult": mult, "mean_cindex": round(mean_ci, 4),
                     "delta_cindex": round(mean_ci - baseline_ci, 4)})
        logger.info(f"  σ×{mult}: C-index={mean_ci:.4f}")
    return pd.DataFrame(rows)


def run_arch_sweep(all_data, index_df, baseline_ci: float) -> pd.DataFrame:
    rows = []
    for n_layers in [1, 2, 3, 4]:
        for hidden in [32, 64, 128]:
            if n_layers == N_LAYERS and hidden == HIDDEN_DIM:
                rows.append({"n_layers": n_layers, "hidden_dim": hidden,
                             "mean_cindex": round(baseline_ci, 4),
                             "delta_cindex": 0.0})
                continue
            mean_ci = run_cv(all_data, index_df, n_layers=n_layers, hidden_dim=hidden)
            rows.append({"n_layers": n_layers, "hidden_dim": hidden,
                         "mean_cindex": round(mean_ci, 4),
                         "delta_cindex": round(mean_ci - baseline_ci, 4)})
            logger.info(f"  layers={n_layers}, hidden={hidden}: C-index={mean_ci:.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    ensure_dirs(ABLATION_DIR)

    index_df = pd.read_csv(INDEX_PATH)
    all_data = load_all_data(index_df)
    logger.info(f"Loaded {len(all_data)} graphs for ablation study")

    # Baseline (full model)
    logger.info("Computing baseline C-index …")
    baseline_ci = run_cv(copy.deepcopy(all_data), index_df)
    logger.info(f"Baseline mean C-index: {baseline_ci:.4f}")

    # 1. Feature ablation
    logger.info("\n=== Feature Ablation ===")
    feat_df = run_feature_ablation(baseline_ci, copy.deepcopy(all_data), index_df)
    feat_df.to_csv(ABLATION_DIR / "feature_ablation.csv", index=False)
    logger.info(f"\n{feat_df.to_string(index=False)}")

    # 2. σ sweep
    logger.info("\n=== Sigma (Distance Weight) Sweep ===")
    sigma_df = run_sigma_sweep(copy.deepcopy(all_data), index_df, baseline_ci)
    sigma_df.to_csv(ABLATION_DIR / "sigma_sweep.csv", index=False)
    logger.info(f"\n{sigma_df.to_string(index=False)}")

    # 3. Architecture sweep
    logger.info("\n=== Architecture Sweep (layers × hidden_dim) ===")
    arch_df = run_arch_sweep(copy.deepcopy(all_data), index_df, baseline_ci)
    arch_df.to_csv(ABLATION_DIR / "arch_sweep.csv", index=False)
    logger.info(f"\n{arch_df.to_string(index=False)}")

    # Combined summary
    all_results = {
        "Feature ablation": feat_df,
        "Sigma sweep":      sigma_df,
        "Arch sweep":       arch_df,
    }
    logger.info(f"\nAll ablation results saved to {ABLATION_DIR}")


if __name__ == "__main__":
    main()
