"""
07_baselines.py
Three comparison baselines under the same patient-level GroupKFold CV:

  1. MLP  — per-sample mean of 39 protein expressions → 3-layer MLP + Cox loss
  2. GCN  — GCNConv replacing SAGEConv, otherwise identical to GraphSAGE
  3. RSF  — Random Survival Forest on per-sample mean expression (sklearn-survival)

Results saved to:
  output/results/baselines/<model>_cv_results.csv
  output/results/baselines/<model>_fold_<k>_predictions.csv

Run:
    python spatial_survival/07_baselines.py
"""

import sys
import copy
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, RESULTS_DIR, N_NODE_FEATURES, HIDDEN_DIM, N_LAYERS,
    DROPOUT, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS,
    PATIENCE, LR_PATIENCE, SEED, N_CV_FOLDS, VAL_FRACTION, N_PROTEINS,
    N_INTERACTION_TYPES, INTERACTION_EMBED_DIM,
)
from utils import get_logger, ensure_dirs, set_seed, compute_cindex, CoxPHLoss

PYG_RAW_DIR  = OUTPUT_DIR / "pyg_dataset" / "raw"
INDEX_PATH   = OUTPUT_DIR / "pyg_dataset" / "dataset_index.csv"
BASELINE_DIR = RESULTS_DIR / "baselines"

logger = get_logger("baselines", RESULTS_DIR / "baselines.log")


# ---------------------------------------------------------------------------
# Helper: load PyG data objects
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
    return scaler


def get_sample_means(data_list: list[Data]) -> np.ndarray:
    """Per-sample mean over the first N_PROTEINS node features."""
    return np.vstack([d.x[:, :N_PROTEINS].numpy().mean(0) for d in data_list])


# ---------------------------------------------------------------------------
# Model 1: MLP baseline
# ---------------------------------------------------------------------------

class MLPSurvival(nn.Module):
    def __init__(self, in_dim: int = N_PROTEINS,
                 hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x_mean = data.x[:, :N_PROTEINS]
        # Graph-level mean across nodes
        from torch_geometric.nn import global_mean_pool
        x_graph = global_mean_pool(x_mean, data.batch)   # (B, N_PROTEINS)
        return self.net(x_graph)


# ---------------------------------------------------------------------------
# Model 2: GCN baseline
# ---------------------------------------------------------------------------

class GCNSurvival(nn.Module):
    def __init__(self, in_channels: int = N_NODE_FEATURES,
                 hidden_dim: int = HIDDEN_DIM, n_layers: int = N_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        dims = [in_channels] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))
        pool_dim = hidden_dim * 2
        self.risk_head = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x = data.x
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        mean_pool = global_mean_pool(x, data.batch)
        max_pool  = global_max_pool(x, data.batch)
        return self.risk_head(torch.cat([mean_pool, max_pool], dim=1))


# ---------------------------------------------------------------------------
# Generic GNN training loop
# ---------------------------------------------------------------------------

def train_gnn(model, train_data, val_data, test_data, device,
              fold_idx: int, model_name: str):
    criterion    = CoxPHLoss()
    optimizer    = Adam(model.parameters(), lr=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY)
    scheduler    = ReduceLROnPlateau(optimizer, mode="max",
                                     patience=LR_PATIENCE, factor=0.5, min_lr=1e-6)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    best_val_ci = -1.0
    no_improve  = 0
    best_state  = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            risk   = model(batch).squeeze(1)
            loss   = criterion(risk, batch.y_time.squeeze(), batch.y_event.squeeze())
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        val_ci, _, _, _ = _eval_gnn(model, val_loader, device)
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

    test_ci, risks, times, events = _eval_gnn(model, test_loader, device)
    return test_ci, risks, times, events, best_val_ci


@torch.no_grad()
def _eval_gnn(model, loader, device):
    model.eval()
    risks, times, events = [], [], []
    for batch in loader:
        batch = batch.to(device)
        r = model(batch).squeeze(1).cpu().numpy()
        t = batch.y_time.squeeze().cpu().numpy()
        e = batch.y_event.squeeze().cpu().numpy()
        risks.append(r); times.append(t); events.append(e)
    r = np.concatenate(risks)
    t = np.concatenate(times)
    e = np.concatenate(events)
    return compute_cindex(r, t, e), r, t, e


# ---------------------------------------------------------------------------
# RSF baseline
# ---------------------------------------------------------------------------

def run_rsf(index_df, all_data, groups):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    fold_results = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(
            gkf.split(index_df, groups=groups)):
        train_data_list = [all_data[i] for i in train_val_idx]
        test_data_list  = [all_data[i] for i in test_idx]

        X_train = get_sample_means(train_data_list)
        X_test  = get_sample_means(test_data_list)

        # Scale on train
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test  = scaler.transform(X_test)

        y_train_times  = np.array([d.y_time.item()  for d in train_data_list])
        y_train_events = np.array([d.y_event.item() for d in train_data_list], dtype=bool)
        y_test_times   = np.array([d.y_time.item()  for d in test_data_list])
        y_test_events  = np.array([d.y_event.item() for d in test_data_list], dtype=bool)

        y_train = Surv.from_arrays(y_train_events, y_train_times)
        y_test  = Surv.from_arrays(y_test_events,  y_test_times)

        set_seed(SEED)
        rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                                   min_samples_leaf=5, n_jobs=-1,
                                   random_state=SEED)
        rsf.fit(X_train, y_train)
        risk_scores = rsf.predict(X_test)
        test_ci = compute_cindex(risk_scores, y_test_times, y_test_events.astype(float))
        logger.info(f"  RSF Fold {fold_idx+1}: test_C={test_ci:.4f}")

        pred_df = pd.DataFrame({
            "acquisition_id": [index_df.iloc[i]["acquisition_id"] for i in test_idx],
            "risk_score": risk_scores,
            "y_time":     y_test_times,
            "y_event":    y_test_events.astype(int),
            "fold":       fold_idx + 1,
        })
        pred_df.to_csv(BASELINE_DIR / f"RSF_fold_{fold_idx+1}_predictions.csv", index=False)

        fold_results.append({"fold": fold_idx+1, "test_ci": test_ci})

    return pd.DataFrame(fold_results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_gnn_baseline(model_name: str, index_df, all_data, groups, device):
    gkf = GroupKFold(n_splits=N_CV_FOLDS)
    fold_results = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(
            gkf.split(index_df, groups=groups)):
        train_val_patients = np.unique(groups[train_val_idx])
        rng = np.random.default_rng(SEED + fold_idx)
        rng.shuffle(train_val_patients)
        n_val = max(1, int(len(train_val_patients) * VAL_FRACTION))
        val_patients = set(train_val_patients[:n_val])

        train_data = [all_data[i] for i in train_val_idx if groups[i] not in val_patients]
        val_data   = [all_data[i] for i in train_val_idx if groups[i] in val_patients]
        test_data  = [all_data[i] for i in test_idx]

        # Fresh copy for scaling
        import copy as _copy
        t_data = [_copy.deepcopy(d) for d in train_data]
        v_data = [_copy.deepcopy(d) for d in val_data]
        te_data = [_copy.deepcopy(d) for d in test_data]
        scale_node_features(t_data, v_data, te_data)

        set_seed(SEED)
        if model_name == "MLP":
            model = MLPSurvival().to(device)
        elif model_name == "GCN":
            model = GCNSurvival().to(device)

        test_ci, risks, times, events, best_val_ci = train_gnn(
            model, t_data, v_data, te_data, device, fold_idx, model_name
        )
        logger.info(f"  {model_name} Fold {fold_idx+1}: val_C={best_val_ci:.4f} test_C={test_ci:.4f}")

        pred_df = pd.DataFrame({
            "acquisition_id": [index_df.iloc[i]["acquisition_id"] for i in test_idx],
            "risk_score": risks, "y_time": times, "y_event": events.astype(int),
            "fold": fold_idx + 1,
        })
        pred_df.to_csv(BASELINE_DIR / f"{model_name}_fold_{fold_idx+1}_predictions.csv", index=False)
        fold_results.append({"fold": fold_idx+1, "best_val_ci": best_val_ci, "test_ci": test_ci})

    return pd.DataFrame(fold_results)


def main():
    set_seed(SEED)
    ensure_dirs(BASELINE_DIR)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index_df = pd.read_csv(INDEX_PATH)
    all_data = load_all_data(index_df)
    groups   = index_df["patient_id"].to_numpy()

    summary = {}

    for model_name in ("MLP", "GCN"):
        logger.info(f"\n{'='*60}\nRunning {model_name} baseline …")
        results = run_gnn_baseline(model_name, index_df, all_data, groups, device)
        results.to_csv(BASELINE_DIR / f"{model_name}_cv_results.csv", index=False)
        summary[model_name] = results["test_ci"].mean()
        logger.info(f"  {model_name} mean test C-index: {summary[model_name]:.4f} "
                    f"± {results['test_ci'].std():.4f}")

    logger.info(f"\n{'='*60}\nRunning RSF baseline …")
    rsf_results = run_rsf(index_df, all_data, groups)
    rsf_results.to_csv(BASELINE_DIR / "RSF_cv_results.csv", index=False)
    summary["RSF"] = rsf_results["test_ci"].mean()
    logger.info(f"  RSF mean test C-index: {summary['RSF']:.4f} "
                f"± {rsf_results['test_ci'].std():.4f}")

    # Summary table
    logger.info("\n=== Baseline Summary ===")
    for name, ci in summary.items():
        logger.info(f"  {name:10s}: mean C-index = {ci:.4f}")


if __name__ == "__main__":
    main()
