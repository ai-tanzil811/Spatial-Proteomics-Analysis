"""
04_normalize_and_save.py
Normalize node features and build PyTorch Geometric Data objects.

NOTE: StandardScaler is fit on the full dataset here for inspection/caching;
      inside the training loop (06_training.py) the scaler is re-fit on the
      training fold only (no leakage). The unscaled PyG dataset is the canonical
      saved version; scaling is applied in-memory during training.

Output:
  output/spatial_graphs/pyg_dataset/raw/   — one <ACQUISITION_ID>.pt per sample
  output/spatial_graphs/pyg_dataset/dataset_index.csv

Run:
    python spatial_survival/04_normalize_and_save.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, EDGES_DIR, SURVIVAL_LABELS_FILE, SEED, N_NODE_FEATURES,
)
from utils import get_logger, ensure_dirs, set_seed

FEATURES_DIR = OUTPUT_DIR / "features"
PYG_RAW_DIR  = OUTPUT_DIR / "pyg_dataset" / "raw"
INDEX_PATH   = OUTPUT_DIR / "pyg_dataset" / "dataset_index.csv"

logger = get_logger("normalize_and_save")


# ---------------------------------------------------------------------------
# Build one PyG Data object
# ---------------------------------------------------------------------------

def build_data_object(acq_id: str,
                       node_feats: np.ndarray,
                       edge_index: np.ndarray,
                       edge_feats: np.ndarray,
                       y_time: float,
                       y_event: int,
                       patient_id: int) -> Data:
    """
    Assemble a torch_geometric.data.Data object.
    node_feats  : (N, 45)  float32
    edge_index  : (2, E)   int64
    edge_feats  : (E, 3)   float32
    """
    data = Data(
        x          = torch.tensor(node_feats, dtype=torch.float32),
        edge_index = torch.tensor(edge_index, dtype=torch.long),
        edge_attr  = torch.tensor(edge_feats, dtype=torch.float32),
        y_time     = torch.tensor([y_time],   dtype=torch.float32),
        y_event    = torch.tensor([y_event],  dtype=torch.float32),
        patient_id = torch.tensor([patient_id], dtype=torch.long),
        acq_id     = acq_id,
    )
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    ensure_dirs(PYG_RAW_DIR)

    # Load survival labels
    survival = pd.read_csv(SURVIVAL_LABELS_FILE)
    survival = survival.set_index("ACQUISITION_ID")

    feature_files = sorted(FEATURES_DIR.glob("*.npz"))
    logger.info(f"Found {len(feature_files)} feature files")

    index_records = []
    skipped = []

    for feat_path in tqdm(feature_files, desc="Building PyG objects"):
        acq_id = feat_path.stem

        # Skip if no survival label
        if acq_id not in survival.index:
            skipped.append(acq_id)
            continue

        # Load features
        feat_data  = np.load(feat_path, allow_pickle=False)
        node_feats = feat_data["node_features"]   # (N, 45)
        edge_feats = feat_data["edge_features"]   # (E, 3)

        # Load edge index
        edge_path  = EDGES_DIR / f"{acq_id}.npz"
        if not edge_path.exists():
            skipped.append(acq_id)
            continue
        edge_data  = np.load(edge_path, allow_pickle=False)
        edge_index = edge_data["edge_index"]      # (2, E)

        # Validate shapes
        E = edge_index.shape[1]
        assert edge_feats.shape[0] == E, \
            f"{acq_id}: edge_feats rows ({edge_feats.shape[0]}) != edge_index cols ({E})"
        assert node_feats.shape[1] == N_NODE_FEATURES, \
            f"{acq_id}: expected {N_NODE_FEATURES} node features, got {node_feats.shape[1]}"

        # Survival info
        row       = survival.loc[acq_id]
        y_time    = float(row["survival_day"])
        y_event   = int(row["survival_status"])
        patient_id = int(row["patient_id"])

        # Build and save PyG Data object
        data = build_data_object(
            acq_id, node_feats, edge_index, edge_feats,
            y_time, y_event, patient_id
        )
        torch.save(data, PYG_RAW_DIR / f"{acq_id}.pt")

        index_records.append({
            "acquisition_id": acq_id,
            "patient_id":     patient_id,
            "y_time":         y_time,
            "y_event":        y_event,
            "n_nodes":        node_feats.shape[0],
            "n_edges":        E,
        })

    if skipped:
        logger.warning(f"Skipped {len(skipped)} samples (no survival label or edge file)")

    # Save dataset index
    index_df = pd.DataFrame(index_records)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(INDEX_PATH, index=False)

    logger.info("=== PyG Dataset Summary ===")
    logger.info(f"  Saved samples : {len(index_records)}")
    logger.info(f"  Patients      : {index_df['patient_id'].nunique()}")
    logger.info(f"  Events        : {index_df['y_event'].sum()} / {len(index_df)}")
    logger.info(f"  Avg nodes     : {index_df['n_nodes'].mean():.0f}")
    logger.info(f"  Avg edges     : {index_df['n_edges'].mean():.0f}")
    logger.info(f"  Index saved   : {INDEX_PATH}")


if __name__ == "__main__":
    main()
