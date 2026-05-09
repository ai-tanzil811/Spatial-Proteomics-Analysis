"""
08_evaluation.py
Evaluate the trained GraphSAGE model:
  - Pool risk scores from all CV folds → overall C-index
  - Time-dependent AUC at t=365, 730, 1095 days
  - Kaplan-Meier curves (high vs low risk, median split)
  - Gradient-based saliency map per node feature
  - Comparison table: GraphSAGE vs GCN vs MLP vs RSF

Run AFTER 06_training.py and 07_baselines.py.

Run:
    python spatial_survival/08_evaluation.py
"""

import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, RESULTS_DIR, N_NODE_FEATURES, HIDDEN_DIM, N_LAYERS,
    DROPOUT, N_INTERACTION_TYPES, INTERACTION_EMBED_DIM, EVAL_TIMES,
    PROTEIN_COLS, SEED, BATCH_SIZE, N_CV_FOLDS,
)
from utils import get_logger, ensure_dirs, set_seed, compute_cindex, plot_km_curves

_model_module = importlib.import_module("05_graphsage_model")
GraphSAGESurvival = _model_module.GraphSAGESurvival

PYG_RAW_DIR  = OUTPUT_DIR / "pyg_dataset" / "raw"
INDEX_PATH   = OUTPUT_DIR / "pyg_dataset" / "dataset_index.csv"
CKPT_DIR     = RESULTS_DIR / "checkpoints"
EVAL_DIR     = RESULTS_DIR / "evaluation"
BASELINE_DIR = RESULTS_DIR / "baselines"

logger = get_logger("evaluation")

FEATURE_NAMES = PROTEIN_COLS + [
    "Local Density", "Neighborhood Entropy", "Boundary Score",
    "Degree Centrality", "Expression Gradient",
]


# ---------------------------------------------------------------------------
# Pool predictions from all CV folds
# ---------------------------------------------------------------------------

def load_pooled_predictions(model_name: str = "GraphSAGE") -> pd.DataFrame:
    if model_name == "GraphSAGE":
        fold_files = sorted(RESULTS_DIR.glob("fold_*_predictions.csv"))
    else:
        fold_files = sorted(BASELINE_DIR.glob(f"{model_name}_fold_*_predictions.csv"))

    if not fold_files:
        logger.warning(f"No prediction files found for {model_name}")
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in fold_files]
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Time-dependent AUC
# ---------------------------------------------------------------------------

def compute_time_auc(df: pd.DataFrame, eval_times: list[int]) -> dict:
    try:
        from sksurv.metrics import cumulative_dynamic_auc
        from sksurv.util import Surv

        y = Surv.from_arrays(df["y_event"].astype(bool), df["y_time"])
        y_train = y  # use same data (no train/test split needed for pooled)

        aucs, mean_auc = cumulative_dynamic_auc(
            y_train, y, df["risk_score"].to_numpy(), eval_times
        )
        return {f"AUC@{t}d": auc for t, auc in zip(eval_times, aucs)}
    except Exception as e:
        logger.warning(f"Time-dependent AUC failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Gradient saliency
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_saliency(model: torch.nn.Module,
                     data_list: list[Data],
                     device: torch.device) -> np.ndarray:
    """
    Simple input-gradient saliency: mean |∂risk/∂x| over all cells and samples.
    Returns array of shape (N_NODE_FEATURES,).
    """
    model.eval()
    saliency_acc = np.zeros(N_NODE_FEATURES, dtype=np.float64)
    count = 0

    loader = DataLoader(data_list, batch_size=4, shuffle=False)
    for batch in loader:
        batch = batch.to(device)
        batch.x.requires_grad_(True)
        risk = model(batch).sum()
        risk.backward()
        grad = batch.x.grad.abs().detach().cpu().numpy()   # (N_total, F)
        saliency_acc += grad.mean(axis=0)
        count += 1

    return saliency_acc / max(count, 1)


def plot_saliency(saliency: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    idx = np.argsort(saliency)[::-1]
    ax.bar(range(len(FEATURE_NAMES)), saliency[idx])
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in idx], rotation=90, fontsize=7)
    ax.set_ylabel("Mean |gradient|")
    ax.set_title("Node Feature Saliency (GraphSAGE)")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(model_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in model_names:
        preds = load_pooled_predictions(name)
        if preds.empty:
            continue
        ci = compute_cindex(preds["risk_score"].to_numpy(),
                            preds["y_time"].to_numpy(),
                            preds["y_event"].to_numpy())
        auc_dict = compute_time_auc(preds, EVAL_TIMES)
        row = {"Model": name, "C-index": round(ci, 4), **{k: round(v, 4) for k, v in auc_dict.items()}}
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    ensure_dirs(EVAL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Pooled GraphSAGE evaluation
    logger.info("Loading pooled GraphSAGE predictions …")
    sage_preds = load_pooled_predictions("GraphSAGE")
    if sage_preds.empty:
        logger.error("No GraphSAGE predictions found. Run 06_training.py first.")
        return

    overall_ci = compute_cindex(sage_preds["risk_score"].to_numpy(),
                                 sage_preds["y_time"].to_numpy(),
                                 sage_preds["y_event"].to_numpy())
    logger.info(f"GraphSAGE overall C-index (pooled): {overall_ci:.4f}")

    # 2. KM curves
    logger.info("Plotting KM curves …")
    plot_km_curves(
        sage_preds["risk_score"].to_numpy(),
        sage_preds["y_time"].to_numpy(),
        sage_preds["y_event"].to_numpy(),
        title="GraphSAGE Survival — High vs Low Risk",
        save_path=EVAL_DIR / "km_curves_graphsage.png",
    )

    # 3. Time-dependent AUC
    auc_dict = compute_time_auc(sage_preds, EVAL_TIMES)
    for k, v in auc_dict.items():
        logger.info(f"  {k}: {v:.4f}")

    # 4. Saliency map (using last fold's checkpoint + its training data)
    try:
        logger.info("Computing saliency map …")
        index_df = pd.read_csv(INDEX_PATH)
        all_data = []
        for _, row in index_df.iterrows():
            pt = PYG_RAW_DIR / f"{row['acquisition_id']}.pt"
            if pt.exists():
                all_data.append(torch.load(pt, weights_only=False))

        ckpt_path = CKPT_DIR / f"fold_{N_CV_FOLDS}_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = GraphSAGESurvival(
                in_channels=N_NODE_FEATURES, hidden_dim=HIDDEN_DIM,
                n_layers=N_LAYERS, dropout=0.0,
                n_interaction_types=N_INTERACTION_TYPES,
                interaction_embed_dim=INTERACTION_EMBED_DIM,
            ).to(device)
            model.load_state_dict(ckpt["state_dict"])

            # Scale with saved scaler
            scaler = ckpt["scaler"]
            scaled_data = []
            for d in all_data[:100]:   # sample 100 graphs for speed
                import copy
                d2 = copy.deepcopy(d)
                d2.x = torch.tensor(scaler.transform(d2.x.numpy()), dtype=torch.float32)
                scaled_data.append(d2)

            saliency = compute_saliency(model, scaled_data, device)
            plot_saliency(saliency, EVAL_DIR / "feature_saliency.png")
            sal_df = pd.DataFrame({"feature": FEATURE_NAMES, "saliency": saliency})
            sal_df.sort_values("saliency", ascending=False).to_csv(
                EVAL_DIR / "feature_saliency.csv", index=False)
            logger.info(f"  Saliency saved to {EVAL_DIR / 'feature_saliency.png'}")
        else:
            logger.warning("No checkpoint found for saliency map")
    except Exception as e:
        logger.warning(f"Saliency computation failed: {e}")

    # 5. Comparison table
    logger.info("Building model comparison table …")
    table = build_comparison_table(["GraphSAGE", "GCN", "MLP", "RSF"])
    if not table.empty:
        table_path = EVAL_DIR / "model_comparison.csv"
        table.to_csv(table_path, index=False)
        logger.info(f"\n{table.to_string(index=False)}")
        logger.info(f"Table saved to {table_path}")

    # 6. Save summary metrics
    summary = {"model": "GraphSAGE", "pooled_cindex": overall_ci}
    summary.update(auc_dict)
    pd.DataFrame([summary]).to_csv(EVAL_DIR / "graphsage_metrics.csv", index=False)
    logger.info(f"\nAll evaluation outputs saved to {EVAL_DIR}")


if __name__ == "__main__":
    main()
