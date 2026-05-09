"""
03_feature_engineering.py
Compute node features (46-dim) and edge features (3-dim) for every sample graph.

Node features (per cell i):
  [0:39]  Protein expression  (39-dim, arcsinh-normalized)
  [39]    Local density        ρ = |N(i)| / (π r²)
  [40]    Neighborhood entropy H = -Σ pₖ log(pₖ)  over cell-type labels in N(i)
  [41]    Boundary score       B = |{j∈N(i): type_j≠type_i}| / |N(i)|
  [42]    Degree centrality    d / (N-1)
  [43]    Expression gradient  mean std of protein vectors across N(i)

Edge features (per directed edge i→j):
  [0]  Cosine similarity    xᵢ·xⱼ / (‖xᵢ‖‖xⱼ‖)
  [1]  Distance weight      exp(-d²/2σ²)
  [2]  Interaction type     0=homotypic, 1=Tumor-Immune, 2=Tumor-Stroma,
                             3=Immune-Immune, 4=other  (integer)

Output per sample saved to  output/spatial_graphs/features/<ACQUISITION_ID>.npz
  Arrays: node_features (N x 44), edge_features (E x 3)

Run:
    python spatial_survival/03_feature_engineering.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MERGED_CELLS_FILE, EDGES_DIR, OUTPUT_DIR,
    PROTEIN_COLS, N_PROTEINS, MACRO_CLASS, INTERACTION_CODES, SEED,
)
from utils import get_logger, ensure_dirs, set_seed

FEATURES_DIR = OUTPUT_DIR / "features"

logger = get_logger("feature_engineering")


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------

def _local_density(neighbors: list[list[int]],
                   r: float,
                   n_cells: int) -> np.ndarray:
    """ρᵢ = |N(i)| / (π r²)  — normalised by disk area."""
    area = np.pi * (r ** 2) if r > 0 else 1.0
    degrees = np.array([len(nb) for nb in neighbors], dtype=np.float32)
    return degrees / area


def _neighborhood_entropy(neighbors: list[list[int]],
                           labels: np.ndarray) -> np.ndarray:
    """Shannon entropy of cell-type distribution in each cell's neighbourhood."""
    n = len(neighbors)
    entropy = np.zeros(n, dtype=np.float32)
    for i, nb in enumerate(neighbors):
        if len(nb) == 0:
            continue
        nb_labels = labels[nb]
        _, counts = np.unique(nb_labels, return_counts=True)
        p = counts / counts.sum()
        entropy[i] = float(-np.sum(p * np.log(p + 1e-12)))
    return entropy


def _boundary_score(neighbors: list[list[int]],
                    labels: np.ndarray) -> np.ndarray:
    """Fraction of neighbours with a different cell-type label."""
    n = len(neighbors)
    score = np.zeros(n, dtype=np.float32)
    for i, nb in enumerate(neighbors):
        if len(nb) == 0:
            continue
        nb_labels = labels[nb]
        score[i] = float((nb_labels != labels[i]).sum()) / len(nb)
    return score


def _degree_centrality(degrees: np.ndarray, n_cells: int) -> np.ndarray:
    denom = max(n_cells - 1, 1)
    return (degrees / denom).astype(np.float32)


def _expression_gradient(neighbors: list[list[int]],
                          expression: np.ndarray) -> np.ndarray:
    """Mean std of 39-dim protein vectors across neighbours of each cell."""
    n = len(neighbors)
    gradient = np.zeros(n, dtype=np.float32)
    for i, nb in enumerate(neighbors):
        if len(nb) == 0:
            continue
        nb_expr = expression[nb]          # (|N|, 39)
        gradient[i] = float(np.mean(np.std(nb_expr, axis=0)))
    return gradient


def compute_node_features(expression: np.ndarray,
                           labels: np.ndarray,
                           edge_index: np.ndarray,
                           sigma: float) -> np.ndarray:
    """
    Assemble the 45-dim node feature matrix for one sample.
    edge_index: (2, E) directed (both directions stored)
    """
    n_cells = len(expression)

    # Build adjacency list from directed edge_index (only unique undirected)
    neighbors: list[list[int]] = [[] for _ in range(n_cells)]
    if edge_index.shape[1] > 0:
        src, dst = edge_index[0], edge_index[1]
        for s, d in zip(src.tolist(), dst.tolist()):
            neighbors[s].append(d)

    degrees = np.array([len(nb) for nb in neighbors], dtype=np.float32)
    r = sigma  # use sigma as characteristic radius

    density    = _local_density(neighbors, r, n_cells)
    entropy    = _neighborhood_entropy(neighbors, labels)
    boundary   = _boundary_score(neighbors, labels)
    deg_cent   = _degree_centrality(degrees, n_cells)
    gradient   = _expression_gradient(neighbors, expression)

    # Concatenate: [expression(39), density(1), entropy(1), boundary(1),
    #               deg_centrality(1), gradient(1)] = 44-dim
    node_feats = np.concatenate([
        expression,            # (N, 39)
        density[:, None],      # (N, 1)
        entropy[:, None],
        boundary[:, None],
        deg_cent[:, None],
        gradient[:, None],
    ], axis=1).astype(np.float32)

    return node_feats


# ---------------------------------------------------------------------------
# Edge features
# ---------------------------------------------------------------------------

def _interaction_type(label_i: str, label_j: str,
                       macro: dict, codes: dict) -> int:
    mc_i = macro.get(label_i, "Unknown")
    mc_j = macro.get(label_j, "Unknown")
    return codes.get((mc_i, mc_j), 4)


def compute_edge_features(expression: np.ndarray,
                           labels: np.ndarray,
                           edge_index: np.ndarray,
                           edge_dist: np.ndarray,
                           sigma: float) -> np.ndarray:
    """
    Assemble the 3-dim edge feature matrix.
    edge_index: (2, E), edge_dist: (E,)
    Returns np.ndarray (E, 3)  [cosine_sim, dist_weight, interaction_type]
    """
    E = edge_index.shape[1]
    if E == 0:
        return np.empty((0, 3), dtype=np.float32)

    src = edge_index[0]
    dst = edge_index[1]

    # Cosine similarity
    xi = expression[src]   # (E, 39)
    xj = expression[dst]   # (E, 39)
    norm_i = np.linalg.norm(xi, axis=1, keepdims=True) + 1e-12
    norm_j = np.linalg.norm(xj, axis=1, keepdims=True) + 1e-12
    cosine_sim = np.sum((xi / norm_i) * (xj / norm_j), axis=1)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

    # Distance weight
    sigma2 = sigma ** 2 if sigma > 0 else 1.0
    dist_weight = np.exp(-(edge_dist ** 2) / (2 * sigma2))

    # Interaction type
    itype = np.array(
        [_interaction_type(labels[s], labels[d], MACRO_CLASS, INTERACTION_CODES)
         for s, d in zip(src.tolist(), dst.tolist())],
        dtype=np.float32,
    )

    edge_feats = np.stack([cosine_sim, dist_weight, itype], axis=1).astype(np.float32)
    return edge_feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    ensure_dirs(FEATURES_DIR)

    logger.info(f"Loading merged cells from {MERGED_CELLS_FILE} …")
    cells = pd.read_parquet(MERGED_CELLS_FILE)

    acquisition_ids = sorted(cells["ACQUISITION_ID"].unique())
    logger.info(f"Computing features for {len(acquisition_ids)} samples …")

    missing_edge_files = []
    for acq_id in tqdm(acquisition_ids, desc="Features"):
        edge_path = EDGES_DIR / f"{acq_id}.npz"
        if not edge_path.exists():
            missing_edge_files.append(acq_id)
            continue

        sample = cells[cells["ACQUISITION_ID"] == acq_id].reset_index(drop=True)
        expression = sample[PROTEIN_COLS].to_numpy(dtype=np.float32)  # (N, 39)
        labels     = sample["CLUSTER_LABEL"].to_numpy()                # (N,)

        edge_data  = np.load(edge_path, allow_pickle=False)
        edge_index = edge_data["edge_index"]   # (2, E)
        edge_dist  = edge_data["edge_dist"]    # (E,)
        sigma      = float(edge_data["sigma"])

        node_feats = compute_node_features(expression, labels, edge_index, sigma)
        edge_feats = compute_edge_features(expression, labels, edge_index, edge_dist, sigma)

        out_path = FEATURES_DIR / f"{acq_id}.npz"
        np.savez_compressed(
            out_path,
            node_features=node_feats,
            edge_features=edge_feats,
        )

    if missing_edge_files:
        logger.warning(
            f"{len(missing_edge_files)} samples skipped (no edge file). "
            "Run 02_graph_construction.py first."
        )

    logger.info("=== Feature Engineering Summary ===")
    # Spot-check feature ranges on a random sample
    sample_files = list(FEATURES_DIR.glob("*.npz"))
    if sample_files:
        spot = np.load(sample_files[0])
        nf = spot["node_features"]
        ef = spot["edge_features"]
        logger.info(f"  Sample: {sample_files[0].stem}")
        logger.info(f"  node_features shape: {nf.shape}  | "
                    f"range [{nf.min():.3f}, {nf.max():.3f}]")
        logger.info(f"  edge_features shape: {ef.shape}  | "
                    f"cosine range [{ef[:,0].min():.3f}, {ef[:,0].max():.3f}] | "
                    f"dist_weight range [{ef[:,1].min():.3f}, {ef[:,1].max():.3f}] | "
                    f"itype unique: {np.unique(ef[:,2]).astype(int).tolist()}")
    logger.info(f"  Features written to {FEATURES_DIR}")


if __name__ == "__main__":
    main()
