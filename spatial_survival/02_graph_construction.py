"""
02_graph_construction.py
Build one hybrid spatial graph per ACQUISITION_ID:
  - Delaunay triangulation (global connectivity)
  - kNN graph (local connectivity, default k=5)
  - Union of both, filter edges > DIST_THRESHOLD pixels, remove self-loops

Output per sample saved to  output/spatial_graphs/edges/<ACQUISITION_ID>.npz
  Arrays: edge_index (2 x E), edge_dist (E,), sigma (scalar mean edge dist)

Run:
    python spatial_survival/02_graph_construction.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MERGED_CELLS_FILE, EDGES_DIR, K_NEIGHBORS, DIST_THRESHOLD, SEED
)
from utils import get_logger, ensure_dirs, set_seed

logger = get_logger("graph_construction")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _delaunay_edges(coords: np.ndarray) -> np.ndarray:
    """Return undirected edge array (E x 2) from Delaunay triangulation."""
    if len(coords) < 3:
        return np.empty((0, 2), dtype=np.int64)
    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = int(simplex[i]), int(simplex[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    return np.array(list(edges), dtype=np.int64)


def _knn_edges(coords: np.ndarray, k: int) -> np.ndarray:
    """Return undirected edge array (E x 2) from kNN graph."""
    n = len(coords)
    if n <= k:
        k = n - 1
    if k < 1:
        return np.empty((0, 2), dtype=np.int64)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    edges = set()
    for i, nbr_row in enumerate(indices):
        for j in nbr_row:
            if i != j:
                edges.add((min(i, j), max(i, j)))
    return np.array(list(edges), dtype=np.int64)


def build_graph(coords: np.ndarray,
                k: int = K_NEIGHBORS,
                dist_threshold: float = DIST_THRESHOLD):
    """
    Build hybrid Delaunay ∪ kNN graph.
    Returns:
        edge_index : np.ndarray (2, E)  — source/target node indices
        edge_dist  : np.ndarray (E,)    — Euclidean distances
        sigma      : float              — mean edge distance (for RBF weight)
    """
    del_edges = _delaunay_edges(coords)
    knn_edges = _knn_edges(coords, k)

    if len(del_edges) == 0 and len(knn_edges) == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        return edge_index, np.array([]), 1.0

    # Union
    if len(del_edges) > 0 and len(knn_edges) > 0:
        all_edges = np.unique(np.vstack([del_edges, knn_edges]), axis=0)
    elif len(del_edges) > 0:
        all_edges = del_edges
    else:
        all_edges = knn_edges

    # Compute distances
    src, dst = all_edges[:, 0], all_edges[:, 1]
    dists = np.linalg.norm(coords[src] - coords[dst], axis=1)

    # Filter edges beyond distance threshold
    mask = dists <= dist_threshold
    src, dst, dists = src[mask], dst[mask], dists[mask]

    if len(dists) == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        return edge_index, np.array([]), 1.0

    sigma = float(np.mean(dists)) if len(dists) > 0 else 1.0

    # Make undirected: add reverse edges
    src_full  = np.concatenate([src, dst])
    dst_full  = np.concatenate([dst, src])
    dists_full = np.concatenate([dists, dists])

    edge_index = np.stack([src_full, dst_full], axis=0).astype(np.int64)
    return edge_index, dists_full.astype(np.float32), sigma


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(k: int = K_NEIGHBORS, dist_threshold: float = DIST_THRESHOLD):
    set_seed(SEED)
    ensure_dirs(EDGES_DIR)

    logger.info(f"Loading merged cells from {MERGED_CELLS_FILE} …")
    cells = pd.read_parquet(MERGED_CELLS_FILE,
                            columns=["ACQUISITION_ID", "CELL_ID", "X", "Y"])

    acquisition_ids = sorted(cells["ACQUISITION_ID"].unique())
    logger.info(f"Building graphs for {len(acquisition_ids)} samples "
                f"(k={k}, dist_threshold={dist_threshold} px) …")

    stats = []
    for acq_id in tqdm(acquisition_ids, desc="Graphs"):
        sample = cells[cells["ACQUISITION_ID"] == acq_id].reset_index(drop=True)
        coords = sample[["X", "Y"]].to_numpy(dtype=np.float32)

        edge_index, edge_dist, sigma = build_graph(
            coords, k=k, dist_threshold=dist_threshold
        )

        out_path = EDGES_DIR / f"{acq_id}.npz"
        np.savez_compressed(
            out_path,
            edge_index=edge_index,
            edge_dist=edge_dist,
            sigma=np.array(sigma, dtype=np.float32),
            n_nodes=np.array(len(coords), dtype=np.int64),
        )

        n_edges = edge_index.shape[1] // 2  # undirected count
        stats.append({
            "acquisition_id": acq_id,
            "n_nodes": len(coords),
            "n_edges": n_edges,
            "sigma": sigma,
        })

    stats_df = pd.DataFrame(stats)
    stats_path = EDGES_DIR / "graph_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    logger.info("=== Graph Construction Summary ===")
    logger.info(f"  Samples built  : {len(stats_df)}")
    logger.info(f"  Avg nodes      : {stats_df['n_nodes'].mean():.0f}")
    logger.info(f"  Avg edges      : {stats_df['n_edges'].mean():.0f}")
    logger.info(f"  Avg sigma (px) : {stats_df['sigma'].mean():.1f}")
    logger.info(f"  Min/Max nodes  : {stats_df['n_nodes'].min()} / {stats_df['n_nodes'].max()}")
    logger.info(f"  Stats saved to : {stats_path}")


if __name__ == "__main__":
    main()
