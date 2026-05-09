"""
config.py
Central configuration: all paths, hyperparameters, and constants.
Adjust values here rather than editing individual scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent          # project root
DATA_DIR = ROOT_DIR / "data" / "dataset_info"
OUTPUT_DIR = ROOT_DIR / "output" / "spatial_graphs"
RESULTS_DIR = ROOT_DIR / "output" / "results"

# Input files
CELL_LOCATIONS_FILE  = DATA_DIR / "cell_locations_and_labels.csv"
EXPRESSION_FILE      = DATA_DIR / "labeled_arcsinh_norm_data.csv"
METADATA_FILE        = DATA_DIR / "sample_metadata.csv"
QC_FILE              = DATA_DIR / "qc_acq_ids_labeled.csv"
MARKER_NAMES_FILE    = DATA_DIR / "marker_names.csv"

# Intermediate outputs
MERGED_CELLS_FILE    = OUTPUT_DIR / "merged_cells.parquet"
SURVIVAL_LABELS_FILE = OUTPUT_DIR / "survival_labels.csv"
EDGES_DIR            = OUTPUT_DIR / "edges"
PYG_DATASET_DIR      = OUTPUT_DIR / "pyg_dataset"

# ---------------------------------------------------------------------------
# Protein marker columns (39 arcsinh-normalized markers)
# These are the exact column names in labeled_arcsinh_norm_data.csv
# ---------------------------------------------------------------------------
PROTEIN_COLS = [
    "CD117", "CD11b", "CD11c", "CD134", "CD14", "CD15", "CD152", "CD16",
    "CD20", "CD21", "CD31", "CD34", "CD38", "CD3e", "CD4", "CD45",
    "CD45RA", "CD45RO", "CD47", "CD49f", "CD56", "CD57", "CD68", "CD69",
    "CD8", "CollagenIV", "FoxP3", "GranzymeB", "HLA-DR", "ICOS", "Ki67",
    "PD1", "PDL1", "PanCK", "Podoplanin", "TMEM16A", "Vimentin", "aSMA", "p16",
]
N_PROTEINS = len(PROTEIN_COLS)   # 39

# Node feature dimension: 39 protein + 5 spatial features = 44
# Spatial: local_density, neighborhood_entropy, boundary_score, degree_centrality, expression_gradient
N_NODE_FEATURES = N_PROTEINS + 5

# ---------------------------------------------------------------------------
# Cell-type macro-class mapping for edge interaction type
# 16 CLUSTER_LABELs → 3 macro-classes: Tumor, Immune, Stroma
# ---------------------------------------------------------------------------
MACRO_CLASS = {
    # Tumor variants
    "Tumor (Podo+)":  "Tumor",
    "Tumor (CD20+)":  "Tumor",
    "Tumor (Ki67+)":  "Tumor",
    "Tumor (p16+)":   "Tumor",
    "Tumor":          "Tumor",
    "Tumor (other)":  "Tumor",
    # Immune cells
    "CD4 T cell":     "Immune",
    "CD8 T cell":     "Immune",
    "B cell":         "Immune",
    "Macrophage":     "Immune",
    "APC":            "Immune",
    "Granulocyte":    "Immune",
    "Naive immune":   "Immune",
    # Stroma / Vessel
    "Stromal/Fibroblast": "Stroma",
    "Vessel":             "Stroma",
    "Lymph vessel":       "Stroma",
}

INTERACTION_CODES = {
    ("Tumor",  "Tumor"):  0,   # homotypic
    ("Immune", "Immune"): 0,   # homotypic
    ("Stroma", "Stroma"): 0,   # homotypic
    ("Tumor",  "Immune"): 1,
    ("Immune", "Tumor"):  1,
    ("Tumor",  "Stroma"): 2,
    ("Stroma", "Tumor"):  2,
    ("Immune", "Stroma"): 3,
    ("Stroma", "Immune"): 3,
}
N_INTERACTION_TYPES = 5   # 0–4 (4 = unknown)

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
K_NEIGHBORS      = 5      # default kNN; sweep {3, 4, 5, 6} in ablation
DIST_THRESHOLD   = 200    # max edge length in pixels (~100 µm at 0.5 µm/px)

# ---------------------------------------------------------------------------
# GraphSAGE model
# ---------------------------------------------------------------------------
HIDDEN_DIM       = 64
N_LAYERS         = 3
DROPOUT          = 0.3
INTERACTION_EMBED_DIM = 8   # embedding size for interaction type

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4
BATCH_SIZE       = 16
MAX_EPOCHS       = 200
PATIENCE         = 20      # early stopping patience (epochs)
LR_PATIENCE      = 10      # ReduceLROnPlateau patience
SEED             = 42

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
N_CV_FOLDS       = 5       # patient-level GroupKFold
VAL_FRACTION     = 0.10    # fraction of train patients held out for validation

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_TIMES = [365, 730, 1095]   # days for time-dependent AUC
