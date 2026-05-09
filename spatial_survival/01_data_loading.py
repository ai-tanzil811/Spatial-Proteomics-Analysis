"""
01_data_loading.py
Load and merge all source CSVs into a single per-cell DataFrame, then save:
  - output/spatial_graphs/merged_cells.parquet   (per-cell: coords, expression, cell-type)
  - output/spatial_graphs/survival_labels.csv    (per-sample: survival_status, survival_day, patient_id)

Run:
    python spatial_survival/01_data_loading.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CELL_LOCATIONS_FILE, EXPRESSION_FILE, METADATA_FILE, QC_FILE,
    PROTEIN_COLS, OUTPUT_DIR, MERGED_CELLS_FILE, SURVIVAL_LABELS_FILE,
)
from utils import get_logger, ensure_dirs, set_seed

logger = get_logger("data_loading")


# ---------------------------------------------------------------------------
# 1. Load QC-passed acquisition IDs
# ---------------------------------------------------------------------------

def load_qc_ids() -> set:
    # The QC file has no header — every row is an acquisition ID
    df = pd.read_csv(QC_FILE, header=None, names=["acquisition_id"])
    ids = set(df["acquisition_id"].str.strip())
    logger.info(f"QC-passed samples: {len(ids)}")
    return ids


# ---------------------------------------------------------------------------
# 2. Load survival labels
# ---------------------------------------------------------------------------

def load_survival_labels(qc_ids: set) -> pd.DataFrame:
    meta = pd.read_csv(METADATA_FILE)
    meta = meta.rename(columns={"acquisition_id": "ACQUISITION_ID"})
    meta["ACQUISITION_ID"] = meta["ACQUISITION_ID"].str.strip()

    # Keep only QC-passed samples
    meta = meta[meta["ACQUISITION_ID"].isin(qc_ids)].copy()

    # Drop samples with unknown patient_id (-1) — cannot safely cross-validate
    before = len(meta)
    meta = meta[meta["patient_id"] != -1].copy()
    logger.info(f"Dropped {before - len(meta)} samples with unknown patient_id (-1)")

    # Keep essential columns
    meta = meta[["ACQUISITION_ID", "patient_id", "survival_status", "survival_day"]].copy()

    # Drop rows missing survival information
    before = len(meta)
    meta = meta.dropna(subset=["survival_status", "survival_day"])
    if before != len(meta):
        logger.warning(f"Dropped {before - len(meta)} samples missing survival labels")

    meta["survival_status"] = meta["survival_status"].astype(int)
    meta["survival_day"]    = meta["survival_day"].astype(float)
    meta["patient_id"]      = meta["patient_id"].astype(int)

    logger.info(
        f"Survival labels: {len(meta)} samples | "
        f"events={meta['survival_status'].sum()} | "
        f"patients={meta['patient_id'].nunique()}"
    )
    return meta.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Load cell locations + cell types
# ---------------------------------------------------------------------------

def load_locations() -> pd.DataFrame:
    logger.info("Loading cell locations …")
    loc = pd.read_csv(CELL_LOCATIONS_FILE,
                      usecols=["ACQUISITION_ID", "CELL_ID", "X", "Y",
                               "CLUSTER_LABEL", "SIZE"])
    loc["ACQUISITION_ID"] = loc["ACQUISITION_ID"].str.strip()
    logger.info(f"  {len(loc):,} cells loaded")
    return loc


# ---------------------------------------------------------------------------
# 4. Load protein expression
# ---------------------------------------------------------------------------

def load_expression() -> pd.DataFrame:
    logger.info("Loading protein expression …")
    expr = pd.read_csv(EXPRESSION_FILE,
                       usecols=["sample_id", "cell_id"] + PROTEIN_COLS)
    expr = expr.rename(columns={"sample_id": "ACQUISITION_ID",
                                 "cell_id":   "CELL_ID"})
    expr["ACQUISITION_ID"] = expr["ACQUISITION_ID"].str.strip()
    logger.info(f"  {len(expr):,} cells loaded, {len(PROTEIN_COLS)} markers")
    return expr


# ---------------------------------------------------------------------------
# 5. Merge and impute
# ---------------------------------------------------------------------------

def merge_and_impute(loc: pd.DataFrame,
                     expr: pd.DataFrame,
                     qc_ids: set,
                     survival_ids: set) -> pd.DataFrame:
    logger.info("Merging locations + expression …")
    df = loc.merge(expr, on=["ACQUISITION_ID", "CELL_ID"], how="inner")
    logger.info(f"  After merge: {len(df):,} cells")

    # Keep only QC-passed samples that also have survival labels
    keep = qc_ids & survival_ids
    df = df[df["ACQUISITION_ID"].isin(keep)].copy()
    logger.info(f"  After QC + survival filter: {len(df):,} cells, "
                f"{df['ACQUISITION_ID'].nunique()} samples")

    # ---- Imputation --------------------------------------------------------
    # Flag cells where all 39 protein values are zero (likely segmentation noise)
    zero_mask = (df[PROTEIN_COLS] == 0).all(axis=1)
    n_zero = zero_mask.sum()
    if n_zero:
        logger.warning(f"  {n_zero} all-zero cells flagged as 'noisy' (kept but flagged)")
    df["all_zero_flag"] = zero_mask.astype(int)

    # Per-sample median imputation for NaN values in protein columns
    n_nan_before = df[PROTEIN_COLS].isna().sum().sum()
    if n_nan_before > 0:
        logger.info(f"  Imputing {n_nan_before} NaN values with per-sample median …")
        def _impute_group(g):
            medians = g[PROTEIN_COLS].median()
            g[PROTEIN_COLS] = g[PROTEIN_COLS].fillna(medians)
            return g
        df = df.groupby("ACQUISITION_ID", group_keys=False).apply(_impute_group)
        n_nan_after = df[PROTEIN_COLS].isna().sum().sum()
        if n_nan_after > 0:
            # Global fallback for any remaining NaNs (sample with all-NaN column)
            df[PROTEIN_COLS] = df[PROTEIN_COLS].fillna(0.0)
            logger.warning("  Remaining NaNs filled with 0 after per-sample median imputation")
    else:
        logger.info("  No NaN values in protein columns")

    # Validate required columns have no NaN
    required = ["ACQUISITION_ID", "CELL_ID", "X", "Y", "CLUSTER_LABEL"]
    n_missing = df[required].isna().sum().sum()
    assert n_missing == 0, f"Missing values in required columns: {df[required].isna().sum()}"

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(42)
    ensure_dirs(OUTPUT_DIR)

    qc_ids       = load_qc_ids()
    survival_df  = load_survival_labels(qc_ids)
    survival_ids = set(survival_df["ACQUISITION_ID"])

    loc_df  = load_locations()
    expr_df = load_expression()

    merged = merge_and_impute(loc_df, expr_df, qc_ids, survival_ids)

    # Save
    logger.info(f"Saving merged cells → {MERGED_CELLS_FILE}")
    MERGED_CELLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(MERGED_CELLS_FILE, index=False, engine="pyarrow")

    logger.info(f"Saving survival labels → {SURVIVAL_LABELS_FILE}")
    survival_df.to_csv(SURVIVAL_LABELS_FILE, index=False)

    # Summary
    logger.info("=== Data Loading Summary ===")
    logger.info(f"  Total cells    : {len(merged):,}")
    logger.info(f"  Total samples  : {merged['ACQUISITION_ID'].nunique()}")
    logger.info(f"  Cell types     : {sorted(merged['CLUSTER_LABEL'].unique())}")
    logger.info(f"  Avg cells/sample: {len(merged) / merged['ACQUISITION_ID'].nunique():.0f}")
    logger.info(f"  Noisy cells    : {merged['all_zero_flag'].sum()}")


if __name__ == "__main__":
    main()
