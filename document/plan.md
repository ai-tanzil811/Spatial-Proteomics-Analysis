# Plan: Spatial Graph-Based Survival Prediction (Protein Expression)

## TL;DR
Directly use the 40 arcsinh-normalized protein expression markers from the MIBI dataset — no gene conversion — to build per-sample hybrid Delaunay+kNN spatial graphs, engineer biologically meaningful node/edge features, train a 3-layer GraphSAGE + Cox partial likelihood model with patient-level LOOCV, compare against baselines, and run ablation studies.

---

## Dataset Facts (Direct Protein Use)
- Cells: 570K+ across 378 ACQUISITION_IDs (samples), 7 patients
- Protein expression: 40 arcsinh-normalized markers in labeled_arcsinh_norm_data.csv — used as-is
- Spatial: X, Y pixel coordinates in cell_locations_and_labels.csv
- Cell types: 16 types in CLUSTER_LABEL column
- Survival: survival_status (0/1) + survival_day (days) in sample_metadata.csv; ~49% events
- ~1500 cells/sample average; QC-passed list in qc_acq_ids_labeled.csv
- Pixel scale: MIBI typically 0.39-0.5 um/px so 100um ~200-256 pixels

---

## File Structure (new folder spatial_survival/)
```
spatial_survival/
├── config.py                  ← all hyperparameters + paths
├── utils.py                   ← seed, C-index, KM helpers
├── 01_data_loading.py         ← merge CSVs → merged_cells.parquet
├── 02_graph_construction.py   ← Delaunay ∪ kNN per sample → edge files
├── 03_feature_engineering.py  ← node + edge feature computation
├── 04_normalize_and_save.py   ← normalization → PyG InMemoryDataset
├── 05_graphsage_model.py      ← GraphSAGE + pooling + Cox risk head
├── 06_training.py             ← LOOCV training loop + early stopping
├── 07_baselines.py            ← MLP, GCN, RSF
├── 08_evaluation.py           ← C-index, AUC, KM curves, saliency
└── 09_ablation.py             ← feature ablation + hyperparam sweep
```

---

## Phases

### Phase 1 - Configuration and Utilities
- config.py: DATA_DIR, OUTPUT_DIR, k_neighbors=5, dist_threshold_pixels=200, sigma=auto, hidden_dim=64, n_layers=3, dropout=0.3, lr=1e-3, weight_decay=1e-4, patience=20, seed=42, batch_size=16
- utils.py: set_seed(), compute_cindex(), plot_km_curves(), logger

### Phase 2 - Data Loading and Merging
- Load cell_locations_and_labels.csv: ACQUISITION_ID, X, Y, CLUSTER_LABEL, CELL_ID
- Load labeled_arcsinh_norm_data.csv: 40 protein cols + cell identifier; merge on ACQUISITION_ID + CELL_ID
- Load sample_metadata.csv; keep acquisition_id, patient_id, survival_status, survival_day
- Filter to 378 QC-passed IDs from qc_acq_ids_labeled.csv
- Impute: median per-marker within each sample for NaN; flag all-zero cells
- Output: output/spatial_graphs/merged_cells.parquet, survival_labels.csv

### Phase 3 - Graph Construction
- For each ACQUISITION_ID (378 graphs):
  - Extract N×2 coordinate array
  - Compute mean kNN distance to auto-set r and sigma_i
  - scipy.spatial.Delaunay triangulation, extract simplex edges
  - sklearn NearestNeighbors(n_neighbors=k=5), extract kNN edges
  - Union both edge sets; remove self-loops and duplicates
  - Filter edges > dist_threshold_pixels (200px); compute edge distances
  - Save edge_index (2xE), edge_distances (E,), sigma_i per sample

### Phase 4 - Node Feature Engineering (46-dim total)
1. Protein Expression (40-dim): arcsinh values direct from labeled_arcsinh_norm_data.csv
2. Local Density (1-dim): |N(i)| / (pi * r^2), r = per-sample mean neighbor dist
3. Neighborhood Entropy (1-dim): -sum(pk * log(pk)) over CLUSTER_LABEL in N(i)
4. Boundary Score (1-dim): |{j in N(i): type_j != type_i}| / |N(i)|
5. Degree Centrality (1-dim): degree(i) / (N-1)
6. Expression Gradient (1-dim): mean std across 40 protein values of N(i) cells
Note: betweenness/closeness skipped - O(N^3) x 378 samples is infeasible

### Phase 5 - Edge Feature Engineering (3-dim total)
1. Cosine Similarity: xi dot xj / (||xi|| * ||xj||) on 40-dim protein vectors
2. Distance Weight: exp(-d^2 / 2*sigma_i^2), sigma_i = per-sample mean edge dist
3. Interaction Type (int): map 16 CLUSTER_LABELs to 3 macro-classes (Tumor/Immune/Stroma+Vessel)
   0=homotypic, 1=Tumor-Immune, 2=Tumor-Stroma, 3=Immune-Immune, 4=other

### Phase 6 - Normalization and PyG Dataset
- StandardScaler on node features: fit on train cells only (inside CV fold), apply to val/test
- Edge: cosine clipped to [-1,1]; dist weight already [0,1]; interaction type as integer (embedded in model)
- Build torch_geometric.data.Data per sample: x(N×46), edge_index(2×E), edge_attr(E×3), y_time, y_event, patient_id
- Save as InMemoryDataset to output/spatial_graphs/pyg_dataset/

### Phase 7 - GraphSAGE Model
- 3x SAGEConv(in->64) + BatchNorm1d + ReLU + Dropout(0.3)
- Global pooling: concat(global_mean_pool, global_max_pool) -> 128-dim
- Risk head: Linear(128->64) -> ReLU -> Dropout(0.3) -> Linear(64->1) -> scalar theta
- Edge attr: distance_weight multiplies neighbor features in custom MessagePassing; interaction type as Embedding(5,8) added to source node features pre-conv
- Cox partial likelihood loss (Breslow approx): L = -sum_i [theta_i - log sum_{j in R(ti)} exp(theta_j)]
- Use pycox CoxPHLoss or hand-rolled; L2 weight decay on Linear layers

### Phase 8 - Training Loop (Patient-Level LOOCV)
- 7-fold LOOCV: test = all samples of 1 held-out patient; train = 6 remaining patients; val = 10% random from train
- Scaler fit on train only each fold (no leakage)
- Adam(lr=1e-3, weight_decay=1e-4) + ReduceLROnPlateau(patience=10) on val C-index
- Early stopping: patience=20, maximize val C-index; save best checkpoint per fold
- DataLoader(batch_size=16); sort by event time within batch for Cox loss
- Log: epoch, train_loss, val_cindex per fold -> output/results/training_logs/

### Phase 9 - Baselines (parallel with Phase 7-8)
- MLP: per-sample global mean of 40 protein expressions -> Linear(40->64->32->1) + Cox; same LOOCV
- GCN: GCNConv replacing SAGEConv, identical otherwise
- RSF: sksurv.ensemble.RandomSurvivalForest on per-sample mean expression (40 features); same LOOCV
- All: same seeds, same patient splits

### Phase 10 - Evaluation and Visualization
- Pool risk scores across all 7 folds -> overall C-index
- Time-dependent AUC at t=365, 730, 1095 days (sksurv.metrics.cumulative_dynamic_auc)
- KM curves: high vs low risk (median split) + log-rank p-value (lifelines)
- Gradient saliency map: which of 46 node features drive predictions
- Summary table: GraphSAGE vs GCN vs MLP vs RSF on C-index + AUC(t)
- Save all to output/results/

### Phase 11 - Ablation and Hyperparameter Tuning
- Feature ablation (7 runs): zero out one group at a time; record delta C-index vs full model
  Groups: Expression(40), Density, Entropy, Boundary, Degree, Gradient, Edge features
- k sweep: k in {3, 4, 5, 6}
- sigma sweep: sigma in {0.5x, 1x, 2x} of mean edge distance
- Depth: 1, 2, 3, 4 GraphSAGE layers
- Hidden dim: 32, 64, 128
- Report all in output/results/ablation_table.csv

---

## Relevant Files
- data/dataset_info/labeled_arcsinh_norm_data.csv — PRIMARY protein expression (40 markers, arcsinh)
- data/dataset_info/cell_locations_and_labels.csv — X, Y, CLUSTER_LABEL, ACQUISITION_ID
- data/dataset_info/sample_metadata.csv — survival_status, survival_day, patient_id
- data/dataset_info/qc_acq_ids_labeled.csv — QC-passed sample list
- data/dataset_info/marker_names.csv — 40 protein marker names
NOT USED: gene_expression_dataset.csv, mibi_as_gene_expression.h5ad, protein_to_genomic/ scripts

---

## Verification
1. Phase 2: merged_cells ~570K rows; no NaN in X/Y/CLUSTER_LABEL; survival_labels = 378 rows
2. Phase 3: avg edge count per sample ~3K-8K; no edge > 200px; zero self-loops; sigma_i stored
3. Phase 4-5: density >= 0; entropy in [0, log(16)]; cosine in [-1,1]; dist_weight in [0,1]
4. Phase 6: PyG Data loads; x.shape=(N,46), edge_attr.shape=(E,3)
5. Phase 8: train loss decreases; val C-index > 0.5 by epoch 10; no NaN loss
6. Phase 10: overall C-index ~0.55-0.70; log-rank p < 0.05 desirable
7. Phase 11: ablation table shows at least 2 features with positive delta C-index

---

## Decisions and Scope
- Protein expression used DIRECTLY - no gene conversion
- Distance threshold: 200px default (~100um at 0.5um/px); tunable
- CV: patient-level LOOCV (7 folds) - only safe option with 7 patients
- Edge attr: custom MessagePassing for distance weight; interaction type as pre-conv node embedding
- Betweenness/closeness centrality excluded (compute cost)
- OUT OF SCOPE: gene-level analysis, CellChat, external validation, temporal dynamics
- 
## Further Considerations
- Pixel scale: If the exact µm/pixel value is in dataset documentation, apply conversion before distance thresholding. Otherwise document 200px = ~100µm assumption throughout.
- Edge features in SAGEConv: Three options if custom MessagePassing proves complex — (A) custom (recommended, simplest), (B) upgrade to GATv2Conv which natively accepts edge attr, (C) GINEConv for full expressivity. Recommend (A) first, test (B) in ablation.
- Low event count per fold: ~49% events overall, but individual patients may have 0 events in test fold → C-index undefined. Fallback: use time-dependent AUC for that fold.


Training is progressing — Fold 1, Epoch 1: loss=6.63, val C-index=0.61. This will run for up to 200 epochs × 5 folds on CPU. You can monitor it via the training log file:
output/results/training_logs/fold_1_log.csv

Or check the live log at training.log. Training will auto-stop early (patience=20 epochs without val C-index improvement) and save checkpoints to output/results/checkpoints/fold_<k>_best.pt.

When training finishes, run in sequence:



python spatial_survival/07_baselines.py   # MLP, GCN, RSF baselines
python spatial_survival/08_evaluation.py  # C-index, AUC, KM curves, comparison table