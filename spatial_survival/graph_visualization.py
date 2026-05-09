import numpy as np, pandas as pd
import matplotlib; print(matplotlib.__version__)
import matplotlib.pyplot as plt; print('matplotlib ok')
cells = pd.read_parquet('output/spatial_graphs/merged_cells.parquet')
acq = cells['ACQUISITION_ID'].unique()[3]
print('Sample:', acq)
sample = cells[cells['ACQUISITION_ID']==acq]
print('Cells:', len(sample))
print('Columns:', list(sample.columns[:8]))
data = np.load(f'output/spatial_graphs/edges/{acq}.npz')
print('edge_index shape:', data['edge_index'].shape)
print('edge_dist shape:', data['edge_dist'].shape)
print('sigma:', data['sigma'])
print('Cluster labels:', sample['CLUSTER_LABEL'].unique().tolist())

import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from pathlib import Path

ACQ_ID   = 'UPMC_c001_v001_r001_reg001'
OUT_PATH = Path('output/results/graph_visualization.png')
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
cells  = pd.read_parquet('output/spatial_graphs/merged_cells.parquet')
sample = cells[cells['ACQUISITION_ID'] == ACQ_ID].reset_index(drop=True)
xy     = sample[['X','Y']].to_numpy(dtype=float)
labels = sample['CLUSTER_LABEL'].to_numpy()

data       = np.load(f'output/spatial_graphs/edges/{ACQ_ID}.npz')
edge_index = data['edge_index']   # (2, E)
edge_dist  = data['edge_dist']    # (E,)
sigma      = float(data['sigma'])

# ── Colour palette ────────────────────────────────────────────────────────────
MACRO = {
    'Tumor':'#E63946','Tumor (CD15+)':'#E63946','Tumor (CD20+)':'#E63946',
    'Tumor (CD21+)':'#E63946','Tumor (Ki67+)':'#E63946','Tumor (Podo+)':'#E63946',
    'CD4 T cell':'#2196F3','CD8 T cell':'#0D47A1','B cell':'#64B5F6',
    'Macrophage':'#4CAF50','APC':'#81C784','Naive immune cell':'#A5D6A7',
    'Granulocyte':'#00BCD4',
    'Stromal / Fibroblast':'#FF9800','Vessel':'#FFCC80','Lymph vessel':'#FFE0B2',
}
node_colors = np.array([MACRO.get(l,'#999999') for l in labels])

# ── Sub-sample edges for readability (keep nearest 15% by distance) ───────────
n_show   = max(len(sample), 800)
rng      = np.random.default_rng(42)
sel_idx  = rng.choice(len(sample), n_show, replace=False)
sel_set  = set(sel_idx)
mask     = np.array([(int(edge_index[0,i]) in sel_set and
                       int(edge_index[1,i]) in sel_set)
                      for i in range(edge_index.shape[1])])
ei_sub   = edge_index[:, mask]
ed_sub   = edge_dist[mask]

# Keep only short edges (< 1.5 sigma) for visual clarity
dist_mask = ed_sub < 1.5 * sigma
ei_sub    = ei_sub[:, dist_mask]
ed_sub    = ed_sub[dist_mask]

print(f'Nodes shown: {n_show}  |  Edges shown: {ei_sub.shape[1]}')

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Panel A: full tissue map coloured by cell type (no edges)
ax = axes[0]
ax.set_facecolor('#111111')
for lbl, col in MACRO.items():
    m = labels == lbl
    if m.sum() > 0:
        ax.scatter(xy[m, 0], xy[m, 1], c=col, s=2, linewidths=0, alpha=0.7, label=lbl)
ax.set_title(f'Cell-type map\n{ACQ_ID}  ({len(sample):,} cells)', fontsize=11, color='white')
ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
ax.tick_params(colors='white'); ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
for spine in ax.spines.values(): spine.set_edgecolor('#444')
ax.set_aspect('equal')
lgd = ax.legend(loc='upper left', fontsize=6, markerscale=3,
                 framealpha=0.3, labelcolor='white',
                 facecolor='#222', edgecolor='#555')
ax.set_title(ax.get_title(), color='white')

# Panel B: sub-sampled graph with edges coloured by distance weight
ax = axes[1]
ax.set_facecolor('#111111')

# Draw edges as a LineCollection, alpha ~ distance weight
w = np.exp(-ed_sub**2 / (2 * sigma**2))
segs    = [(xy[ei_sub[0,i]], xy[ei_sub[1,i]]) for i in range(ei_sub.shape[1])]
lc      = LineCollection(segs, linewidths=0.4, alpha=0.35, color='#AAAAAA', zorder=1)
ax.add_collection(lc)

# Node scatter (sub-sample only)
ax.scatter(xy[sel_idx, 0], xy[sel_idx, 1],
           c=node_colors[sel_idx], s=12, linewidths=0, alpha=0.9, zorder=2)

ax.autoscale()
ax.set_aspect('equal')
ax.set_title(f'Hybrid Delaunay+kNN graph\n(sub-sample: {n_show} nodes, {ei_sub.shape[1]:,} edges)',
             fontsize=11, color='white')
ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
ax.tick_params(colors='white'); ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
for spine in ax.spines.values(): spine.set_edgecolor('#444')

# Macro-class legend
macro_patches = [
    mpatches.Patch(color='#E63946', label='Tumor'),
    mpatches.Patch(color='#2196F3', label='CD4 T cell'),
    mpatches.Patch(color='#0D47A1', label='CD8 T cell'),
    mpatches.Patch(color='#64B5F6', label='B cell'),
    mpatches.Patch(color='#4CAF50', label='Macrophage'),
    mpatches.Patch(color='#81C784', label='APC'),
    mpatches.Patch(color='#A5D6A7', label='Naive immune'),
    mpatches.Patch(color='#00BCD4', label='Granulocyte'),
    mpatches.Patch(color='#FF9800', label='Stroma/Fibroblast'),
    mpatches.Patch(color='#FFCC80', label='Vessel'),
    mpatches.Patch(color='#FFE0B2', label='Lymph vessel'),
]
ax.legend(handles=macro_patches, loc='upper left', fontsize=7,
           framealpha=0.3, labelcolor='white',
           facecolor='#222', edgecolor='#555')

fig.patch.set_facecolor('#1a1a2e')
plt.tight_layout()
fig.savefig(OUT_PATH, dpi=160, bbox_inches='tight')
print('Saved:', OUT_PATH)