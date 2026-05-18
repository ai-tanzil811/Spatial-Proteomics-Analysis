# Comparative Analysis: Cell-Cell Communication & Integration Pipelines

**Document Date:** May 2026  
**Focus:** MaxFUSE, NATMI, RNAchat, CellChat — Comparative boundary analysis and mitigation strategies

---

## Executive Summary

Four major computational frameworks address cell-cell communication and multi-omics integration:
- **MaxFUSE**: Data fusion engine for integrating spatial + single-cell omics
- **NATMI**: Network-based ligand-receptor inference from single-cell data
- **RNAchat**: Clinical phenotype-linked pathway analysis via ICA + ML
- **CellChat**: Curated ligand-receptor database + communication probability scoring

Each excels in different contexts. **No single tool handles the complete workflow** from raw spatial proteomics → communication networks → clinical phenotype associations. This document maps their boundaries and proposes integrated mitigation strategies.

---

## 1. Comparative Analysis Table

| **Dimension** | **MaxFUSE** | **NATMI** | **RNAchat** | **CellChat** |
|---|---|---|---|---|
| **Primary Goal** | Data fusion: align spatial + scRNA-seq | Network inference: cell type communication | Pathway phenotype association | Direct L-R communication probability |
| **Input Data Types** | Spatial (image-based), scRNA-seq, proteomics | scRNA-seq (cell type + expression) | Bulk expression + clinical metadata | scRNA-seq, spatial transcriptomics |
| **Core Algorithm** | Graph alignment, matrix factorization | Co-expression network + hypergeometric test | ICA-derived metapathways + ML | Ligand-receptor DB matching + NMF |
| **Cell Type Requirement** | Required (cell type labels) | Required (cell type labels) | Optional (uses clinical labels instead) | Required (cell type labels) |
| **Spatial Information** | Explicitly leveraged (coordinates) | Ignored (use distance cutoff post-hoc) | Not used | Can incorporate spatial proximity |
| **Clinical Integration** | Indirect (via fused representations) | Not designed for clinical phenotypes | Direct (integrated in pipeline) | Not designed for clinical phenotypes |
| **Output Types** | Fused embeddings, correspondence matrix | Co-expression edges, p-values | Metagene weights, SHAP explanations, causal effects | Interaction pairs, probability scores, spatial plots |
| **Methodology Detail** | Multi-view learning (e.g., Seurat, scVI) | Statistical network inference | Unsupervised feature extraction → supervised ML | Curated database + probabilistic communication |
| **Key Strengths** | ✓ Multi-omics integration ✓ Preserves spatial context ✓ Directly aligns modalities | ✓ Data-driven ✓ No DB dependency ✓ Fast ✓ Statistical rigor | ✓ Clinical phenotype links ✓ Interpretability (SHAP) ✓ Causal inference (DML) ✓ Multi-omics friendly | ✓ Curated biology ✓ Probability-based ✓ Visualization ✓ Well-validated |
| **Key Limitations** | ✗ Assumes matched spatial/scRNA ✗ Computationally expensive ✗ No native clinical output ✗ Requires hyperparameter tuning | ✗ Limited to co-expression (not direct L-R) ✗ No spatial context ✗ Ignores phenotypes ✗ Parameter sensitivity | ✗ Depends on ICA preprocessing ✗ Limited to bulk-like expression ✗ No direct L-R communication ✗ Requires ID_REF matching | ✗ DB-dependent (outdated curations) ✗ No statistical inference of new interactions ✗ Weak at novel ligands/receptors ✗ Does not infer functional outcome |
| **Scalability** | Medium (10k–100k cells) | High (100k+ cells) | High (bulk samples) | Medium (memory-intensive at 10k+ pairs) |
| **Computational Cost** | High (GPU beneficial) | Low-moderate | Low-moderate | Medium |
| **Dependency Profile** | Python, deep learning library (PyTorch/TF) | Python, scipy, networkx | Python (scikit-learn, causal-ml) | Python/R (Seurat compatible) |
| **Maturity & Adoption** | Growing (recent methods) | Moderate (published, less widely adopted) | Established (peer-reviewed web platform) | High (widely used, curated DB maintenance) |
| **Native Visualization** | Limited (embeddings, heatmaps) | Network plots | Dashboard (correlation, SHAP, DML) | Rich (spatial plots, network, dot plots) |
| **Typical Use Case** | "Map cell types from spatial to scRNA donor cohort" | "Which cell types talk to which?" | "Which pathways predict drug response?" | "What ligands mediate immune checkpoint?" |

---

## 2. Pipeline Descriptions & Boundaries

### 2.1 MaxFUSE: Multi-Modal Data Fusion

**Pipeline Overview:**
```
Spatial Proteomics (image coords + proteins)  ─┐
                                               ├─> Data Normalization
scRNA-seq (cell type + expression)            ─┤    & Alignment
                                               ├─> Multiview Embedding
Optional: Genomics, Metabolomics              ─┤    (e.g., NMF, Seurat)
                                               │
                                               ├─> Feature Correspondence Matrix
                                               │    (which features align across modes)
                                               │
                                               └─> Fused Representation
                                                   (joint latent space)
```

**What It Does:**
- Aligns multiple omics modalities (spatial, scRNA, proteomics, etc.) into a shared latent space
- Creates cross-modal correspondence (e.g., "protein X in space maps to gene Y in scRNA")
- Leverages spatial coordinates to inform alignment
- Outputs fused embeddings suitable for downstream analysis

**Boundaries & Limitations:**
| Boundary | Impact | Example |
|----------|--------|---------|
| Assumes matched samples | Fails if spatial and scRNA are from different donors/conditions | Can't directly use public spatial + internal scRNA unless carefully matched |
| Cell type labels required for spatial | Preprocessing burden; quality depends on segmentation | Tissue images without cell segmentation are incompatible |
| Computationally expensive | Not practical for 100k+ cell studies on CPU | Requires GPU for iterative optimization |
| Hyperparameter sensitivity | Fusion quality depends on weights, dimensions, regularization | Results may vary widely; requires validation |
| No inherent statistical testing | Hard to answer "are these associations significant?" | Downstream testing needed (e.g., permutation tests) |
| Indirect clinical link | Does not natively connect fused features to phenotype outcomes | Requires secondary ML pipeline to link embeddings to outcomes |

**When to Use:**
- ✓ You have matched spatial + scRNA-seq from same samples
- ✓ Goal is to map cell types or features across modalities
- ✓ Computational resources (GPU) available
- ✓ You can validate alignment quality independently

**When NOT to Use:**
- ✗ Only single modality (scRNA or spatial, not both)
- ✗ Unmatched samples (different donors, batches)
- ✗ Need direct ligand-receptor communication (use CellChat)
- ✗ Need direct clinical outcome link (use RNAchat)

---

### 2.2 NATMI: Network Analysis of Cellular Interactions

**Pipeline Overview:**
```
scRNA-seq (cell type + expression) ─┐
                                    ├─> Cell type separation
                                    │   (subset each type)
                                    ├─> Co-expression network
                                    │   (within cell type)
                                    │
                                    ├─> Hypergeometric test:
                                    │   P(ligand & receptor co-expressed)
                                    │
                                    └─> Communication pairs
                                        (cell type A → B via L-R)
```

**What It Does:**
- Computes co-expression networks within and between cell types
- Tests whether ligand-receptor pairs are significantly co-expressed
- No external database dependency (data-driven)
- Outputs edge list: source cell type → target cell type, ligand, receptor, p-value

**Boundaries & Limitations:**
| Boundary | Impact | Example |
|----------|--------|---------|
| Confuses co-expression with direct binding | May infer indirect signaling as direct | Transcription factor A + ligand B co-expressed doesn't mean A secretes B |
| No spatial distance filtering | All cell types assumed equally likely to interact | Predicts immune cell–epithelial communication even if physically distant |
| Limited to known ligands/receptors | Requires pre-curated gene sets (e.g., CellChatDB) | Novel or context-specific ligands/receptors not discovered |
| Single-cell-only | Bulk data (spatial proteomics or tissue) not compatible | Spatial proteomics must be converted to cell-type signatures first |
| No statistical power assessment | P-values can be misleading in sparse matrices | Small cell type populations → inflated p-values |
| Parameter sensitivity | Results depend on correlation threshold, network density | No clear guidance on robustness |
| Ignores phenotype outcomes | Communication network exists independent of clinical significance | May identify interactions irrelevant to disease or treatment |

**When to Use:**
- ✓ Have high-quality single-cell data with clear cell types
- ✓ Want data-driven communication inference (no DB bias)
- ✓ Need fast computation on large cohorts
- ✓ Interested in novel ligand-receptor pairs

**When NOT to Use:**
- ✗ Need spatial proximity constraints (use CellChat + spatial)
- ✗ Want clinically validated ligand-receptor pairs (use CellChat DB)
- ✗ Have bulk data without cell type decomposition
- ✗ Need to link interactions to disease outcomes (use RNAchat)

---

### 2.3 RNAchat: Clinical Phenotype-Pathway Integration

**Pipeline Overview:**
```
Bulk Expression (CSV: gene × sample) ─┐
                                      ├─> Data Integration on ID_REF
Clinical Metadata (CSV: ID_REF + label) ┤
                                      │
                                      ├─> Independent Component Analysis (ICA)
                                      │   (compress genes → metagenes)
                                      │
                                      ├─> Metagene-Phenotype Association
                                      │   (correlation, ML models)
                                      │
                                      ├─> Interpretation (SHAP, DML)
                                      │   (which genes drive phenotype?)
                                      │
                                      └─> Causal Heterogeneity (Causal Forest)
                                          (subgroup-specific effects)
```

**What It Does:**
- Integrates bulk expression with clinical covariates on sample ID (ID_REF)
- Decomposes genes into interpretable metagene modules via ICA
- Links metagenes to clinical phenotypes (Random Forest, XGBoost, LightGBM)
- Provides SHAP-based feature importance and causal effect heterogeneity
- Web platform for interactive exploration

**Boundaries & Limitations:**
| Boundary | Impact | Example |
|----------|--------|---------|
| Bulk expression only | Cell type signals averaged out; single-cell lost | Spatial proteomics per-cell data must be aggregated to cell types or regions |
| ICA preprocessing dependency | Metagenes only meaningful if ICA converges well | Highly sparse or batch-heavy data → poor ICA decomposition |
| Requires ID_REF matching | Integration fails if sample IDs don't overlap | Different ID formats across expression and metadata → data loss |
| No direct L-R communication | Does not infer ligand-receptor pairs | Cannot answer "which ligands mediate effect X?" directly |
| No spatial coordinates | Loses location information even if spatial data available | Spatial proteomics must be spatially-aggregate or region-level |
| Limited database validation | Metagenes are data-driven; may lack biological validation | A metagene may be statistical artifact, not biological pathway |
| Phenotype-centric only | Focuses on clinical labels; ignores cell-type-specific biology | Cannot answer "what do T cells do in this condition?" |

**When to Use:**
- ✓ Have bulk expression + clinical metadata from same cohort
- ✓ Goal is to predict or explain clinical phenotypes (response, severity, outcome)
- ✓ Want interpretability via SHAP or causal inference
- ✓ Multi-omics available (protein, metabolomics, etc. with same ID_REF)

**When NOT to Use:**
- ✗ Need cell-cell communication inference (use CellChat or NATMI)
- ✗ Have high-dimensional spatial data with cell-level resolution
- ✗ No clinical phenotype; purely exploratory cell biology
- ✗ Need validated biological pathways (use CellChat DB or curated pathways)

---

### 2.4 CellChat: Ligand-Receptor Communication Framework

**Pipeline Overview:**
```
scRNA-seq (cell type + expression) ┐
OR                                  ├─> Cell type expression inference
Spatial Transcriptomics (SPT)       ┤   (gene expression per cell type)
                                    │
                                    ├─> CellChatDB Ligand-Receptor Pairs
                                    │   (curated interactions: L from A, R on B)
                                    │
                                    ├─> Interaction Probability
                                    │   (NMF decomposition of L×R expression)
                                    │
                                    ├─> Communication Network
                                    │   (cell type A → B, ligand, strength)
                                    │
                                    ├─> Pathway Analysis
                                    │   (which biological pathways drive communication?)
                                    │
                                    └─> Visualization & Inference
                                        (dot plots, spatial maps, network diagrams)
```

**What It Does:**
- Matches expressed ligands in source cell type with expressed receptors in target cell type
- Scores interaction strength via non-negative matrix factorization
- Aggregates ligand-receptor interactions into signaling pathways
- Visualizes communication networks and spatial distribution
- Handles spatial transcriptomics natively (proximity-weighted scoring)

**Boundaries & Limitations:**
| Boundary | Impact | Example |
|----------|--------|---------|
| Curated DB dependency | Misses novel or context-specific interactions; DB outdated curations | Novel immune checkpoints not yet in CellChatDB go undetected |
| No de novo inference | Cannot statistically test novel L-R hypotheses | Suspected but uncurated interactions require external validation |
| Probabilistic, not statistical | No confidence intervals or p-values; scores are relative | Hard to compare interaction strength across datasets/conditions |
| Cell-type-level only | Cannot infer cell-to-cell communication within a cell type | Tumor cell–tumor cell paracrine signaling lumped as intracrine |
| Assumes ligand from source, receptor on target | Bidirectional communication requires running twice | A↔B communication requires two separate analyses |
| Does not test functional outcome | Identifies communication, not phenotypic consequence | May prioritize abundant but functionally silent interactions |
| Limited to well-characterized pathways | Non-canonical signaling (e.g., exosomal, metabolic) underrepresented | Metabolic crosstalk (lactate, arginine) not captured in current CellChatDB |
| Spatial distance is heuristic | Proximity weighting is ad-hoc; no formal statistical modeling | Spatial cutoff (e.g., 100 µm) is arbitrary |

**When to Use:**
- ✓ Have scRNA-seq or spatial transcriptomics with clear cell types
- ✓ Goal is to map ligand-receptor communication networks
- ✓ Want curated, biologically validated interactions
- ✓ Spatial data available (can leverage proximity information)
- ✓ Need rich visualization and downstream pathway analysis

**When NOT to Use:**
- ✗ Need novel ligand-receptor discovery (use NATMI + validation)
- ✗ Want to link communication to clinical phenotypes (use RNAchat + CellChat)
- ✗ Have only bulk data without cell-type resolution
- ✗ Need statistical significance testing of interactions
- ✗ Require intra-cell-type or cell-cell paracrine inference

---

## 3. Gap Analysis: What No Single Tool Provides

### 3.1 Missing Integrations

| Gap | Why It Matters | Severity |
|-----|---|---|
| **No tool links L-R communication to clinical phenotype** | You can identify which cell types talk, but not whether that talk matters for patient outcome | 🔴 High |
| **No tool jointly optimizes spatial + spatial proteomics + clinical data** | Spatial proteomics has cell-level protein + location; current tools optimize for transcriptomics or bulk | 🔴 High |
| **No tool discovers novel L-R pairs AND validates statistical significance** | NATMI finds novel pairs; CellChat validates known pairs. Combined approach missing | 🔴 High |
| **No tool handles cell-type-by-cell-type interactions with spatial distance** | CellChat proxies distance; no formal statistical model of local communication | 🟡 Medium |
| **No tool infers functional consequence of communication** | Can identify "A talks to B"; cannot infer "A→B causes phenotype X" | 🟡 Medium |
| **No tool integrates multi-omics (protein, metabolite, phospho-proteomics) + communication** | MaxFUSE does multi-omics fusion; doesn't link to communication inference | 🟡 Medium |
| **No tool addresses spatial context in bulk/region-level analysis** | RNAchat works at bulk level; CellChat needs cells. No middle ground | 🟡 Medium |

### 3.2 Critical Workflow Gaps

**Current Bottlenecks:**

1. **Data Format Incompatibility**
   - MaxFUSE expects cell-level annotations + spatial coordinates
   - RNAchat expects bulk expression + sample-level metadata
   - CellChat expects scRNA-seq with cell types
   - **Spatial proteomics** has per-cell proteins + location — fits MaxFUSE input, but output is not directly compatible with CellChat or RNAchat

2. **Loss of Information**
   - Spatial proteomics → MaxFUSE (fused embeddings) → lose spatial coordinates
   - Spatial proteomics → region-level aggregation → RNAchat (loses cell-type signal)
   - scRNA-seq → NATMI or CellChat (ignores patient phenotype)

3. **Downstream Analysis**
   - CellChat + NATMI output communication networks, but **no native way to link to clinical outcomes**
   - RNAchat predicts phenotypes, but **does not identify cell-type-specific or ligand-receptor drivers**

4. **Validation & Interpretation**
   - NATMI is data-driven; CellChat is DB-driven. **No consensus on true interactions**.
   - RNAchat metagenes may not correspond to known pathways; **biological validation burden on user**.
   - CellChat scores are relative; **no cross-dataset or cross-condition comparisons**.

---

## 4. Proposed Mitigation Strategies

### 4.1 Integration Framework: "Multi-Layer Analysis Pipeline"

**Workflow:**
```
SPATIAL PROTEOMICS (cell-level proteins + location)
        ↓
├─ Layer 1: Protein-to-Gene Mapping
│  (map proteins to gene expression surrogates via public databases)
│  Output: Gene expression matrix + cell coordinates
│
├─ Layer 2: MaxFUSE-Based Fusion
│  (integrate Layer 1 output with scRNA-seq if available)
│  Input: Spatial protein (as pseudo-expression) + scRNA-seq + clinical metadata
│  Output: Fused embeddings + cross-modal correspondence
│
├─ Layer 3: Cell-Type Annotation
│  (use MaxFUSE embeddings or existing cell types)
│  Output: Cell type labels per spatial location
│
├─ Layer 4: Dual Communication Inference
│  Branch A: CellChat (curated L-R, strong visualization)
│  Branch B: NATMI (data-driven L-R discovery)
│  Output: Two communication networks; consensus = validated interactions
│
├─ Layer 5: Phenotype Linkage
│  (RNAchat-style: aggregate cell-type-specific communication by region or cell type)
│  Input: Communication scores (from Layer 4) + clinical metadata
│  Method: ML model (Random Forest, XGBoost) to predict clinical variable from communication strength
│  Output: SHAP + DML → which cell-cell pairs or pathways drive phenotype?
│
└─ Layer 6: Interpretation & Validation
   (SHAP explanations, permutation tests, external cohort validation)
   Output: Ranked list of cell type interactions associated with phenotype
```

**Advantages:**
- ✓ Retains spatial information end-to-end
- ✓ Leverages MaxFUSE for multi-omics, CellChat for validated biology, RNAchat for phenotype linking
- ✓ Dual L-R inference (curated + data-driven) → robust consensus
- ✓ SHAP + DML → interpretability + causal heterogeneity

**Challenges:**
- ⚠ Computational cost: 3 major pipelines run serially
- ⚠ Requires protein-to-gene mapping (non-trivial; introduces noise)
- ⚠ Integration points may lose information
- ⚠ Hyperparameter tuning across 6 layers

---

### 4.2 Practical Mitigation Checklist

#### **Problem: Spatial Proteomics → MaxFUSE → Loss of Spatial Context**
**Mitigation:**
- [ ] Preserve spatial coordinates throughout MaxFUSE (add as metadata in output embeddings)
- [ ] Use spatial-aware clustering post-MaxFUSE (e.g., SpotClean, MANATEE)
- [ ] Validate fused embeddings retain local structure (compute spatial autocorrelation of embedding dimensions)

#### **Problem: NATMI Ignores Phenotype; CellChat Ignores Clinical Outcomes**
**Mitigation:**
- [ ] Post-hoc linkage: use communication scores as features in RNAchat-style ML
- [ ] Aggregate CellChat output (interaction strength per cell type pair per region) + clinical metadata → logistic regression or RF
- [ ] SHAP to identify which L-R pairs matter for phenotype

#### **Problem: RNAchat Metagenes May Not Be Biologically Validated**
**Mitigation:**
- [ ] Annotate metagenes: correlate with known pathway databases (msigdb, KEGG, Reactome)
- [ ] Validate against CellChat outputs: if a metagene correlates with phenotype AND overlaps with active L-R pathway, higher confidence
- [ ] Use biological priors: initialize ICA with known pathway signatures instead of random

#### **Problem: No Consensus on L-R Pairs (NATMI vs. CellChat Disagreement)**
**Mitigation:**
- [ ] Rank pairs by intersection: pairs in both NATMI and CellChat get higher score
- [ ] External validation: literature mining, protein-protein interaction networks, pathway DBs
- [ ] Experimental follow-up: prioritize discordant pairs for validation (where NATMI says yes, CellChat says no, or vice versa)

#### **Problem: MaxFUSE Computationally Expensive; Not Practical for Large Cohorts**
**Mitigation:**
- [ ] Use reference-based projection: learn fusion on small annotated cohort, project new samples (faster)
- [ ] GPU acceleration + distributed computing (dask, Rapids)
- [ ] Approximate: use fast linear fusion (PCA + CCA) instead of deep learning if speed critical

#### **Problem: Spatial Coordinates Wasted in Bulk/Region-Level Analysis**
**Mitigation:**
- [ ] Preserve spatial resolution: instead of whole-tissue bulk, use region-level (e.g., 10 regions per slide)
- [ ] Spatial GLM: use region-level communication strength + clinical variable, with spatial autocorrelation structure
- [ ] Multiplex validation: test findings in high-resolution spatial + low-resolution bulk cohorts

---

### 4.3 Recommended Tool Selection by Use Case

| **Your Question** | **Primary Tool** | **Secondary Tool** | **Tertiary Tool** | **Rationale** |
|---|---|---|---|---|
| "Which cell types talk to each other?" | CellChat | NATMI | – | CellChat for curated DB; NATMI for data-driven validation |
| "How do spatial cell-cell interactions relate to patient outcome?" | Multi-Layer Pipeline (all 4) | – | – | Requires end-to-end integration |
| "What ligands mediate immune activation?" | CellChat | NATMI | – | CellChat for pathway level; NATMI for discovery |
| "Which pathway modules predict drug response?" | RNAchat | CellChat | – | RNAchat is purpose-built; validate pathways with CellChat |
| "How to integrate spatial proteomics + scRNA?" | MaxFUSE | – | – | Direct alignment of modalities |
| "Novel L-R discovery (with external validation)" | NATMI | Literature/DB | – | NATMI finds candidates; external sources validate |
| "Infer subgroup-specific communication networks" | Multi-Layer + DML | – | – | RNAchat's DML engine for heterogeneity |
| "Link immune checkpoints to spatial tumor microenvironment" | CellChat + RNAchat | MaxFUSE (if multi-omics) | – | CellChat for immune biology; RNAchat for phenotype |

---

## 5. Boundary Conditions & Edge Cases

### 5.1 When Each Tool Breaks Down

| **Tool** | **Breaking Point** | **Symptom** | **Workaround** |
|---|---|---|---|
| **MaxFUSE** | Unmatched spatial + scRNA (different donors) | Fused embeddings don't align; poor correspondence | Validate sample matching before fusion; if unmatched, use separate analyses |
| **NATMI** | Very sparse expression (dropout > 90%) | Co-expression networks unreliable; inflated p-values | Impute or filter genes; use robust correlation (Spearman, Biweight) |
| **RNAchat** | ICA non-convergence (highly sparse or batch-heavy data) | Metagenes are noise; SHAP artifacts | Apply variance-stabilizing transform; batch correction (ComBat); reduce dims first (PCA) |
| **CellChat** | Rare cell type (< 10 cells) | Receptor/ligand scores undefined; network incomplete | Merge with similar cell types; use pseudobulk aggregation |
| **All Tools** | Highly imbalanced phenotype (95% one class) | ML models overfit; SHAP artifacts | Class balancing (SMOTE, class weights); stratified CV |

### 5.2 Data Quality Requirements

| **Tool** | **Min. Sample Size** | **Min. Cells/Cell Type** | **Batch Tolerance** | **Dropout Tolerance** |
|---|---|---|---|---|
| **MaxFUSE** | ≥ 5 matched pairs | ≥ 50 per type | Low (alignment breaks) | Low (< 50% dropout) |
| **NATMI** | ≥ 1 (but stat. power ↑ with replicates) | ≥ 30 per type | Medium (use batch-aware correlation) | Medium (40–80%) |
| **RNAchat** | ≥ 20–30 samples | N/A (bulk) | Medium-High (batch correction) | High (60%+) |
| **CellChat** | ≥ 3 (groups to compare) | ≥ 10 per type | Medium (if cell type composition similar) | High (handles log-normal) |

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Month 1)
- [ ] Set up standardized pipeline for protein-to-gene mapping (spatial proteomics → gene-level expression)
- [ ] Implement Layer 1–2 of Multi-Layer Pipeline (data prep + MaxFUSE if multi-omics available)
- [ ] Validate spatial coordinate preservation post-MaxFUSE

### Phase 2: Communication Inference (Month 2)
- [ ] Implement Layer 4: Run CellChat + NATMI in parallel on same dataset
- [ ] Benchmark: which pairs overlap? Which discord? Build consensus ranking
- [ ] Visualize dual networks; annotate with confidence scores

### Phase 3: Phenotype Linkage (Month 3)
- [ ] Implement Layer 5: RNAchat-style ML on communication features
- [ ] Input: cell type pair interaction strength + clinical variable
- [ ] Output: SHAP importance (which L-R pairs predict phenotype?)
- [ ] Run DML for heterogeneity (which cell pairs matter for subgroups?)

### Phase 4: Validation & Interpretation (Month 4)
- [ ] Permutation tests: shuffle phenotype labels, re-run ML → null distribution
- [ ] External validation: test findings on held-out cohort if available
- [ ] Biological validation: correlate top hits with literature, pathway databases

### Phase 5: Deployment & Iteration
- [ ] Package pipeline as modular Snakemake + Nextflow
- [ ] Document data format specifications (input/output contracts)
- [ ] Create interactive dashboard (e.g., Streamlit, Shiny) for exploration

---

## 7. Conclusion

| **Tool** | **Best For** | **Avoid If** |
|---|---|---|
| **MaxFUSE** | Multi-omics alignment (spatial + scRNA) | Single modality; unmatched samples |
| **NATMI** | Data-driven L-R discovery; large cohorts | Need validated biology; low sample counts |
| **RNAchat** | Phenotype prediction & interpretation; clinical links | Cell-cell communication focus |
| **CellChat** | Ligand-receptor biology; spatial context; visualization | Novel L-R discovery; clinical phenotype links |

**No single tool solves the complete spatial proteomics → communication → phenotype workflow.** A **multi-layer integration** (MaxFUSE + CellChat/NATMI + RNAchat) with careful validation at each boundary is recommended for robust, interpretable results. Invest in **protein-to-gene mapping quality**, **dual L-R inference consensus**, and **rigorous phenotype linkage** to overcome current tool limitations.

