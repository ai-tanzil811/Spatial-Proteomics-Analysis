# Work Summary Report: Comparative Pipeline Analysis

# Weekly Research Progress Report
## Analysis of Spatial Proteomics Data for Survival and Treatment Insights

Prepared On: May 18, 2026
Team: 253_I1
Report Period: Week of April 28 – May 6, 2026
Project Stage: Phase 2 — Graph Enrichment & Cell Communication Analysis
| Name | Student ID |
|------|-----------|
| Tabasum Sumaiya | 0112230295 |
| Md. Hasibul Hossain | 0112230293 |
| Marjia Islam | 0112230958 |
| Chowdhury Nafisa Binte Ershad | 0112230369 |
| Ashraful Islam Tanzil | 0112230028 |

---

## Objective

Conduct a comprehensive comparative analysis of four major cell-cell communication and multi-omics integration pipelines (MaxFUSE, NATMI, RNAchat, CellChat v2) to identify their boundaries, gaps, and propose an integrated mitigation strategy for spatial proteomics analysis.

## Deliverables

### 1. **COMPARATIVE_PIPELINE_ANALYSIS.md** (Markdown)
- **15-dimensional comparative table** across 4 tools
  - Input/output types, algorithms, scalability, computational costs, maturity
  - Strengths, limitations, and typical use cases
  
- **Detailed pipeline breakdowns** (Section 2)
  - Visual flowcharts for each tool
  - Boundary conditions and when to use/avoid each tool
  
- **Gap analysis** (Section 3)
  - 7 critical missing integrations (ranked by severity)
  - 4 workflow bottlenecks (data format, information loss, downstream analysis, validation)
  
- **Mitigation strategies** (Section 4)
  - **Six-layer integration framework** combining all tools end-to-end
  - Practical checklists for each gap
  - Tool selection guide by research question
  - Data quality requirements and edge cases
  
- **4-month implementation roadmap** (Phase 1–5)

### 2. **comparative.tex** (LaTeX)
- Professional academic document with team report header
- Updated to CellChat v2 throughout (including v2-specific enhancements)
- **28 peer-reviewed references** including:
  - MaxFUSE, NATMI, RNAchat, CellChat v2 foundational papers
  - Spatial proteomics and multi-omics integration literature
  - Causal inference (DML, Causal Forests) and SHAP interpretability
  - Cell-cell communication review papers
  - Spatial statistics and machine learning methods

---

## Key Findings

| Finding | Impact | Recommendation |
|---------|--------|-----------------|
| **No single tool solves complete workflow** | Spatial proteomics → communication → phenotype requires 4+ tools | Use six-layer multi-tool pipeline |
| **Data format incompatibility** | Information loss at integration points (spatial context, cell types) | Preserve metadata throughout; use region-level aggregation |
| **L-R inference gap** | CellChat v2 finds validated pairs; NATMI finds novel pairs; no consensus | Run both; rank by intersection; prioritize discordant pairs for validation |
| **Phenotype linkage missing** | Communication networks exist independent of clinical significance | Post-hoc ML (Random Forest + SHAP) on interaction scores |
| **Spatial context lost in bulk analysis** | RNAchat is bulk-level; CellChat v2 needs cells | Use region-level (10 regions/slide) instead of whole-tissue |
| **ICA metagenes lack validation** | RNAchat outputs may be statistical artifacts | Correlate with KEGG, Reactome; validate against CellChat v2 pathways |

---

## Proposed Workflow: Six-Layer Integration

```
Layer 1: Protein-to-Gene Mapping (spatial proteomics → pseudo-expression)
   ↓
Layer 2: MaxFUSE Fusion (spatial protein + scRNA-seq + clinical)
   ↓
Layer 3: Cell-Type Annotation (from fused embeddings)
   ↓
Layer 4: Dual Communication Inference
   ├─→ CellChat v2 (curated L-R)
   └─→ NATMI (data-driven L-R) → consensus ranking
   ↓
Layer 5: Phenotype Linkage (ML on communication features + RNAchat-style causal inference)
   ↓
Layer 6: Interpretation & Validation (SHAP + permutation tests + external cohort)
```

**Advantages:**
- ✓ Retains spatial information end-to-end
- ✓ Dual L-R inference (curated + data-driven)
- ✓ Direct clinical phenotype linking via causal inference
- ✓ SHAP-based interpretability at each layer

**Challenges:**
- Computational cost (GPU required for MaxFUSE)
- Protein-to-gene mapping noise
- Hyperparameter tuning across 6 layers

---

## Tool Selection Summary

| Question | Primary Tool | Secondary Tool |
|----------|---|---|
| Which cell types talk? | CellChat v2 | NATMI |
| How do interactions relate to patient outcome? | **Multi-Layer Pipeline** | – |
| Pathway prediction of drug response? | RNAchat | CellChat v2 |
| Multi-omics alignment? | MaxFUSE | – |
| Novel L-R discovery? | NATMI | Literature |

---

## Data Quality Requirements

- **Sample size:** 5+ matched pairs (MaxFUSE), 20–30 bulk samples (RNAchat), ≥3 groups (CellChat v2)
- **Cells/cell type:** 10–50 minimum (cell-type-dependent tools)
- **Batch tolerance:** Low (MaxFUSE) to High (RNAchat, CellChat v2)
- **Dropout tolerance:** Low (<50%, MaxFUSE) to High (>60%, RNAchat, CellChat v2)

---

## Implementation Roadmap

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **1: Foundation** | Weeks 1–2 | Protein-to-gene mapping; Layer 1–2 setup; spatial validation |
| **2: Communication** | Weeks 3–4 | CellChat v2 + NATMI parallel runs; consensus ranking |
| **3: Phenotype** | Weeks 5–6 | RNAchat-style ML; DML for heterogeneity; SHAP plots |
| **4: Validation** | Weeks 7–8 | Permutation tests; external cohort; biological validation |
| **5: Deployment** | Weeks 9–10 | Snakemake/Nextflow pipeline; documentation; interactive dashboard |

---

## Next Steps

1. **Immediate:** Set up protein-to-gene mapping standardization
2. **Week 1–2:** Validate MaxFUSE on matched spatial + scRNA-seq samples
3. **Week 3–4:** Implement dual L-R inference; benchmark NATMI vs. CellChat v2
4. **Week 5–6:** Build phenotype-linkage ML pipeline; generate SHAP explanations
5. **Ongoing:** Document data format specifications; prepare for external validation

---

## Conclusion

A **six-layer multi-tool integration** leveraging MaxFUSE (fusion), CellChat v2 + NATMI (L-R inference), and RNAchat (phenotype linking) is feasible and recommended for robust spatial proteomics analysis. Each layer addresses a specific gap; together they enable end-to-end analysis from raw spatial data to clinical phenotype associations with interpretability and causal inference.

**Estimated Timeline:** 10 weeks to full implementation and validation.

---

**Documents Created:**
- `COMPARATIVE_PIPELINE_ANALYSIS.md` — Full analysis with examples, tables, and checklists
- `comparative.tex` — Peer-reviewed academic version with 28 references and team header
