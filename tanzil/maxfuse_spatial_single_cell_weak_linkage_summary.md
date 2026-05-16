# Integration of Spatial and Single-Cell Data Across Modalities with Weakly Linked Features

Paper link: https://www.nature.com/articles/s41587-023-01935-0

Github code link: https://github.com/Tabassum-Sumaiya13/Spatial-Proteomics-Analysis

The original MaxFuse paper is a better conceptual fit for our spatial-proteomics use case than scMCGF, because it is designed for weakly linked cross-modal data rather than scRNA-seq-only clustering.

## **Integration of spatial and single-cell data across modalities with weakly linked features**

### Core Idea

MaxFuse is a cross-modal integration method for datasets that do not share strongly correlated features. It starts with a small set of weakly linked features between modalities, uses them to get a rough cross-modal alignment, then iteratively refines that alignment by building a shared embedding, smoothing within each modality, and matching cells across the two datasets. The result is a joint representation that can recover biological structure even when the direct feature overlap is poor.

In practice, the method is useful for spatial proteomic data paired with single-cell transcriptomic, chromatin, or other single-cell assays. Rather than depending on one-to-one feature correspondence, MaxFuse uses the full information available in each modality and gradually strengthens the cross-modal match.

### Model Process

MaxFuse works in an iterative loop:

1. Start with two unpaired modalities.
   - Example: spatial proteomics on tissue sections and single-cell sequencing from a related reference.
   - Only weakly linked features may be available.

2. Build an initial cross-modal signal.
   - Weak feature links provide a first approximation of how the two modalities correspond.
   - This first step is intentionally rough.

3. Construct a shared coembedding.
   - Both modalities are projected into a common latent space.
   - The embedding is not final yet; it is only the starting point for refinement.

4. Smooth each modality within its own neighborhood structure.
   - Local structure is used to propagate information.
   - This helps recover signal from noisy or sparse weak links.

5. Match cells across modalities.
   - The smoothed representations are used to update cell pairings.
   - The matching step is repeated rather than done once.

6. Recompute the shared embedding.
   - Better matches produce a better joint representation.
   - The improved embedding then supports better matching in the next round.

7. Stop when alignment stabilizes.
   - The output is a final matched embedding and aligned cross-modal structure.
   - The result can be used for downstream biological interpretation.

### Pipeline

```markdown
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                   MAXFUSE COMPLETE PIPELINE                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  STEP 1: Input Two Weakly Linked Modalities                                         │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                │ │
│  │   Modality A: Spatial proteomics or spatial multi-omics                        │ │
│  │   Modality B: Single-cell transcriptome / chromatin / other single-cell data   │ │
│  │                                                                                │ │
│  │   Shared signal is weak, incomplete, or only indirectly related                │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                          │
│                                          ▼                                          │
│  STEP 2: Weak Feature Link Initialization                                           │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                │ │
│  │   Use weakly linked features to get a rough cross-modal pairing                │ │
│  │   This gives a starting correspondence but not the final alignment             │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                          │
│                                          ▼                                          │
│  STEP 3: Coembedding                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                │ │
│  │   Project both modalities into a shared latent space                           │ │
│  │   The joint embedding captures broad biological structure                      │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                          │
│                                          ▼                                          │
│  STEP 4: Fuzzy Smoothing                                                            │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                │ │
│  │   Smooth local structure inside each modality                                  │ │
│  │   This denoises weakly linked signals and preserves neighborhood context       │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                          │
│                                          ▼                                          │
│  STEP 5: Cell Matching                                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                │ │
│  │   Match cells across modalities using the smoothed coembedding                 │ │
│  │   Update correspondences iteratively                                           │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                          │
│                                          ▼                                          │
│  STEP 6: Iterate Until Stable                                                       │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                │ │
│  │   Recompute coembedding using the updated matches                              │ │
│  │   Repeat smoothing + matching until convergence                                │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                          │                                          │
│                                          ▼                                          │
│  STEP 7: Output                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                │ │
│  │   • Matched cross-modal cells                                                  │ │
│  │   • Shared embedding                                                           │ │
│  │   • Aligned proteomic / transcriptomic / epigenomic information                │ │
│  │   • Downstream cell annotation or tissue-level interpretation                  │ │
│  └────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### How to use in our work

```markdown
STEP 1: Use MaxFuse instead of scMCGF for spatially linked modalities
        ┌────────────────────────────────────────────────────────────────┐
        │  We do NOT start from scRNA-seq-only clustering.               │
        │  We start from weakly linked spatial + single-cell modalities  │
        │  that reflect the tissue context of our data.                  │
        └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STEP 2: Replace the raw linked features with our spatial-proteomics inputs
        ┌────────────────────────────────────────────────────────────────┐
        │  Example inputs for our use case:                              │
        │  • Spatial protein expression                                  │
        │  • Cell coordinates / tissue neighborhood features             │
        │  • Cell-type or marker-derived weak links                      │
        │  • Optional transcriptomic or epigenomic reference features    │
        └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STEP 3: Build the initial weak alignment
        ┌────────────────────────────────────────────────────────────────┐
        │  Use weak biological links to create a rough correspondence    │
        │  between spatial cells and the reference single-cell profile   │
        │  This is the seed for the iterative MaxFuse loop               │
        └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STEP 4: Run iterative coembedding + smoothing + matching
        ┌────────────────────────────────────────────────────────────────┐
        │  MaxFuse repeatedly refines the shared embedding and matches   │
        │  This is the core mechanism we should adapt conceptually       │
        └────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
STEP 5: Use the aligned output for spatial proteomics analysis
        ┌────────────────────────────────────────────────────────────────┐
        │  Downstream tasks can include:                                 │
        │  • Spatial interaction analysis                                │
        │  • Cell neighborhood discovery                                 │
        │  • Protein localization change tracking                        │
        │  • Cross-modal marker transfer                                 │
        └────────────────────────────────────────────────────────────────┘
```

### Datasets Used

| Aspect | Details |
| --- | --- |
| Main setting | Weakly linked spatial and single-cell modalities |
| Example use case | Spatial proteomics + single-cell sequencing |
| Input requirement | No need for strong one-to-one feature overlap |
| Output | Matched cells, shared embedding, aligned multi-modal biology |
| Strength | Works when feature linkage is weak rather than strongly correlated |

### To Adapt or Not?

| If you... | Then... |
| --- | --- |
| Need a method that matches spatial proteomics to single-cell reference data | Use MaxFuse as the starting point |
| Want to keep spatial context central | MaxFuse is a better fit than scMCGF |
| Need a pure clustering method for scRNA-seq | scMCGF is more direct, but less suitable here |
| Want weak-link integration across modalities | Adapt MaxFuse, not scMCGF |
| Need interpretability through iterative alignment | MaxFuse gives a clearer integration story |

**Final Recommendation:** For this project, MaxFuse is the better paper to build from. It is designed for weakly linked cross-modal integration, which matches spatial proteomics much better than an scRNA-seq clustering method.


## Reference

Nature Biotechnology: Integration of spatial and single-cell data across modalities with weakly linked features.