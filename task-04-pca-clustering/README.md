# Task 4: Dimensionality Reduction & Unsupervised Clustering

**RabTech Academy · Data Science with Python & R Internship**

**Notebook:** [`notebooks/pca_clustering.ipynb`](notebooks/pca_clustering.ipynb) (executed, all outputs included)

**Question:** can we discover natural wine types from 13 chemical measurements, without labels?
**Data:** UCI Wine, 178 wines × 13 features (`data/raw/wine.csv`). The 3 grape cultivars are hidden during clustering and used only for validation.

## What's covered

| Requirement | Notebook section |
|---|---|
| Standardize high-dimensional features, then PCA | §2–3 |
| Explained-variance-ratio scree plot, choosing the number of components (Kaiser, elbow, 80%) | §3 |
| PCA loadings heatmap and biplot | §3 |
| K-Means: Elbow Method and Silhouette scores for k = 2…10, silhouette plots | §4 |
| DBSCAN with a k-distance plot and parameter grid | §5 |
| Hierarchical clustering (Ward dendrogram) | §6 |
| Algorithm comparison: silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI | §7 |
| 2D projections (Seaborn), 3D (Matplotlib) and interactive 3D (Plotly) | §8 · [`reports/interactive_3d_clusters.html`](reports/interactive_3d_clusters.html) |
| Cluster profiles and findings | §9–10 |

## Key findings

1. **5 principal components keep 80.2%** of the variance (13 → 5 dims). PC1 = phenolic richness, PC2 = colour and alcohol.
2. **Silhouette peaks at k = 3** (0.369). Davies-Bouldin and Calinski-Harabasz agree; the elbow is ambiguous (3–4).
3. **K-Means (k = 3) recovers the hidden cultivars**: ARI 0.90, with 172 of 178 wines grouped with their own cultivar. Ward agrees on k = 3 (ARI 0.76).
4. **DBSCAN is a poor fit**: the groups have similar density, so it either finds 1 cluster or labels 29% of the wines as noise (ARI 0.56).

## Reproduce

```bash
uv sync --locked
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/pca_clustering.ipynb
```
