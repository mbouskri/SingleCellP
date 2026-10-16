# Learning RNA velocity and generative modelling on pancreas endocrinogenesis

A small learning project on the public scVelo pancreas endocrinogenesis dataset
(3,696 cells, developing mouse pancreas). I wanted hands-on experience with single-cell
data and to test a small generative model on it. This is a learning project, not novel
research.

**Follows standard tutorials:** preprocessing, clustering, scVelo, CellRank.
**My own additions:** the comparison of clusters with published labels, the CellRank
checks, and the VAE with its evaluation against PCA.

## Questions
1. How well do unsupervised clusters match the published cell-type labels?
2. Does a velocity-based CellRank kernel recover the known endpoints?
3. Does a small VAE separate cell types better than PCA, and does its latent space follow
   a sensible Ductal-to-Beta path?

## Data and preprocessing
`scv.datasets.pancreas()`: 3,696 cells, spliced and unspliced counts, 8 published cell types.
For clustering and the VAE: `normalize_total` (1e4), `log1p`, 2,000 highly variable genes.
For velocity: scVelo `filter_and_normalize` (7,197 genes kept).

## 1. Clustering vs published labels
Leiden clustering on a 30-PC, 30-neighbour graph. [Leiden is seeded with random_state=0;
re-copy these numbers after fixing the seed.]

| Resolution | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 |
|---|---|---|---|---|---|---|
| Clusters | 5 | 7 | 9 | 9 | 9 | 12 |
| ARI vs labels | 0.563 | 0.551 | 0.520 | 0.616 | 0.619 | 0.520 |

Across five seeds at resolution 0.5 the ARI ranged from 0.44 to 0.62, so differences between
resolutions are within noise. At resolution 0.8, Delta and Epsilon each fall (99%) into their
own cluster. The real disagreements are Ductal and Ngn3 low EP (mixed across three clusters)
and Pre-endocrine (spread over several), which lie on a continuum.

## 2. RNA velocity and CellRank
Velocity arrows run from Ductal toward the endocrine types, with a circular pattern inside
Ductal [that I think reflects the cell cycle; state what the S/G2M plot showed].
CellRank: velocity kernel (80%) + connectivity kernel (20%), GPCCA.

- **Data-driven terminal states:** Alpha and Beta at every setting; a Ductal sub-state
  at 8 and 9 macrostates that disappears at 11. Delta and Epsilon were never found.
- **Endpoints specified by hand** (Alpha, Beta, Delta, Epsilon, from prior biology):

| Mean fate probability | Alpha | Beta | Delta | Epsilon |
|---|---|---|---|---|
| Ductal / Ngn3 low EP / Ngn3 high EP | 0.06 | 0.91 | 0.02 | 0.02 |
| Pre-endocrine | 0.06 | 0.88 | 0.03 | 0.03 |
| Beta | 0.02 | 0.97 | 0.00 | 0.00 |
| Alpha | 0.59 | 0.38 | 0.02 | 0.01 |
| Delta | 0.02 | 0.23 | 0.75 | 0.00 |
| Epsilon | 0.32 | 0.30 | 0.04 | 0.33 |

Early populations have almost identical fate distributions, and Alpha and Epsilon cells do
not lead strongly to their own fate, so the fate probabilities distinguish early stages
poorly. Specifying the endpoints
uses prior knowledge, so this shows the method can use them, not that it found them.

## 3. VAE
PyTorch VAE: 2,000 genes → 256 → 15 latent → 256 → 2,000 (ReLU hidden layers, linear output),
MSE reconstruction plus a KL term averaged over latent dimensions and weighted by `KLw`.
Cells split 90/10 (3,327 train / 369 validation), Adam (lr 1e-3), batch 128, 100 epochs,
best-validation-epoch weights kept, seed 0.

| KL weight | 0.0005 | 0.005 | 0.05 | 0.5 |
|---|---|---|---|---|
| Best epoch | 37 | 92 | 85 | 98 |
| Active latent dims (of 15) | 15 | 13 | 3 | 0 |

Cell-type recall (5-fold kNN, k=15, on the latent means; PCA uses the same 15 dimensions):

| Cell type | Cells | PCA | VAE 0.0005 | VAE 0.005 | VAE 0.05 | VAE 0.5 |
|---|---|---|---|---|---|---|
| Alpha | 481 | 0.94 | 0.95 | 0.94 | 0.91 | 0.32 |
| Beta | 591 | 0.96 | 0.96 | 0.95 | 0.94 | 0.71 |
| Delta | 70 | 0.93 | 0.96 | 0.93 | 0.84 | 0.00 |
| Ductal | 916 | 0.97 | 0.97 | 0.98 | 0.96 | 0.96 |
| Epsilon | 142 | 0.96 | 0.94 | 0.92 | 0.92 | 0.00 |
| Ngn3 high EP | 642 | 0.98 | 0.98 | 0.96 | 0.98 | 0.94 |
| Ngn3 low EP | 262 | 0.40 | 0.32 | 0.31 | 0.45 | 0.04 |
| Pre-endocrine | 592 | 0.93 | 0.91 | 0.89 | 0.90 | 0.60 |

Between repeated runs, small-group recall moved by up to about 5 points, so differences of
that size are not meaningful.

**Interpolation.** Decoding a straight line in latent space from the mean Ductal cell to the
mean Beta cell (KL 0.005), against the real cluster means:

| Gene | Real Ductal → Beta | Decoded start → end |
|---|---|---|
| Sox9 | 1.74 → 0.23 | 1.81 → 0.31 |
| Ins1 | 0.11 → 2.16 | -0.01 → 1.36 |
| Ins2 | 0.02 → 2.20 | 0.00 → 1.84 |
| Neurog3 | peaks at 3.28 in Ngn3 high EP | peak 0.38, 0.12 → 0.34 |


## Findings
- A small VAE matches PCA but does not beat it at KL weights up to 0.05 (differences are
  within run-to-run noise).
- Delta and Epsilon are separable in expression space (PCA recall 0.93 and 0.96, own
  Leiden clusters at resolution 0.8), so CellRank's failure to find them as terminal states
  is not a representation problem.
- A KL weight of 0.5 collapses the posterior (0 active dimensions): the decoded path is flat
  and rare cell types are lost first.
- The latent space follows a Ductal-to-Beta direction for Sox9 and the insulin genes, but
  reproduces only a small fraction of the transient Neurog3 peak.

## Limitations
- One dataset and mostly one seed. Delta (70 cells) and Epsilon (142) make differences of a
  few points unreliable.
- The published labels were presumably derived from the same expression data, so agreement
  with them is partly circular.
- scVI is the standard VAE for single-cell counts. This is an educational model with a
  Gaussian loss on log-normalised data.
- Terminal states for Delta and Epsilon were specified by hand; kernel weights were not tuned.


Tested with Python 3.10, Windows, CUDA GPU. Versions: scanpy: 1.11.5 , scvelo: 0.3.4 , cellrank: 2.0.7 , pandas: 2.0.1 , torch: 2.0.0+cu118.
