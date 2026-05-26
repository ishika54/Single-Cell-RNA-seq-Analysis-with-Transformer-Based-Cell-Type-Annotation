# ============================================================
# stage3_preprocess.py
#
# Preprocess PBMC10k to MATCH PBMC3k training pipeline.
#
# CRITICAL:
# The Transformer expects the EXACT SAME 2013 genes
# in the EXACT SAME ORDER as training.
# ============================================================

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import warnings

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

print("=" * 70)
print("STAGE 3 — PREPROCESS NEW PBMC DATASET")
print("=" * 70)

# ============================================================
# LOAD TRAINING METADATA
# ============================================================

print("\nLoading training checkpoint metadata...")

checkpoint = torch.load(
    'single_cell_transformer.pt',
    map_location='cpu'
)

gene_names_3k = checkpoint['gene_names']

N_GENES = checkpoint['n_genes']

print(f"Reference gene set loaded: {N_GENES} genes")

# ============================================================
# LOAD PBMC10k
# ============================================================

print("\nLoading PBMC10k dataset...")

adata_new = sc.read_10x_mtx(
    'data/pbmc5k/filtered_feature_bc_matrix/',
    var_names='gene_symbols',
    cache=True
)

adata_new.var_names_make_unique()

print(f"Raw dataset shape: {adata_new.shape}")

# ============================================================
# QUALITY CONTROL
# ============================================================

print("\nApplying quality control...")

# Filter low-quality cells
sc.pp.filter_cells(
    adata_new,
    min_genes=200
)

# Filter rare genes
sc.pp.filter_genes(
    adata_new,
    min_cells=3
)

# Mitochondrial genes
adata_new.var['mt'] = adata_new.var_names.str.startswith('MT-')

# Compute QC metrics
sc.pp.calculate_qc_metrics(
    adata_new,
    qc_vars=['mt'],
    percent_top=None,
    log1p=False,
    inplace=True
)

# Same thresholds as PBMC3k
adata_new = adata_new[
    adata_new.obs.n_genes_by_counts < 2500,
    :
]

adata_new = adata_new[
    adata_new.obs.pct_counts_mt < 5,
    :
]

print(f"After QC: {adata_new.shape}")

# ============================================================
# NORMALIZATION
# ============================================================

print("\nNormalizing counts...")

# Same normalization as PBMC3k
sc.pp.normalize_total(
    adata_new,
    target_sum=1e4
)

# Same log transform
sc.pp.log1p(adata_new)

print("Normalization complete.")

# ============================================================
# GENE ALIGNMENT
# ============================================================

print("\nAligning genes to PBMC3k training genes...")

genes_in_new = set(adata_new.var_names)

genes_in_3k = set(gene_names_3k)

shared_genes = genes_in_3k & genes_in_new

missing_genes = genes_in_3k - genes_in_new

print("\nGene overlap analysis:")
print(f"PBMC3k training genes : {len(gene_names_3k)}")
print(f"PBMC10k total genes   : {len(adata_new.var_names)}")
print(f"Shared genes          : {len(shared_genes)}")
print(f"Missing genes         : {len(missing_genes)}")

if len(missing_genes) > 0:

    print("\nFirst missing genes:")

    print(list(missing_genes)[:10])

# ============================================================
# CONVERT TO DENSE MATRIX
# ============================================================

print("\nConverting expression matrix...")

X_new_raw = adata_new.X

if sp.issparse(X_new_raw):

    X_new_raw = X_new_raw.toarray()

X_new_raw = X_new_raw.astype(np.float32)

print(f"Dense matrix shape: {X_new_raw.shape}")

# ============================================================
# BUILD GENE INDEX
# ============================================================

new_gene_list = list(adata_new.var_names)

new_gene_idx = {
    g: i for i, g in enumerate(new_gene_list)
}

# ============================================================
# CREATE ALIGNED MATRIX
# ============================================================

print("\nBuilding aligned matrix...")

n_cells = adata_new.shape[0]

X_aligned = np.zeros(
    (n_cells, N_GENES),
    dtype=np.float32
)

for j, gene in enumerate(gene_names_3k):

    if gene in new_gene_idx:

        X_aligned[:, j] = X_new_raw[
            :,
            new_gene_idx[gene]
        ]

print(f"Aligned matrix shape: {X_aligned.shape}")

# ============================================================
# SCALE FEATURES
# ============================================================

print("\nScaling features...")

# Match PBMC3k scaling behavior
scaler = StandardScaler()

X_aligned = scaler.fit_transform(X_aligned)

# Same clipping behavior as Scanpy scaling
X_aligned = np.clip(
    X_aligned,
    -10,
    10
)

X_aligned = X_aligned.astype(np.float32)

print("Scaling complete.")

print(f"Final value range:")
print(f"Min: {X_aligned.min():.3f}")
print(f"Max: {X_aligned.max():.3f}")

# ============================================================
# SAVE OUTPUT
# ============================================================

print("\nSaving aligned dataset...")

np.save(
    'X_pbmc10k_aligned.npy',
    X_aligned
)

print("\nSaved:")
print("X_pbmc10k_aligned.npy")

print(f"\nFinal shape: {X_aligned.shape}")

print(f"Total cells processed: {n_cells}")

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print("\nNext step:")
print("python stage3_transfer.py")
