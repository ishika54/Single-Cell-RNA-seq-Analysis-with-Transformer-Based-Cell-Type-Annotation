# Single-Cell RNA-seq Analysis with Transformer-Based Cell Type Annotation

Transformer-based single-cell RNA-seq analysis pipeline featuring:

- Scanpy preprocessing
- Leiden clustering
- Cell type annotation
- Transformer-based classification
- Attention visualization
- Cross-dataset transfer learning
- Comparison with Random Forest baseline

## Key Features

- Biologically interpretable Transformer attention
- PBMC3k → PBMC10k transfer experiment
- Gene-level immune representation learning
- Attention visualization for unseen datasets


## Dataset

The project uses publicly available PBMC datasets from 10x Genomics.

Datasets:
- PBMC3k
- PBMC5k / PBMC10k

Download links:
https://www.10xgenomics.com/resources/datasets

## Transformer Architecture
![Transformer](single_cell_transformer_architecture.svg)

## Outputs

### Attention Visualization
![Attention](outputs/attention_comprehensive.png)

### Transfer Attention
![Transfer](outputs/stage3_attention_new_dataset.png)

### Model Comparison
![Comparison](outputs/stage3_summary.png)

