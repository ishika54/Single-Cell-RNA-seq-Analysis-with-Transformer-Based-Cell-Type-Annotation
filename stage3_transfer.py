# ============================================================
# stage3_transfer.py
# The KEY experiment: train on PBMC3k, predict on PBMC10k.
# No retraining. Tests true cross-dataset transfer.
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings, json
warnings.filterwarnings('ignore')

print("=" * 60)
print("Stage 3 — Cross-Dataset Transfer Experiment")
print("=" * 60)

# ── Load everything ───────────────────────────────────────────
checkpoint = torch.load('single_cell_transformer.pt', map_location='cpu')
X_train    = np.load('X_train.npy')
y_train    = np.load('y_train.npy')
X_test_3k  = np.load('X_test.npy')    # original PBMC3k test set
y_test_3k  = np.load('y_test.npy')
X_new      = np.load('X_pbmc10k_aligned.npy')  # new PBMC10k

label_encoder          = LabelEncoder()
label_encoder.classes_ = checkpoint['label_encoder_classes']
gene_names             = checkpoint['gene_names']
N_GENES                = checkpoint['n_genes']
N_CLASSES              = checkpoint['n_classes']
rf_acc_samedataset     = checkpoint['rf_accuracy']
transformer_acc_same   = checkpoint['best_test_acc']

print(f"PBMC3k train: {X_train.shape[0]} cells")
print(f"PBMC3k test:  {X_test_3k.shape[0]} cells")
print(f"PBMC10k new:  {X_new.shape[0]} cells")
print(f"Genes:        {N_GENES}")
print(f"Classes:      {list(label_encoder.classes_)}")

# ── Rebuild model ─────────────────────────────────────────────

class GeneEmbedding(nn.Module):

    def __init__(self, n_genes, d_model):

        super().__init__()

        self.gene_embed = nn.Embedding(
            n_genes,
            d_model
        )

        self.value_proj = nn.Linear(
            1,
            d_model
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        batch_size, n_genes = x.shape

        gene_ids = torch.arange(
            n_genes,
            device=x.device
        )

        gene_ids = gene_ids.unsqueeze(0).expand(
            batch_size,
            -1
        )

        gene_embeds = self.gene_embed(gene_ids)

        value_embeds = self.value_proj(
            x.unsqueeze(-1)
        )

        return self.norm(
            gene_embeds + value_embeds
        )


class SingleCellTransformerCPU(nn.Module):

    def __init__(
        self,
        n_genes,
        n_classes,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1
    ):

        super().__init__()

        self.gene_embedding = GeneEmbedding(
            n_genes,
            d_model
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):

        batch_size = x.shape[0]

        gene_tokens = self.gene_embedding(x)

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1
        )

        tokens = torch.cat(
            [cls, gene_tokens],
            dim=1
        )

        encoded = self.transformer(tokens)

        cls_output = encoded[:, 0]

        return self.classifier(cls_output)

    def get_attention_weights(self, x):

        self.eval()

        with torch.no_grad():

            batch_size = x.shape[0]

            gene_tokens = self.gene_embedding(x)

            cls = self.cls_token.expand(
                batch_size,
                -1,
                -1
            )

            tokens = torch.cat(
                [cls, gene_tokens],
                dim=1
            )

            _, attn = self.transformer.layers[0].self_attn(
                tokens,
                tokens,
                tokens,
                need_weights=True,
                average_attn_weights=True
            )

        return attn

model = SingleCellTransformerCPU(N_GENES, N_CLASSES)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("\nModel loaded.")

# ── Random Forest on PBMC10k (no retraining) ─────────────────
print("\n[STEP 1] RF predictions on PBMC10k (no retraining)...")
rf = RandomForestClassifier(n_estimators=200, random_state=42,
                            class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)

# PBMC3k performance (already known, for reference)
rf_preds_3k  = rf.predict(X_test_3k)
rf_acc_3k    = accuracy_score(y_test_3k, rf_preds_3k)

# PBMC10k performance (transfer — no labels available, so we measure
# by comparing to Transformer predictions as pseudo-labels, and also
# by looking at prediction confidence distribution)
rf_preds_new = rf.predict(X_new)
rf_proba_new = rf.predict_proba(X_new)
rf_conf_new  = rf_proba_new.max(axis=1)  # confidence per cell

print(f"RF on PBMC3k (same dataset): {rf_acc_3k*100:.2f}%")
print(f"RF on PBMC10k prediction distribution:")
unique, counts = np.unique(rf_preds_new, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {label_encoder.classes_[u]}: {c} cells ({c/len(rf_preds_new)*100:.1f}%)")
print(f"RF mean confidence on PBMC10k: {rf_conf_new.mean()*100:.1f}%")

# ── Transformer on PBMC10k (no retraining) ───────────────────
print("\n[STEP 2] Transformer predictions on PBMC10k (no retraining)...")
all_logits = []
with torch.no_grad():
    for start in range(0, len(X_new), 256):
        batch   = torch.FloatTensor(X_new[start:start+256])
        logits  = model(batch)
        all_logits.append(logits.numpy())
all_logits   = np.concatenate(all_logits, axis=0)
tr_preds_new = all_logits.argmax(axis=1)
tr_proba_new = torch.softmax(torch.FloatTensor(all_logits), dim=1).numpy()
tr_conf_new  = tr_proba_new.max(axis=1)

# PBMC3k performance (same dataset)
with torch.no_grad():
    tr_preds_3k = model(torch.FloatTensor(X_test_3k)).argmax(1).numpy()
tr_acc_3k = accuracy_score(y_test_3k, tr_preds_3k)

print(f"Transformer on PBMC3k (same dataset): {tr_acc_3k*100:.2f}%")
print(f"Transformer on PBMC10k prediction distribution:")
unique, counts = np.unique(tr_preds_new, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {label_encoder.classes_[u]}: {c} cells ({c/len(tr_preds_new)*100:.1f}%)")
print(f"Transformer mean confidence on PBMC10k: {tr_conf_new.mean()*100:.1f}%")

# ── Agreement analysis ────────────────────────────────────────
# Since PBMC10k has no ground truth labels, we measure:
# 1. Agreement between RF and Transformer on PBMC10k
# 2. Prediction confidence (higher = more certain = more reliable)
# 3. Distribution of predicted cell types (should match known PBMC biology)

agreement = np.mean(rf_preds_new == tr_preds_new)
print(f"\nRF vs Transformer agreement on PBMC10k: {agreement*100:.1f}%")
print("(Higher agreement = both models see the same biology)")

# ── Expected PBMC biology check ───────────────────────────────
# In any PBMC sample, T cells should be ~40-60% of cells
# Monocytes ~15-25%, B cells ~5-15%, NK cells ~5-15%
print("\nBiological plausibility check:")
print("Expected PBMC composition vs Transformer predictions:")
expected = {
    'CD4+ T cells': (30, 60),
    'CD14+ Monocytes': (10, 30),
    'B cells': (5, 20),
    'NK / CD8+ T cells': (5, 20),
    'Dendritic cells': (0, 5),
    'Platelets': (0, 5),
}
n_new = len(tr_preds_new)
for ci, cname in enumerate(label_encoder.classes_):
    count = np.sum(tr_preds_new == ci)
    pct   = count / n_new * 100
    lo, hi = expected.get(cname, (0, 100))
    status = "✓ plausible" if lo <= pct <= hi else "? check"
    print(f"  {cname:<25} predicted {pct:5.1f}%  (expected {lo}-{hi}%)  {status}")

# ── Confidence comparison (key metric) ───────────────────────
print(f"\n{'Model':<15} {'Same-Dataset Acc':>18} {'New-Dataset Conf':>18}")
print("-" * 55)
print(f"{'Random Forest':<15} {rf_acc_3k*100:>17.2f}% {rf_conf_new.mean()*100:>17.1f}%")
print(f"{'Transformer':<15} {tr_acc_3k*100:>17.2f}% {tr_conf_new.mean()*100:>17.1f}%")
print("\nHigher confidence on new dataset = better transfer")

# ── PLOTS ─────────────────────────────────────────────────────
print("\nGenerating plots...")

# Plot 1: Prediction distribution on PBMC10k — RF vs Transformer
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
class_names = list(label_encoder.classes_)

for ax, preds, title, conf in zip(
    axes,
    [rf_preds_new, tr_preds_new],
    ['Random Forest → PBMC10k', 'Transformer → PBMC10k'],
    [rf_conf_new, tr_conf_new]
):
    counts = [np.sum(preds == i) for i in range(N_CLASSES)]
    colors = ['#1f77b4','#9467bd','#ff7f0e','#e377c2','#2ca02c','#bcbd22']
    bars = ax.bar(class_names, counts, color=colors,
                  edgecolor='black', linewidth=0.8)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 20,
                str(c), ha='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted Cell Type', fontsize=11)
    ax.set_ylabel('Number of Cells', fontsize=11)
    ax.set_title(f'{title}\nMean confidence: {conf.mean()*100:.1f}%', fontsize=12)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Cell Type Predictions on Unseen PBMC10k Dataset\n'
             '(Both models trained on PBMC3k only — no retraining)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('stage3_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: stage3_predictions.png")

# Plot 2: Confidence distribution — RF vs Transformer on new data
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, conf, title, color in zip(
    axes,
    [rf_conf_new, tr_conf_new],
    ['Random Forest', 'Transformer'],
    ['#7fbfff', '#ff8c42']
):
    ax.hist(conf, bins=30, color=color, edgecolor='black',
            linewidth=0.5, alpha=0.85)
    ax.axvline(conf.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {conf.mean()*100:.1f}%')
    ax.set_xlabel('Prediction Confidence', fontsize=11)
    ax.set_ylabel('Number of Cells', fontsize=11)
    ax.set_title(f'{title}\nConfidence on PBMC10k', fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
plt.suptitle('Prediction Confidence on Unseen Dataset\n'
             'Higher = more certain the model knows the cell type',
             fontsize=12)
plt.tight_layout()
plt.savefig('stage3_confidence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: stage3_confidence.png")

# Plot 3: RF vs Transformer agreement heatmap on PBMC10k
fig, ax = plt.subplots(figsize=(9, 7))
agree_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for rf_p, tr_p in zip(rf_preds_new, tr_preds_new):
    agree_matrix[rf_p, tr_p] += 1

sns.heatmap(agree_matrix, annot=True, fmt='d',
            xticklabels=[f'T:{c}' for c in class_names],
            yticklabels=[f'RF:{c}' for c in class_names],
            cmap='YlOrRd', ax=ax, linewidths=0.5)
ax.set_title(f'RF vs Transformer Prediction Agreement on PBMC10k\n'
             f'Overall agreement: {agreement*100:.1f}%', fontsize=12)
ax.tick_params(axis='x', rotation=30)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig('stage3_agreement.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: stage3_agreement.png")

# Plot 4: Summary comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Same-dataset accuracy
axes[0].bar(['RF\n(PBMC3k)', 'Transformer\n(PBMC3k)'],
            [rf_acc_3k*100, tr_acc_3k*100],
            color=['#7fbfff', '#ff8c42'], edgecolor='black', width=0.4)
for i, v in enumerate([rf_acc_3k*100, tr_acc_3k*100]):
    axes[0].text(i, v + 0.2, f'{v:.2f}%', ha='center',
                 fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy (%)'); axes[0].set_ylim([88, 100])
axes[0].set_title('Same Dataset (PBMC3k)\nLabels available — direct comparison')
axes[0].grid(axis='y', alpha=0.3)

# New dataset confidence
axes[1].bar(['RF\n(PBMC10k)', 'Transformer\n(PBMC10k)'],
            [rf_conf_new.mean()*100, tr_conf_new.mean()*100],
            color=['#7fbfff', '#ff8c42'], edgecolor='black', width=0.4)
for i, v in enumerate([rf_conf_new.mean()*100, tr_conf_new.mean()*100]):
    axes[1].text(i, v + 0.2, f'{v:.1f}%', ha='center',
                 fontsize=12, fontweight='bold')
axes[1].set_ylabel('Mean Prediction Confidence (%)')
axes[1].set_ylim([50, 100])
axes[1].set_title('New Dataset (PBMC10k)\nNo labels — confidence as proxy')
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Stage 3 Summary: Same-Dataset vs Cross-Dataset Performance',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('stage3_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: stage3_summary.png")

# Attention on PBMC10k cells
print("\nGenerating attention on PBMC10k cells...")
marker_genes  = ['IL7R','LTB','CD3D','CD3E', 'TRAC','GIMAP5','MAL','NKG7','GNLY','GZMA','GZMB', 'PRF1','CTSW','FGFBP2','MS4A1','CD79A','CD79B', 'CD19','HLA-DMB','HLA-DMA','LYZ','TYROBP','FCER1G', 'LGALS2','FCN1','S100A8', 'S100A9','PYCARD','CTSS','FCER1A','CLEC10A', 'HLA-DPA1','HLA-DPB1','PPBP','PF4','GNG11', 'ITGA2B','TREML1','TOR1A','C1orf162']

found_markers = [g for g in marker_genes if g in gene_names]

all_attn = []
for start in range(0, min(len(X_new), 512), 64):
    chunk = torch.FloatTensor(X_new[start:start+64])
    w     = model.get_attention_weights(chunk)
    all_attn.append(w[:, 0, 1:].numpy())
cls_attn_new = np.concatenate(all_attn, axis=0)
y_new_sample = tr_preds_new[:len(cls_attn_new)]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
for ci, (ax, cname) in enumerate(zip(axes, label_encoder.classes_)):
    mask = (y_new_sample == ci)
    if mask.sum() == 0:
        ax.axis('off'); continue
    avg   = cls_attn_new[mask].mean(0)
    top_i = np.argsort(avg)[-15:][::-1]
    tg    = [gene_names[i] for i in top_i]
    tv    = avg[top_i]
    cols  = ['#d62728' if g in found_markers else '#4878cf' for g in tg]
    ax.barh(range(len(tg)), tv[::-1], color=cols[::-1],
            edgecolor='black', linewidth=0.4)
    ax.set_yticks(range(len(tg)))
    ax.set_yticklabels(tg[::-1], fontsize=9)
    ax.set_title(f'{cname}  (n={mask.sum()} PBMC10k cells)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Attention Score', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

fig.legend(handles=[
    mpatches.Patch(facecolor='#d62728', label='Known marker gene'),
    mpatches.Patch(facecolor='#4878cf', label='Other gene')
], loc='lower right', fontsize=11)
plt.suptitle('Attention on PBMC10k (Unseen Dataset)\n'
             'Same attention patterns = model learned biology, not dataset artifacts',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('stage3_attention_new_dataset.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: stage3_attention_new_dataset.png")

# ── Save results ──────────────────────────────────────────────
results = {
    'same_dataset': {
        'rf_accuracy':          float(rf_acc_3k),
        'transformer_accuracy': float(tr_acc_3k),
    },
    'cross_dataset_pbmc10k': {
        'rf_mean_confidence':          float(rf_conf_new.mean()),
        'transformer_mean_confidence': float(tr_conf_new.mean()),
        'rf_transformer_agreement':    float(agreement),
        'n_new_cells':                 int(len(X_new)),
    }
}
with open('stage3_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: stage3_results.json")

print("\n" + "=" * 60)
print("STAGE 3 COMPLETE")
print("=" * 60)
print(f"Same dataset  — RF: {rf_acc_3k*100:.2f}% | Transformer: {tr_acc_3k*100:.2f}%")
print(f"New dataset   — RF conf: {rf_conf_new.mean()*100:.1f}% | "
      f"Transformer conf: {tr_conf_new.mean()*100:.1f}%")
print(f"Model agreement on PBMC10k: {agreement*100:.1f}%")
print("\nFiles saved:")
print("  stage3_predictions.png")
print("  stage3_confidence.png")
print("  stage3_agreement.png")
print("  stage3_summary.png")
print("  stage3_attention_new_dataset.png")
print("  stage3_results.json")
