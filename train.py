# ============================================================
# train.py — corrected and complete
# Run: nohup python train.py > output.log 2>&1 &
#      tail -f output.log
# ============================================================

import os, sys, time, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import scanpy as sc
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

print("=" * 60)
print("Single-Cell Transformer Training Script")
print("=" * 60)

# ── device ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device}")
if device.type == 'cuda':
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    # Use all available CPU cores for PyTorch ops
    n_cores = min(64, os.cpu_count() or 8)
    torch.set_num_threads(n_cores)
    print(f"[INFO] CPU cores used: {n_cores}")
sys.stdout.flush()

# ── load data ─────────────────────────────────────────────────
print("\n[STEP 1] Loading pbmc3k_annotated.h5ad ...")
# File must be in the SAME directory as train.py
# Run this script from: ~/transformer_project/
if not os.path.exists('pbmc3k_annotated.h5ad'):
    print("[ERROR] pbmc3k_annotated.h5ad not found in current directory.")
    print(f"[ERROR] Current directory: {os.getcwd()}")
    print("[ERROR] Run: cd ~/transformer_project && python train.py")
    sys.exit(1)

adata      = sc.read_h5ad('pbmc3k_annotated.h5ad')
gene_names = list(adata.var_names)
print(f"[INFO] Shape: {adata.shape}")
print(f"[INFO] Cell types available: {list(adata.obs['cell_type'].unique())}")

# Convert sparse matrix to dense numpy array
X_raw = adata.X
X = X_raw.toarray().astype(np.float32) if sp.issparse(X_raw) \
    else np.array(X_raw, dtype=np.float32)

# Encode string labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(adata.obs['cell_type'].values)

print(f"[INFO] X shape: {X.shape}  |  value range: {X.min():.2f} to {X.max():.2f}")
print("[INFO] Label encoding:")
for i, name in enumerate(label_encoder.classes_):
    print(f"         {i} → {name}  ({np.sum(y == i)} cells)")
sys.stdout.flush()

# ── train/test split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
N_GENES   = X_train.shape[1]
N_CLASSES = len(label_encoder.classes_)

np.save('X_train.npy', X_train)
np.save('X_test.npy',  X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy',  y_test)
print(f"[INFO] Train: {len(X_train)} cells | Test: {len(X_test)} cells")
print(f"[INFO] N_GENES={N_GENES} | N_CLASSES={N_CLASSES}")

# ── random forest baseline ─────────────────────────────────────
print("\n[STEP 2] Training Random Forest baseline ...")
sys.stdout.flush()
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=200, random_state=42,
    class_weight='balanced', n_jobs=-1    # uses all CPU cores
)
rf.fit(X_train, y_train)
rf_preds    = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"[INFO] RF accuracy: {rf_accuracy*100:.2f}%  (took {time.time()-t0:.1f}s)")
print("[INFO] RF per-class report:")
print(classification_report(y_test, rf_preds,
      target_names=label_encoder.classes_, digits=3))
sys.stdout.flush()

# ── model definition ──────────────────────────────────────────
class GeneEmbedding(nn.Module):
    def __init__(self, n_genes, d_model):
        super().__init__()
        self.gene_embed = nn.Embedding(n_genes, d_model)
        self.value_proj = nn.Linear(1, d_model)
        self.norm       = nn.LayerNorm(d_model)

    def forward(self, x):
        b, g = x.shape
        ids  = torch.arange(g, device=x.device)
        ge   = self.gene_embed(ids).unsqueeze(0).expand(b, -1, -1)
        ve   = self.value_proj(x.unsqueeze(-1))
        return self.norm(ge + ve)


class SingleCellTransformer(nn.Module):
    def __init__(self, n_genes, n_classes,
                 d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.gene_embedding = GeneEmbedding(n_genes, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            enc, num_layers=n_layers, enable_nested_tensor=False
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        b   = x.shape[0]
        ge  = self.gene_embedding(x)
        cls = self.cls_token.expand(b, -1, -1)
        out = self.transformer(torch.cat([cls, ge], dim=1))
        return self.classifier(out[:, 0, :])

    def get_attention_weights(self, x):
        self.eval()
        with torch.no_grad():
            b   = x.shape[0]
            ge  = self.gene_embedding(x)
            cls = self.cls_token.expand(b, -1, -1)
            tok = torch.cat([cls, ge], dim=1)
            _, w = self.transformer.layers[0].self_attn(
                tok, tok, tok,
                need_weights=True, average_attn_weights=True
            )
        return w   # shape: (batch, seq_len, seq_len)


# ── dataset & loaders ─────────────────────────────────────────
class SCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):          return len(self.y)
    def __getitem__(self, i):   return self.X[i], self.y[i]

class_counts   = np.bincount(y_train)
class_weights  = 1.0 / class_counts
sample_weights = class_weights[y_train]

sampler = WeightedRandomSampler(
    torch.FloatTensor(sample_weights), len(sample_weights), replacement=True
)
train_loader = DataLoader(SCDataset(X_train, y_train),
                          batch_size=64, sampler=sampler,
                          num_workers=2, pin_memory=(device.type=='cuda'))
test_loader  = DataLoader(SCDataset(X_test, y_test),
                          batch_size=128, shuffle=False,
                          num_workers=2, pin_memory=(device.type=='cuda'))

# ── training setup ────────────────────────────────────────────
print("\n[STEP 3] Training Transformer ...")
model = SingleCellTransformer(N_GENES, N_CLASSES).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"[INFO] Parameters: {total_params:,}")
print(f"[INFO] Running on: {device}")
sys.stdout.flush()

cw_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=cw_tensor)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

N_EPOCHS      = 60
train_losses  = []
train_accs    = []
test_accs     = []
best_test_acc = 0.0
best_state    = None
t_start       = time.time()

print(f"\n{'Epoch':>6} | {'Loss':>8} | {'TrainAcc':>9} | {'TestAcc':>8} | {'LR':>10} | {'Time':>6}")
print("-" * 65)
sys.stdout.flush()

for epoch in range(N_EPOCHS):
    # ── train phase ──────────────────────────────────────────
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        # FIX: single forward pass (old script called model(xb) twice)
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (logits.detach().argmax(1) == yb).sum().item()
        total      += len(yb)

    train_loss = total_loss / len(train_loader)
    train_acc  = correct / total

    # ── eval phase ───────────────────────────────────────────
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)

    test_acc = accuracy_score(y_test, all_preds)
    scheduler.step(test_acc)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        # Store on CPU to avoid holding GPU memory
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    lr      = optimizer.param_groups[0]['lr']
    elapsed = time.time() - t_start

    print(f"{epoch+1:>6} | {train_loss:>8.4f} | {train_acc*100:>8.2f}% | "
          f"{test_acc*100:>7.2f}% | {lr:>10.6f} | {elapsed:>5.0f}s",
          flush=True)

print(f"\n[RESULT] Best transformer accuracy : {best_test_acc*100:.2f}%")
print(f"[RESULT] Random Forest accuracy    : {rf_accuracy*100:.2f}%")
print(f"[RESULT] Total training time       : {time.time()-t_start:.1f}s")
sys.stdout.flush()

# ── save model ────────────────────────────────────────────────
print("\n[STEP 4] Saving model and arrays ...")
torch.save({
    'model_state_dict':      best_state,
    'label_encoder_classes': label_encoder.classes_,
    'gene_names':            gene_names,
    'n_genes':               N_GENES,
    'n_classes':             N_CLASSES,
    'best_test_acc':         best_test_acc,
    'rf_accuracy':           rf_accuracy,
    'train_losses':          train_losses,
    'train_accs':            train_accs,
    'test_accs':             test_accs,
}, 'single_cell_transformer.pt')
print("[INFO] Saved: single_cell_transformer.pt")

with open('results_summary.json', 'w') as f:
    json.dump({
        'transformer_accuracy': float(best_test_acc),
        'rf_accuracy':          float(rf_accuracy),
        'n_epochs':             N_EPOCHS,
        'n_genes':              N_GENES,
        'n_classes':            N_CLASSES,
        'classes':              list(label_encoder.classes_),
    }, f, indent=2)
print("[INFO] Saved: results_summary.json")

# ── plots ─────────────────────────────────────────────────────
print("\n[STEP 5] Generating plots (saved as PNG) ...")

# Reload best model on CPU for all plotting/attention work
model_cpu = SingleCellTransformer(N_GENES, N_CLASSES)
model_cpu.load_state_dict(best_state)
model_cpu.eval()

with torch.no_grad():
    final_preds = model_cpu(torch.FloatTensor(X_test)).argmax(1).numpy()
transformer_acc = accuracy_score(y_test, final_preds)

# Plot 1: Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ep = range(1, N_EPOCHS + 1)
axes[0].plot(ep, train_losses, color='steelblue', linewidth=2, label='Train Loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss Curve')
axes[0].grid(alpha=0.3); axes[0].legend()

axes[1].plot(ep, [a*100 for a in train_accs], 'b--', linewidth=2, label='Train Acc')
axes[1].plot(ep, [a*100 for a in test_accs],  color='darkorange', linewidth=2, label='Test Acc')
axes[1].axhline(rf_accuracy*100, color='red', linestyle=':', linewidth=2,
                label=f'RF Baseline ({rf_accuracy*100:.1f}%)')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Transformer vs Random Forest')
axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylim([40, 102])
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("[INFO] Saved: training_curves.png")

# Plot 2: Side-by-side confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, preds, title, acc in zip(
    axes,
    [rf_preds, final_preds],
    ['Random Forest', 'Transformer'],
    [rf_accuracy, transformer_acc]
):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                cmap='Blues', ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{title}\nAccuracy: {acc*100:.2f}%')
    ax.tick_params(axis='x', rotation=30)
plt.suptitle('RF vs Transformer — Confusion Matrices', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("[INFO] Saved: confusion_matrices.png")

# Plot 3: Model accuracy comparison bar chart
fig, ax = plt.subplots(figsize=(7, 5))
models = ['Random Forest\n(Baseline)', 'Transformer\n(This Work)']
accs   = [rf_accuracy*100, best_test_acc*100]
colors = ['#7fbfff', '#ff8c42']
bars = ax.bar(models, accs, color=colors, edgecolor='black', linewidth=1.2, width=0.4)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{acc:.2f}%', ha='center', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)'); ax.set_ylim([88, 100])
ax.set_title('PBMC3k Cell Type Classification\nRF vs Transformer')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[INFO] Saved: model_comparison.png")

# Plot 4: Attention visualization per cell type
print("\n[STEP 6] Attention visualization ...")
marker_genes  = ['NKG7','GZMA','MS4A1','CD79A','CST3','PPBP',
                 'LYZ','IL7R','CD3D','CD8A','GNLY','LTB']
found_markers = [g for g in marker_genes if g in gene_names]
print(f"[INFO] Marker genes found in HVG set: {found_markers}")

# Use all test cells for more stable attention averages
X_sample = torch.FloatTensor(X_test)
y_sample = y_test

# Process in small chunks to avoid RAM spike
CHUNK = 64
all_attn = []
for start in range(0, len(X_sample), CHUNK):
    chunk = X_sample[start:start+CHUNK]
    w     = model_cpu.get_attention_weights(chunk)
    all_attn.append(w[:, 0, 1:].numpy())   # CLS→gene attention
cls_attn = np.concatenate(all_attn, axis=0)   # (n_test, n_genes)
print(f"[INFO] Attention shape: {cls_attn.shape}")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for ci, (ax, cname) in enumerate(zip(axes, label_encoder.classes_)):
    mask = (y_sample == ci)
    n    = mask.sum()
    if n == 0:
        ax.text(0.5, 0.5, f'{cname}\nNo test cells',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        continue

    avg_attn = cls_attn[mask].mean(axis=0)       # average over all cells of this type
    top_idx  = np.argsort(avg_attn)[-15:][::-1]  # top 15 genes
    top_g    = [gene_names[i] for i in top_idx]
    top_v    = avg_attn[top_idx]
    colors   = ['#d62728' if g in found_markers else '#4878cf' for g in top_g]

    ax.barh(range(len(top_g)), top_v[::-1],
            color=colors[::-1], edgecolor='black', linewidth=0.4)
    ax.set_yticks(range(len(top_g)))
    ax.set_yticklabels(top_g[::-1], fontsize=9)
    ax.set_title(f'{cname}  (n={n} cells)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Attention Score', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Print top genes to log
    hits = [g for g in top_g if g in found_markers]
    print(f"[ATTN] {cname}: top genes = {top_g[:5]} | markers found = {hits}")

legend_handles = [
    mpatches.Patch(facecolor='#d62728', edgecolor='black', label='Known biological marker gene'),
    mpatches.Patch(facecolor='#4878cf', edgecolor='black', label='Other high-attention gene')
]
fig.legend(handles=legend_handles, loc='lower right', fontsize=11)
plt.suptitle(
    'Transformer Self-Attention: Genes Attended to per Cell Type\n'
    'Red = Known PBMC Marker Genes',
    fontsize=13, fontweight='bold', y=1.01
)
plt.tight_layout()
plt.savefig('attention_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("[INFO] Saved: attention_visualization.png")

# ── final summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE — FILES SAVED:")
print("=" * 60)
print("  single_cell_transformer.pt   — model weights + metadata")
print("  results_summary.json         — accuracy results")
print("  training_curves.png          — loss and accuracy plots")
print("  confusion_matrices.png       — RF vs Transformer comparison")
print("  model_comparison.png         — bar chart comparison")
print("  attention_visualization.png  — biological interpretability")
print("  X_train/test.npy, y_train/test.npy")
print(f"\n  Random Forest accuracy  : {rf_accuracy*100:.2f}%")
print(f"  Transformer accuracy    : {best_test_acc*100:.2f}%")
print("=" * 60)
