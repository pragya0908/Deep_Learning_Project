import os
import shutil
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from src.model import DiabetesMLP

os.makedirs('figures', exist_ok=True)

# 1. Copy correlation heatmap
shutil.copy('artifacts/correlation_heatmap.png', 'figures/correlation_heatmap.png')
print('Copied correlation_heatmap.png to figures/')

# Set general aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Georgia']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Load experiment summaries and histories
with open('artifacts/experiments_summary.json', 'r') as f:
    exp_summary = json.load(f)

experiments = exp_summary['experiments']
model_histories = {}
colors = {
    'Model A': '#1f77b4',  # Blue
    'Model B': '#ff7f0e',  # Orange
    'Model C': '#2ca02c',  # Green
    'Model D': '#d62728',  # Red
    'Model E': '#9467bd'   # Purple
}
labels = {
    'Model A': 'Model A: ReLU [32, 16]',
    'Model B': 'Model B: LeakyReLU [32, 16]',
    'Model C': 'Model C: Tanh [32, 16]',
    'Model D': 'Model D: Shallow [16]',
    'Model E': 'Model E: Deep Reg. [64, 32, 16]'
}

for exp in experiments:
    mid = exp['model_id']
    safe_name = exp['name'].replace(' ', '_').replace('(', '').replace(')', '')
    hist_file = f'artifacts/exp_{mid.replace(" ", "_")}_{safe_name}_history.json'
    if os.path.exists(hist_file):
        with open(hist_file, 'r') as hf:
            model_histories[mid] = json.load(hf)

# 2. Generate figures/loss_curves.png (Side-by-side Train & Val BCE Loss)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

for mid, hist in model_histories.items():
    epochs = hist['epoch']
    axes[0].plot(epochs, hist['train_loss'], label=labels[mid], color=colors[mid], linewidth=1.8)
    axes[1].plot(epochs, hist['val_loss'], label=labels[mid], color=colors[mid], linewidth=1.8)
    
    # Mark best validation epoch
    exp_entry = next(e for e in exp_summary['experiments'] if e['model_id'] == mid)
    best_ep = exp_entry['training']['best_val_epoch']
    best_vl = exp_entry['training']['best_val_loss']
    axes[1].scatter(best_ep, best_vl, color=colors[mid], s=45, zorder=5, edgecolor='black', linewidth=0.6)

axes[0].set_title('(a) Training BCE Loss Across Epochs', fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Binary Cross-Entropy Loss', fontsize=11)
axes[0].set_ylim(0.3, 0.75)
axes[0].legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

axes[1].set_title('(b) Validation BCE Loss (Markers: Early Stopping Points)', fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Binary Cross-Entropy Loss', fontsize=11)
axes[1].set_ylim(0.45, 0.75)
axes[1].legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.savefig('figures/loss_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print('Generated figures/loss_curves.png')

# 3. Generate figures/roc_curves.png
X_test = np.load('artifacts/X_test.npy')
y_test = np.load('artifacts/y_test.npy')
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

for exp in experiments:
    mid = exp['model_id']
    safe_name = exp['name'].replace(' ', '_').replace('(', '').replace(')', '')
    ckpt_file = f'artifacts/exp_{mid.replace(" ", "_")}_{safe_name}.pt'
    
    cfg = exp['hyperparameters']
    model = DiabetesMLP(
        input_dim=8,
        hidden_layers=cfg['hidden_layers'],
        activation=cfg['activation'],
        dropout_rate=cfg['dropout_rate']
    )
    checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).numpy().flatten()
    
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{labels[mid]} (AUC = {roc_auc:.4f})", color=colors[mid], linewidth=2.0)

ax.plot([0, 1], [0, 1], linestyle='--', color='#777777', linewidth=1.2, label='No-Skill Baseline (AUC = 0.5000)')
ax.set_title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.92, fontsize=9.5)
plt.tight_layout()
plt.savefig('figures/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print('Generated figures/roc_curves.png')

# 4. Generate figures/confusion_matrices.png (Winning Model - Model A: ReLU [32, 16])
cm = np.array([[66, 9], [20, 21]])
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

group_names = ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)']
group_counts = [f"{value:0.0f}" for value in cm.flatten()]
group_percentages = [f"{value:.1%}" for value in cm.flatten() / np.sum(cm)]
annot_labels = [f"{v1}\n{v2} ({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
annot_labels = np.asarray(annot_labels).reshape(2, 2)

sns.heatmap(
    cm,
    annot=annot_labels,
    fmt='',
    cmap='Blues',
    cbar=False,
    ax=ax,
    linewidths=1.5,
    linecolor='white',
    annot_kws={'fontsize': 10.5, 'fontweight': 'bold'}
)

ax.set_title('Confusion Matrix — Model A (ReLU [32, 16])\nHeld-Out Test Split (N = 116)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Predicted Clinical Label', fontsize=11, fontweight='bold')
ax.set_ylabel('True Clinical Label', fontsize=11, fontweight='bold')
ax.set_xticklabels(['Negative (0)', 'Positive (1)'], fontsize=10.5)
ax.set_yticklabels(['Negative (0)', 'Positive (1)'], fontsize=10.5, rotation=0)

plt.tight_layout()
plt.savefig('figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print('Generated figures/confusion_matrices.png')
