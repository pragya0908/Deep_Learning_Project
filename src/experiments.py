"""
Experiments module for Deep Learning CIA 1.
Assignment: Early Diabetes Risk Prediction

Runs controlled experiments on the exact same preprocessed stratified split (seed=42):

Experiment Set 1 — Activation Function Comparison:
  - Model A: ReLU               (hidden_layers=[32, 16], lr=0.001, dropout=0.2, wd=1e-4)
  - Model B: LeakyReLU (a=0.01) (hidden_layers=[32, 16], lr=0.001, dropout=0.2, wd=1e-4)
  - Model C: Tanh               (hidden_layers=[32, 16], lr=0.001, dropout=0.2, wd=1e-4)

Experiment Set 2 — Architecture & Regularization Comparison:
  (Using best-performing activation function from Set 1)
  - Model D (Shallow):          hidden_layers=[16],         dropout=0.0, weight_decay=0.0
  - Model E (Deep Regularized): hidden_layers=[64, 32, 16], dropout=0.3, weight_decay=1e-4

Outputs:
  - artifacts/experiments_summary.json
  - artifacts/comparison_table.csv
  - artifacts/loss_curves_comparison.png
  - artifacts/roc_curves_comparison.png
  - artifacts/confusion_matrices_comparison.png
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure project root and src are available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError as err:
    raise ImportError(
        "\n[PyTorch Import Error] 'torch' was not found in the current Python environment.\n"
        "To resolve this in VS Code:\n"
        "  1. Press Ctrl+Shift+P -> 'Python: Select Interpreter'\n"
        "  2. Choose '.venv' (Python 3.11.9) located at d:\\Deep_Learning\\.venv\\Scripts\\python.exe\n"
    ) from err

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

from src.model import DiabetesMLP
from src.train import TabularDataset, train_model, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluate trained model on the held-out test split.

    Returns:
    --------
    metrics : Dict[str, Any]
        Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix (TN, FP, FN, TP).
    all_probs : np.ndarray
        Predicted probabilities for ROC curve plotting.
    all_targets : np.ndarray
        True binary target values.
    """
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            probs = model.predict_proba(features).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy().flatten())

    all_targets = np.array(all_targets, dtype=int)
    all_probs = np.array(all_probs, dtype=float)
    all_preds = (all_probs >= 0.5).astype(int)

    acc = float(accuracy_score(all_targets, all_preds))
    prec = float(precision_score(all_targets, all_preds, zero_division=0))
    rec = float(recall_score(all_targets, all_preds, zero_division=0))
    f1 = float(f1_score(all_targets, all_preds, zero_division=0))
    try:
        auc = float(roc_auc_score(all_targets, all_probs))
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        },
        "cm_matrix": cm.tolist()
    }
    return metrics, all_probs, all_targets


def plot_loss_curves(
    experiments_data: List[Dict[str, Any]],
    output_path: str = "artifacts/loss_curves_comparison.png"
) -> str:
    """
    Generate high-resolution comparison plot of train/validation loss curves.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    n_models = len(experiments_data)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.2), sharey=True)

    if n_models == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments_data):
        hist = exp["history"]
        epochs = hist["epoch"]
        train_loss = hist["train_loss"]
        val_loss = hist["val_loss"]
        best_epoch = exp["best_val_epoch"]
        best_loss = exp["best_val_loss"]

        ax.plot(epochs, train_loss, label="Train Loss", color="#2b5c8f", lw=2)
        ax.plot(epochs, val_loss, label="Val Loss", color="#d95f02", lw=2, linestyle="--")

        # Mark best validation epoch
        ax.scatter([best_epoch], [best_loss], color="#d62728", s=80, zorder=5,
                   label=f"Best Ep {best_epoch} ({best_loss:.3f})")
        ax.axvline(best_epoch, color="#d62728", linestyle=":", alpha=0.6)

        ax.set_title(f"{exp['model_id']}: {exp['name']}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("BCE Loss" if ax == axes[0] else "", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0.2, 0.8)

    plt.suptitle("Overlaid Training & Validation Loss Curves (Early Stopping Comparison)",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved loss curves comparison to: {output_path}")
    return output_path


def plot_combined_roc_curves(
    experiments_data: List[Dict[str, Any]],
    output_path: str = "artifacts/roc_curves_comparison.png"
) -> str:
    """
    Generate high-resolution combined ROC curves plot for all 5 models.
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 7))

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, exp in enumerate(experiments_data):
        y_true = exp["y_true"]
        y_probs = exp["y_probs"]
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = exp["metrics"]["roc_auc"]
        color = palette[idx % len(palette)]

        label = f"{exp['model_id']} ({exp['name']}): AUC = {auc:.4f}"
        ax.plot(fpr, tpr, lw=2.2, color=color, label=label)

    # Diagonal no-skill baseline
    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle=":", label="No-Skill Baseline (AUC = 0.50)")

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Combined ROC Curves — 5 Model Comparison\n(Held-out Test Split, N=116)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved combined ROC curves to: {output_path}")
    return output_path


def plot_side_by_side_confusion_matrices(
    experiments_data: List[Dict[str, Any]],
    output_path: str = "artifacts/confusion_matrices_comparison.png"
) -> str:
    """
    Generate high-resolution side-by-side confusion matrix heatmaps.
    """
    sns.set_theme(style="white")
    n_models = len(experiments_data)
    fig, axes = plt.subplots(1, n_models, figsize=(3.8 * n_models, 3.8))

    if n_models == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments_data):
        cm = np.array(exp["metrics"]["cm_matrix"])
        tn, fp, fn, tp = exp["metrics"]["confusion_matrix"]["TN"], exp["metrics"]["confusion_matrix"]["FP"], exp["metrics"]["confusion_matrix"]["FN"], exp["metrics"]["confusion_matrix"]["TP"]
        acc = exp["metrics"]["accuracy"]
        f1 = exp["metrics"]["f1_score"]

        # Annotations with counts and labels
        annot = np.array([
            [f"TN\n{tn}", f"FP\n{fp}"],
            [f"FN\n{fn}", f"TP\n{tp}"]
        ])

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            linewidths=1.2,
            linecolor="white",
            annot_kws={"fontsize": 11, "fontweight": "bold"}
        )

        ax.set_title(f"{exp['model_id']}: {exp['name']}\nAcc: {acc*100:.1f}% | F1: {f1:.3f}",
                     fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Label", fontsize=9.5)
        ax.set_ylabel("True Label" if ax == axes[0] else "", fontsize=9.5)
        ax.set_xticklabels(["Neg (0)", "Pos (1)"], fontsize=9)
        ax.set_yticklabels(["Neg (0)", "Pos (1)"], fontsize=9)

    plt.suptitle("Side-by-Side Confusion Matrices (Holdout Test Split)",
                 fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved side-by-side confusion matrices to: {output_path}")
    return output_path


def run_controlled_experiments(
    artifacts_dir: str = "artifacts",
    batch_size: int = 32,
    epochs: int = 150,
    patience: int = 15
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute controlled Experiment Set 1 and Set 2, saving all summaries, tables, and plots.
    """
    logger.info("===================================================================")
    logger.info("   STARTING CONTROLLED DEEP LEARNING EXPERIMENTS (SEED = 42)      ")
    logger.info("===================================================================")

    os.makedirs(artifacts_dir, exist_ok=True)
    npz_path = os.path.join(artifacts_dir, "processed_data.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Processed dataset '{npz_path}' not found. Run src/preprocess.py first.")

    # Load the exact preprocessed split
    data = np.load(npz_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    logger.info(f"Loaded preprocessed splits: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TabularDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Executing experiments on device: {device}")

    # =========================================================================
    # EXPERIMENT SET 1: Activation Function Comparison
    # Fixed: hidden_layers=[32, 16], lr=0.001, dropout_rate=0.2, weight_decay=1e-4
    # =========================================================================
    set1_configs = [
        {
            "model_id": "Model A",
            "name": "ReLU",
            "activation": "relu",
            "hidden_layers": [32, 16],
            "dropout_rate": 0.2,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "set": "Set 1: Activation Comparison"
        },
        {
            "model_id": "Model B",
            "name": "LeakyReLU",
            "activation": "leaky_relu",
            "hidden_layers": [32, 16],
            "dropout_rate": 0.2,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "set": "Set 1: Activation Comparison"
        },
        {
            "model_id": "Model C",
            "name": "Tanh",
            "activation": "tanh",
            "hidden_layers": [32, 16],
            "dropout_rate": 0.2,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "set": "Set 1: Activation Comparison"
        }
    ]

    all_experiments_data = []

    logger.info("\n>>> RUNNING EXPERIMENT SET 1: ACTIVATION FUNCTION COMPARISON <<<")
    for cfg in set1_configs:
        logger.info(f"\n--- Training {cfg['model_id']}: {cfg['name']} (Activation={cfg['activation']}) ---")
        set_seed(42)

        model = DiabetesMLP(
            input_dim=8,
            hidden_layers=cfg["hidden_layers"],
            activation=cfg["activation"],
            dropout_rate=cfg["dropout_rate"]
        )

        ckpt_path = os.path.join(artifacts_dir, f"exp_{cfg['model_id'].replace(' ', '_')}_{cfg['name']}.pt")
        hist_path = os.path.join(artifacts_dir, f"exp_{cfg['model_id'].replace(' ', '_')}_{cfg['name']}_history.json")

        t_start = time.time()
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            patience=patience,
            save_path=ckpt_path,
            history_path=hist_path,
            device=device
        )
        elapsed_sec = round(time.time() - t_start, 2)

        best_val_loss = min(history["val_loss"])
        best_val_epoch = history["epoch"][history["val_loss"].index(best_val_loss)]

        metrics, y_probs, y_true = evaluate_model_on_test(trained_model, test_loader, device)

        exp_entry = {
            **cfg,
            "trainable_parameters": model.get_trainable_parameters(),
            "best_val_loss": round(best_val_loss, 4),
            "best_val_epoch": best_epoch if (best_epoch := best_val_epoch) else 0,
            "training_time_sec": elapsed_sec,
            "metrics": metrics,
            "history": history,
            "y_probs": y_probs,
            "y_true": y_true,
            "checkpoint_path": ckpt_path
        }
        all_experiments_data.append(exp_entry)
        logger.info(
            f"Finished {cfg['model_id']} in {elapsed_sec}s | Best Val Loss: {best_val_loss:.4f} (Ep {best_val_epoch}) | "
            f"Test Acc: {metrics['accuracy']*100:.2f}% | Test AUC: {metrics['roc_auc']:.4f}"
        )

    # Determine best-performing activation function from Set 1
    # Ranked by lowest best_val_loss (or highest val_roc_auc)
    best_set1_entry = min(all_experiments_data[:3], key=lambda x: x["best_val_loss"])
    best_activation = best_set1_entry["activation"]
    logger.info(
        f"\n===================================================================\n"
        f"Selected Best Activation from Set 1: '{best_activation.upper()}' "
        f"(from {best_set1_entry['model_id']} with Val Loss: {best_set1_entry['best_val_loss']:.4f})\n"
        f"==================================================================="
    )

    # =========================================================================
    # EXPERIMENT SET 2: Architecture & Regularization Comparison
    # Using best_activation from Set 1
    # =========================================================================
    set2_configs = [
        {
            "model_id": "Model D",
            "name": f"Shallow ({best_activation.capitalize()})",
            "activation": best_activation,
            "hidden_layers": [16],
            "dropout_rate": 0.0,
            "lr": 0.001,
            "weight_decay": 0.0,
            "set": "Set 2: Architecture & Regularization"
        },
        {
            "model_id": "Model E",
            "name": f"Deep Regularized ({best_activation.capitalize()})",
            "activation": best_activation,
            "hidden_layers": [64, 32, 16],
            "dropout_rate": 0.3,
            "lr": 0.001,
            "weight_decay": 1e-4,
            "set": "Set 2: Architecture & Regularization"
        }
    ]

    logger.info(f"\n>>> RUNNING EXPERIMENT SET 2: ARCHITECTURE & REGULARIZATION (USING {best_activation.upper()}) <<<")
    for cfg in set2_configs:
        logger.info(f"\n--- Training {cfg['model_id']}: {cfg['name']} ---")
        set_seed(42)

        model = DiabetesMLP(
            input_dim=8,
            hidden_layers=cfg["hidden_layers"],
            activation=cfg["activation"],
            dropout_rate=cfg["dropout_rate"]
        )

        safe_name = cfg['name'].replace(' ', '_').replace('(', '').replace(')', '')
        ckpt_path = os.path.join(artifacts_dir, f"exp_{cfg['model_id'].replace(' ', '_')}_{safe_name}.pt")
        hist_path = os.path.join(artifacts_dir, f"exp_{cfg['model_id'].replace(' ', '_')}_{safe_name}_history.json")

        t_start = time.time()
        trained_model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            patience=patience,
            save_path=ckpt_path,
            history_path=hist_path,
            device=device
        )
        elapsed_sec = round(time.time() - t_start, 2)

        best_val_loss = min(history["val_loss"])
        best_val_epoch = history["epoch"][history["val_loss"].index(best_val_loss)]

        metrics, y_probs, y_true = evaluate_model_on_test(trained_model, test_loader, device)

        exp_entry = {
            **cfg,
            "trainable_parameters": model.get_trainable_parameters(),
            "best_val_loss": round(best_val_loss, 4),
            "best_val_epoch": best_epoch if (best_epoch := best_val_epoch) else 0,
            "training_time_sec": elapsed_sec,
            "metrics": metrics,
            "history": history,
            "y_probs": y_probs,
            "y_true": y_true,
            "checkpoint_path": ckpt_path
        }
        all_experiments_data.append(exp_entry)
        logger.info(
            f"Finished {cfg['model_id']} in {elapsed_sec}s | Best Val Loss: {best_val_loss:.4f} (Ep {best_val_epoch}) | "
            f"Test Acc: {metrics['accuracy']*100:.2f}% | Test AUC: {metrics['roc_auc']:.4f}"
        )

    # =========================================================================
    # BUILD COMPARISON TABLE & SUMMARY JSON
    # =========================================================================
    table_rows = []
    json_summary_list = []

    for exp in all_experiments_data:
        cm = exp["metrics"]["confusion_matrix"]
        row = {
            "Model_ID": exp["model_id"],
            "Model_Name": exp["name"],
            "Experiment_Set": exp["set"],
            "Architecture": str(exp["hidden_layers"]),
            "Activation": exp["activation"],
            "Dropout": exp["dropout_rate"],
            "Weight_Decay": exp["weight_decay"],
            "Parameters": exp["trainable_parameters"],
            "Best_Val_Epoch": exp["best_val_epoch"],
            "Best_Val_Loss": exp["best_val_loss"],
            "Test_Accuracy": exp["metrics"]["accuracy"],
            "Test_Precision": exp["metrics"]["precision"],
            "Test_Recall": exp["metrics"]["recall"],
            "Test_F1": exp["metrics"]["f1_score"],
            "Test_ROC_AUC": exp["metrics"]["roc_auc"],
            "TN": cm["TN"],
            "FP": cm["FP"],
            "FN": cm["FN"],
            "TP": cm["TP"],
            "Train_Time_s": exp["training_time_sec"]
        }
        table_rows.append(row)

        # JSON record (excluding raw arrays)
        json_record = {
            "model_id": exp["model_id"],
            "name": exp["name"],
            "set": exp["set"],
            "hyperparameters": {
                "hidden_layers": exp["hidden_layers"],
                "activation": exp["activation"],
                "dropout_rate": exp["dropout_rate"],
                "learning_rate": exp["lr"],
                "weight_decay": exp["weight_decay"],
                "parameters": exp["trainable_parameters"]
            },
            "training": {
                "best_val_loss": exp["best_val_loss"],
                "best_val_epoch": exp["best_val_epoch"],
                "training_time_seconds": exp["training_time_sec"],
                "total_epochs_run": len(exp["history"]["epoch"])
            },
            "test_metrics": exp["metrics"]
        }
        json_summary_list.append(json_record)

    df_comparison = pd.DataFrame(table_rows)

    # 1. Save artifacts/comparison_table.csv
    csv_path = os.path.join(artifacts_dir, "comparison_table.csv")
    df_comparison.to_csv(csv_path, index=False)
    logger.info(f"Saved comparison table CSV to: {csv_path}")

    # Also sync artifacts/experiment_results.csv for app.py
    alt_csv = os.path.join(artifacts_dir, "experiment_results.csv")
    df_comparison.to_csv(alt_csv, index=False)

    # 2. Save artifacts/experiments_summary.json
    json_summary = {
        "best_activation_selected_from_set1": best_activation,
        "seed": 42,
        "total_experiments": len(all_experiments_data),
        "experiments": json_summary_list
    }
    json_path = os.path.join(artifacts_dir, "experiments_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=4)
    logger.info(f"Saved experiments summary JSON to: {json_path}")

    # Also sync artifacts/experiment_results.json
    alt_json = os.path.join(artifacts_dir, "experiment_results.json")
    with open(alt_json, "w") as f:
        json.dump(json_summary_list, f, indent=4)

    # =========================================================================
    # GENERATE HIGH-RESOLUTION COMPARISON PLOTS
    # =========================================================================
    plot_loss_curves(
        all_experiments_data,
        output_path=os.path.join(artifacts_dir, "loss_curves_comparison.png")
    )
    plot_combined_roc_curves(
        all_experiments_data,
        output_path=os.path.join(artifacts_dir, "roc_curves_comparison.png")
    )
    plot_side_by_side_confusion_matrices(
        all_experiments_data,
        output_path=os.path.join(artifacts_dir, "confusion_matrices_comparison.png")
    )

    # =========================================================================
    # PRINT FORMATTED COMPARISON TABLE TO TERMINAL
    # =========================================================================
    print("\n" + "=" * 115)
    print("                     DEEP LEARNING CIA 1 — EXPERIMENTAL BENCHMARK COMPARISON TABLE")
    print("=" * 115)
    print(df_comparison.to_string(index=False))
    print("=" * 115 + "\n")

    return df_comparison, json_summary


if __name__ == "__main__":
    run_controlled_experiments()
