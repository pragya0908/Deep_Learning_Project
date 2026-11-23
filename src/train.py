"""
Training module for Early Diabetes Risk Prediction.
Assignment: Deep Learning CIA 1

Features:
- Deterministic random seed configuration (torch, numpy, random = 42)
- Tabular Dataset & DataLoader construction (batch_size=32)
- train_model function using:
  - Adam optimizer
  - BCEWithLogitsLoss
  - Early stopping on validation loss (patience=15)
  - Up to 150 training epochs
  - Per-epoch logging: train_loss, val_loss, train_acc, and val_acc
  - Checkpoint saving (.pt) and history logging (.json) into artifacts/
  - Returns (trained_model, history)
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# Ensure project root and src are available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as err:
    raise ImportError(
        "\n[PyTorch Import Error] 'torch' was not found in the current Python environment.\n"
        "To resolve this in VS Code:\n"
        "  1. Press Ctrl+Shift+P -> 'Python: Select Interpreter'\n"
        "  2. Choose '.venv' (Python 3.11.9) located at d:\\Deep_Learning\\.venv\\Scripts\\python.exe\n"
    ) from err

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.model import DiabetesMLP

# Backward compatibility alias
EarlyDiabetesMLP = DiabetesMLP

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set deterministic seeds across torch, numpy, and python random for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Deterministic random seed set to {seed}")


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular medical features and binary outcome labels.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)
        self.y = y_tensor

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def compute_epoch_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Compute average loss, accuracy, and ROC-AUC over a DataLoader.
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy().flatten())

    avg_loss = total_loss / len(data_loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5

    return float(avg_loss), float(acc), float(auc)


def train_model(
    model: Optional[nn.Module] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    data_path: str = "artifacts/processed_data.npz",
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 15,
    save_path: str = "artifacts/best_model.pt",
    history_path: str = "artifacts/training_history.json",
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train DiabetesMLP with Adam optimizer, BCEWithLogitsLoss, and early stopping.

    Parameters:
    -----------
    model : nn.Module, optional
        Instantiated PyTorch model. If None, default DiabetesMLP(input_dim=8, hidden_layers=[32, 16]) is created.
    train_loader : DataLoader, optional
        DataLoader for training split. If None, loaded from `data_path`.
    val_loader : DataLoader, optional
        DataLoader for validation split. If None, loaded from `data_path`.
    data_path : str, default='artifacts/processed_data.npz'
        Path to preprocessed data archive if loaders are not supplied.
    epochs : int, default=150
        Maximum number of training epochs.
    batch_size : int, default=32
        Batch size for training.
    lr : float, default=0.001
        Learning rate for Adam optimizer.
    weight_decay : float, default=1e-4
        L2 regularization weight decay for Adam optimizer.
    patience : int, default=15
        Early stopping patience on validation loss.
    save_path : str, default='artifacts/best_model.pt'
        Filepath to save the best model weights checkpoint.
    history_path : str, default='artifacts/training_history.json'
        Filepath to save epoch-by-epoch training history.
    device : torch.device, optional
        Device to run training on (CPU or CUDA).

    Returns:
    --------
    Tuple[nn.Module, Dict[str, list]]
        Best trained model (weights restored) and history dictionary containing
        epoch-by-epoch train_loss, val_loss, train_acc, and val_acc.
    """
    set_seed(42)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If loaders not passed, load from artifacts
    if train_loader is None or val_loader is None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Processed dataset archive not found at '{data_path}'. Run src/preprocess.py first.")
        data = np.load(data_path)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]

        train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        input_dim = X_train.shape[1]
    else:
        sample_x, _ = next(iter(train_loader))
        input_dim = sample_x.shape[1]

    # Instantiate default model if none supplied
    if model is None:
        model = DiabetesMLP(
            input_dim=input_dim,
            hidden_layers=[32, 16],
            activation="relu",
            dropout_rate=0.2
        )

    model = model.to(device)

    # Loss function and Adam optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Tracking & early stopping state
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_roc_auc": []
    }

    logger.info(f"Starting training on {device} for up to {epochs} epochs (Early Stopping Patience={patience})")
    logger.info(f"Model Architecture:\n{getattr(model, 'summary', lambda: str(model))()}")

    for epoch in range(1, epochs + 1):
        # ---------------- Training Phase ---------------- #
        model.train()
        running_train_loss = 0.0
        train_targets_epoch = []
        train_preds_epoch = []

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * len(targets)
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

            train_preds_epoch.extend(preds)
            train_targets_epoch.extend(targets.cpu().numpy().flatten())

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(train_targets_epoch, train_preds_epoch)

        # ---------------- Validation Phase ---------------- #
        epoch_val_loss, epoch_val_acc, epoch_val_auc = compute_epoch_metrics(
            model, val_loader, criterion, device
        )

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        history["val_roc_auc"].append(round(epoch_val_auc, 4))

        # Log progress per epoch
        logger.info(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | "
            f"Val AUC: {epoch_val_auc:.4f}"
        )

        # Early stopping check on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"  --> Best model checkpoint saved to '{save_path}' (Val Loss: {epoch_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} (Best epoch was {best_epoch} with Val Loss: {best_val_loss:.4f})")
                break

    # Save training history JSON
    hist_dir = os.path.dirname(history_path)
    if hist_dir:
        os.makedirs(hist_dir, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Saved complete training history JSON to '{history_path}'")

    # Also save copy as artifacts/history.json for convenience
    alt_hist_path = os.path.join(hist_dir or "artifacts", "history.json")
    with open(alt_hist_path, "w") as f:
        json.dump(history, f, indent=4)

    # Load best model weights before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        logger.info(f"Restored best weights from '{save_path}'")

    return model, history


def run_training_pipeline():
    """
    Full pipeline: trains DiabetesMLP and evaluates on the holdout test set.
    """
    logger.info("=== STARTING MODEL TRAINING WORKFLOW ===")
    trained_model, history = train_model()

    # Evaluate on holdout test set
    data_path = "artifacts/processed_data.npz"
    data = np.load(data_path)
    X_test, y_test = data["X_test"], data["y_test"]
    test_loader = DataLoader(TabularDataset(X_test, y_test), batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()

    trained_model.eval()
    test_loss, test_acc, test_auc = compute_epoch_metrics(trained_model, test_loader, criterion, device)

    # Additional test metrics
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for feat, _ in test_loader:
            probs = trained_model.predict_proba(feat.to(device)).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))

    precision = float(precision_score(y_test, all_preds, zero_division=0))
    recall = float(recall_score(y_test, all_preds, zero_division=0))
    f1 = float(f1_score(y_test, all_preds, zero_division=0))

    test_metrics = {
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "test_precision": round(precision, 4),
        "test_recall": round(recall, 4),
        "test_f1": round(f1, 4),
        "test_roc_auc": round(test_auc, 4)
    }

    metrics_path = "artifacts/test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=4)

    logger.info("=== FINAL HOLDOUT TEST SET EVALUATION ===")
    for metric_name, val in test_metrics.items():
        logger.info(f"  {metric_name.upper():<16}: {val:.4f}")
    logger.info("=== TRAINING PIPELINE COMPLETED SUCCESSFULLY ===")

    return trained_model, history, test_metrics


if __name__ == "__main__":
    run_training_pipeline()
