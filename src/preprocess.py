"""
Preprocess module for Early Diabetes Risk Prediction (Pima Indians Diabetes Dataset).
Assignment: Deep Learning CIA 1

This module handles:
1. Loading the dataset and replacing physiologically impossible zeros in
   ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] with NaN.
2. Stratified 70/15/15 train/validation/test split using random_state=42.
3. Strict computation of column medians on the training split only and imputing
   missing values across train, val, and test splits to prevent data leakage.
4. Fitting StandardScaler strictly on training features, transforming all splits,
   and saving artifacts (medians.json, scaler.joblib, tensors/arrays).
5. Generating and saving EDA plots (class distribution & feature correlation heatmap).
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# Ensure project root and src are in sys.path
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
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
IMPOSSIBLE_ZERO_COLS: List[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]
FEATURE_COLS: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]
TARGET_COL: str = "Outcome"


def load_dataset(csv_path: str = "data/diabetes.csv") -> pd.DataFrame:
    """Load the Pima Indians Diabetes CSV dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset from '{csv_path}' with shape {df.shape}")
    return df


def replace_impossible_zeros(
    df: pd.DataFrame,
    cols: List[str] = IMPOSSIBLE_ZERO_COLS
) -> pd.DataFrame:
    """
    Replace physiologically impossible zero values with NaN.
    
    A living human cannot have zero glucose, blood pressure, skin fold thickness,
    serum insulin, or body mass index. In the Pima Indians dataset, 0 in these
    columns represents missing data.
    """
    df_clean = df.copy()
    zero_counts = {}
    for col in cols:
        if col in df_clean.columns:
            count = (df_clean[col] == 0).sum()
            zero_counts[col] = count
            df_clean[col] = df_clean[col].replace(0, np.nan)
    
    logger.info("Physiologically impossible zeros replaced with NaN:")
    for col, count in zero_counts.items():
        pct = (count / len(df_clean)) * 100
        logger.info(f"  - {col}: {count} zeros ({pct:.2f}% missing)")
    return df_clean


def perform_stratified_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified 70/15/15 train/val/test split.
    
    Step 1: Split into Train (70%) and Temp (30%), stratified by target.
    Step 2: Split Temp into Validation (50% of 30% = 15%) and Test (50% of 30% = 15%),
            stratified by target.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df[target_col],
        random_state=random_state
    )
    
    test_sub_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_sub_ratio,
        stratify=temp_df[target_col],
        random_state=random_state
    )
    
    logger.info("Stratified 70/15/15 split completed:")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pos_count = (split[target_col] == 1).sum()
        neg_count = (split[target_col] == 0).sum()
        pos_pct = (pos_count / len(split)) * 100
        logger.info(
            f"  - {name:>5} set: {len(split)} samples ({len(split)/len(df)*100:.2f}%) "
            f"| Class 0: {neg_count} | Class 1: {pos_count} ({pos_pct:.2f}% positive)"
        )
    return train_df, val_df, test_df


def compute_and_impute_medians(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Compute column medians strictly on the training split and impute missing
    values across train, val, and test splits to strictly prevent data leakage.
    """
    train_medians_series = train_df[feature_cols].median()
    train_medians: Dict[str, float] = train_medians_series.to_dict()
    
    logger.info("Training Split Column Medians (computed strictly on train set):")
    for feat, med in train_medians.items():
        logger.info(f"  - {feat}: {med:.2f}")
        
    train_imputed = train_df.copy()
    val_imputed = val_df.copy()
    test_imputed = test_df.copy()
    
    train_imputed[feature_cols] = train_imputed[feature_cols].fillna(train_medians_series)
    val_imputed[feature_cols] = val_imputed[feature_cols].fillna(train_medians_series)
    test_imputed[feature_cols] = test_imputed[feature_cols].fillna(train_medians_series)
    
    assert train_imputed[feature_cols].isna().sum().sum() == 0, "NaNs remain in train set after imputation"
    assert val_imputed[feature_cols].isna().sum().sum() == 0, "NaNs remain in val set after imputation"
    assert test_imputed[feature_cols].isna().sum().sum() == 0, "NaNs remain in test set after imputation"
    
    logger.info("Missing value imputation successfully applied to all splits with zero NaNs remaining.")
    return train_imputed, val_imputed, test_imputed, train_medians


def fit_and_scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler strictly on training features, then transform train, val,
    and test feature splits.
    """
    scaler = StandardScaler()
    X_train_scaled = np.asarray(scaler.fit_transform(train_df[feature_cols].values), dtype=np.float64)
    X_val_scaled = np.asarray(scaler.transform(val_df[feature_cols].values), dtype=np.float64)
    X_test_scaled = np.asarray(scaler.transform(test_df[feature_cols].values), dtype=np.float64)
    
    logger.info("StandardScaler fitted strictly on training features and applied to train/val/test.")
    logger.info(f"  - Train scaled feature mean (approx 0.0): {float(X_train_scaled.mean()):.4f}")
    logger.info(f"  - Train scaled feature std (approx 1.0):  {float(X_train_scaled.std()):.4f}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler  # type: ignore


def save_eda_plots(
    df_raw: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts_dir: str = "artifacts"
) -> Tuple[str, str]:
    """Generate and save class distribution and correlation heatmap plots."""
    os.makedirs(artifacts_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Class Distribution Plot
    class_dist_path = os.path.join(artifacts_dir, "class_distribution.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    class_counts = df_raw[TARGET_COL].value_counts().sort_index()
    labels = ["Non-Diabetic (0)", "Diabetic (1)"]
    colors = ["#2b5c8f", "#d95f02"]
    
    bars = axes[0].bar(labels, class_counts.values, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(
            f"{height} ({height/len(df_raw)*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold"
        )
    axes[0].set_title("Overall Class Distribution (Pima Indians Dataset)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Sample Count", fontsize=11)
    axes[0].set_ylim(0, max(class_counts.values) * 1.15)
    
    splits_summary = []
    for split_name, split in [("Train (70%)", train_df), ("Val (15%)", val_df), ("Test (15%)", test_df)]:
        c0 = (split[TARGET_COL] == 0).sum()
        c1 = (split[TARGET_COL] == 1).sum()
        total = len(split)
        splits_summary.append({
            "Split": split_name,
            "Non-Diabetic (0)": c0,
            "Diabetic (1)": c1,
            "Pct Positive": (c1 / total) * 100
        })
    df_splits_summary = pd.DataFrame(splits_summary)
    
    x = np.arange(len(df_splits_summary))
    w = 0.35
    b1 = axes[1].bar(x - w/2, df_splits_summary["Non-Diabetic (0)"], width=w, label="Non-Diabetic (0)", color="#2b5c8f", edgecolor="black", alpha=0.85)
    b2 = axes[1].bar(x + w/2, df_splits_summary["Diabetic (1)"], width=w, label="Diabetic (1)", color="#d95f02", edgecolor="black", alpha=0.85)
    
    for bar in list(b1) + list(b2):
        height = bar.get_height()
        axes[1].annotate(
            f"{height}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9
        )
        
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_splits_summary["Split"], fontweight="bold")
    axes[1].set_title("Stratified Splits Distribution (70/15/15)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Sample Count", fontsize=11)
    axes[1].legend(frameon=True)
    axes[1].set_ylim(0, max(df_splits_summary["Non-Diabetic (0)"]) * 1.15)
    
    plt.tight_layout()
    plt.savefig(class_dist_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved class distribution plot to: {class_dist_path}")
    
    # 2. Correlation Heatmap
    heatmap_path = os.path.join(artifacts_dir, "correlation_heatmap.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = df_raw.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.4,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax
    )
    ax.set_title("Pima Indians Diabetes Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved correlation heatmap to: {heatmap_path}")
    
    return class_dist_path, heatmap_path


def save_artifacts(
    artifacts_dir: str,
    train_medians: Dict[str, float],
    scaler: StandardScaler,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str]
) -> Dict[str, str]:
    """Save all preprocessed arrays, scaler, medians, and tensors into artifacts/."""
    os.makedirs(artifacts_dir, exist_ok=True)
    saved_files = {}
    
    # 1. medians.json
    medians_path = os.path.join(artifacts_dir, "medians.json")
    with open(medians_path, "w") as f:
        json.dump(train_medians, f, indent=4)
    saved_files["medians"] = medians_path
    
    # 2. scaler.joblib
    scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    saved_files["scaler"] = scaler_path
    
    # 3. Individual numpy arrays
    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }
    for name, arr in splits.items():
        arr_path = os.path.join(artifacts_dir, f"{name}.npy")
        np.save(arr_path, arr)
        saved_files[name] = arr_path
        
    # 4. Consolidated npz file
    npz_path = os.path.join(artifacts_dir, "processed_data.npz")
    np.savez_compressed(
        npz_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=np.array(feature_names)
    )
    saved_files["processed_npz"] = npz_path
    
    # 5. PyTorch tensors
    if HAS_TORCH:
        tensors_dict = {
            "X_train": torch.tensor(X_train, dtype=torch.float32),
            "y_train": torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
            "X_val": torch.tensor(X_val, dtype=torch.float32),
            "y_val": torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
            "X_test": torch.tensor(X_test, dtype=torch.float32),
            "y_test": torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
            "feature_names": feature_names
        }
        pt_path = os.path.join(artifacts_dir, "diabetes_tensors.pt")
        torch.save(tensors_dict, pt_path)
        saved_files["tensors_pt"] = pt_path
        
    return saved_files


def run_pipeline(
    data_path: str = "data/diabetes.csv",
    artifacts_dir: str = "artifacts"
) -> Dict[str, str]:
    """Execute the complete end-to-end preprocessing pipeline."""
    logger.info("=== STARTING PREPROCESSING PIPELINE ===")
    df_raw = load_dataset(data_path)
    df_clean = replace_impossible_zeros(df_raw, cols=IMPOSSIBLE_ZERO_COLS)
    train_df, val_df, test_df = perform_stratified_split(
        df_clean, target_col=TARGET_COL,
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
    )
    train_imputed, val_imputed, test_imputed, train_medians = compute_and_impute_medians(
        train_df, val_df, test_df, feature_cols=FEATURE_COLS
    )
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = fit_and_scale_features(
        train_imputed, val_imputed, test_imputed, feature_cols=FEATURE_COLS
    )
    y_train = np.asarray(train_imputed[TARGET_COL].values, dtype=np.float32)
    y_val = np.asarray(val_imputed[TARGET_COL].values, dtype=np.float32)
    y_test = np.asarray(test_imputed[TARGET_COL].values, dtype=np.float32)
    
    saved_artifacts = save_artifacts(
        artifacts_dir=artifacts_dir,
        train_medians=train_medians,
        scaler=scaler,
        X_train=X_train_scaled, y_train=y_train,
        X_val=X_val_scaled, y_val=y_val,
        X_test=X_test_scaled, y_test=y_test,
        feature_names=FEATURE_COLS
    )
    class_dist_path, heatmap_path = save_eda_plots(
        df_raw=df_raw, train_df=train_df, val_df=val_df, test_df=test_df,
        artifacts_dir=artifacts_dir
    )
    saved_artifacts["class_distribution_plot"] = class_dist_path
    saved_artifacts["correlation_heatmap"] = heatmap_path
    
    logger.info("=== PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY ===")
    return saved_artifacts


if __name__ == "__main__":
    run_pipeline()
