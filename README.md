# Early Diabetes Risk Prediction | Deep Learning CIA 1

**Course:** Deep Learning (CSC403-8) — CIA 1: Component I A  
**Student Name:** Pragya  
**Academic Year:** 2026–2027  

An end-to-end, reproducible deep learning system for early diabetes risk classification on the Pima Indians Diabetes Dataset ($N=768$). Featuring leakage-free clinical preprocessing, modular PyTorch artificial neural network (ANN) architectures, controlled experimental benchmarks, an interactive Streamlit diagnostic laboratory web application, and a publication-quality LaTeX research report.

---

## 📁 Repository Structure

```text
Deep_Learning/
├── data/
│   └── diabetes.csv                     # Raw Pima Indians Diabetes Dataset (768 rows, 9 cols)
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── preprocess.py                    # Zero imputation, stratified 70/15/15 split, scaling, EDA plots
│   ├── model.py                         # Configurable DiabetesMLP (PyTorch nn.Module)
│   ├── train.py                         # Deterministic training loop, BCE loss, early stopping
│   └── experiments.py                   # Automated benchmarks (Set 1: Activations, Set 2: Architectures)
├── artifacts/                           # Saved weights, histories, scaler, and JSON summaries
│   ├── best_model.pt                    # Best model PyTorch checkpoint (833 params)
│   ├── comparison_table.csv             # Benchmark comparison table (Models A through E)
│   ├── experiments_summary.json         # Comprehensive experimental metadata
│   ├── dataset_summary.json             # Dataset split counts and invalid zero counts
│   ├── medians.json                     # Training split column medians (leakage-free)
│   ├── scaler.joblib                    # Fitted StandardScaler
│   ├── processed_data.npz               # Train/val/test feature & label arrays
│   ├── diabetes_tensors.pt              # PyTorch tensors dictionary
│   └── test_metrics.json                # Evaluated test metrics
├── figures/                             # Publication-quality figures (300 DPI) for report
│   ├── correlation_heatmap.png          # Feature correlation matrix
│   ├── loss_curves.png                  # Training and validation loss curves (all 5 runs)
│   ├── roc_curves.png                   # Overlaid ROC curves with AUC in legend
│   └── confusion_matrices.png           # Annotated confusion matrix of winning model
├── app.py                               # 4-Tab interactive Streamlit diagnostic laboratory app
├── report.tex                           # Complete, submission-ready LaTeX research report
├── requirements.txt                     # Project dependencies
├── pyproject.toml                       # Build & type checking configuration
├── pyrefly.toml                         # Pyrefly LSP configuration
├── .gitignore                           # Production Git ignore rules
└── README.md                            # Project documentation
```

---

## 🩺 Dataset & Preprocessing Pipeline

- **Cohort:** $N = 768$ female Pima Indian patients, 8 clinical features, 1 binary outcome (500 non-diabetic [65.10%], 268 diabetic [34.90%]).
- **Zero Imputation:** Biologically impossible zero values in `Glucose` (5), `BloodPressure` (35), `SkinThickness` (227), `Insulin` (374), and `BMI` (11) are converted to `NaN`. (Zero pregnancies is biologically valid and preserved).
- **Stratified Partition:** $70/15/15$ split ($N_{\text{train}}=537$, $N_{\text{val}}=115$, $N_{\text{test}}=116$) using seed 42.
- **Zero Data Leakage:** Column medians and standardisation parameters ($\mu, \sigma$) are computed strictly on the training partition and broadcast to validation and test splits.

---

## 🧠 Model Architectures & Controlled Experiments

Models were trained with deterministic seeds (42), Adam optimizer ($\eta = 0.001$), batch size 32, max 150 epochs, and early stopping ($K=15$ epochs on validation loss):

| Model ID | Architecture | Activation | Dropout | Weight Decay | Parameters | Best Val Epoch | Best Val Loss | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** | $[32, 16]$ | **ReLU** | $0.2$ | $10^{-4}$ | 833 | **9** | **0.5028** | **75.00%** | **70.00%** | **51.22%** | **0.5915** | 0.8459 |
| **Model B** | $[32, 16]$ | LeakyReLU | $0.2$ | $10^{-4}$ | 833 | 9 | 0.5033 | 75.00% | 70.00% | 51.22% | 0.5915 | 0.8455 |
| **Model C** | $[32, 16]$ | Tanh | $0.2$ | $10^{-4}$ | 833 | 48 | 0.5094 | 75.00% | 70.00% | 51.22% | 0.5915 | **0.8641** |
| **Model D** | $[16]$ | ReLU | $0.0$ | $0.0$ | 161 | 24 | 0.5090 | 74.14% | 68.97% | 48.78% | 0.5714 | 0.8602 |
| **Model E** | $[64, 32, 16]$ | **ReLU** | $0.3$ | $10^{-4}$ | 3,201 | **14** | **0.4992** | **75.00%** | **70.00%** | **51.22%** | **0.5915** | **0.8637** |

---

## 🚀 Quick Start Guide

### 1. Environment Setup

```powershell
# Clone repository
git clone https://github.com/<username>/Early-Diabetes-Risk-Prediction.git
cd Early-Diabetes-Risk-Prediction

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run Preprocessing, Training & Benchmarks

```powershell
# 1. Ingest raw CSV, impute zeros, split, scale, and save artifacts
python src/preprocess.py

# 2. Train baseline / best model with early stopping
python src/train.py

# 3. Run all controlled benchmark experiments (Models A through E)
python src/experiments.py
```

### 3. Launch Interactive Streamlit Diagnostic Web App

```powershell
streamlit run app.py
```
App features 4 comprehensive tabs:
1. **Dataset & Preprocessing:** Dataset overview metrics, zero-imputation statistics, interactive feature distributions, and correlation heatmap.
2. **ANN Math & Architecture:** Mathematical equations for artificial neurons, activations, backpropagation, and interactive layer shape tables.
3. **Experimental Comparison:** Benchmark table, interactive Plotly loss curves, ROC comparisons, and live interactive training sandbox.
4. **Live Risk Predictor:** Clinical input controls, training median imputation, feature z-score deviation bars, and real-time risk tier prediction.

---

## 📄 Compiling the LaTeX Report

The research report `report.tex` is fully formatted with `booktabs`, analytical derivations, and embedded figures in `figures/`.

Compile using `pdflatex`:
```powershell
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex   # Run twice to resolve references and labels
```
Or upload `report.tex` alongside the `figures/` directory directly to [Overleaf](https://www.overleaf.com/).
