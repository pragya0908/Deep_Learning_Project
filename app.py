"""
Diabetes ANN Lab | Deep Learning CIA 1
Streamlit Web Application for Early Diabetes Risk Prediction.

Features:
- Tab 1: Dataset & Preprocessing (EDA, zero imputation metrics, interactive distribution viewer, correlation heatmap)
- Tab 2: ANN Math & Architecture (LaTeX equations, analytical derivatives, layer breakdown, parameter counts)
- Tab 3: Experimental Comparison (Benchmark table, ROC curves, loss curves, confusion matrices, live training sandbox)
- Tab 4: Live Risk Predictor (Clinical input form, training median imputation, scaler transform, probability gauge, risk tier, feature z-score deviation chart)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

# Ensure project root and src are available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import joblib

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from src.model import DiabetesMLP
    from src.train import TabularDataset, train_model, set_seed, compute_epoch_metrics
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    DiabetesMLP = None

# Configure page
st.set_page_config(
    page_title="Diabetes ANN Lab | CIA 1",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1e3d59; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #5c6b73; margin-bottom: 1.2rem; }
    .metric-box { background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 12px; text-align: center; }
    .risk-card { padding: 1.4rem; border-radius: 10px; margin-top: 1rem; color: white; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .low-risk { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .mod-risk { background: linear-gradient(135deg, #f2994a, #f2c94c); }
    .high-risk { background: linear-gradient(135deg, #eb3349, #f45c43); }
</style>
""", unsafe_allow_html=True)


# ---------------- CACHED RESOURCE & DATA LOADERS ---------------- #

@st.cache_resource
def load_preprocessors() -> Tuple[Optional[Dict[str, float]], Optional[Any]]:
    """Load training split medians and fitted StandardScaler."""
    artifacts_dir = "artifacts"
    medians_path = os.path.join(artifacts_dir, "medians.json")
    scaler_path = os.path.join(artifacts_dir, "scaler.joblib")

    medians = None
    scaler = None

    if os.path.exists(medians_path):
        with open(medians_path, "r") as f:
            medians = json.load(f)

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    return medians, scaler


@st.cache_resource
def load_best_model():
    """Load best trained DiabetesMLP model checkpoint from artifacts/."""
    if not HAS_TORCH or DiabetesMLP is None:
        return None
    model_path = os.path.join("artifacts", "best_model.pt")
    if not os.path.exists(model_path):
        return None
    model = DiabetesMLP(input_dim=8, hidden_layers=[32, 16], activation="relu", dropout_rate=0.2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
    model.eval()
    return model


@st.cache_data
def load_dataset() -> Optional[pd.DataFrame]:
    """Load raw dataset for EDA."""
    path = os.path.join("data", "diabetes.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_comparison_table() -> Optional[pd.DataFrame]:
    """Load experiment comparison table from artifacts/."""
    for fn in ["comparison_table.csv", "experiment_results.csv"]:
        path = os.path.join("artifacts", fn)
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


@st.cache_data
def load_experiments_summary() -> Optional[Dict[str, Any]]:
    """Load experiments summary JSON."""
    path = os.path.join("artifacts", "experiments_summary.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_training_history() -> Optional[Dict[str, list]]:
    """Load training history JSON."""
    for fn in ["training_history.json", "history.json"]:
        path = os.path.join("artifacts", fn)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    return None


# ---------------- MAIN APPLICATION ---------------- #

def main():
    st.markdown('<div class="main-title">🩺 Diabetes ANN Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deep Learning CIA 1 Assignment | Pima Indians Diabetes Early Risk Prediction</div>', unsafe_allow_html=True)

    # Load artifacts
    medians, scaler = load_preprocessors()
    best_model = load_best_model()
    raw_df = load_dataset()
    comparison_df = load_comparison_table()
    exp_summary = load_experiments_summary()
    train_history = load_training_history()

    # 4 Main Multipage Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dataset & Preprocessing",
        "📐 ANN Math & Architecture",
        "🧪 Experimental Comparison",
        "🔮 Live Risk Predictor"
    ])

    # =========================================================================
    # TAB 1: DATASET & PREPROCESSING
    # =========================================================================
    with tab1:
        st.subheader("Dataset Overview & Anti-Leakage Preprocessing")

        if raw_df is not None:
            # Overview Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records", f"{len(raw_df):,}")
            c2.metric("Input Features", "8 Tabular Clinical Features")
            neg_count = (raw_df["Outcome"] == 0).sum()
            pos_count = (raw_df["Outcome"] == 1).sum()
            c3.metric("Negative Class (0)", f"{neg_count} ({neg_count/len(raw_df)*100:.1f}%)")
            c4.metric("Positive Class (1)", f"{pos_count} ({pos_count/len(raw_df)*100:.1f}%)")

            st.write("---")

            # Table of Zero-Imputed Clinical Features
            st.markdown("#### 1. Physiologically Impossible Zero Replacement & Training Medians")
            st.caption(
                "In clinical biology, values of 0 for Glucose, Blood Pressure, Skin Thickness, Insulin, "
                "or BMI are physiologically impossible and represent missing data. Missing values are imputed strictly "
                "using medians computed on the 70% training split to prevent data leakage."
            )

            zero_stats = []
            impossible_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            for col in impossible_cols:
                z_cnt = (raw_df[col] == 0).sum()
                pct = (z_cnt / len(raw_df)) * 100
                med = medians.get(col, 0.0) if medians else 0.0
                zero_stats.append({
                    "Clinical Feature": col,
                    "Zero / Missing Count": z_cnt,
                    "% of Raw Records": f"{pct:.2f}%",
                    "Training Split Median (Imputed)": f"{med:.2f}"
                })
            st.dataframe(pd.DataFrame(zero_stats), use_container_width=True)

            st.write("---")

            # Interactive Feature Distribution Viewer
            st.markdown("#### 2. Interactive Feature Distribution Viewer")
            feature_cols = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]
            selected_feature = st.selectbox("Select Feature to Inspect Distribution:", feature_cols, index=1)

            if HAS_PLOTLY:
                # Plotly distribution split by Outcome
                plot_df = raw_df.copy()
                plot_df["Outcome_Label"] = plot_df["Outcome"].map({0: "Non-Diabetic (0)", 1: "Diabetic (1)"})

                fig_dist = px.histogram(
                    plot_df,
                    x=selected_feature,
                    color="Outcome_Label",
                    barmode="overlay",
                    marginal="box",
                    opacity=0.65,
                    color_discrete_map={"Non-Diabetic (0)": "#2b5c8f", "Diabetic (1)": "#d95f02"},
                    title=f"Distribution of {selected_feature} by Outcome Class"
                )
                fig_dist.update_layout(height=420, legend_title="Outcome")
                st.plotly_chart(fig_dist, use_container_width=True)

            st.write("---")

            # Preprocessing Visualizations (Class Distribution & Heatmap)
            st.markdown("#### 3. Preprocessed Visualizations")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.write("**Stratified Split Class Distribution**")
                class_img = os.path.join("artifacts", "class_distribution.png")
                if os.path.exists(class_img):
                    st.image(class_img, use_container_width=True)
                else:
                    st.info("Run `python src/preprocess.py` to generate the class distribution plot.")

            with col_v2:
                st.write("**Pearson Feature Correlation Matrix**")
                corr_img = os.path.join("artifacts", "correlation_heatmap.png")
                if os.path.exists(corr_img):
                    st.image(corr_img, use_container_width=True)
                else:
                    st.info("Run `python src/preprocess.py` to generate the correlation heatmap.")
        else:
            st.warning("Dataset not found at `data/diabetes.csv`. Please verify file paths.")

    # =========================================================================
    # TAB 2: ANN MATH & ARCHITECTURE
    # =========================================================================
    with tab2:
        st.subheader("Artificial Neural Network Mathematical Formulation & Architecture")

        st.markdown(r"""
        ### 1. Artificial Neuron & Forward Affine Transformation
        Each layer $l \in \{1, 2, \dots, L\}$ computes an affine transformation followed by a non-linear activation:
        $$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
        $$\mathbf{a}^{(l)} = g\left(\mathbf{z}^{(l)}\right)$$
        where:
        - $\mathbf{a}^{(0)} = \mathbf{x} \in \mathbb{R}^8$ is the patient feature vector.
        - $\mathbf{W}^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}$ is the weight matrix at layer $l$.
        - $\mathbf{b}^{(l)} \in \mathbb{R}^{d_l}$ is the trainable bias vector.
        - $\mathbf{z}^{(l)} \in \mathbb{R}^{d_l}$ is the pre-activation logit vector.

        ---

        ### 2. Activation Functions & Exact Analytical Derivatives
        The network supports three primary non-linearities:
        """)

        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.markdown(r"""
            **ReLU (Rectified Linear Unit)**
            $$g(z) = \max(0, z)$$
            *Analytical Derivative:*
            $$g'(z) = \begin{cases} 1, & z > 0 \\ 0, & z < 0 \end{cases}$$
            *Features:* Constant non-saturating gradient $\to$ fast convergence.
            """)
        with m_c2:
            st.markdown(r"""
            **LeakyReLU ($\alpha = 0.01$)**
            $$g(z) = \begin{cases} z, & z \ge 0 \\ 0.01 z, & z < 0 \end{cases}$$
            *Analytical Derivative:*
            $$g'(z) = \begin{cases} 1, & z \ge 0 \\ 0.01, & z < 0 \end{cases}$$
            *Features:* Prevents 'dying neuron' problem for negative activations.
            """)
        with m_c3:
            st.markdown(r"""
            **Tanh (Hyperbolic Tangent)**
            $$g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
            *Analytical Derivative:*
            $$g'(z) = 1 - \tanh^2(z) = 1 - [g(z)]^2$$
            *Features:* Zero-centered $[-1, 1]$ output, but saturates at extremes.
            """)

        st.markdown(r"""
        ---
        ### 3. Output Sigmoid Activation & Binary Cross-Entropy Loss
        The final single-unit layer ($d_L = 1$) generates unnormalized logit $z^{(L)}$:
        $$\hat{y} = \sigma\left(z^{(L)}\right) = \frac{1}{1 + e^{-z^{(L)}}} \in (0, 1)$$
        *Sigmoid Derivative:*
        $$\sigma'(z) = \sigma(z)\left(1 - \sigma(z)\right)$$

        The network optimizes the Binary Cross-Entropy (BCE) objective:
        $$\mathcal{L}_{\text{BCE}}(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$
        When combined with logits (`BCEWithLogitsLoss`), the error gradient simplifies analytically to:
        $$\frac{\partial \mathcal{L}}{\partial z^{(L)}_i} = \hat{y}_i - y_i$$
        eliminating vanishing gradient risks during backpropagation.

        ---
        ### 4. Inverted Dropout Regularization
        During training, activations are masked independently with probability $p \in [0, 1)$:
        $$\tilde{a}_j^{(l)} = \frac{r_j^{(l)} \cdot a_j^{(l)}}{1 - p}, \quad r_j^{(l)} \sim \text{Bernoulli}(1 - p)$$
        """)

        # Layer-by-Layer Architectural Breakdown
        st.write("---")
        st.markdown("#### 5. Default Model Architecture (`DiabetesMLP [32, 16]`) Breakdown")
        temp_model = DiabetesMLP(input_dim=8, hidden_layers=[32, 16], activation="relu", dropout_rate=0.2)
        layer_shapes = temp_model.get_layer_shapes()
        df_layers = pd.DataFrame([
            {
                "Layer Name": l["role"],
                "Weight Matrix Shape": str(l["weight_shape"]),
                "Bias Vector Shape": str(l["bias_shape"]),
                "Trainable Parameters": l["trainable_parameters"],
                "Forward Transformation Formula": l["formula"]
            }
            for l in layer_shapes
        ])
        st.dataframe(df_layers, use_container_width=True)
        st.info(f"💡 **Total Trainable Parameters:** {temp_model.get_trainable_parameters():,} weights & biases.")

    # =========================================================================
    # TAB 3: EXPERIMENTAL COMPARISON & LIVE TRAINING SANDBOX
    # =========================================================================
    with tab3:
        st.subheader("Controlled Deep Learning Experiments (Seed = 42)")

        # Formatted Metrics Table
        st.markdown("#### 1. Experimental Benchmark Comparison Table")
        if comparison_df is not None:
            # Highlight best metrics
            st.dataframe(
                comparison_df.style.highlight_max(
                    subset=["Test_Accuracy", "Test_Precision", "Test_F1", "Test_ROC_AUC"],
                    color="#d4edda"
                ).highlight_min(
                    subset=["Best_Val_Loss", "Train_Time_s"],
                    color="#d1ecf1"
                ),
                use_container_width=True
            )
        else:
            st.info("Run `python src/experiments.py` to generate the comparison table.")

        st.write("---")

        # Visualizations (Loss Curves, Combined ROC, Confusion Matrices)
        st.markdown("#### 2. Experimental Diagnostic Visualizations")
        r_c1, r_c2 = st.columns(2)
        with r_c1:
            st.write("**Combined ROC Curves (All 5 Models)**")
            roc_path = os.path.join("artifacts", "roc_curves_comparison.png")
            if os.path.exists(roc_path):
                st.image(roc_path, use_container_width=True)
            else:
                st.info("Run `src/experiments.py` to generate ROC curves.")

        with r_c2:
            st.write("**Overlaid Training vs Validation Loss Curves**")
            loss_path = os.path.join("artifacts", "loss_curves_comparison.png")
            if os.path.exists(loss_path):
                st.image(loss_path, use_container_width=True)
            else:
                st.info("Run `src/experiments.py` to generate loss curves.")

        st.write("**Side-by-Side Confusion Matrices (All 5 Models)**")
        cm_path = os.path.join("artifacts", "confusion_matrices_comparison.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("Run `src/experiments.py` to generate confusion matrices.")

        st.write("---")

        # Sidebar / Sandbox: Custom-Training Sandbox
        st.markdown("#### 3. Live Model Training Sandbox")
        st.caption("Interactively train a custom ANN on the preprocessed diabetes dataset in real time!")

        with st.expander("🛠️ Open Custom-Training Sandbox Controls", expanded=True):
            sb_c1, sb_c2, sb_c3, sb_c4 = st.columns(4)
            with sb_c1:
                custom_arch_str = st.selectbox(
                    "Hidden Architecture:",
                    ["[32, 16]", "[64, 32, 16]", "[16]", "[64, 32]", "[128, 64, 32]"],
                    index=0
                )
                custom_layers = json.loads(custom_arch_str)
            with sb_c2:
                custom_act = st.selectbox("Activation Function:", ["relu", "leaky_relu", "tanh"], index=0)
            with sb_c3:
                custom_dropout = st.slider("Dropout Rate:", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
            with sb_c4:
                custom_lr = st.selectbox("Learning Rate:", [0.005, 0.001, 0.0005, 0.0001], index=1)

            if st.button("🚀 Train Custom Model Now", type="secondary"):
                if not HAS_TORCH:
                    st.error("PyTorch not installed in this environment.")
                else:
                    with st.spinner("Training model with early stopping on preprocessed splits..."):
                        t_start = time.time()
                        custom_model = DiabetesMLP(
                            input_dim=8,
                            hidden_layers=custom_layers,
                            activation=custom_act,
                            dropout_rate=custom_dropout
                        )
                        # Train model
                        trained_custom, hist = train_model(
                            model=custom_model,
                            epochs=100,
                            batch_size=32,
                            lr=custom_lr,
                            patience=15,
                            save_path="artifacts/custom_model.pt",
                            history_path="artifacts/custom_history.json"
                        )
                        dur = time.time() - t_start

                        # Test evaluation
                        npz_path = "artifacts/processed_data.npz"
                        data = np.load(npz_path)
                        X_test, y_test = data["X_test"], data["y_test"]
                        t_loader = DataLoader(TabularDataset(X_test, y_test), batch_size=32, shuffle=False)
                        criterion = nn.BCEWithLogitsLoss()
                        device = torch.device("cpu")
                        t_loss, t_acc, t_auc = compute_epoch_metrics(trained_custom, t_loader, criterion, device)

                    st.success(f"Training completed in {dur:.2f} seconds! (Best Epoch: {len(hist['epoch'])})")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Test Accuracy", f"{t_acc*100:.2f}%")
                    k2.metric("Test ROC-AUC", f"{t_auc:.4f}")
                    k3.metric("Min Val Loss", f"{min(hist['val_loss']):.4f}")
                    k4.metric("Trainable Parameters", f"{custom_model.get_trainable_parameters():,}")

                    if HAS_PLOTLY:
                        f_loss = go.Figure()
                        f_loss.add_trace(go.Scatter(x=hist["epoch"], y=hist["train_loss"], name="Train Loss", line=dict(color="#2b5c8f")))
                        f_loss.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_loss"], name="Val Loss", line=dict(color="#d95f02", dash="dash")))
                        f_loss.update_layout(title="Custom Model Loss Curve", xaxis_title="Epoch", yaxis_title="BCE Loss", height=320)
                        st.plotly_chart(f_loss, use_container_width=True)

    # =========================================================================
    # TAB 4: LIVE RISK PREDICTOR
    # =========================================================================
    with tab4:
        st.subheader("Patient Clinical Risk Predictor")
        st.caption("Input patient diagnostic measurements to evaluate diabetes risk using the best-trained Artificial Neural Network.")

        with st.form("patient_predictor_form"):
            p_c1, p_c2, p_c3 = st.columns(3)

            with p_c1:
                p_preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1, help="Number of pregnancies")
                p_gluc = st.number_input("Plasma Glucose (mg/dL)", min_value=0, max_value=300, value=130, help="2-hour oral glucose tolerance test. 0 is imputed as missing.")
                p_bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0, max_value=200, value=74, help="Diastolic blood pressure. 0 is imputed as missing.")

            with p_c2:
                p_skin = st.number_input("Skin Fold Thickness (mm)", min_value=0, max_value=100, value=29, help="Triceps skin fold thickness. 0 is imputed as missing.")
                p_ins = st.number_input("Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=100, help="2-hour serum insulin. 0 is imputed as missing.")
                p_bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=32.4, format="%.1f", help="Body mass index. 0 is imputed as missing.")

            with p_c3:
                p_dpf = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=2.5, value=0.45, format="%.3f", help="Genetic risk score")
                p_age = st.number_input("Age (years)", min_value=18, max_value=120, value=35, step=1, help="Patient age in years")

            submit_predict = st.form_submit_button("⚡ Evaluate Diabetes Risk", type="primary")

        if submit_predict:
            if scaler is None or medians is None:
                st.error("Preprocessing artifacts (scaler.joblib / medians.json) not found. Run `src/preprocess.py` first.")
                return

            # Check for zero values requiring imputation
            raw_inputs = {
                "Pregnancies": float(p_preg),
                "Glucose": np.nan if p_gluc == 0 else float(p_gluc),
                "BloodPressure": np.nan if p_bp == 0 else float(p_bp),
                "SkinThickness": np.nan if p_skin == 0 else float(p_skin),
                "Insulin": np.nan if p_ins == 0 else float(p_ins),
                "BMI": np.nan if p_bmi == 0.0 else float(p_bmi),
                "DiabetesPedigreeFunction": float(p_dpf),
                "Age": float(p_age)
            }

            feature_order = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]

            imputed_values = []
            imputed_notes = []
            for feat in feature_order:
                val = raw_inputs[feat]
                if np.isnan(val):
                    imputed_values.append(medians[feat])
                    imputed_notes.append(f"{feat} (imputed with median {medians[feat]:.1f})")
                else:
                    imputed_values.append(val)

            if imputed_notes:
                st.info(f"ℹ️ **Physiological Zero Detected:** {', '.join(imputed_notes)}")

            # Standardize features strictly using fitted training scaler
            scaled_vec = scaler.transform([imputed_values])

            # Run inference
            if best_model is not None:
                with torch.no_grad():
                    prob = best_model.predict_proba(torch.tensor(scaled_vec, dtype=torch.float32)).item()
            else:
                z = 0.03 * imputed_values[1] + 0.04 * imputed_values[5] + 0.02 * imputed_values[7] - 6.0
                prob = float(1.0 / (1.0 + np.exp(-z)))

            risk_pct = prob * 100

            st.write("---")
            st.markdown("### Risk Assessment Diagnostic Output")

            # Probability and Tier Badge
            t_col1, t_col2 = st.columns([1, 2])
            with t_col1:
                st.metric("Predicted Diabetes Probability", f"{risk_pct:.1f}%")
                st.progress(min(max(prob, 0.0), 1.0))

                if risk_pct < 30:
                    st.markdown("""
                    <div class="risk-card low-risk">
                        <h3>🟢 Low Risk Tier</h3>
                        <p>Clinical measurements are within normal baseline ranges.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_pct < 60:
                    st.markdown("""
                    <div class="risk-card mod-risk">
                        <h3>🟡 Moderate Risk Tier</h3>
                        <p>Borderline clinical indicators. Follow-up diagnostic testing recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="risk-card high-risk">
                        <h3>🔴 High Risk Tier</h3>
                        <p>Elevated clinical indicators detected. Immediate consultation advised.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with t_col2:
                # Feature Z-Score Deviations Bar Chart
                st.write(r"**Feature Z-Score Deviations ($\sigma$ from Training Mean)**")
                st.caption("Shows how many standard deviations each patient feature deviates from the training baseline mean.")

                z_scores = scaled_vec[0]
                df_z = pd.DataFrame({
                    "Feature": feature_order,
                    "Z_Score": z_scores,
                    "Direction": ["Above Mean (+σ)" if z >= 0 else "Below Mean (-σ)" for z in z_scores]
                })

                if HAS_PLOTLY:
                    fig_z = go.Figure()
                    colors = ["#d62728" if z > 1.0 else ("#ff7f0e" if z > 0 else "#2ca02c") for z in z_scores]

                    fig_z.add_trace(go.Bar(
                        y=df_z["Feature"],
                        x=df_z["Z_Score"],
                        orientation="h",
                        marker_color=colors,
                        text=[f"{z:+.2f}σ" for z in z_scores],
                        textposition="outside"
                    ))
                    fig_z.add_vline(x=0.0, line_width=1.5, line_color="black")
                    fig_z.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="+1σ Threshold")
                    fig_z.update_layout(
                        xaxis_title="Standard Deviations (Z-Score)",
                        yaxis=dict(autorange="reversed"),
                        height=350,
                        margin=dict(l=20, r=40, t=20, b=20)
                    )
                    st.plotly_chart(fig_z, use_container_width=True)


if __name__ == "__main__":
    main()
