"""
app.py: Streamlit Dashboard for Network Intrusion Detection (CICIDS2017).

Features:
- Cached artifact and model loading.
- Sidebar model selector (Stacked Ensemble [default] / XGBoost).
- Demo files selector or custom CSV upload.
- Academic disclaimer and documentation about box.
- Flow statistics metric cards (analysed, attack %, skipped, latency, throughput).
- Predicted class distribution and Normal vs Attack visual cards.
- Interactive predictions table with class filter and confidence threshold slider.
- Full prediction CSV download button.
- Ground-truth evaluation (Accuracy, Macro F1, Confusion Matrix, Per-class metrics)
  when 'Label' is present in the dataset.
- Clean, user-friendly error messages without raw tracebacks.
"""
from sklearn.metrics import confusion_matrix
from pathlib import Path
import sys
import io
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Setup paths
APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"
MODELS_DIR = APP_DIR / "models"
DEMO_DIR = APP_DIR / "demo"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from predict_utils import evaluate, load_artifacts, predict, prepare_features

# Set page configuration
st.set_page_config(
    page_title="Network Intrusion Detection System (CICIDS2017)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading machine learning models and preprocessors...")
def get_cached_artifacts():
    """Load and cache artifacts once across dashboard user sessions."""
    return load_artifacts(MODELS_DIR)


def main():
    # -------------------------------------------------------------
    # 1. LOAD CACHED ARTIFACTS
    # -------------------------------------------------------------
    try:
        artifacts = get_cached_artifacts()
    except Exception as e:
        st.error(f"Failed to initialize detection pipeline: {e}")
        st.stop()

    class_names = artifacts["class_names_15"]

    # -------------------------------------------------------------
    # 2. SIDEBAR CONTROLS & ABOUT BOX
    # -------------------------------------------------------------
    st.sidebar.title("🛡️ NIDS Controls")

    # Model selector
    model_choice = st.sidebar.selectbox(
        "Select Detection Model",
        options=["Stacked Ensemble", "XGBoost"],
        index=0,
        help="Choose between the Stacked Ensemble (LR meta-learner over LR+RF+XGB) or the standalone XGBoost baseline.",
    )

    st.sidebar.markdown("---")

    # Data Source selector
    st.sidebar.subheader("Data Source")
    data_source_mode = st.sidebar.radio(
        "Choose Traffic Input",
        options=[
            "Pick a Demo Dataset",
            "Upload Custom Flow CSV",
        ],
        index=0,
    )

    df_raw: Optional[pd.DataFrame] = None
    source_label = ""

    if data_source_mode == "Pick a Demo Dataset":
        demo_choice = st.sidebar.selectbox(
            "Select Demo File",
            options=[
                "demo_balanced.csv (1,213 flows, ~100/class with Labels)",
                "demo_realistic.csv (5,000 flows, natural test distribution)",
                "demo_unlabeled.csv (1,213 flows, no Label column)",
            ],
            index=0,
        )
        file_map = {
            "demo_balanced.csv (1,213 flows, ~100/class with Labels)": DEMO_DIR / "demo_balanced.csv",
            "demo_realistic.csv (5,000 flows, natural test distribution)": DEMO_DIR / "demo_realistic.csv",
            "demo_unlabeled.csv (1,213 flows, no Label column)": DEMO_DIR / "demo_unlabeled.csv",
        }
        chosen_path = file_map[demo_choice]
        source_label = chosen_path.name
        try:
            if not chosen_path.exists():
                st.sidebar.error(f"Demo file not found at: {chosen_path}")
                st.stop()
            df_raw = pd.read_csv(chosen_path)
        except Exception as e:
            st.sidebar.error(f"Error loading demo file: {e}")
            st.stop()

    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Network Flow CSV (CICFlowMeter format)",
            type=["csv"],
            help="Upload a CSV file containing the 65 required CICIDS2017 flow features. 'Label' column is optional.",
        )
        if uploaded_file is not None:
            source_label = uploaded_file.name
            try:
                # Check for empty file
                if uploaded_file.size == 0:
                    st.error("Uploaded CSV file is completely empty (0 bytes). Please upload a valid flow file.")
                    st.stop()
                df_raw = pd.read_csv(uploaded_file)
                if df_raw.empty:
                    st.error("Uploaded CSV file contains no data rows. Please upload a valid flow file.")
                    st.stop()
            except pd.errors.EmptyDataError:
                st.error("The uploaded CSV file contains no data or columns.")
                st.stop()
            except pd.errors.ParserError as e:
                st.error(f"Failed to parse CSV file: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Error reading uploaded CSV file: {e}")
                st.stop()
        else:
            st.info("👈 Please select a Demo Dataset or upload a CSV file from the sidebar to begin inference.")
            # Still show about box in sidebar
            render_about_box()
            return

    # Render sidebar About Box
    render_about_box()

    # -------------------------------------------------------------
    # 3. MAIN DASHBOARD: TITLE & FEATURE PREPARATION
    # -------------------------------------------------------------
    st.title("Network Intrusion Detection System (CICIDS2017)")
    st.caption(f"Active Model: **{model_choice}** | Active Dataset: **{source_label}**")

    if df_raw is None or len(df_raw) == 0:
        st.warning("No data available to analyze.")
        return

    # Feature Preparation with friendly error handling
    try:
        X, kept_positions, skipped_df = prepare_features(df_raw, artifacts["selected_features"])
    except ValueError as e:
        st.error(f"⚠️ Validation Error: {e}")
        st.info("Tip: The input CSV must contain all 65 flow features selected during model training.")
        return
    except Exception as e:
        st.error(f"Unexpected error during feature preparation: {e}")
        return

    total_rows = len(df_raw)
    n_kept = len(X)
    n_skipped = len(skipped_df)

    if n_kept == 0:
        st.error("All rows in the provided dataset were skipped due to missing, NaN, or infinite feature values.")
        if n_skipped > 0:
            with st.expander(f"View Skipped Rows ({n_skipped:,})"):
                st.dataframe(skipped_df[["skip_reason"]], width="stretch")
        return

    # Run Prediction
    try:
        preds = predict(model_choice, X, artifacts)
    except Exception as e:
        st.error(f"Inference error during model prediction: {e}")
        return

    # Extract Prediction Statistics
    elapsed = preds.elapsed_time
    throughput = (n_kept / elapsed) if elapsed > 0 else 0.0
    attack_count = int(np.sum(preds.is_attack))
    benign_count = n_kept - attack_count
    attack_pct = (attack_count / n_kept) * 100.0 if n_kept > 0 else 0.0

    # -------------------------------------------------------------
    # 4. METRIC CARDS
    # -------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Flows Analysed", f"{n_kept:,}", help=f"Total rows in file: {total_rows:,}")
    col2.metric("Flagged as Attack", f"{attack_count:,} ({attack_pct:.1f}%)", delta=f"{attack_pct:.1f}% attack rate", delta_color="inverse")
    col3.metric("Rows Skipped", f"{n_skipped:,}", help="Rows containing NaN or infinite values in required features")
    col4.metric("Prediction Time", f"{elapsed * 1000:.1f} ms", help="Total time taken by model.predict_proba")
    col5.metric("Throughput", f"{throughput:,.0f} flows/s", help="Batch processing inference speed")

    if n_skipped > 0:
        with st.expander(f"⚠️ Warning: {n_skipped:,} rows were skipped due to invalid values"):
            st.dataframe(skipped_df, width="stretch")

    st.markdown("---")

    # -------------------------------------------------------------
    # 5. VISUALIZATIONS: PREDICTED CLASSES & NORMAL VS ATTACK
    # -------------------------------------------------------------
    vcol1, vcol2 = st.columns([3, 2])

    with vcol1:
        st.subheader("Predicted Traffic Classes")
        pred_counts = pd.Series(preds.pred_class_name).value_counts()
        fig_bar, ax_bar = plt.subplots(figsize=(8, 4.5), facecolor="white")
        bars = ax_bar.barh(pred_counts.index, pred_counts.values, color="#1f77b4", edgecolor="black", height=0.6)
        ax_bar.set_xlabel("Number of Flows", fontsize=10, fontweight="bold")
        ax_bar.grid(axis="x", linestyle="--", alpha=0.6)
        for bar in bars:
            w = bar.get_width()
            ax_bar.text(w + (max(pred_counts.values) * 0.02), bar.get_y() + bar.get_height() / 2, f"{int(w):,}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with vcol2:
        st.subheader("Traffic Composition")
        comp_df = pd.DataFrame({
            "Traffic Type": ["BENIGN (Normal)", "Malicious Attack"],
            "Count": [benign_count, attack_count]
        })
        fig_pie, ax_pie = plt.subplots(figsize=(5, 4.5), facecolor="white")
        colors = ["#2ca02c", "#d62728"]
        wedges, texts, autotexts = ax_pie.pie(
            comp_df["Count"],
            labels=comp_df["Traffic Type"],
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            textprops={"fontsize": 9, "fontweight": "bold"},
            wedgeprops={"edgecolor": "black", "linewidth": 1.2}
        )
        for at in autotexts:
            at.set_color("white")
        plt.tight_layout()
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    st.markdown("---")

    # -------------------------------------------------------------
    # 6. INTERACTIVE PREDICTIONS TABLE & DOWNLOAD
    # -------------------------------------------------------------
    st.subheader("Traffic Flow Predictions Explorer")

    # Build predictions dataframe
    has_ground_truth = "Label" in df_raw.columns
    pred_table = pd.DataFrame({
        "Row": kept_positions,
        "Predicted Class": preds.pred_class_name,
        "Confidence": np.round(preds.confidence, 4),
        "Attack Score": np.round(preds.attack_score, 4),
        "Status": np.where(preds.is_attack, "🚨 Attack", "✅ Normal"),
    })
    if has_ground_truth:
        y_true_all = df_raw.iloc[kept_positions]["Label"].astype(str).values
        pred_table["True Label"] = y_true_all
        pred_table["Correct"] = np.where(pred_table["Predicted Class"] == pred_table["True Label"], "✓", "✗")

    # Filters
    fcol1, fcol2 = st.columns([2, 2])
    with fcol1:
        unique_predicted = sorted(list(set(preds.pred_class_name)))
        selected_classes = st.multiselect(
            "Filter by Predicted Class",
            options=unique_predicted,
            default=unique_predicted,
        )
    with fcol2:
        min_conf = st.slider(
            "Minimum Confidence Filter",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
        )

    # Filtered dataframe
    filtered_df = pred_table[
        (pred_table["Predicted Class"].isin(selected_classes)) &
        (pred_table["Confidence"] >= min_conf)
    ]

    total_filtered = len(filtered_df)
    st.caption(f"Displaying {min(total_filtered, 1000):,} of {total_filtered:,} matching flows.")

    # Show at most 1,000 rows in the UI
    st.dataframe(
        filtered_df.head(1000),
        width="stretch",
        height=320,
    )

    # Full predictions download button
    full_csv_buffer = io.StringIO()
    pred_table.to_csv(full_csv_buffer, index=False)
    st.download_button(
        label="📥 Download Full Predictions CSV",
        data=full_csv_buffer.getvalue(),
        file_name=f"nids_predictions_{model_choice.lower().replace(' ', '_')}_{source_label}.csv",
        mime="text/csv",
        help="Download the complete prediction results table for all analyzed flows.",
    )

    # -------------------------------------------------------------
    # 7. GROUND TRUTH EVALUATION (WHEN 'Label' IS PRESENT)
    # -------------------------------------------------------------
    if has_ground_truth:
        st.markdown("---")
        st.subheader("Model Evaluation Against Ground-Truth Labels")
        st.info("ℹ️ Note: These evaluation scores describe a sample from the held-out test set, not unobserved live network traffic.")

        y_true = df_raw.iloc[kept_positions]["Label"].astype(str).values
        y_pred = preds.pred_class_name

        eval_res = evaluate(y_true, y_pred, class_names)

        # High-level metrics
        ecol1, ecol2, ecol3, ecol4 = st.columns(4)
        ecol1.metric("15-Class Accuracy", f"{eval_res.accuracy * 100:.2f}%")
        ecol2.metric("15-Class Macro F1", f"{eval_res.macro_f1:.4f}")

        # Compute Binary False Alarms & Missed Attacks
        binary_true_attack = (y_true != "BENIGN")
        binary_pred_attack = (y_pred != "BENIGN")

        false_alarms = int(np.sum((~binary_true_attack) & binary_pred_attack))
        missed_attacks = int(np.sum(binary_true_attack & (~binary_pred_attack)))

        ecol3.metric("False Alarms (FP on Benign)", f"{false_alarms:,}")
        ecol4.metric("Missed Attacks (FN)", f"{missed_attacks:,}")

        # Confusion Matrix Heatmap & Per-Class Metrics Table
        rcol1, rcol2 = st.columns([3, 2])

        with rcol1:
            st.markdown("#### Row-Normalized Confusion Matrix")
            # Filter classes present in this subset
            labels_present = [c for c in class_names if (c in y_true or c in y_pred)]
            sub_cm = confusion_matrix(y_true, y_pred, labels=labels_present, normalize="true")

            annot_sub = np.full(sub_cm.shape, "", dtype=object)
            for r in range(len(labels_present)):
                for c in range(len(labels_present)):
                    if sub_cm[r, c] >= 0.01:
                        annot_sub[r, c] = f"{sub_cm[r, c]:.2f}"

            fig_cm, ax_cm = plt.subplots(figsize=(8, 6), facecolor="white")
            sns.heatmap(
                sub_cm,
                annot=annot_sub,
                fmt="",
                cmap="Blues",
                xticklabels=labels_present,
                yticklabels=labels_present,
                cbar_kws={"label": "Normalized Recall"},
                ax=ax_cm,
                linewidths=0.5,
                linecolor="lightgrey"
            )
            ax_cm.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
            ax_cm.set_ylabel("True Class", fontsize=10, fontweight="bold")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with rcol2:
            st.markdown("#### Per-Class Performance")
            # Filter per-class dataframe to classes with support > 0
            active_pc = eval_res.per_class_table[eval_res.per_class_table["Support"] > 0]
            st.dataframe(active_pc, width="stretch", height=380)


def render_about_box():
    """Render the project documentation and limitation notice in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("About This System")
    with st.sidebar.expander("📌 Project Context & Limitations", expanded=False):
        st.markdown(
            """
            - **Training Scope**: Trained on the **CICIDS2017** benchmark dataset captured in an emulated lab environment, not on live production network traffic.
            - **Feature Requirements**: Input data must conform to **CICFlowMeter-style** flow statistics with the expected 65 feature columns.
            - **Probability Calibration**: Confidence values represent raw softmax or meta-learner probabilities and are **not calibrated** posterior attack likelihoods.
            - **Minority Attack Classes**: Detection metrics for ultra-rare classes (**Heartbleed**, **Infiltration**, **SQL Injection**) rest on only 2–7 test flows and are statistically unreliable.
            - **Demo Files**: The demo files are exact stratified samples drawn from the **held-out test set** ($20\\%$ split).
            - **Academic Prototype**: This system is an **academic research prototype**, not a hardened production intrusion prevention system (IPS).
            """
        )


if __name__ == "__main__":
    main()
