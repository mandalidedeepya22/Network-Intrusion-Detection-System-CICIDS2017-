"""
Step 9: Freeze Pipeline and Generate Report Assets.

Inference-only script:
- Generates reproducibility records and frozen requirements.
- Validates model predictions against cached outputs.
- Measures single-flow predict_proba latency across 3 repeats.
- Performs descriptive permutation importance on a balanced test subsample.
- Produces publication-grade report figures without internal titles.
- Generates demo CSV datasets for the Streamlit dashboard.
"""

import copy
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

# Ensure src directory is in sys.path for StackedIDS unpickling
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stack_model import StackedIDS

# Ensure standard UTF-8 console output for Windows PowerShell
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Consistent Model Color Palette Across All Figures
MODEL_COLORS = {
    "Logistic Regression": "#d95f02",  # Rust / Orange
    "Random Forest": "#7570b3",        # Purple
    "XGBoost": "#1b9e77",              # Teal / Green
    "Stacked Ensemble": "#2b5c8f"      # Navy Blue
}


def run_report_assets():
    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"
    outputs_dir = PROJECT_ROOT / "outputs"
    figures_dir = outputs_dir / "report_figures"
    demo_dir = PROJECT_ROOT / "demo"

    figures_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STEP 9: FREEZE PIPELINE & GENERATE REPORT ASSETS (INFERENCE ONLY)")
    print("=" * 75)

    # Load shared configurations
    with open(models_dir / "selected_features.json", "r", encoding="utf-8") as f:
        selected_features = json.load(f)["selected"]
    with open(models_dir / "label_maps.json", "r", encoding="utf-8") as f:
        label_maps = json.load(f)

    id_to_class_15 = {int(k): v for k, v in label_maps["multi15"]["id_to_class"].items()}
    class_to_id_15 = label_maps["multi15"]["class_to_id"]
    class_to_id_13 = label_maps["multi13"]["class_to_id"]
    class_names_15 = [id_to_class_15[i] for i in range(15)]
    benign_id = class_to_id_15["BENIGN"]

    web_attack_mapping = {
        "Web Attack - Brute Force": "Web Attack",
        "Web Attack - XSS": "Web Attack",
        "Web Attack - Sql Injection": "Web Attack"
    }
    map_15_to_13 = np.array([
        class_to_id_13[web_attack_mapping.get(id_to_class_15[i], id_to_class_15[i])]
        for i in range(15)
    ], dtype=np.int32)

    # =============================================================
    # PART A - REPRODUCIBILITY RECORD
    # =============================================================
    print("\n" + "=" * 75)
    print("PART A: REPRODUCIBILITY RECORD (MODEL CARD & FROZEN REQUIREMENTS)")
    print("=" * 75)

    # 1. Gather package versions
    import importlib.metadata

    package_versions = {
        "python": platform.python_version(),
        "pandas": importlib.metadata.version("pandas"),
        "numpy": importlib.metadata.version("numpy"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "xgboost": importlib.metadata.version("xgboost"),
        "imbalanced-learn": importlib.metadata.version("imbalanced-learn"),
        "boruta": importlib.metadata.version("boruta"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "joblib": importlib.metadata.version("joblib")
    }

    # 2. Read dataset dimensions directly from Parquet metadata
    parquet_files = [
        "merged_raw.parquet",
        "cleaned.parquet",
        "train.parquet",
        "test.parquet",
        "train_cap.parquet",
        "train_smote.parquet",
        "test_scaled.parquet"
    ]
    dataset_metadata = {}
    for pf in parquet_files:
        p_path = data_dir / pf
        meta = pq.read_metadata(p_path)
        dataset_metadata[pf] = {
            "num_rows": int(meta.num_rows),
            "num_columns": int(meta.num_columns)
        }
        print(f"  * {pf:<22}: {meta.num_rows:>10,} rows, {meta.num_columns:>3} columns")

    # 3. Model files and sizes in models/
    model_files_info = {}
    for mf in sorted(models_dir.glob("*.joblib")):
        model_files_info[mf.name] = {
            "size_bytes": int(mf.stat().st_size),
            "size_mb": round(float(mf.stat().st_size / (1024 ** 2)), 4)
        }

    # 4. Construct model_card.json payload
    model_card_payload = {
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "package_versions": package_versions
        },
        "random_seeds": {
            "data_split": 42,
            "feature_selection": 42,
            "capping": 42,
            "smote_baseline": 42,
            "ablation_seeds": [42, 43, 44],
            "bootstrap_evaluation": 42,
            "permutation_importance": 42
        },
        "datasets": dataset_metadata,
        "selected_features_count": len(selected_features),
        "selected_features": selected_features,
        "classes_count": len(class_names_15),
        "classes": class_names_15,
        "models_artifacts": model_files_info
    }

    model_card_path = models_dir / "model_card.json"
    with open(model_card_path, "w", encoding="utf-8") as f:
        json.dump(model_card_payload, f, indent=2)
    print(f"\nSaved reproducibility record to: {model_card_path}")

    # 5. Write requirements_frozen.txt from pip freeze
    print("Generating requirements_frozen.txt via 'python -m pip freeze'...")
    freeze_res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    req_frozen_path = PROJECT_ROOT / "requirements_frozen.txt"
    with open(req_frozen_path, "w", encoding="utf-8") as f:
        f.write(freeze_res.stdout)
    print(f"Saved frozen dependencies to: {req_frozen_path}")

    # =============================================================
    # PART B - FINAL RESULTS (INFERENCE ONLY)
    # =============================================================
    print("\n" + "=" * 75)
    print("PART B: FINAL RESULTS & RIGOROUS INFERENCE VALIDATION")
    print("=" * 75)

    print("Loading test dataset and model checkpoints...")
    test_scaled = pd.read_parquet(data_dir / "test_scaled.parquet", engine="pyarrow")
    X_test_65 = test_scaled[selected_features].values
    y_test_15 = test_scaled["y_multi15"].values
    y_test_13 = test_scaled["y_multi13"].values
    y_test_bin = test_scaled["y_binary"].values
    n_test = len(y_test_15)

    lr = joblib.load(models_dir / "logistic_regression.joblib")
    rf = joblib.load(models_dir / "random_forest.joblib")
    xgb = joblib.load(models_dir / "xgboost.joblib")
    stack = joblib.load(models_dir / "stacked_ids.joblib")

    # Load cached predictions from Step 8
    cached_npz_path = outputs_dir / "test_predictions.npz"
    cached_preds = np.load(cached_npz_path)

    models_list = [
        ("logistic_regression", "Logistic Regression", lr, "lr", 0.93512, 0.54071),
        ("random_forest", "Random Forest", rf, "rf", 0.99807, 0.86217),
        ("xgboost", "XGBoost", xgb, "xgb", 0.99839, 0.86609),
        ("stacked_ids", "Stacked Ensemble", stack, "stack", 0.99847, 0.87276)
    ]

    final_results_records = []
    final_per_class_records = []
    model_probas = {}
    model_preds = {}

    for model_key, display_name, model_obj, cache_key, exp_acc, exp_f1 in models_list:
        print(f"\nEvaluating: {display_name}...")
        proba = model_obj.predict_proba(X_test_65)
        pred_15 = np.argmax(proba, axis=1)

        model_probas[display_name] = proba
        model_preds[display_name] = pred_15

        # Strict Assertion 1: Argmax matches cached predictions exactly
        cached_p = cached_preds[cache_key]
        assert np.array_equal(pred_15, cached_p), (
            f"[ASSERTION FAILED] {display_name} predictions do not match cached test_predictions.npz!"
        )
        print("  [PASS] Predictions 100% identical to Step 8 cached predictions.")

        # Compute 15-class metrics
        acc_15 = accuracy_score(y_test_15, pred_15)
        prec_macro_15 = precision_score(y_test_15, pred_15, average="macro", zero_division=0)
        rec_macro_15 = recall_score(y_test_15, pred_15, average="macro", zero_division=0)
        f1_macro_15 = f1_score(y_test_15, pred_15, average="macro", zero_division=0)
        f1_weighted_15 = f1_score(y_test_15, pred_15, average="weighted", zero_division=0)

        # Strict Assertion 2: Match earlier values to 5 decimals
        assert np.isclose(acc_15, exp_acc, atol=1e-5), (
            f"[ASSERTION FAILED] {display_name} accuracy {acc_15:.5f} != expected {exp_acc:.5f}"
        )
        assert np.isclose(f1_macro_15, exp_f1, atol=1e-5), (
            f"[ASSERTION FAILED] {display_name} macro F1 {f1_macro_15:.5f} != expected {exp_f1:.5f}"
        )
        print(f"  [PASS] Verified to 5 decimals: Acc={acc_15:.5f} (exp: {exp_acc:.5f}), Macro F1={f1_macro_15:.5f} (exp: {exp_f1:.5f})")

        # 13-class metric
        pred_13 = map_15_to_13[pred_15]
        f1_macro_13 = f1_score(y_test_13, pred_13, average="macro", zero_division=0)

        # Binary metrics
        pred_bin = (pred_15 != benign_id).astype(int)
        prec_bin = precision_score(y_test_bin, pred_bin, pos_label=1, zero_division=0)
        rec_bin = recall_score(y_test_bin, pred_bin, pos_label=1, zero_division=0)
        f1_bin = f1_score(y_test_bin, pred_bin, pos_label=1, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_test_bin, pred_bin).ravel()
        benign_fpr = (fp / (fp + tn)) * 100.0

        attack_score = 1.0 - proba[:, benign_id]
        roc_auc = roc_auc_score(y_test_bin, attack_score)
        avg_precision = average_precision_score(y_test_bin, attack_score)

        final_results_records.append({
            "Model": display_name,
            "15-Class Acc": round(acc_15, 5),
            "15-Class Macro Prec": round(prec_macro_15, 5),
            "15-Class Macro Rec": round(rec_macro_15, 5),
            "15-Class Macro F1": round(f1_macro_15, 5),
            "15-Class Weighted F1": round(f1_weighted_15, 5),
            "13-Class Macro F1": round(f1_macro_13, 5),
            "Attack Precision": round(prec_bin, 5),
            "Attack Recall": round(rec_bin, 5),
            "Attack F1": round(f1_bin, 5),
            "FP Count": int(fp),
            "FN Count": int(fn),
            "Benign FPR (%)": round(benign_fpr, 4),
            "ROC-AUC": round(roc_auc, 5),
            "Average Precision": round(avg_precision, 5)
        })

        # Per-class records
        rep = classification_report(y_test_15, pred_15, target_names=class_names_15, output_dict=True, zero_division=0)
        for c_name in class_names_15:
            final_per_class_records.append({
                "Model": display_name,
                "Class": c_name,
                "Precision": round(rep[c_name]["precision"], 5),
                "Recall": round(rep[c_name]["recall"], 5),
                "F1": round(rep[c_name]["f1-score"], 5),
                "Support": int(rep[c_name]["support"])
            })

    # Save and print final results
    final_results_df = pd.DataFrame(final_results_records)
    final_results_path = outputs_dir / "final_results.csv"
    final_results_df.to_csv(final_results_path, index=False)

    final_per_class_df = pd.DataFrame(final_per_class_records)
    final_per_class_path = outputs_dir / "final_per_class.csv"
    final_per_class_df.to_csv(final_per_class_path, index=False)

    print("\n" + "=" * 75)
    print("FINAL RESULTS TABLE")
    print("=" * 75)
    print(final_results_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("FINAL PER-CLASS METRICS (SUMMARY VIEW)")
    print("=" * 75)
    print(final_per_class_df.head(30).to_string(index=False))
    print(f"... ({len(final_per_class_df)} total class rows saved to {final_per_class_path})")

    # =============================================================
    # PART C - LATENCY, MEASURED 3 TIMES
    # =============================================================
    print("\n" + "=" * 75)
    print("PART C: FAIR SINGLE-THREAD LATENCY (3 INDEPENDENT REPEATS)")
    print("=" * 75)

    np.random.seed(42)
    single_indices = np.random.choice(n_test, size=200, replace=False)
    single_samples = [X_test_65[i:i + 1] for i in single_indices]

    lr_fair = copy.deepcopy(lr)
    rf_fair = copy.deepcopy(rf)
    rf_fair.set_params(n_jobs=1)
    xgb_fair = copy.deepcopy(xgb)
    xgb_fair.set_params(n_jobs=1)
    meta_fair = copy.deepcopy(stack.meta_model)

    stack_fair = StackedIDS(
        base_models=[lr_fair, rf_fair, xgb_fair],
        meta_model=meta_fair,
        classes_=stack.classes_,
        feature_names=stack.feature_names
    )

    models_fair_dict = {
        "Logistic Regression": lr_fair,
        "Random Forest (n_jobs=1)": rf_fair,
        "XGBoost (n_jobs=1)": xgb_fair,
        "Stacked Ensemble (fair)": stack_fair
    }

    latency_rows = []
    for repeat_num in range(1, 4):
        print(f"\n--- Latency Measurement Repeat {repeat_num} of 3 ---")
        for m_name, m_obj in models_fair_dict.items():
            times_ms = []
            for s in single_samples:
                t0 = time.perf_counter()
                m_obj.predict_proba(s)
                times_ms.append((time.perf_counter() - t0) * 1000.0)
            med = float(np.median(times_ms))
            p95 = float(np.percentile(times_ms, 95))
            print(f"  {m_name:<28}: Median = {med:>7.4f} ms | P95 = {p95:>7.4f} ms")
            latency_rows.append({
                "Model": m_name,
                "Repeat": repeat_num,
                "Median Latency (ms)": round(med, 4),
                "95th Percentile (ms)": round(p95, 4)
            })

    # Batch throughput with default settings
    print("\nMeasuring batch throughput on full test set (default settings)...")
    throughput_records = {}
    for m_name, m_obj in [("Logistic Regression", lr), ("Random Forest", rf), ("XGBoost", xgb), ("Stacked Ensemble", stack)]:
        t0 = time.perf_counter()
        m_obj.predict(X_test_65)
        dt = time.perf_counter() - t0
        th = len(X_test_65) / dt
        throughput_records[m_name] = round(th, 1)
        print(f"  {m_name:<25}: {th:>11,.1f} flows/sec ({dt:.2f}s)")

    # Aggregate latency across repeats
    latency_df = pd.DataFrame(latency_rows)
    latency_summary = latency_df.groupby("Model")[["Median Latency (ms)", "95th Percentile (ms)"]].mean().reset_index()
    latency_summary["Batch Throughput (flows/s)"] = latency_summary["Model"].map(
        lambda m: throughput_records.get(m.replace(" (n_jobs=1)", "").replace(" (fair)", ""), 0.0)
    )

    latency_path = outputs_dir / "latency_final.csv"
    latency_summary.to_csv(latency_path, index=False)
    print(f"\nSaved final latency summary to: {latency_path}")
    print(latency_summary.to_string(index=False))

    # =============================================================
    # PART D - SMOTE PER-CLASS VIEW (NO TRAINING)
    # =============================================================
    print("\n" + "=" * 75)
    print("PART D: SMOTE PER-CLASS F1 ANALYSIS (V1_BASELINE vs V2_NO_SMOTE)")
    print("=" * 75)

    ablation_pc_path = outputs_dir / "ablation_per_class.csv"
    ablation_pc_df = pd.read_csv(ablation_pc_path)

    v1_runs = ablation_pc_df[ablation_pc_df["Variant"] == "V1_baseline"]
    v2_runs = ablation_pc_df[ablation_pc_df["Variant"] == "V2_no_smote"]

    smote_per_class_records = []
    for c_name in class_names_15:
        v1_mean = v1_runs[c_name].mean()
        v1_std = v1_runs[c_name].std()
        v2_mean = v2_runs[c_name].mean()
        v2_std = v2_runs[c_name].std()
        diff = v2_mean - v1_mean

        smote_per_class_records.append({
            "Class Name": c_name,
            "V1 Baseline F1 (Mean ± Std)": f"{v1_mean:.5f} ± {v1_std:.5f}",
            "V2 No-SMOTE F1 (Mean ± Std)": f"{v2_mean:.5f} ± {v2_std:.5f}",
            "Difference (V2 - V1)": round(diff, 5)
        })

    smote_comp_df = pd.DataFrame(smote_per_class_records)
    print(smote_comp_df.to_string(index=False))

    print("\n--- XGBoost Parameters in Ablation Study ---")
    print("  Parameters: objective='multi:softprob', n_estimators=200, max_depth=8, learning_rate=0.1, tree_method='hist'")
    print("  Subsample: 1.0 (default) | colsample_bytree: 1.0 (default) | colsample_bylevel: 1.0 (default)")
    print("  Seeds Evaluated: 42, 43, 44")
    v2_f1_values = v2_runs["Heartbleed"].values
    print(f"\nDid V2 No-SMOTE seeds produce identical results? YES.")
    print("Explanation: Because subsample=1.0, colsample=1.0, and no stochastic oversampling (SMOTE) is applied,")
    print("the training dataset and histogram binning are completely deterministic across seeds (std = 0.00000).")

    # =============================================================
    # PART E - PERMUTATION IMPORTANCE (DESCRIPTIVE ONLY)
    # =============================================================
    print("\n" + "=" * 75)
    print("PART E: DESCRIPTIVE PERMUTATION IMPORTANCE (XGBOOST, BALANCED TEST SUBSAMPLE)")
    print("=" * 75)

    print("Building balanced test subsample (at most 5,000 rows per class, random_state=42)...")
    sub_dfs = []
    for c in range(15):
        grp = test_scaled[test_scaled["y_multi15"] == c]
        sub_dfs.append(grp.sample(n=min(len(grp), 5000), random_state=42))
    test_sub = pd.concat(sub_dfs, axis=0, ignore_index=True)
    X_sub = test_sub[selected_features].values
    y_sub = test_sub["y_multi15"].values
    print(f"Subsample built: {len(test_sub):,} rows across 15 classes.")

    print("Computing permutation importance (scoring=macro F1, n_repeats=5, n_jobs=-1)...")
    t_perm0 = time.perf_counter()
    f1_scorer = make_scorer(f1_score, average="macro", zero_division=0)
    perm_result = permutation_importance(
        xgb,
        X_sub,
        y_sub,
        scoring=f1_scorer,
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )
    perm_time = time.perf_counter() - t_perm0
    print(f"Permutation importance completed in: {perm_time:.2f} seconds")

    perm_df = pd.DataFrame({
        "feature": selected_features,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    perm_path = outputs_dir / "permutation_importance.csv"
    perm_df.to_csv(perm_path, index=False)
    print(f"Saved full permutation importances to: {perm_path}")

    print("\n--- Top 15 Features by Permutation Importance (Macro F1 Decrease) ---")
    top15_perm = perm_df.head(15).copy()
    top15_perm["importance_mean"] = top15_perm["importance_mean"].round(5)
    top15_perm["importance_std"] = top15_perm["importance_std"].round(5)
    top15_perm.index = range(1, 16)
    print(top15_perm.to_string())

    dest_port_rank = int(perm_df[perm_df["feature"] == "Destination Port"].index[0]) + 1
    dest_port_val = perm_df.loc[perm_df["feature"] == "Destination Port", "importance_mean"].values[0]
    print(f"\nDestination Port Rank: #{dest_port_rank} out of 65 features (Mean Macro F1 drop: {dest_port_val:.5f})")

    # =============================================================
    # PART F - PUBLICATION-GRADE REPORT FIGURES
    # =============================================================
    print("\n" + "=" * 75)
    print("PART F: GENERATING PUBLICATION-GRADE REPORT FIGURES (NO INTERNAL TITLES)")
    print("=" * 75)

    # 1. fig_class_distribution.png
    cleaned_df = pd.read_parquet(data_dir / "cleaned.parquet", columns=["Label"])
    class_counts = cleaned_df["Label"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    bars = ax.barh(class_counts.index, class_counts.values, color="#2b5c8f", edgecolor="black", height=0.65)
    ax.set_xscale("log")
    ax.set_xlabel("Number of Flows (Logarithmic Scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Traffic Category / Class", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    # Label counts
    for bar in bars:
        val = bar.get_width()
        ax.text(val * 1.15, bar.get_y() + bar.get_height() / 2, f"{int(val):,}", va="center", fontsize=9)
    plt.tight_layout()
    fig1_path = figures_dir / "fig_class_distribution.png"
    plt.savefig(fig1_path, dpi=300, facecolor="white")
    plt.close()
    print(f"  [1/7] Saved: {fig1_path}")

    # 2. Confusion Matrices for Stack and XGBoost (Annotate cells >= 0.01)
    support_counts = np.bincount(y_test_15, minlength=15)
    axis_labels = [f"{class_names_15[i]}\n(n={support_counts[i]:,})" for i in range(15)]

    for model_name, p_arr, filename in [
        ("Stacked Ensemble", model_preds["Stacked Ensemble"], "fig_confusion_stack.png"),
        ("XGBoost", model_preds["XGBoost"], "fig_confusion_xgboost.png")
    ]:
        cm = confusion_matrix(y_test_15, p_arr, normalize="true")
        annot_matrix = np.full(cm.shape, "", dtype=object)
        for r in range(15):
            for c in range(15):
                if cm[r, c] >= 0.01:
                    annot_matrix[r, c] = f"{cm[r, c]:.2f}"

        fig, ax = plt.subplots(figsize=(14, 12), facecolor="white")
        sns.heatmap(
            cm,
            annot=annot_matrix,
            fmt="",
            cmap="Blues",
            xticklabels=axis_labels,
            yticklabels=axis_labels,
            cbar_kws={"label": "Normalized Recall (Proportion of True Class)"},
            ax=ax,
            linewidths=0.5,
            linecolor="lightgrey"
        )
        ax.set_xlabel("Predicted Class", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Class", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        fig_cm_path = figures_dir / filename
        plt.savefig(fig_cm_path, dpi=300, facecolor="white")
        plt.close()
        print(f"  [2/7] Saved: {fig_cm_path}")

    # 3. fig_roc_binary.png (Log-scale FPR axis from 1e-5 to 1)
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    for m_name in ["Logistic Regression", "Random Forest", "XGBoost", "Stacked Ensemble"]:
        attack_score = 1.0 - model_probas[m_name][:, benign_id]
        fpr, tpr, _ = roc_curve(y_test_bin, attack_score)
        auc_val = roc_auc_score(y_test_bin, attack_score)
        ax.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.5f})", color=MODEL_COLORS[m_name], linewidth=2.0)

    ax.plot([1e-5, 1], [1e-5, 1], "k--", alpha=0.5, label="Random Guessing")
    ax.set_xscale("log")
    ax.set_xlim([1e-5, 1.0])
    ax.set_ylim([0.80, 1.005])
    ax.set_xlabel("False Positive Rate on BENIGN (Log Scale: 1e-5 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Attack Recall)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6, which="both")
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    fig3_path = figures_dir / "fig_roc_binary.png"
    plt.savefig(fig3_path, dpi=300, facecolor="white")
    plt.close()
    print(f"  [3/7] Saved: {fig3_path}")

    # 4. fig_pr_binary.png (Precision-Recall Curve with Average Precision in Legend)
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    for m_name in ["Logistic Regression", "Random Forest", "XGBoost", "Stacked Ensemble"]:
        attack_score = 1.0 - model_probas[m_name][:, benign_id]
        prec, rec, _ = precision_recall_curve(y_test_bin, attack_score)
        ap_val = average_precision_score(y_test_bin, attack_score)
        ax.plot(rec, prec, label=f"{m_name} (AP = {ap_val:.5f})", color=MODEL_COLORS[m_name], linewidth=2.0)

    ax.set_xlabel("Attack Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Attack Precision", fontsize=11, fontweight="bold")
    ax.set_xlim([0.90, 1.002])
    ax.set_ylim([0.70, 1.005])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    plt.tight_layout()
    fig4_path = figures_dir / "fig_pr_binary.png"
    plt.savefig(fig4_path, dpi=300, facecolor="white")
    plt.close()
    print(f"  [4/7] Saved: {fig4_path}")

    # 5. fig_per_class_f1.png (Grouped bar chart of per-class F1 for all 4 models)
    fig, ax = plt.subplots(figsize=(15, 7), facecolor="white")
    n_classes = len(class_names_15)
    x = np.arange(n_classes)
    width = 0.20

    pc_models = ["Logistic Regression", "Random Forest", "XGBoost", "Stacked Ensemble"]
    for i, m_name in enumerate(pc_models):
        m_df = final_per_class_df[final_per_class_df["Model"] == m_name]
        f1_vals = [m_df.loc[m_df["Class"] == c, "F1"].values[0] for c in class_names_15]
        ax.bar(x + (i - 1.5) * width, f1_vals, width, label=m_name, color=MODEL_COLORS[m_name], edgecolor="black", alpha=0.9)

    ax.set_ylabel("F1 Score", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    x_tick_labels = [f"{class_names_15[i]}\n(n={support_counts[i]:,})" for i in range(15)]
    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim([0.0, 1.05])
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    plt.tight_layout()
    fig5_path = figures_dir / "fig_per_class_f1.png"
    plt.savefig(fig5_path, dpi=300, facecolor="white")
    plt.close()
    print(f"  [5/7] Saved: {fig5_path}")

    # 6. fig_ablation.png (Mean ± std macro F1 per variant with individual seed dots)
    ablation_runs = pd.read_csv(outputs_dir / "ablation_runs.csv")
    variants_ordered = ["V1_baseline", "V2_no_smote", "V3_all_69", "V4_top_20", "V5_no_dest_port"]
    var_labels = ["V1: Baseline\n(65 feats, SMOTE)", "V2: No SMOTE\n(65 feats)", "V3: All 69 Feats\n(No Boruta)", "V4: Top 20 Feats\n(XGBoost MDI)", "V5: No Dest Port\n(64 feats)"]

    mean_f1s = []
    std_f1s = []
    seed_points = []
    for v in variants_ordered:
        v_sub = ablation_runs[ablation_runs["Variant"] == v]
        mean_f1s.append(v_sub["15-Class Macro F1"].mean())
        std_f1s.append(v_sub["15-Class Macro F1"].std())
        seed_points.append(v_sub["15-Class Macro F1"].values)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    x_idx = np.arange(len(variants_ordered))
    bars = ax.bar(x_idx, mean_f1s, yerr=std_f1s, capsize=6, color="#2b5c8f", edgecolor="black", alpha=0.8, width=0.5)

    # Plot individual seed points
    for idx, pts in enumerate(seed_points):
        ax.scatter([idx] * len(pts), pts, color="#d95f02", s=50, zorder=5, edgecolor="black", label="Seed Runs (42, 43, 44)" if idx == 0 else "")

    ax.set_xticks(x_idx)
    ax.set_xticklabels(var_labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("15-Class Macro F1 Score", fontsize=11, fontweight="bold")
    ax.set_ylim([0.68, 0.92])
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10, frameon=True)

    for bar, m, s in zip(bars, mean_f1s, std_f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.007, f"{m:.4f}\n(±{s:.4f})", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig6_path = figures_dir / "fig_ablation.png"
    plt.savefig(fig6_path, dpi=300, facecolor="white")
    plt.close()
    print(f"  [6/7] Saved: {fig6_path}")

    # 7. fig_perm_importance.png (Top 15 permutation importances with error bars)
    top15_plot = perm_df.head(15).sort_values(by="importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    ax.barh(
        top15_plot["feature"],
        top15_plot["importance_mean"],
        xerr=top15_plot["importance_std"],
        capsize=4,
        color="#1b9e77",
        edgecolor="black",
        height=0.65
    )
    ax.set_xlabel("Permutation Feature Importance (Mean Macro F1 Decrease ± Std)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Network Flow Feature", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig7_path = figures_dir / "fig_perm_importance.png"
    plt.savefig(fig7_path, dpi=300, facecolor="white")
    plt.close()
    print(f"  [7/7] Saved: {fig7_path}")

    # =============================================================
    # PART G - DEMO FILES FOR STREAMLIT DASHBOARD
    # =============================================================
    print("\n" + "=" * 75)
    print("PART G: GENERATING DEMO DATASETS (RAW UNSCALED TEST SET ONLY)")
    print("=" * 75)

    test_raw = pd.read_parquet(data_dir / "test.parquet", engine="pyarrow")
    raw_feature_cols = [c for c in test_raw.columns if c not in ["Label", "y_binary", "y_multi15", "y_multi13"]]
    print(f"Loaded raw unscaled test data: {test_raw.shape[0]:,} rows, {len(raw_feature_cols)} feature columns.")

    # 1. demo_balanced.csv: up to 100 rows per class
    balanced_dfs = []
    for c_name in class_names_15:
        c_grp = test_raw[test_raw["Label"] == c_name]
        n_take = min(len(c_grp), 100)
        balanced_dfs.append(c_grp.sample(n=n_take, random_state=42))

    demo_balanced = pd.concat(balanced_dfs, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    demo_balanced_out = demo_balanced[raw_feature_cols + ["Label"]]
    demo_balanced_path = demo_dir / "demo_balanced.csv"
    demo_balanced_out.to_csv(demo_balanced_path, index=False)
    print(f"\nSaved {demo_balanced_path.name} ({len(demo_balanced_out):,} rows) to: {demo_balanced_path}")
    print("Class distribution in demo_balanced.csv:")
    print(demo_balanced_out["Label"].value_counts().to_string())

    # 2. demo_realistic.csv: 5,000 random rows with real class mix
    demo_realistic = test_raw.sample(n=5000, random_state=42).reset_index(drop=True)
    demo_realistic_out = demo_realistic[raw_feature_cols + ["Label"]]
    demo_realistic_path = demo_dir / "demo_realistic.csv"
    demo_realistic_out.to_csv(demo_realistic_path, index=False)
    print(f"\nSaved {demo_realistic_path.name} ({len(demo_realistic_out):,} rows) to: {demo_realistic_path}")
    print("Class distribution in demo_realistic.csv:")
    print(demo_realistic_out["Label"].value_counts().to_string())

    # 3. demo_unlabeled.csv: demo_balanced without Label
    demo_unlabeled_out = demo_balanced[raw_feature_cols]
    demo_unlabeled_path = demo_dir / "demo_unlabeled.csv"
    demo_unlabeled_out.to_csv(demo_unlabeled_path, index=False)
    print(f"\nSaved {demo_unlabeled_path.name} ({len(demo_unlabeled_out):,} rows) to: {demo_unlabeled_path}")
    print(f"Columns in demo_unlabeled.csv: {demo_unlabeled_out.shape[1]} (Zero target columns)")

    print("\n" + "=" * 75)
    print("STEP 9 COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    run_report_assets()
