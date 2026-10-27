"""
Step 6: Train and Evaluate Baseline Models on the 15-Class Target.

Why this matters for viva:
Evaluating diverse model families (linear via Logistic Regression, bagging via Random Forest,
and boosting via XGBoost) establishes rigorous benchmarks before building an ensemble.
Evaluating across fine-grained 15-class, grouped 13-class, and binary viewpoints ensures that
both broad detection efficacy and extreme minority attack sensitivity are thoroughly analyzed.
"""

import json
import sys
import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Ensure standard UTF-8 console output for Windows PowerShell
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_project_root() -> Path:
    """Return the absolute path of the nids-project directory."""
    return Path(__file__).resolve().parent.parent


def train_and_eval_baselines():
    project_root = get_project_root()
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    outputs_dir = project_root / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_smote.parquet"
    test_path = data_dir / "test_scaled.parquet"
    selected_features_path = models_dir / "selected_features.json"
    label_maps_path = models_dir / "label_maps.json"
    comparison_csv_path = outputs_dir / "baseline_metrics.csv"

    print("=" * 70)
    print("STEP 6: BASELINE MODEL TRAINING & MULTI-PERSPECTIVE EVALUATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Load Data, Feature Names, and Mappings
    # -------------------------------------------------------------
    print("[1/5] Loading datasets and mappings...")
    with open(selected_features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)["selected"]
    print(f"Selected features count: {len(selected_features)}")

    with open(label_maps_path, "r", encoding="utf-8") as f:
        label_maps = json.load(f)

    id_to_class_15 = {int(k): v for k, v in label_maps["multi15"]["id_to_class"].items()}
    class_to_id_15 = label_maps["multi15"]["class_to_id"]
    class_to_id_13 = label_maps["multi13"]["class_to_id"]
    benign_id = class_to_id_15["BENIGN"]
    class_names_15 = [id_to_class_15[i] for i in range(15)]

    # Mapping dictionary from 15-class integer IDs to 13-class integer IDs
    web_attack_mapping = {
        "Web Attack - Brute Force": "Web Attack",
        "Web Attack - XSS": "Web Attack",
        "Web Attack - Sql Injection": "Web Attack"
    }
    map_15_to_13 = {}
    for id_15 in range(15):
        name_15 = id_to_class_15[id_15]
        name_13 = web_attack_mapping.get(name_15, name_15)
        map_15_to_13[id_15] = class_to_id_13[name_13]

    train_smote = pd.read_parquet(train_path, engine="pyarrow")
    test_scaled = pd.read_parquet(test_path, engine="pyarrow")
    print(f"Loaded train_smote: {train_smote.shape[0]:,} rows, {train_smote.shape[1]} columns")
    print(f"Loaded test_scaled:  {test_scaled.shape[0]:,} rows, {test_scaled.shape[1]} columns")

    # -------------------------------------------------------------
    # 2. Strict Sanity Assertions Before Training
    # -------------------------------------------------------------
    print("\n[2/5] Running sanity assertions before training...")

    # A) Target column consistency in train_smote
    expected_labels = train_smote["y_multi15"].map(id_to_class_15)
    expected_binary = (expected_labels != "BENIGN").astype(np.int32)
    expected_13_labels = expected_labels.replace(web_attack_mapping)
    expected_multi13 = expected_13_labels.map(class_to_id_13).astype(np.int32)

    assert (train_smote["Label"] == expected_labels).all(), "Label column inconsistent with y_multi15 in train_smote"
    assert (train_smote["y_binary"] == expected_binary).all(), "y_binary column inconsistent with y_multi15 in train_smote"
    assert (train_smote["y_multi13"] == expected_multi13).all(), "y_multi13 column inconsistent with y_multi15 in train_smote"
    print("  [PASS] Target column consistency across all rows in train_smote verified.")

    # B) Identical feature columns & ordering
    assert list(train_smote[selected_features].columns) == list(test_scaled[selected_features].columns), (
        "Feature columns or column ordering mismatch between train and test sets!"
    )
    print("  [PASS] Feature column names and ordering are identical.")

    # C) No NaN or Inf in either set
    X_train = train_smote[selected_features].values
    y_train = train_smote["y_multi15"].values

    X_test = test_scaled[selected_features].values
    y_test_15 = test_scaled["y_multi15"].values
    y_test_13 = test_scaled["y_multi13"].values
    y_test_bin = test_scaled["y_binary"].values

    assert not np.isnan(X_train).any(), "NaN found in X_train"
    assert not np.isinf(X_train).any(), "Inf found in X_train"
    assert not np.isnan(X_test).any(), "NaN found in X_test"
    assert not np.isinf(X_test).any(), "Inf found in X_test"
    print("  [PASS] Zero NaN and zero Inf values in train and test feature matrices.")

    # -------------------------------------------------------------
    # 3. Model Definitions
    # -------------------------------------------------------------
    models_config = [
        (
            "logistic_regression",
            "Logistic Regression",
            LogisticRegression(max_iter=1000, random_state=42)
        ),
        (
            "random_forest",
            "Random Forest",
            RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        ),
        (
            "xgboost",
            "XGBoost",
            XGBClassifier(
                objective="multi:softprob",
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                tree_method="hist",
                n_jobs=-1,
                random_state=42
            )
        )
    ]

    # Pre-select 200 random rows for single-flow latency measurement
    np.random.seed(42)
    single_indices = np.random.choice(len(X_test), size=200, replace=False)
    single_samples = [X_test[i:i + 1] for i in single_indices]

    comparison_results = []

    # -------------------------------------------------------------
    # 4. Training and Evaluation Loop
    # -------------------------------------------------------------
    print("\n[3/5] & [4/5] Training models and evaluating on full test set (499,596 rows)...")

    for model_key, display_name, model in models_config:
        print("\n" + "-" * 60)
        print(f"TRAINING: {display_name.upper()}")
        print("-" * 60)

        # Train model
        t_train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t_train_start
        print(f"Training completed in: {train_time:.2f} seconds ({train_time / 60:.2f} minutes)")

        # Save model
        model_save_path = models_dir / f"{model_key}.joblib"
        joblib.dump(model, model_save_path, compress=3)
        model_size_mb = model_save_path.stat().st_size / (1024 ** 2)
        print(f"Saved model to: {model_save_path} ({model_size_mb:.2f} MB)")

        # Predict on full test set
        print("Evaluating on full test set...")
        t_batch_start = time.perf_counter()
        y_pred_15 = model.predict(X_test)
        t_batch = time.perf_counter() - t_batch_start
        throughput = len(X_test) / t_batch
        print(f"Batch inference time: {t_batch:.2f}s | Throughput: {throughput:,.1f} flows/sec")

        # Predict probabilities for ROC-AUC
        y_proba = model.predict_proba(X_test)

        # Single-flow latency benchmark (200 flows)
        single_times_ms = []
        for sample in single_samples:
            t_s0 = time.perf_counter()
            model.predict(sample)
            single_times_ms.append((time.perf_counter() - t_s0) * 1000.0)
        median_lat = float(np.median(single_times_ms))
        p95_lat = float(np.percentile(single_times_ms, 95))
        print(f"Single-flow latency: Median = {median_lat:.3f} ms | 95th Percentile = {p95_lat:.3f} ms")

        # 4a) 15-Class Metrics
        acc_15 = accuracy_score(y_test_15, y_pred_15)
        prec_macro_15 = precision_score(y_test_15, y_pred_15, average="macro", zero_division=0)
        rec_macro_15 = recall_score(y_test_15, y_pred_15, average="macro", zero_division=0)
        f1_macro_15 = f1_score(y_test_15, y_pred_15, average="macro", zero_division=0)
        f1_weighted_15 = f1_score(y_test_15, y_pred_15, average="weighted", zero_division=0)

        # Save per-class metrics CSV
        report_dict = classification_report(
            y_test_15, y_pred_15, target_names=class_names_15, output_dict=True, zero_division=0
        )
        per_class_records = []
        for cls_name in class_names_15:
            per_class_records.append({
                "class": cls_name,
                "precision": round(report_dict[cls_name]["precision"], 5),
                "recall": round(report_dict[cls_name]["recall"], 5),
                "f1": round(report_dict[cls_name]["f1-score"], 5),
                "support": int(report_dict[cls_name]["support"])
            })
        per_class_df = pd.DataFrame(per_class_records)
        per_class_csv_path = outputs_dir / f"{model_key}_per_class.csv"
        per_class_df.to_csv(per_class_csv_path, index=False)
        print(f"Saved per-class metrics to: {per_class_csv_path}")

        # 4b) 13-Class View
        y_pred_13 = np.array([map_15_to_13[p] for p in y_pred_15])
        f1_macro_13 = f1_score(y_test_13, y_pred_13, average="macro", zero_division=0)

        # 4c) Binary View
        y_pred_bin = (y_pred_15 != benign_id).astype(int)
        acc_bin = accuracy_score(y_test_bin, y_pred_bin)
        prec_bin = precision_score(y_test_bin, y_pred_bin, pos_label=1, zero_division=0)
        rec_bin = recall_score(y_test_bin, y_pred_bin, pos_label=1, zero_division=0)
        f1_bin = f1_score(y_test_bin, y_pred_bin, pos_label=1, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred_bin).ravel()
        fpr_benign = (fp / (fp + tn)) * 100.0

        # Score = 1 - P(BENIGN)
        attack_score = 1.0 - y_proba[:, benign_id]
        roc_auc = roc_auc_score(y_test_bin, attack_score)

        # 4d) Confusion Matrix Heatmap (Row-Normalized)
        cm = confusion_matrix(y_test_15, y_pred_15, normalize="true")
        plt.figure(figsize=(13, 11))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names_15,
            yticklabels=class_names_15,
            cbar_kws={"label": "Normalized Recall (Row Proportion)"}
        )
        plt.xlabel("Predicted Class", fontsize=11)
        plt.ylabel("True Class", fontsize=11)
        plt.title(f"15-Class Normalized Confusion Matrix: {display_name}", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        cm_path = outputs_dir / f"{model_key}_confusion.png"
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix heatmap to: {cm_path}")

        # Store record for comparison table
        comparison_results.append({
            "Model": display_name,
            "Train Time (s)": round(train_time, 2),
            "15-Class Acc": round(acc_15, 5),
            "15-Class Macro F1": round(f1_macro_15, 5),
            "15-Class Weighted F1": round(f1_weighted_15, 5),
            "13-Class Macro F1": round(f1_macro_13, 5),
            "Binary Acc": round(acc_bin, 5),
            "Attack Precision": round(prec_bin, 5),
            "Attack Recall": round(rec_bin, 5),
            "Attack F1": round(f1_bin, 5),
            "FP Count": int(fp),
            "TN Count": int(tn),
            "Benign FPR (%)": round(fpr_benign, 4),
            "ROC-AUC": round(roc_auc, 5),
            "Throughput (flows/s)": round(throughput, 1),
            "Single Median (ms)": round(median_lat, 3),
            "Single P95 (ms)": round(p95_lat, 3)
        })

    # -------------------------------------------------------------
    # 5. Summary Table & Output
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINE MODELS COMPARISON SUMMARY")
    print("=" * 70)
    comp_df = pd.DataFrame(comparison_results)
    comp_df.to_csv(comparison_csv_path, index=False)
    print(comp_df.to_string(index=False))
    print(f"\nSaved comparison summary table to: {comparison_csv_path}")

    # Print rare class recall comparison
    rare_classes = ["Heartbleed", "Infiltration", "Web Attack - Sql Injection", "Web Attack - XSS"]
    print("\n--- Rare Attack Class Recall & Support ---")
    rare_records = []
    for model_key, display_name, _ in models_config:
        df_pc = pd.read_csv(outputs_dir / f"{model_key}_per_class.csv")
        row_dict = {"Model": display_name}
        for rc in rare_classes:
            rc_row = df_pc[df_pc["class"] == rc].iloc[0]
            row_dict[f"{rc} (Recall)"] = rc_row["recall"]
            row_dict[f"{rc} (Support)"] = int(rc_row["support"])
        rare_records.append(row_dict)
    rare_df = pd.DataFrame(rare_records)
    print(rare_df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    train_and_eval_baselines()
