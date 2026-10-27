"""
Step 7: Build and Evaluate the Stacking Ensemble Classifier.

Why this matters for viva:
Stacking combines complementary model architectures (linear + bagging + boosting).
Crucially, out-of-fold (OOF) cross-validation prevents the meta-learner from overfitting
to the training predictions of the base learners, and applying SMOTE inside each fold
guarantees that synthetic samples never leak into out-of-fold validation sets.
"""

import copy
import json
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
import seaborn as sns
from imblearn.over_sampling import SMOTE
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

# Ensure src directory is on sys.path for StackedIDS import and unpickling
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


def train_stack_pipeline():
    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"
    outputs_dir = PROJECT_ROOT / "outputs"

    train_cap_path = data_dir / "train_cap.parquet"
    train_smote_path = data_dir / "train_smote.parquet"
    test_scaled_path = data_dir / "test_scaled.parquet"
    selected_features_path = models_dir / "selected_features.json"
    label_maps_path = models_dir / "label_maps.json"
    stack_model_path = models_dir / "stacked_ids.joblib"

    print("=" * 70)
    print("STEP 7: STACKING ENSEMBLE TRAINING & EVALUATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Load Data, Features, and Label Mappings
    # -------------------------------------------------------------
    print("[1/9] Loading datasets and configuration...")
    with open(selected_features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)["selected"]
    print(f"Features: {len(selected_features)} selected columns.")

    with open(label_maps_path, "r", encoding="utf-8") as f:
        label_maps = json.load(f)

    id_to_class_15 = {int(k): v for k, v in label_maps["multi15"]["id_to_class"].items()}
    class_to_id_15 = label_maps["multi15"]["class_to_id"]
    class_to_id_13 = label_maps["multi13"]["class_to_id"]
    benign_id = class_to_id_15["BENIGN"]
    class_names_15 = [id_to_class_15[i] for i in range(15)]

    web_attack_mapping = {
        "Web Attack - Brute Force": "Web Attack",
        "Web Attack - XSS": "Web Attack",
        "Web Attack - Sql Injection": "Web Attack"
    }
    map_15_to_13 = {
        i: class_to_id_13[web_attack_mapping.get(id_to_class_15[i], id_to_class_15[i])]
        for i in range(15)
    }

    train_cap = pd.read_parquet(train_cap_path, engine="pyarrow")
    test_scaled = pd.read_parquet(test_scaled_path, engine="pyarrow")
    print(f"Loaded train_cap:   {train_cap.shape[0]:,} rows (real scaled)")
    print(f"Loaded test_scaled: {test_scaled.shape[0]:,} rows")

    X_cap = train_cap[selected_features].values
    y_cap = train_cap["y_multi15"].values

    X_test = test_scaled[selected_features].values
    y_test_15 = test_scaled["y_multi15"].values
    y_test_13 = test_scaled["y_multi13"].values
    y_test_bin = test_scaled["y_binary"].values

    # -------------------------------------------------------------
    # 2. Out-of-Fold Meta-Feature Generation (5 Folds on train_cap)
    # -------------------------------------------------------------
    print("\n[2/9] Generating Out-Of-Fold (OOF) meta-features on REAL rows via 5-Fold CV...")
    print("  Base Learners: LogisticRegression, RandomForest (200), XGBoost (200)")
    print("  SMOTE strictly applied INSIDE each training fold only.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_matrix = np.full((len(train_cap), 45), np.nan, dtype=np.float32)

    total_cv_start = time.perf_counter()

    for fold_idx, (fold_tr_idx, fold_val_idx) in enumerate(skf.split(X_cap, y_cap), 1):
        fold_start = time.perf_counter()
        print(f"\n--- Fold {fold_idx} / 5 ---")

        X_f_tr, y_f_tr = X_cap[fold_tr_idx], y_cap[fold_tr_idx]
        X_f_val = X_cap[fold_val_idx]

        # Check minority classes and assert k_neighbors < min_class_count
        classes_in_fold, counts_in_fold = np.unique(y_f_tr, return_counts=True)
        counts_dict = dict(zip(classes_in_fold, counts_in_fold))

        classes_to_oversample = [c for c, cnt in counts_dict.items() if cnt < 2000]
        k_neighbors = 5
        for cls_id in classes_to_oversample:
            cnt = counts_dict[cls_id]
            assert k_neighbors < cnt, (
                f"[ASSERTION FAILED] Fold {fold_idx}: class {id_to_class_15[cls_id]} has {cnt} rows <= k_neighbors ({k_neighbors})"
            )
        print(f"  Fold {fold_idx} assertion passed: k_neighbors ({k_neighbors}) < min class size ({min(counts_in_fold)}).")

        # Apply SMOTE inside the fold
        smote_strategy = {c: 2000 for c in classes_to_oversample}
        smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=k_neighbors, random_state=42)
        X_f_tr_sm, y_f_tr_sm = smote.fit_resample(X_f_tr, y_f_tr)
        print(f"  Fold train set: {len(X_f_tr):,} raw -> {len(X_f_tr_sm):,} after SMOTE. Validation: {len(X_f_val):,} real rows.")

        # Train Base 1: Logistic Regression
        t_m = time.perf_counter()
        lr_fold = LogisticRegression(max_iter=1000, random_state=42)
        lr_fold.fit(X_f_tr_sm, y_f_tr_sm)
        p_lr = lr_fold.predict_proba(X_f_val)
        print(f"    * Logistic Regression fit + predict: {time.perf_counter() - t_m:.2f}s")

        # Train Base 2: Random Forest
        t_m = time.perf_counter()
        rf_fold = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        rf_fold.fit(X_f_tr_sm, y_f_tr_sm)
        p_rf = rf_fold.predict_proba(X_f_val)
        print(f"    * Random Forest fit + predict:       {time.perf_counter() - t_m:.2f}s")

        # Train Base 3: XGBoost
        t_m = time.perf_counter()
        xgb_fold = XGBClassifier(
            objective="multi:softprob",
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=-1,
            random_state=42
        )
        xgb_fold.fit(X_f_tr_sm, y_f_tr_sm)
        p_xgb = xgb_fold.predict_proba(X_f_val)
        print(f"    * XGBoost fit + predict:             {time.perf_counter() - t_m:.2f}s")

        # Stack fold probabilities (45 columns)
        oof_matrix[fold_val_idx] = np.hstack([p_lr, p_rf, p_xgb]).astype(np.float32)
        fold_elapsed = time.perf_counter() - fold_start
        print(f"  Fold {fold_idx} completed in: {fold_elapsed:.2f}s ({fold_elapsed / 60:.2f}m)")

    total_cv_time = time.perf_counter() - total_cv_start
    print(f"\nAll 5 folds completed in: {total_cv_time:.2f}s ({total_cv_time / 60:.2f} minutes)")

    # Assertions on OOF matrix
    assert oof_matrix.shape == (227338, 45), f"Unexpected OOF matrix shape: {oof_matrix.shape}"
    assert not np.isnan(oof_matrix).any(), "NaN found in OOF matrix!"
    assert not np.isinf(oof_matrix).any(), "Inf found in OOF matrix!"
    print("OOF Assertions passed: Shape is (227,338, 45), zero NaNs, zero Infs.")

    # -------------------------------------------------------------
    # 3. Base-Learner CV Performance Estimates
    # -------------------------------------------------------------
    print("\n[3/9] Base-Learner Out-of-Fold CV Performance Estimates (on train_cap):")
    oof_preds_lr = np.argmax(oof_matrix[:, 0:15], axis=1)
    oof_preds_rf = np.argmax(oof_matrix[:, 15:30], axis=1)
    oof_preds_xgb = np.argmax(oof_matrix[:, 30:45], axis=1)

    base_cv_records = [
        {
            "Base Learner": "Logistic Regression",
            "OOF Accuracy": round(accuracy_score(y_cap, oof_preds_lr), 5),
            "OOF Macro F1": round(f1_score(y_cap, oof_preds_lr, average="macro", zero_division=0), 5)
        },
        {
            "Base Learner": "Random Forest",
            "OOF Accuracy": round(accuracy_score(y_cap, oof_preds_rf), 5),
            "OOF Macro F1": round(f1_score(y_cap, oof_preds_rf, average="macro", zero_division=0), 5)
        },
        {
            "Base Learner": "XGBoost",
            "OOF Accuracy": round(accuracy_score(y_cap, oof_preds_xgb), 5),
            "OOF Macro F1": round(f1_score(y_cap, oof_preds_xgb, average="macro", zero_division=0), 5)
        }
    ]
    print(pd.DataFrame(base_cv_records).to_string(index=False))

    # -------------------------------------------------------------
    # 4. Meta-Learner Cross-Validation & Fitting
    # -------------------------------------------------------------
    print("\n[4/9] Evaluating and fitting Meta-Learner on full OOF matrix...")
    meta_model = LogisticRegression(max_iter=1000, random_state=42)

    # 5-fold CV estimate on meta-features
    print("Running 5-fold CV on meta-features (labeled as meta-level estimate)...")
    meta_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof_preds = cross_val_predict(meta_model, oof_matrix, y_cap, cv=meta_skf, n_jobs=-1)
    meta_cv_acc = accuracy_score(y_cap, meta_oof_preds)
    meta_cv_f1 = f1_score(y_cap, meta_oof_preds, average="macro", zero_division=0)
    print(f"  [Meta-Level 5-Fold CV Estimate] Accuracy: {meta_cv_acc:.5f} | Macro F1: {meta_cv_f1:.5f}")

    # Fit meta-learner on full OOF matrix
    print("Fitting Meta-Learner on the complete (227,338, 45) OOF matrix...")
    t_meta_fit = time.perf_counter()
    meta_model.fit(oof_matrix, y_cap)
    print(f"Meta-Learner fit completed in: {time.perf_counter() - t_meta_fit:.2f}s")

    # -------------------------------------------------------------
    # 5. Build and Save StackedIDS Model
    # -------------------------------------------------------------
    print("\n[5/9] Assembling final StackedIDS pipeline...")
    # Load full base models trained on full train_smote from Step 6
    lr_full = joblib.load(models_dir / "logistic_regression.joblib")
    rf_full = joblib.load(models_dir / "random_forest.joblib")
    xgb_full = joblib.load(models_dir / "xgboost.joblib")

    for name, m in [("LR", lr_full), ("RF", rf_full), ("XGB", xgb_full)]:
        assert hasattr(m, "classes_"), f"Model {name} missing classes_ attribute"
        assert np.array_equal(m.classes_, np.arange(15)), f"Model {name} classes_ mismatch: {m.classes_}"
    print("  [PASS] All 3 base models loaded from Step 6 with valid classes_ [0..14].")

    stacked_ensemble = StackedIDS(
        base_models=[lr_full, rf_full, xgb_full],
        meta_model=meta_model,
        classes_=np.arange(15),
        feature_names=selected_features
    )

    joblib.dump(stacked_ensemble, stack_model_path, compress=3)
    file_size_mb = stack_model_path.stat().st_size / (1024 ** 2)
    print(f"Saved StackedIDS ensemble to: {stack_model_path} ({file_size_mb:.2f} MB)")

    # Verify in fresh Python subprocess on 1,000 test rows
    print("Verifying saved StackedIDS in an independent, fresh Python subprocess...")
    sample_1k = X_test[:1000]
    in_process_preds = stacked_ensemble.predict(sample_1k)

    temp_sample_path = PROJECT_ROOT / "models" / "_temp_1k_sample.npy"
    np.save(temp_sample_path, sample_1k)

    subprocess_code = f"""
import sys
from pathlib import Path
import numpy as np
import joblib

project_root = Path(r'{PROJECT_ROOT}')
src_dir = project_root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

model = joblib.load(project_root / 'models' / 'stacked_ids.joblib')
sample = np.load(project_root / 'models' / '_temp_1k_sample.npy')
preds = model.predict(sample)
np.save(project_root / 'models' / '_temp_1k_preds.npy', preds)
"""
    sub_res = subprocess.run([sys.executable, "-c", subprocess_code], capture_output=True, text=True)
    if sub_res.returncode != 0:
        print("Subprocess stderr:", sub_res.stderr)
        raise RuntimeError("Subprocess verification script failed!")

    temp_preds_path = PROJECT_ROOT / "models" / "_temp_1k_preds.npy"
    subprocess_preds = np.load(temp_preds_path)
    assert np.array_equal(in_process_preds, subprocess_preds), (
        "[ASSERTION FAILED] Predictions differ between in-process and subprocess deserialization!"
    )
    print("  [PASS] Fresh subprocess test: 1,000 predictions match with 100% parity.")

    # Cleanup temp files
    temp_sample_path.unlink(missing_ok=True)
    temp_preds_path.unlink(missing_ok=True)

    # -------------------------------------------------------------
    # 6. Full Test Set Evaluation (499,596 rows)
    # -------------------------------------------------------------
    print("\n[6/9] Evaluating StackedIDS on full test set (499,596 rows)...")
    t_test_start = time.perf_counter()
    y_pred_stack = stacked_ensemble.predict(X_test)
    t_test_batch = time.perf_counter() - t_test_start
    throughput_stack = len(X_test) / t_test_batch

    y_proba_stack = stacked_ensemble.predict_proba(X_test)

    # 15-Class Metrics
    acc_15 = accuracy_score(y_test_15, y_pred_stack)
    prec_macro_15 = precision_score(y_test_15, y_pred_stack, average="macro", zero_division=0)
    rec_macro_15 = recall_score(y_test_15, y_pred_stack, average="macro", zero_division=0)
    f1_macro_15 = f1_score(y_test_15, y_pred_stack, average="macro", zero_division=0)
    f1_weighted_15 = f1_score(y_test_15, y_pred_stack, average="weighted", zero_division=0)

    # Per-class table
    report_dict = classification_report(
        y_test_15, y_pred_stack, target_names=class_names_15, output_dict=True, zero_division=0
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
    stack_per_class_df = pd.DataFrame(per_class_records)
    stack_per_class_path = outputs_dir / "stack_per_class.csv"
    stack_per_class_df.to_csv(stack_per_class_path, index=False)
    print(f"Saved per-class metrics to: {stack_per_class_path}")

    # 13-Class View
    y_pred_13_stack = np.array([map_15_to_13[p] for p in y_pred_stack])
    f1_macro_13 = f1_score(y_test_13, y_pred_13_stack, average="macro", zero_division=0)

    # Binary View
    y_pred_bin_stack = (y_pred_stack != benign_id).astype(int)
    acc_bin = accuracy_score(y_test_bin, y_pred_bin_stack)
    prec_bin = precision_score(y_test_bin, y_pred_bin_stack, pos_label=1, zero_division=0)
    rec_bin = recall_score(y_test_bin, y_pred_bin_stack, pos_label=1, zero_division=0)
    f1_bin = f1_score(y_test_bin, y_pred_bin_stack, pos_label=1, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred_bin_stack).ravel()
    fpr_benign = (fp / (fp + tn)) * 100.0

    attack_score_stack = 1.0 - y_proba_stack[:, benign_id]
    roc_auc_stack = roc_auc_score(y_test_bin, attack_score_stack)

    # Confusion matrix heatmap
    cm_stack = confusion_matrix(y_test_15, y_pred_stack, normalize="true")
    plt.figure(figsize=(13, 11))
    sns.heatmap(
        cm_stack,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names_15,
        yticklabels=class_names_15,
        cbar_kws={"label": "Normalized Recall (Row Proportion)"}
    )
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.title("15-Class Normalized Confusion Matrix: Stacking Ensemble", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_path = outputs_dir / "stack_confusion.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to: {cm_path}")

    # -------------------------------------------------------------
    # 7. Latency Benchmarks
    # -------------------------------------------------------------
    print("\n[7/9] Latency Benchmarking (Batch Throughput & Single-Flow Latency over 200 rows)...")
    np.random.seed(42)
    single_indices = np.random.choice(len(X_test), size=200, replace=False)
    single_samples = [X_test[i:i + 1] for i in single_indices]

    # Measure Stack single-flow latency
    stack_single_times = []
    for s in single_samples:
        t0 = time.perf_counter()
        stacked_ensemble.predict(s)
        stack_single_times.append((time.perf_counter() - t0) * 1000.0)
    median_lat_stack = float(np.median(stack_single_times))
    p95_lat_stack = float(np.percentile(stack_single_times, 95))

    # Also re-measure RF and XGB single-flow latency with n_jobs=1
    rf_single = copy.deepcopy(rf_full)
    rf_single.set_params(n_jobs=1)
    xgb_single = copy.deepcopy(xgb_full)
    xgb_single.set_params(n_jobs=1)

    rf_single_times_n1 = []
    xgb_single_times_n1 = []
    for s in single_samples:
        t0 = time.perf_counter()
        rf_single.predict(s)
        rf_single_times_n1.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        xgb_single.predict(s)
        xgb_single_times_n1.append((time.perf_counter() - t0) * 1000.0)

    print(f"Stack Latency: Batch Throughput = {throughput_stack:,.1f} flows/s | Median = {median_lat_stack:.3f} ms | P95 = {p95_lat_stack:.3f} ms")
    print(f"Random Forest  (n_jobs=1): Median = {np.median(rf_single_times_n1):.3f} ms | P95 = {np.percentile(rf_single_times_n1, 95):.3f} ms")
    print(f"XGBoost        (n_jobs=1): Median = {np.median(xgb_single_times_n1):.3f} ms | P95 = {np.percentile(xgb_single_times_n1, 95):.3f} ms")

    # -------------------------------------------------------------
    # 8. Comparison Table & Rare Attack Class Recall
    # -------------------------------------------------------------
    print("\n[8/9] Compiling 4-Model Comparison Table...")
    baseline_csv_path = outputs_dir / "baseline_metrics.csv"
    if baseline_csv_path.exists():
        comp_df = pd.read_csv(baseline_csv_path)
    else:
        comp_df = pd.DataFrame()

    stack_row = {
        "Model": "Stacked Ensemble",
        "Train Time (s)": round(total_cv_time + t_meta_fit, 2),
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
        "ROC-AUC": round(roc_auc_stack, 5),
        "Throughput (flows/s)": round(throughput_stack, 1),
        "Single Median (ms)": round(median_lat_stack, 3),
        "Single P95 (ms)": round(p95_lat_stack, 3)
    }

    model_comparison_df = pd.concat([comp_df, pd.DataFrame([stack_row])], ignore_index=True)
    comparison_save_path = outputs_dir / "model_comparison.csv"
    model_comparison_df.to_csv(comparison_save_path, index=False)
    print("\n" + "=" * 70)
    print("ALL MODELS COMPARISON TABLE (4 MODELS)")
    print("=" * 70)
    print(model_comparison_df.to_string(index=False))

    # Rare classes table
    rare_classes = ["Heartbleed", "Infiltration", "Web Attack - Sql Injection", "Web Attack - XSS"]
    print("\n--- Rare Attack Class Recall & Support (All 4 Models) ---")
    model_keys = [
        ("logistic_regression", "Logistic Regression"),
        ("random_forest", "Random Forest"),
        ("xgboost", "XGBoost"),
        ("stack", "Stacked Ensemble")
    ]
    rare_rows = []
    for m_key, m_disp in model_keys:
        pc_path = outputs_dir / f"{m_key}_per_class.csv"
        df_pc = pd.read_csv(pc_path)
        r_dict = {"Model": m_disp}
        for rc in rare_classes:
            row_match = df_pc[df_pc["class"] == rc].iloc[0]
            r_dict[f"{rc} (Recall)"] = row_match["recall"]
            r_dict[f"{rc} (Support)"] = int(row_match["support"])
        rare_rows.append(r_dict)
    rare_df = pd.DataFrame(rare_rows)
    print(rare_df.to_string(index=False))

    # -------------------------------------------------------------
    # 9. Comparison: Stack vs XGBoost Differences
    # -------------------------------------------------------------
    print("\n[9/9] Difference Analysis: Stack vs XGBoost Predictions...")
    xgb_preds = xgb_full.predict(X_test)
    diff_mask = (y_pred_stack != xgb_preds)
    num_diff = int(diff_mask.sum())
    pct_diff = (num_diff / len(X_test)) * 100
    print(f"Total test flows where Stack and XGBoost disagree: {num_diff:,} out of {len(X_test):,} ({pct_diff:.3f}%)")
    print("=" * 70)


if __name__ == "__main__":
    train_stack_pipeline()
