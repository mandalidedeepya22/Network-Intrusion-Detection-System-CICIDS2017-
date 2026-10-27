"""
Step 8: In-Depth Diagnostics and Comprehensive Ablation Study.

Why this matters for viva:
Ablation isolates the individual contribution of each pipeline component (SMOTE resampling,
Boruta feature selection, port-specific features, and model architectures) under multi-seed
validation. Paired bootstrapping quantifies test-set sampling uncertainty, while fine-grained
web attack confusion matrices reveal the exact decision boundaries between related attack vectors.
"""

import copy
import gc
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
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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


def run_diagnostics_and_ablation():
    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STEP 8: DIAGNOSTICS, PAIRED BOOTSTRAP & SYSTEMATIC ABLATION STUDY")
    print("=" * 75)

    # Load label maps and selected features
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

    # Load test set
    test_scaled = pd.read_parquet(data_dir / "test_scaled.parquet", engine="pyarrow")
    y_test_15 = test_scaled["y_multi15"].values
    y_test_13 = test_scaled["y_multi13"].values
    y_test_bin = test_scaled["y_binary"].values
    X_test_65 = test_scaled[selected_features].values
    n_test = len(y_test_15)

    # =============================================================
    # PART A - WEB-ATTACK CONFUSION ANALYSIS
    # =============================================================
    print("\n" + "=" * 75)
    print("PART A: WEB-ATTACK CONFUSION & FINE-GRAINED ANALYSIS (NO TRAINING)")
    print("=" * 75)

    print("Loading pre-trained baseline and ensemble models...")
    lr = joblib.load(models_dir / "logistic_regression.joblib")
    rf = joblib.load(models_dir / "random_forest.joblib")
    xgb = joblib.load(models_dir / "xgboost.joblib")
    stack = joblib.load(models_dir / "stacked_ids.joblib")

    print(f"Generating predictions on full test set ({n_test:,} flows)...")
    preds_lr = lr.predict(X_test_65)
    preds_rf = rf.predict(X_test_65)
    preds_xgb = xgb.predict(X_test_65)
    preds_stack = stack.predict(X_test_65)

    # Cache predictions to outputs/test_predictions.npz
    preds_cache_path = outputs_dir / "test_predictions.npz"
    np.savez_compressed(
        preds_cache_path,
        lr=preds_lr,
        rf=preds_rf,
        xgb=preds_xgb,
        stack=preds_stack
    )
    print(f"Cached all 4 model predictions to: {preds_cache_path}")

    web_classes = ["Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection"]
    models_dict = {
        "Logistic Regression": preds_lr,
        "Random Forest": preds_rf,
        "XGBoost": preds_xgb,
        "Stacked Ensemble": preds_stack
    }

    for model_name, preds in models_dict.items():
        cm = confusion_matrix(y_test_15, preds, labels=np.arange(15))
        print(f"\n--- {model_name} Confusion Matrix Rows (Raw Non-Zero Counts) ---")
        for cls_name in web_classes:
            cid = class_to_id_15[cls_name]
            row = cm[cid]
            total_true = row.sum()
            non_zero_cells = [(class_names_15[j], int(row[j])) for j in range(15) if row[j] > 0]
            print(f"  * True '{cls_name}' (Total: {total_true} rows):")
            for pred_name, cnt in non_zero_cells:
                pct = (cnt / total_true) * 100
                print(f"      -> Predicted '{pred_name:<26}': {cnt:>4} flows ({pct:>6.2f}%)")

    # Precision, Recall, and F1 Table for Web Attacks
    print("\n--- Web Attack Family Metrics Across Models ---")
    web_metrics = []
    for model_name, preds in models_dict.items():
        cm = confusion_matrix(y_test_15, preds, labels=np.arange(15))
        for cls_name in web_classes:
            cid = class_to_id_15[cls_name]
            tp = cm[cid, cid]
            fp = cm[:, cid].sum() - tp
            fn = cm[cid, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            web_metrics.append({
                "Model": model_name,
                "Class": cls_name,
                "Precision": round(prec, 5),
                "Recall": round(rec, 5),
                "F1": round(f1, 5),
                "Support": int(cm[cid, :].sum())
            })
    print(pd.DataFrame(web_metrics).to_string(index=False))

    # =============================================================
    # PART B - PAIRED BOOTSTRAP ON THE TEST SET
    # =============================================================
    print("\n" + "=" * 75)
    print("PART B: PAIRED BOOTSTRAP STATISTICAL SIGNIFICANCE (1,000 RESAMPLES)")
    print("=" * 75)
    print("Note: This bootstrap captures test-set sampling noise only, not training randomness.")
    print("      Classes with 2-7 test rows (e.g. Heartbleed=2, SQLi=4, Infiltration=7) make macro F1 sensitive to sampling variation.")

    n_boot = 1000
    rng = np.random.default_rng(42)

    diff_macro15 = np.empty(n_boot, dtype=np.float64)
    diff_macro13 = np.empty(n_boot, dtype=np.float64)
    diff_fpr = np.empty(n_boot, dtype=np.float64)

    t_b0 = time.perf_counter()

    for b in range(n_boot):
        boot_idx = rng.integers(0, n_test, size=n_test)
        yt = y_test_15[boot_idx]
        yt_13 = y_test_13[boot_idx]

        yp_stack = preds_stack[boot_idx]
        yp_xgb = preds_xgb[boot_idx]

        # 15-class fast CM via np.bincount
        cm_stack = np.bincount(yt * 15 + yp_stack, minlength=225).reshape(15, 15)
        cm_xgb = np.bincount(yt * 15 + yp_xgb, minlength=225).reshape(15, 15)

        # 15-class macro F1 (absent classes get F1 = 0)
        def get_macro_f1(cm):
            tp = np.diag(cm).astype(np.float64)
            pred_sum = cm.sum(axis=0).astype(np.float64)
            true_sum = cm.sum(axis=1).astype(np.float64)
            prec = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=(pred_sum > 0))
            rec = np.divide(tp, true_sum, out=np.zeros_like(tp), where=(true_sum > 0))
            pr_sum = prec + rec
            f1 = np.divide(2 * prec * rec, pr_sum, out=np.zeros_like(tp), where=(pr_sum > 0))
            return float(np.mean(f1))

        f1_15_stack = get_macro_f1(cm_stack)
        f1_15_xgb = get_macro_f1(cm_xgb)
        diff_macro15[b] = f1_15_stack - f1_15_xgb

        # 13-class fast CM
        yp_13_stack = map_15_to_13[yp_stack]
        yp_13_xgb = map_15_to_13[yp_xgb]
        cm13_stack = np.bincount(yt_13 * 13 + yp_13_stack, minlength=169).reshape(13, 13)
        cm13_xgb = np.bincount(yt_13 * 13 + yp_13_xgb, minlength=169).reshape(13, 13)

        f1_13_stack = get_macro_f1(cm13_stack)
        f1_13_xgb = get_macro_f1(cm13_xgb)
        diff_macro13[b] = f1_13_stack - f1_13_xgb

        # Benign FPR (class 0)
        fpr_stack = cm_stack[0, 1:].sum() / cm_stack[0].sum()
        fpr_xgb = cm_xgb[0, 1:].sum() / cm_xgb[0].sum()
        diff_fpr[b] = fpr_stack - fpr_xgb

    boot_time = time.perf_counter() - t_b0
    print(f"Completed 1,000 paired bootstrap iterations in: {boot_time:.2f} seconds")

    def print_boot_metric(name, diffs, lower_is_better=False):
        mean_d = float(np.mean(diffs))
        ci_lower = float(np.percentile(diffs, 2.5))
        ci_upper = float(np.percentile(diffs, 97.5))
        frac_win = float(np.mean(diffs > 0) if not lower_is_better else np.mean(diffs < 0))
        print(f"\n  * {name}:")
        print(f"      Mean Difference (Stack - XGBoost): {mean_d:+.6f}")
        print(f"      95% Percentile Confidence Interval: [{ci_lower:+.6f}, {ci_upper:+.6f}]")
        print(f"      Fraction of Resamples where Stack is Superior: {frac_win * 100:.2f}%")

    print_boot_metric("15-Class Macro F1 (Stack minus XGBoost)", diff_macro15)
    print_boot_metric("13-Class Macro F1 (Stack minus XGBoost)", diff_macro13)
    print_boot_metric("Benign False-Positive Rate (Stack minus XGBoost)", diff_fpr, lower_is_better=True)

    # =============================================================
    # PART C - FAIR LATENCY BENCHMARK FOR THE STACK
    # =============================================================
    print("\n" + "=" * 75)
    print("PART C: FAIR LATENCY BENCHMARK (n_jobs=1 SINGLE-THREADED PREDICTION)")
    print("=" * 75)
    print("Deep-copying base models with n_jobs=1 to isolate true CPU single-thread predict_proba latency...")

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

    models_latency_test = [
        ("Logistic Regression", lr_fair),
        ("Random Forest (n_jobs=1)", rf_fair),
        ("XGBoost (n_jobs=1)", xgb_fair),
        ("Full Stacked Ensemble (fair)", stack_fair)
    ]

    latency_records = []
    for m_name, model_obj in models_latency_test:
        latencies = []
        for s in single_samples:
            t0 = time.perf_counter()
            model_obj.predict_proba(s)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        med = float(np.median(latencies))
        p95 = float(np.percentile(latencies, 95))
        latency_records.append({
            "Component / Model": m_name,
            "Median Latency (ms)": round(med, 4),
            "95th Percentile (ms)": round(p95, 4)
        })

    print(pd.DataFrame(latency_records).to_string(index=False))

    # =============================================================
    # PART D - SYSTEMATIC ABLATION STUDY WITH XGBOOST
    # =============================================================
    print("\n" + "=" * 75)
    print("PART D: SYSTEMATIC ABLATION STUDY (XGBOOST, 5 VARIANTS x 3 SEEDS)")
    print("=" * 75)
    print("Fixed XGBoost Params: 200 trees, max_depth=8, lr=0.1, tree_method='hist', multi:softprob, n_jobs=-1")
    print("SMOTE Recipe: classes < 2000 oversampled to 2000, k_neighbors=5, applied on 50k capped training set.")

    train_cap = pd.read_parquet(data_dir / "train_cap.parquet", engine="pyarrow")
    y_cap = train_cap["y_multi15"].values
    seeds = [42, 43, 44]

    # Precompute Top-20 features from Step 6 XGBoost model
    xgb_step6 = joblib.load(models_dir / "xgboost.joblib")
    importances = xgb_step6.feature_importances_
    top20_indices = np.argsort(importances)[::-1][:20]
    top20_features = [selected_features[i] for i in top20_indices]
    print("\nTop 20 Features from Step 6 XGBoost:")
    for rank, f_name in enumerate(top20_features, 1):
        print(f"  {rank:>2}. {f_name:<30} (MDI Importance: {importances[selected_features.index(f_name)]:.5f})")

    # Precompute V3 (All 69 Features) scaled dataset
    print("\nPreparing V3 dataset (All 69 Features from train.parquet & test.parquet)...")
    train_full = pd.read_parquet(data_dir / "train.parquet", engine="pyarrow")
    test_full = pd.read_parquet(data_dir / "test.parquet", engine="pyarrow")
    feature_cols_69 = [c for c in train_full.columns if c not in ["Label", "y_binary", "y_multi15", "y_multi13"]]
    print(f"Found {len(feature_cols_69)} numeric features in raw train dataset.")

    scaler_69 = StandardScaler()
    scaler_69.fit(train_full[feature_cols_69].values)
    X_test_69 = scaler_69.transform(test_full[feature_cols_69].values).astype(np.float32)

    # Replicate exact train_cap row selection (50k per class, random_state=42)
    cap_indices = []
    classes_sorted = sorted(np.unique(train_full["y_multi15"]))
    for cls_id in classes_sorted:
        idx_c = np.where(train_full["y_multi15"].values == cls_id)[0]
        if len(idx_c) > 50000:
            rng_c = np.random.default_rng(42)
            idx_c = rng_c.choice(idx_c, size=50000, replace=False)
        cap_indices.extend(idx_c)
    cap_indices = np.array(cap_indices)
    X_cap_69 = scaler_69.transform(train_full[feature_cols_69].iloc[cap_indices].values).astype(np.float32)
    y_cap_69 = train_full["y_multi15"].iloc[cap_indices].values

    del train_full, test_full
    gc.collect()
    print(f"V3 Prepared: X_cap_69 shape: {X_cap_69.shape}, X_test_69 shape: {X_test_69.shape}")

    # Define the 5 Variants
    # Each variant provides: (name, description, uses_smote, get_data_func)
    variants = [
        ("V1_baseline", "V1: Baseline (65 selected features, capped, SMOTE)", True),
        ("V2_no_smote", "V2: No SMOTE (65 selected features, capped only)", False),
        ("V3_all_69", "V3: All 69 Features (No Boruta, capped, SMOTE)", True),
        ("V4_top_20", "V4: Top 20 Features (from XGBoost importances, SMOTE)", True),
        ("V5_no_dest_port", "V5: Without Destination Port (64 features, SMOTE)", True)
    ]

    all_run_records = []
    all_per_class_f1_records = []

    for v_code, v_name, uses_smote in variants:
        print("\n" + "-" * 70)
        print(f"STARTING ABLATION: {v_name}")
        print("-" * 70)

        # Select feature subsets
        if v_code == "V1_baseline" or v_code == "V2_no_smote":
            X_tr_base = train_cap[selected_features].values
            X_te_base = X_test_65
            y_tr_base = y_cap
        elif v_code == "V3_all_69":
            X_tr_base = X_cap_69
            X_te_base = X_test_69
            y_tr_base = y_cap_69
        elif v_code == "V4_top_20":
            X_tr_base = train_cap[top20_features].values
            X_te_base = test_scaled[top20_features].values
            y_tr_base = y_cap
        elif v_code == "V5_no_dest_port":
            feats_no_port = [c for c in selected_features if c != "Destination Port"]
            X_tr_base = train_cap[feats_no_port].values
            X_te_base = test_scaled[feats_no_port].values
            y_tr_base = y_cap

        for seed in seeds:
            t_run_0 = time.perf_counter()

            # Resample with SMOTE if applicable
            if uses_smote:
                classes_in_train, counts_in_train = np.unique(y_tr_base, return_counts=True)
                counts_dict = dict(zip(classes_in_train, counts_in_train))
                oversample_dict = {c: 2000 for c, cnt in counts_dict.items() if cnt < 2000}

                # Assertion: k_neighbors < class size
                for c, cnt in counts_dict.items():
                    if cnt < 2000:
                        assert 5 < cnt, f"[ASSERTION FAILED] Class {id_to_class_15[c]} has {cnt} rows <= k_neighbors (5)"

                smote = SMOTE(sampling_strategy=oversample_dict, k_neighbors=5, random_state=seed)
                X_fit, y_fit = smote.fit_resample(X_tr_base, y_tr_base)
            else:
                X_fit, y_fit = X_tr_base, y_tr_base

            # Train XGBoost
            xgb_model = XGBClassifier(
                objective="multi:softprob",
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                tree_method="hist",
                n_jobs=-1,
                random_state=seed
            )
            xgb_model.fit(X_fit, y_fit)
            train_time = time.perf_counter() - t_run_0

            # Evaluate on full test set
            y_pred = xgb_model.predict(X_te_base)

            acc_15 = accuracy_score(y_test_15, y_pred)
            f1_15 = f1_score(y_test_15, y_pred, average="macro", zero_division=0)

            # 13-class view
            y_pred_13 = map_15_to_13[y_pred]
            f1_13 = f1_score(y_test_13, y_pred_13, average="macro", zero_division=0)

            # Binary view
            y_pred_bin = (y_pred != benign_id).astype(int)
            att_recall = recall_score(y_test_bin, y_pred_bin, pos_label=1, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_test_bin, y_pred_bin).ravel()
            benign_fpr = (fp / (fp + tn)) * 100.0

            # Per-class F1
            cm_run = confusion_matrix(y_test_15, y_pred, labels=np.arange(15))
            tp_diag = np.diag(cm_run).astype(np.float64)
            pred_sum = cm_run.sum(axis=0).astype(np.float64)
            true_sum = cm_run.sum(axis=1).astype(np.float64)
            prec_arr = np.divide(tp_diag, pred_sum, out=np.zeros_like(tp_diag), where=(pred_sum > 0))
            rec_arr = np.divide(tp_diag, true_sum, out=np.zeros_like(tp_diag), where=(true_sum > 0))
            pr_sum_arr = prec_arr + rec_arr
            f1_per_class = np.divide(2 * prec_arr * rec_arr, pr_sum_arr, out=np.zeros_like(tp_diag), where=(pr_sum_arr > 0))

            run_record = {
                "Variant": v_code,
                "Description": v_name,
                "Seed": seed,
                "Train Time (s)": round(train_time, 2),
                "15-Class Acc": round(acc_15, 5),
                "15-Class Macro F1": round(f1_15, 5),
                "13-Class Macro F1": round(f1_13, 5),
                "Attack Recall": round(att_recall, 5),
                "Benign FPR (%)": round(benign_fpr, 4),
                "FP Count": int(fp),
                "TN Count": int(tn)
            }
            all_run_records.append(run_record)

            per_class_record = {"Variant": v_code, "Seed": seed}
            for cls_idx, c_name in enumerate(class_names_15):
                per_class_record[c_name] = round(float(f1_per_class[cls_idx]), 5)
            all_per_class_f1_records.append(per_class_record)

            print(f"  [{v_code} | Seed {seed}] Done ({train_time:.1f}s) -> 15-Class Acc: {acc_15:.5f} | 15-Class Macro F1: {f1_15:.5f} | 13-Class Macro F1: {f1_13:.5f} | FPR: {benign_fpr:.4f}% ({fp} FPs)")

            # Check reproducibility for V1 seed 42
            if v_code == "V1_baseline" and seed == 42:
                step6_acc = 0.99839
                step6_f1 = 0.86609
                acc_diff = abs(acc_15 - step6_acc)
                f1_diff = abs(f1_15 - step6_f1)
                print(f"\n  >>> REPRODUCIBILITY CHECK (V1 Seed 42 vs Step 6 XGBoost):")
                print(f"      Step 6 XGBoost : Accuracy = {step6_acc:.5f}, Macro F1 = {step6_f1:.5f}")
                print(f"      Rebuilt V1 (42): Accuracy = {acc_15:.5f}, Macro F1 = {f1_15:.5f}")
                print(f"      Delta          : ΔAcc = {acc_diff:.5f}, ΔF1 = {f1_diff:.5f}")
                if f1_diff > 0.001:
                    print(f"      [WARNING] Macro F1 differs by more than 0.001 ({f1_diff:.5f})!")
                else:
                    print("      [PASS] Rebuilt V1 matches Step 6 XGBoost within 0.001 parity.")
                print()

    # Save individual runs and per-class tables
    runs_df = pd.DataFrame(all_run_records)
    runs_path = outputs_dir / "ablation_runs.csv"
    runs_df.to_csv(runs_path, index=False)
    print(f"\nSaved all 15 ablation runs to: {runs_path}")

    per_class_df = pd.DataFrame(all_per_class_f1_records)
    per_class_path = outputs_dir / "ablation_per_class.csv"
    per_class_df.to_csv(per_class_path, index=False)
    print(f"Saved ablation per-class F1 metrics to: {per_class_path}")

    # Compute Aggregated Summary (Mean ± Std over 3 seeds)
    summary_records = []
    for v_code, v_name, _ in variants:
        v_runs = runs_df[runs_df["Variant"] == v_code]
        summary_records.append({
            "Variant": v_code,
            "Description": v_name,
            "15-Class Acc (Mean)": round(v_runs["15-Class Acc"].mean(), 5),
            "15-Class Acc (Std)": round(v_runs["15-Class Acc"].std(), 5),
            "15-Class Macro F1 (Mean)": round(v_runs["15-Class Macro F1"].mean(), 5),
            "15-Class Macro F1 (Std)": round(v_runs["15-Class Macro F1"].std(), 5),
            "13-Class Macro F1 (Mean)": round(v_runs["13-Class Macro F1"].mean(), 5),
            "13-Class Macro F1 (Std)": round(v_runs["13-Class Macro F1"].std(), 5),
            "Attack Recall (Mean)": round(v_runs["Attack Recall"].mean(), 5),
            "Attack Recall (Std)": round(v_runs["Attack Recall"].std(), 5),
            "Benign FPR (%) (Mean)": round(v_runs["Benign FPR (%)"].mean(), 4),
            "Benign FPR (%) (Std)": round(v_runs["Benign FPR (%)"].std(), 4),
            "Avg Train Time (s)": round(v_runs["Train Time (s)"].mean(), 1)
        })

    summary_df = pd.DataFrame(summary_records)
    summary_path = outputs_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved aggregated ablation summary to: {summary_path}")

    print("\n" + "=" * 75)
    print("ABLATION STUDY AGGREGATED SUMMARY (MEAN ± STD OVER 3 SEEDS)")
    print("=" * 75)
    print(summary_df.to_string(index=False))

    # Generate Bar Chart with Error Bars for 15-Class Macro F1
    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(variants))
    means = summary_df["15-Class Macro F1 (Mean)"].values
    stds = summary_df["15-Class Macro F1 (Std)"].values
    labels = [v[0].replace("_", "\n") for v in variants]

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    bars = plt.bar(x_positions, means, yerr=stds, capsize=6, color=colors, edgecolor="black", alpha=0.88, width=0.55)

    plt.ylim([max(0.70, float(min(means - stds) - 0.05)), min(1.0, float(max(means + stds) + 0.03))])
    plt.xticks(x_positions, labels, fontsize=10, fontweight="bold")
    plt.ylabel("15-Class Macro F1 Score", fontsize=11, fontweight="bold")
    plt.title("Ablation Study: 15-Class Macro F1 Across Variants (Mean ± 1 Std Dev)", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Add value annotations on top of bars
    for bar, m, s in zip(bars, means, stds):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + s + 0.005,
            f"{m:.4f}\n(±{s:.4f})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    plt.tight_layout()
    plot_path = outputs_dir / "ablation_macro_f1.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved ablation bar chart to: {plot_path}")
    print("=" * 75)
    print("STEP 8 COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    run_diagnostics_and_ablation()
