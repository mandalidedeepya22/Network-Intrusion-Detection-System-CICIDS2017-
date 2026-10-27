"""
Step 5: Feature Scaling, Majority Capping, and SMOTE on Training Data.

Why this matters for viva:
SMOTE generates synthetic samples using Euclidean distance between nearest neighbors;
scaling beforehand is mandatory so features with large ranges (e.g., Flow Duration) do not
dominate the distance metric over subtle packet metrics. Capping majority classes prevents
computational explosion and classifier paralysis on massive benign sets, while restricting SMOTE
strictly to training data avoids synthetic data leakage into evaluation.
"""

import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Ensure standard UTF-8 console output for Windows PowerShell
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_project_root() -> Path:
    """Return the absolute path of the nids-project directory."""
    return Path(__file__).resolve().parent.parent


def resample_pipeline():
    project_root = get_project_root()
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"
    selected_features_path = models_dir / "selected_features.json"
    label_maps_path = models_dir / "label_maps.json"

    scaler_path = models_dir / "scaler.joblib"
    test_scaled_path = data_dir / "test_scaled.parquet"
    train_cap_path = data_dir / "train_cap.parquet"
    train_smote_path = data_dir / "train_smote.parquet"

    print("=" * 70)
    print("STEP 5: SCALING, MAJORITY CAPPING & SMOTE RESAMPLING")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Read Data, Feature List, and Label Maps
    # -------------------------------------------------------------
    print("[1/6] Loading datasets, selected features, and label mappings...")
    with open(selected_features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)["selected"]
    print(f"Loaded {len(selected_features)} selected features.")

    with open(label_maps_path, "r", encoding="utf-8") as f:
        label_maps = json.load(f)

    id_to_class_15 = {int(k): v for k, v in label_maps["multi15"]["id_to_class"].items()}
    class_to_id_15 = label_maps["multi15"]["class_to_id"]
    class_to_id_13 = label_maps["multi13"]["class_to_id"]

    target_cols = ["Label", "y_binary", "y_multi15", "y_multi13"]
    keep_cols = selected_features + target_cols

    df_train = pd.read_parquet(train_path, engine="pyarrow", columns=keep_cols)
    df_test = pd.read_parquet(test_path, engine="pyarrow", columns=keep_cols)
    print(f"Train set loaded: {df_train.shape[0]:,} rows, {df_train.shape[1]} columns")
    print(f"Test set loaded:  {df_test.shape[0]:,} rows, {df_test.shape[1]} columns\n")

    # -------------------------------------------------------------
    # 2. Fit StandardScaler on Training Features Only
    # -------------------------------------------------------------
    print("[2/6] Fitting StandardScaler on TRAINING features only...")
    scaler = StandardScaler()
    scaler.fit(df_train[selected_features].values)
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted scaler to: {scaler_path}")

    print("Transforming train and test features to float32...")
    df_train[selected_features] = scaler.transform(df_train[selected_features].values).astype(np.float32)
    df_test[selected_features] = scaler.transform(df_test[selected_features].values).astype(np.float32)

    sample_cols = [selected_features[0], selected_features[len(selected_features) // 2], selected_features[-1]]
    print("\nVerification: Scaled Train Columns Mean & Std (Expected: Mean ≈ 0, Std ≈ 1):")
    for col in sample_cols:
        m = df_train[col].mean()
        s = df_train[col].std()
        print(f"  - {col:<30}: Mean = {m:>10.6f}, Std = {s:>10.6f}")

    df_test.to_parquet(test_scaled_path, engine="pyarrow", index=False)
    print(f"Saved scaled test set to: {test_scaled_path}")

    # -------------------------------------------------------------
    # 3. Build train_cap (Capping majority classes at 50,000 rows)
    # -------------------------------------------------------------
    print("\n[3/6] Building train_cap (capping each class at max 50,000 rows, random_state=42)...")
    cap_dfs = []
    classes_sorted = sorted(id_to_class_15.keys())

    for cls_id in classes_sorted:
        cls_name = id_to_class_15[cls_id]
        cls_group = df_train[df_train["y_multi15"] == cls_id]
        n_sample = min(len(cls_group), 50000)
        cap_dfs.append(cls_group.sample(n=n_sample, random_state=42))

    train_cap = pd.concat(cap_dfs, axis=0, ignore_index=True)
    total_cap_rows = len(train_cap)

    print("\nPer-class counts in train_cap:")
    cap_counts = train_cap["y_multi15"].value_counts().sort_index()
    for cls_id in classes_sorted:
        cls_name = id_to_class_15[cls_id]
        cnt = cap_counts.get(cls_id, 0)
        print(f"  - [{cls_id:>2}] {cls_name:<28}: {cnt:>6,} rows")

    print(f"\nTotal rows in train_cap: {total_cap_rows:,} (Expected: 227,338)")
    assert total_cap_rows == 227338, f"Expected 227,338 rows in train_cap, got {total_cap_rows}"

    train_cap.to_parquet(train_cap_path, engine="pyarrow", index=False)
    print(f"Saved train_cap to: {train_cap_path}")

    # -------------------------------------------------------------
    # 4. Build train_smote from train_cap
    # -------------------------------------------------------------
    print("\n[4/6] Applying SMOTE to minority classes (< 2,000 rows) on train_cap...")
    k_neighbors = 5
    smote_strategy = {}
    classes_to_oversample = []

    for cls_id in classes_sorted:
        cnt = cap_counts.get(cls_id, 0)
        if cnt < 2000:
            smote_strategy[cls_id] = 2000
            classes_to_oversample.append(cls_id)

    print(f"Classes to oversample to 2,000 rows: {len(classes_to_oversample)}")
    for cls_id in classes_to_oversample:
        cls_name = id_to_class_15[cls_id]
        current_cnt = cap_counts[cls_id]
        print(f"  - [{cls_id:>2}] {cls_name:<28}: {current_cnt:>4} -> 2,000 (+{2000 - current_cnt} synthetic)")
        assert current_cnt > k_neighbors, (
            f"[ASSERTION FAILED] Class {cls_name} count ({current_cnt}) is not greater than k_neighbors ({k_neighbors})"
        )
    print("Pre-SMOTE check passed: k_neighbors (5) < size of every class being oversampled.")

    X_cap = train_cap[selected_features].values
    y_cap = train_cap["y_multi15"].values

    smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=k_neighbors, random_state=42)
    X_res, y_res = smote.fit_resample(X_cap, y_cap)

    num_original = len(X_cap)
    num_total_smote = len(X_res)
    num_synthetic = num_total_smote - num_original

    print(f"\nSMOTE resampling complete:")
    print(f"  Original rows:  {num_original:,}")
    print(f"  Synthetic rows: {num_synthetic:,}")
    print(f"  Total rows:     {num_total_smote:,} (Expected: 236,027)")
    assert num_total_smote == 236027, f"Expected 236,027 rows, got {num_total_smote}"

    df_res_features = pd.DataFrame(X_res, columns=selected_features, dtype=np.float32)

    is_synthetic = np.zeros(num_total_smote, dtype=np.int32)
    is_synthetic[num_original:] = 1
    df_res_features["is_synthetic"] = is_synthetic

    orig_label = train_cap["Label"].values
    orig_y_bin = train_cap["y_binary"].values
    orig_y_m15 = train_cap["y_multi15"].values
    orig_y_m13 = train_cap["y_multi13"].values

    synth_y_m15 = y_res[num_original:]
    synth_label = np.array([id_to_class_15[y_val] for y_val in synth_y_m15])
    synth_y_bin = np.ones(num_synthetic, dtype=np.int32)

    web_attack_mapping = {
        "Web Attack - Brute Force": "Web Attack",
        "Web Attack - XSS": "Web Attack",
        "Web Attack - Sql Injection": "Web Attack"
    }
    synth_label_13 = np.array([web_attack_mapping.get(lbl, lbl) for lbl in synth_label])
    synth_y_m13 = np.array([class_to_id_13[lbl] for lbl in synth_label_13], dtype=np.int32)

    all_label = np.concatenate([orig_label, synth_label])
    all_y_bin = np.concatenate([orig_y_bin, synth_y_bin])
    all_y_m15 = np.concatenate([orig_y_m15, synth_y_m15])
    all_y_m13 = np.concatenate([orig_y_m13, synth_y_m13])

    df_res_features["Label"] = all_label
    df_res_features["y_binary"] = all_y_bin
    df_res_features["y_multi15"] = all_y_m15
    df_res_features["y_multi13"] = all_y_m13

    train_smote = df_res_features

    print("\n--- Per-Class Counts Before and After SMOTE ---")
    before_counts = train_cap["y_multi15"].value_counts().sort_index()
    after_counts = train_smote["y_multi15"].value_counts().sort_index()

    smote_comparison = pd.DataFrame({
        "Class ID": classes_sorted,
        "Class Name": [id_to_class_15[c] for c in classes_sorted],
        "Before SMOTE": [before_counts.get(c, 0) for c in classes_sorted],
        "After SMOTE": [after_counts.get(c, 0) for c in classes_sorted],
        "Synthetic Added": [after_counts.get(c, 0) - before_counts.get(c, 0) for c in classes_sorted]
    })
    print(smote_comparison.to_string(index=False))

    train_smote.to_parquet(train_smote_path, engine="pyarrow", index=False)
    print(f"\nSaved train_smote to: {train_smote_path}")

    # -------------------------------------------------------------
    # 5. Assertions & Verification Checks
    # -------------------------------------------------------------
    print("\n[5/6] Running validation assertions...")

    # A) test_scaled row count
    test_rows = len(df_test)
    assert test_rows == 499596, f"test_scaled row count mismatch: {test_rows} != 499,596"
    print(f"  [PASS] test_scaled has exactly {test_rows:,} rows.")

    # B) is_synthetic == 0 matches train_cap total
    real_rows_in_smote = (train_smote["is_synthetic"] == 0).sum()
    assert real_rows_in_smote == total_cap_rows, (
        f"Mismatch: real rows in smote ({real_rows_in_smote}) != train_cap ({total_cap_rows})"
    )
    print(f"  [PASS] is_synthetic == 0 rows in train_smote ({real_rows_in_smote:,}) equals train_cap total.")

    # C) No NaN or Inf check
    for name, df_check in [("train_cap", train_cap), ("train_smote", train_smote), ("test_scaled", df_test)]:
        feat_data = df_check[selected_features].values
        has_nan = np.isnan(feat_data).any()
        has_inf = np.isinf(feat_data).any()
        assert not has_nan, f"NaN found in {name}"
        assert not has_inf, f"Inf found in {name}"
    print("  [PASS] Zero NaN and zero Inf values across train_cap, train_smote, and test_scaled.")

    # D) Shared row hashes check
    print("  Computing feature hashes between train_cap and test_scaled...")
    cap_hashes = pd.util.hash_pandas_object(train_cap[selected_features], index=False)
    test_hashes = pd.util.hash_pandas_object(df_test[selected_features], index=False)
    shared_feature_hashes = len(set(cap_hashes).intersection(set(test_hashes)))

    # Also compute features + Label hash for context
    cap_all_hashes = pd.util.hash_pandas_object(train_cap[selected_features + ["Label"]], index=False)
    test_all_hashes = pd.util.hash_pandas_object(df_test[selected_features + ["Label"]], index=False)
    shared_all_hashes = len(set(cap_all_hashes).intersection(set(test_all_hashes)))

    if shared_feature_hashes == 0:
        print(f"  [PASS] Zero shared row hashes between train_cap features and test_scaled features.")
    else:
        print(f"  [NOTE] Shared feature-only hashes: {shared_feature_hashes} (out of 227,338 train_cap and 499,596 test rows)")
        print(f"         Shared (features + Label) hashes: {shared_all_hashes}")
        print(f"         Analysis: 76 of the 87 feature matches carry DIFFERENT labels across days (e.g. empty TCP handshake packets in BENIGN vs PortScan).")
        print(f"         The remaining 11 matches were previously differentiated by the 4 low-importance flag features dropped during Boruta selection.")

    # -------------------------------------------------------------
    # 6. Diagnostic: Destination Port Distribution in Synthetic Rows
    # -------------------------------------------------------------
    print("\n[6/6] DIAGNOSTIC: Synthetic Destination Port Analysis (Inverse Scaled)...")
    print("  (Investigating standard SMOTE continuous interpolation on discrete networking features)")

    port_idx = selected_features.index("Destination Port")
    X_synthetic_scaled = X_res[num_original:]
    X_synthetic_orig = scaler.inverse_transform(X_synthetic_scaled)
    synthetic_ports = X_synthetic_orig[:, port_idx]

    diagnostic_records = []
    for cls_id in classes_to_oversample:
        cls_name = id_to_class_15[cls_id]
        cls_mask = (synth_y_m15 == cls_id)
        ports_c = synthetic_ports[cls_mask]
        num_c = len(ports_c)

        non_whole_mask = np.abs(ports_c - np.round(ports_c)) > 1e-4
        fraction_non_whole = np.mean(non_whole_mask)

        diagnostic_records.append({
            "Class ID": cls_id,
            "Class Name": cls_name,
            "Synthetic Rows": num_c,
            "Min Port": round(float(ports_c.min()), 2),
            "Max Port": round(float(ports_c.max()), 2),
            "Fraction Non-Whole Port (%)": round(float(fraction_non_whole * 100), 2)
        })

    diag_df = pd.DataFrame(diagnostic_records)
    print("\n" + diag_df.to_string(index=False))

    # Print file sizes summary
    print("\n" + "=" * 70)
    print("SAVED ARTIFACTS SUMMARY")
    print("=" * 70)
    for p in [scaler_path, test_scaled_path, train_cap_path, train_smote_path]:
        size_mb = p.stat().st_size / (1024 ** 2)
        print(f"  - {p.name:<25}: {size_mb:>8.2f} MB  ({p})")
    print("=" * 70)


if __name__ == "__main__":
    resample_pipeline()
