"""
Step 3: Label encoding and stratified train/test split.

Why this matters for viva:
Splitting the dataset BEFORE any resampling, scaling, or feature selection is the fundamental
rule of machine learning hygiene to guarantee zero data leakage. Stratifying across all 15 original
classes preserves the exact real-world proportions (including rare zero-day-like attacks with only dozens
of samples) across both the training and test sets.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure standard UTF-8 console output for Windows PowerShell
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_project_root() -> Path:
    """Return the absolute path of the nids-project directory."""
    return Path(__file__).resolve().parent.parent


def split_dataset():
    project_root = get_project_root()
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    input_path = data_dir / "cleaned.parquet"
    train_output_path = data_dir / "train.parquet"
    test_output_path = data_dir / "test.parquet"
    label_maps_path = models_dir / "label_maps.json"

    print("=" * 70)
    print("STEP 3: LABEL ENCODING & STRATIFIED TRAIN/TEST SPLIT")
    print("=" * 70)
    print(f"Reading cleaned data from: {input_path}...")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path, engine="pyarrow")
    total_rows, total_cols = df.shape
    print(f"Loaded: {total_rows:,} rows, {total_cols} columns\n")

    # -------------------------------------------------------------
    # 1. Build Target Encodings (y_binary, y_multi15, y_multi13)
    # -------------------------------------------------------------
    print("[1/5] Building target encodings...")

    # y_binary: 0 for BENIGN, 1 for any attack
    df["y_binary"] = (df["Label"] != "BENIGN").astype(np.int32)
    binary_map = {
        "class_to_id": {"BENIGN": 0, "ATTACK": 1},
        "id_to_class": {0: "BENIGN", 1: "ATTACK"}
    }

    # y_multi15: integer-encoded original 15 classes (sorted)
    classes_15 = sorted(df["Label"].unique())
    multi15_map = {
        "class_to_id": {cls_name: idx for idx, cls_name in enumerate(classes_15)},
        "id_to_class": {idx: cls_name for idx, cls_name in enumerate(classes_15)}
    }
    df["y_multi15"] = df["Label"].map(multi15_map["class_to_id"]).astype(np.int32)

    # y_multi13: merged Web Attacks into "Web Attack" (13 classes, sorted)
    web_attack_mapping = {
        "Web Attack - Brute Force": "Web Attack",
        "Web Attack - XSS": "Web Attack",
        "Web Attack - Sql Injection": "Web Attack"
    }
    label_13_series = df["Label"].replace(web_attack_mapping)
    classes_13 = sorted(label_13_series.unique())
    multi13_map = {
        "class_to_id": {cls_name: idx for idx, cls_name in enumerate(classes_13)},
        "id_to_class": {idx: cls_name for idx, cls_name in enumerate(classes_13)}
    }
    df["y_multi13"] = label_13_series.map(multi13_map["class_to_id"]).astype(np.int32)

    # Save label mappings to models/label_maps.json
    label_maps_payload = {
        "binary": binary_map,
        "multi15": multi15_map,
        "multi13": multi13_map
    }
    with open(label_maps_path, "w", encoding="utf-8") as f:
        json.dump(label_maps_payload, f, indent=2)
    print(f"Saved label mappings to: {label_maps_path}")

    # -------------------------------------------------------------
    # 2. Stratified 80/20 Train-Test Split on Indices
    # -------------------------------------------------------------
    print("\n[2/5] Performing 80/20 stratified split based on 15-class Label...")
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=42,
        shuffle=True,
        stratify=df["Label"]
    )

    df_train = df.loc[train_idx].copy().reset_index(drop=True)
    df_test = df.loc[test_idx].copy().reset_index(drop=True)

    # -------------------------------------------------------------
    # 3. Verification Checks & Assertions
    # -------------------------------------------------------------
    print("\n[3/5] Running verification checks & assertions...")

    # A) Check Shapes
    print(f"  * Train Shape: {df_train.shape[0]:,} rows, {df_train.shape[1]} columns")
    print(f"  * Test Shape:  {df_test.shape[0]:,} rows, {df_test.shape[1]} columns")

    # B) Index Disjointness Assertion
    overlap = len(set(train_idx).intersection(set(test_idx)))
    assert overlap == 0, f"[ASSERTION FAILED] Overlapping indices found: {overlap}"
    print("  * Index Check: PASSED (Zero overlapping indices)")

    # C) Full Row Hash Check (Features + Label)
    print("  * Computing row content hashes across all features + Label...")
    feature_and_label_cols = [c for c in df.columns if c not in ["y_binary", "y_multi15", "y_multi13"]]
    train_hashes = pd.util.hash_pandas_object(df_train[feature_and_label_cols], index=False)
    test_hashes = pd.util.hash_pandas_object(df_test[feature_and_label_cols], index=False)
    shared_hashes = len(set(train_hashes).intersection(set(test_hashes)))
    assert shared_hashes == 0, f"[ASSERTION FAILED] Shared row hashes detected: {shared_hashes}"
    print("  * Hash Integrity Check: PASSED (Zero shared hashes between train and test)")

    # D) 15-Class Presence in Both Sets
    train_unique_classes = set(df_train["Label"].unique())
    test_unique_classes = set(df_test["Label"].unique())
    expected_classes = set(classes_15)
    assert train_unique_classes == expected_classes, "[ASSERTION FAILED] Missing classes in train set"
    assert test_unique_classes == expected_classes, "[ASSERTION FAILED] Missing classes in test set"
    print("  * Class Completeness: PASSED (All 15 classes present in both train and test)")

    # E) Per-Class Counts & Test Share Table
    print("\n--- Per-Class Distribution (15 Classes) ---")
    train_counts = df_train["Label"].value_counts()
    test_counts = df_test["Label"].value_counts()
    class_dist_df = pd.DataFrame({
        "Train Count": train_counts,
        "Test Count": test_counts,
        "Total Count": train_counts + test_counts,
        "Test Share (%)": (test_counts / (train_counts + test_counts) * 100).round(2)
    }).loc[classes_15]  # Keep sorted order
    print(class_dist_df.to_string())

    # F) Binary Class Counts & Attack Percentages
    print("\n--- Binary Class Distribution ---")
    train_b_counts = df_train["y_binary"].value_counts()
    test_b_counts = df_test["y_binary"].value_counts()
    train_attack_pct = (train_b_counts[1] / len(df_train)) * 100
    test_attack_pct = (test_b_counts[1] / len(df_test)) * 100

    binary_summary = pd.DataFrame({
        "Train Count": [train_b_counts[0], train_b_counts[1]],
        "Train %": [(train_b_counts[0] / len(df_train)) * 100, train_attack_pct],
        "Test Count": [test_b_counts[0], test_b_counts[1]],
        "Test %": [(test_b_counts[0] / len(df_test)) * 100, test_attack_pct]
    }, index=["BENIGN (0)", "ATTACK (1)"])
    print(binary_summary.round(3).to_string())
    print(f"Train Attack Rate: {train_attack_pct:.2f}% | Test Attack Rate: {test_attack_pct:.2f}%")

    # -------------------------------------------------------------
    # 4. Verify Column Types & Structure
    # -------------------------------------------------------------
    print("\n[4/5] Verifying column types before saving...")
    feature_cols = [c for c in df_train.columns if c not in ["Label", "y_binary", "y_multi15", "y_multi13"]]
    for col in feature_cols:
        assert df_train[col].dtype == np.float32, f"Column {col} is not float32 in train"
        assert df_test[col].dtype == np.float32, f"Column {col} is not float32 in test"
    print(f"Verified all {len(feature_cols)} feature columns are float32.")

    # -------------------------------------------------------------
    # 5. Save Train and Test Parquet Datasets
    # -------------------------------------------------------------
    print("\n[5/5] Saving train and test Parquet files...")
    df_train.to_parquet(train_output_path, engine="pyarrow", index=False)
    df_test.to_parquet(test_output_path, engine="pyarrow", index=False)

    train_size_mb = train_output_path.stat().st_size / (1024 ** 2)
    test_size_mb = test_output_path.stat().st_size / (1024 ** 2)
    print(f"Saved: {train_output_path} ({train_size_mb:.2f} MB)")
    print(f"Saved: {test_output_path} ({test_size_mb:.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    split_dataset()
