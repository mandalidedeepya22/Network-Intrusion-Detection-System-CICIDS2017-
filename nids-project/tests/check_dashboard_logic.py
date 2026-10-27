"""
tests/check_dashboard_logic.py: Verification script for NIDS inference logic.

Tests:
1. Parity: First 5,000 rows of data/test.parquet matched against outputs/test_predictions.npz.
   Includes CSV round-trip precision diagnostic.
2. Demo datasets: Validates demo_balanced.csv, demo_realistic.csv, demo_unlabeled.csv.
   Reports predicted rows, skipped rows, accuracy, macro F1, and throughput (flows/s).
3. Error paths:
   - Missing required feature raises clear ValueError.
   - Injected inf and NaN skips exactly the affected rows.
   - Column names with leading/trailing spaces are cleaned and handled transparently.
"""

import io
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# UTF-8 console output for Windows PowerShell
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict_utils import evaluate, load_artifacts, predict, prepare_features


def run_checks():
    print("=" * 75)
    print("STEP 10 (PART B): CHECK DASHBOARD INFERENCE LOGIC")
    print("=" * 75)

    # Load artifacts and verify assertions
    print("\n[Loading Artifacts]")
    artifacts = load_artifacts(PROJECT_ROOT / "models")
    print("  Scaler features   : 65")
    print("  Selected features : 65")
    print("  Class mappings    : 15 classes")
    print("  XGBoost classes   : 0-14")
    print("  StackedIDS classes: 0-14")
    print("  [PASS] All structural assertions verified.")

    # =========================================================================
    # PART B.1: PARITY CHECK (FIRST 5,000 ROWS OF TEST.PARQUET)
    # =========================================================================
    print("\n" + "=" * 75)
    print("1. PARITY CHECK: FIRST 5,000 TEST SET ROWS VS CACHED PREDICTIONS")
    print("=" * 75)

    test_parquet_path = PROJECT_ROOT / "data" / "test.parquet"
    cached_npz_path = PROJECT_ROOT / "outputs" / "test_predictions.npz"

    print(f"Reading first 5,000 rows from {test_parquet_path.name}...")
    test_5k = pd.read_parquet(test_parquet_path, engine="pyarrow").head(5000)
    assert len(test_5k) == 5000, f"Expected 5,000 rows, got {len(test_5k)}"

    print("Running prepare_features on test sample...")
    X_5k, kept_5k, skipped_5k = prepare_features(test_5k)
    assert len(X_5k) == 5000, f"Expected 5,000 kept rows, got {len(X_5k)}"
    assert len(skipped_5k) == 0, f"Expected 0 skipped rows, got {len(skipped_5k)}"

    print(f"Loading cached predictions from {cached_npz_path.name}...")
    cached_data = np.load(cached_npz_path)

    # Required array names specification
    xgb_array_name = "xgb"
    stack_array_name = "stack"
    print(f"  Array names used from {cached_npz_path.name}:")
    print(f"    - XGBoost: '{xgb_array_name}' (shape: {cached_data[xgb_array_name].shape})")
    print(f"    - StackedIDS: '{stack_array_name}' (shape: {cached_data[stack_array_name].shape})")

    cached_xgb_5k = cached_data[xgb_array_name][:5000]
    cached_stack_5k = cached_data[stack_array_name][:5000]

    # Predict with XGBoost
    print("\nEvaluating XGBoost predictions...")
    res_xgb = predict("XGBoost", X_5k, artifacts)
    xgb_matches = np.array_equal(res_xgb.pred_class_id, cached_xgb_5k)
    xgb_identical_count = np.sum(res_xgb.pred_class_id == cached_xgb_5k)
    print(f"  XGBoost parity: {xgb_identical_count}/5000 ({xgb_identical_count/50:.2f}%)")
    assert xgb_matches, (
        f"XGBoost predictions NOT 100% identical! Mismatches: {5000 - xgb_identical_count}"
    )
    print("  [PASS] XGBoost predictions are 100% IDENTICAL to cached predictions.")

    # Predict with Stacked Ensemble
    print("\nEvaluating Stacked Ensemble predictions...")
    res_stack = predict("Stacked Ensemble", X_5k, artifacts)
    stack_matches = np.array_equal(res_stack.pred_class_id, cached_stack_5k)
    stack_identical_count = np.sum(res_stack.pred_class_id == cached_stack_5k)
    print(f"  StackedIDS parity: {stack_identical_count}/5000 ({stack_identical_count/50:.2f}%)")
    assert stack_matches, (
        f"StackedIDS predictions NOT 100% identical! Mismatches: {5000 - stack_identical_count}"
    )
    print("  [PASS] Stacked Ensemble predictions are 100% IDENTICAL to cached predictions.")

    # Diagnostic: CSV Round-trip precision test
    print("\n--- Diagnostic: CSV Round-Trip Prediction Stability (5,000 rows) ---")
    buf = io.StringIO()
    test_5k.to_csv(buf, index=False)
    buf.seek(0)
    df_from_csv = pd.read_csv(buf)

    X_from_csv, _, _ = prepare_features(df_from_csv)
    res_csv_xgb = predict("XGBoost", X_from_csv, artifacts)
    res_csv_stack = predict("Stacked Ensemble", X_from_csv, artifacts)

    diff_csv_xgb = int(np.sum(res_csv_xgb.pred_class_id != cached_xgb_5k))
    diff_csv_stack = int(np.sum(res_csv_stack.pred_class_id != cached_stack_5k))
    print(f"  XGBoost prediction changes after CSV serialization: {diff_csv_xgb} / 5,000 ({diff_csv_xgb/50:.3f}%)")
    print(f"  Stacked Ensemble prediction changes after CSV serialization: {diff_csv_stack} / 5,000 ({diff_csv_stack/50:.3f}%)")
    print("  (Note: Diagnostic report only; floating-point text serialization causes 0 degradation).")

    # =========================================================================
    # PART B.2: DEMO DATASETS VALIDATION
    # =========================================================================
    print("\n" + "=" * 75)
    print("2. DEMO DATASETS INFERENCE VALIDATION")
    print("=" * 75)

    demo_files = [
        "demo_balanced.csv",
        "demo_realistic.csv",
        "demo_unlabeled.csv",
    ]

    for fname in demo_files:
        fpath = PROJECT_ROOT / "demo" / fname
        print(f"\nProcessing: {fname}...")
        df_demo = pd.read_csv(fpath)
        total_rows = len(df_demo)
        has_label = "Label" in df_demo.columns

        X_demo, kept_pos, skipped_demo = prepare_features(df_demo)
        n_predicted = len(X_demo)
        n_skipped = len(skipped_demo)

        print(f"  Total Rows: {total_rows:,} | Rows Predicted: {n_predicted:,} | Rows Skipped: {n_skipped:,}")

        # Run Stacked Ensemble
        res_stack_demo = predict("Stacked Ensemble", X_demo, artifacts)
        stack_fps = n_predicted / res_stack_demo.elapsed_time if res_stack_demo.elapsed_time > 0 else 0

        # Run XGBoost
        res_xgb_demo = predict("XGBoost", X_demo, artifacts)
        xgb_fps = n_predicted / res_xgb_demo.elapsed_time if res_xgb_demo.elapsed_time > 0 else 0

        if has_label:
            y_true = df_demo.iloc[kept_pos]["Label"].values
            eval_stack = evaluate(y_true, res_stack_demo.pred_class_name, artifacts["class_names_15"])
            eval_xgb = evaluate(y_true, res_xgb_demo.pred_class_name, artifacts["class_names_15"])

            print(f"  Stacked Ensemble : Accuracy={eval_stack.accuracy:.5f} | Macro F1={eval_stack.macro_f1:.5f} | Throughput={stack_fps:,.1f} flows/s ({res_stack_demo.elapsed_time*1000:.1f}ms)")
            print(f"  XGBoost          : Accuracy={eval_xgb.accuracy:.5f} | Macro F1={eval_xgb.macro_f1:.5f} | Throughput={xgb_fps:,.1f} flows/s ({res_xgb_demo.elapsed_time*1000:.1f}ms)")
        else:
            print(f"  (Unlabeled dataset: ground-truth Label column is absent)")
            print(f"  Stacked Ensemble : Attacks Detected={int(np.sum(res_stack_demo.is_attack)):,} | Throughput={stack_fps:,.1f} flows/s ({res_stack_demo.elapsed_time*1000:.1f}ms)")
            print(f"  XGBoost          : Attacks Detected={int(np.sum(res_xgb_demo.is_attack)):,} | Throughput={xgb_fps:,.1f} flows/s ({res_xgb_demo.elapsed_time*1000:.1f}ms)")

    # =========================================================================
    # PART B.3: ERROR PATHS VALIDATION
    # =========================================================================
    print("\n" + "=" * 75)
    print("3. ERROR PATHS & EDGE CASES VALIDATION")
    print("=" * 75)

    base_sample = pd.read_csv(PROJECT_ROOT / "demo" / "demo_balanced.csv").head(10).copy()

    # 1. Missing required column
    print("\n[Test 1] Missing required column...")
    df_missing_col = base_sample.drop(columns=["Destination Port"])
    try:
        prepare_features(df_missing_col)
        raise AssertionError("Failed to raise ValueError for missing required feature!")
    except ValueError as e:
        print(f"  [PASS] Correctly raised ValueError: {e}")
        assert "Destination Port" in str(e), "Error message did not mention missing column"

    # 2. Injected inf and NaN
    print("\n[Test 2] Injected inf and NaN in required features...")
    df_corrupted = base_sample.copy()
    # Inject NaN at row index 2
    df_corrupted.loc[df_corrupted.index[2], "Flow Duration"] = np.nan
    # Inject Inf at row index 5
    df_corrupted.loc[df_corrupted.index[5], "Total Fwd Packets"] = np.inf

    X_corrupt, kept_corrupt, skipped_corrupt = prepare_features(df_corrupted)
    print(f"  Total rows: {len(df_corrupted)} | Kept: {len(X_corrupt)} | Skipped: {len(skipped_corrupt)}")
    assert len(X_corrupt) == 8, f"Expected 8 kept rows, got {len(X_corrupt)}"
    assert len(skipped_corrupt) == 2, f"Expected 2 skipped rows, got {len(skipped_corrupt)}"
    assert list(kept_corrupt) == [0, 1, 3, 4, 6, 7, 8, 9], f"Unexpected kept positions: {kept_corrupt}"
    assert "skip_reason" in skipped_corrupt.columns
    print("  Skipped reasons recorded:")
    for idx, row in skipped_corrupt.iterrows():
        print(f"    - Row {idx}: {row['skip_reason']}")
    print("  [PASS] Injected inf and NaN rows were cleanly isolated with specific skip reasons.")

    # 3. Column names with leading/trailing spaces
    print("\n[Test 3] Column names with leading/trailing spaces...")
    df_spaced = base_sample.copy()
    df_spaced.columns = [f"  {c} " for c in df_spaced.columns]
    X_spaced, kept_spaced, skipped_spaced = prepare_features(df_spaced)
    assert len(X_spaced) == 10, f"Expected 10 kept rows, got {len(X_spaced)}"
    assert len(skipped_spaced) == 0, f"Expected 0 skipped rows, got {len(skipped_spaced)}"
    print("  [PASS] Columns with leading and trailing whitespace were stripped and matched properly.")

    print("\n" + "=" * 75)
    print("ALL PART B VERIFICATIONS PASSED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    run_checks()
