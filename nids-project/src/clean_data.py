"""
Step 2: Clean and preprocess the merged CICIDS2017 dataset.

Why this matters for viva:
Raw network captures contain artifacts like duplicate flows, missing/infinite rates,
constant columns with zero variance, and encoding glitches. Cleaning these ensures
that the models learn genuine network signatures rather than recording anomalies,
and prevents data leakage where identical flows might appear in both train and test sets.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure standard UTF-8 console output for Windows PowerShell
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_project_root() -> Path:
    """Return the absolute path of the nids-project directory."""
    return Path(__file__).resolve().parent.parent


def clean_dataset():
    project_root = get_project_root()
    input_path = project_root / "data" / "merged_raw.parquet"
    output_path = project_root / "data" / "cleaned.parquet"

    print("=" * 70)
    print("STEP 2: DATA CLEANING PIPELINE")
    print("=" * 70)
    print(f"Reading raw data from: {input_path}...")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path, engine="pyarrow")
    initial_rows, initial_cols = df.shape
    initial_mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Loaded: {initial_rows:,} rows, {initial_cols} columns")
    print(f"Initial Memory Usage: {initial_mem_mb:.2f} MB ({initial_mem_mb / 1024:.2f} GB)\n")

    # -------------------------------------------------------------
    # 1. Strip column whitespace & clean Label encoding
    # -------------------------------------------------------------
    print("[1/8] Stripping column whitespace & cleaning label text...")
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise KeyError("Column 'Label' not found in dataset columns.")

    # Fix broken dash in Web Attack labels (replace non-ASCII characters with '-')
    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.replace(r"[^\x00-\x7F]+", "-", regex=True)
        .str.strip()
    )

    unique_labels = sorted(df["Label"].unique())
    print(f"Found {len(unique_labels)} unique labels (expect 15):")
    for lbl in unique_labels:
        print(f"  - {lbl}")

    # -------------------------------------------------------------
    # 2. Detect and drop duplicate columns (e.g. '.1' suffix)
    # -------------------------------------------------------------
    print("\n[2/8] Checking for duplicate columns ending with '.1'...")
    dot_one_cols = [c for c in df.columns if c.endswith(".1")]
    dropped_cols = []
    for col in dot_one_cols:
        base_col = col[:-2]
        if base_col in df.columns and df[col].equals(df[base_col]):
            dropped_cols.append(col)

    if dropped_cols:
        df = df.drop(columns=dropped_cols)
        print(f"Dropped duplicate columns ({len(dropped_cols)}): {dropped_cols}")
    else:
        print("No duplicate '.1' columns found.")

    # -------------------------------------------------------------
    # 3. Downcast feature columns to float32
    # -------------------------------------------------------------
    print("\n[3/8] Downcasting feature columns to float32...")
    feature_cols = [c for c in df.columns if c != "Label"]
    df[feature_cols] = df[feature_cols].astype(np.float32)
    downcast_mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Memory after float32 downcasting: {downcast_mem_mb:.2f} MB ({downcast_mem_mb / 1024:.2f} GB)")

    # -------------------------------------------------------------
    # 4. Replace +inf/-inf with NaN and drop missing rows
    # -------------------------------------------------------------
    print("\n[4/8] Checking for infinite and NaN values...")
    # Identify infs and NaNs per feature column
    inf_nan_counts = {}
    for col in feature_cols:
        col_series = df[col]
        num_inf = np.isinf(col_series).sum()
        num_nan = col_series.isna().sum()
        total_invalid = num_inf + num_nan
        if total_invalid > 0:
            inf_nan_counts[col] = {"Inf": int(num_inf), "NaN": int(num_nan), "Total": int(total_invalid)}

    print("Columns with NaN/Inf values:")
    for col, counts in inf_nan_counts.items():
        print(f"  - {col}: {counts['Total']:,} invalid ({counts['Inf']:,} Inf, {counts['NaN']:,} NaN)")

    # Replace infs with NaN across feature columns
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    rows_before_na = len(df)
    df = df.dropna()
    rows_dropped_na = rows_before_na - len(df)
    print(f"Dropped {rows_dropped_na:,} rows containing NaN/Inf. Remaining rows: {len(df):,}")

    # -------------------------------------------------------------
    # 5. Drop duplicate rows
    # -------------------------------------------------------------
    print("\n[5/8] Checking for duplicate rows...")
    rows_before_dups = len(df)
    df = df.drop_duplicates()
    rows_dropped_dups = rows_before_dups - len(df)
    print(f"Dropped {rows_dropped_dups:,} duplicate rows. Remaining rows: {len(df):,}")

    # -------------------------------------------------------------
    # 6. Drop constant columns (nunique <= 1)
    # -------------------------------------------------------------
    print("\n[6/8] Checking for constant columns (nunique <= 1)...")
    constant_cols = [c for c in df.columns if c != "Label" and df[c].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        print(f"Dropped {len(constant_cols)} constant column(s):")
        for c in constant_cols:
            print(f"  - {c}")
    else:
        print("No constant columns detected.")

    # -------------------------------------------------------------
    # 7. Reset index & print cleaning summary
    # -------------------------------------------------------------
    print("\n[7/8] Resetting index and generating cleaning summary...")
    df = df.reset_index(drop=True)

    final_rows, final_cols = df.shape
    final_mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    print("\n" + "=" * 70)
    print("CLEANING SUMMARY")
    print("=" * 70)
    print(f"Initial Shape:       {initial_rows:,} rows, {initial_cols} columns")
    print(f"Final Shape:         {final_rows:,} rows, {final_cols} columns")
    print(f"Rows Removed:        {initial_rows - final_rows:,} ({(initial_rows - final_rows) / initial_rows * 100:.2f}%)")
    print(f"Columns Removed:     {initial_cols - final_cols}")
    print(f"Final Memory Usage:  {final_mem_mb:.2f} MB ({final_mem_mb / 1024:.2f} GB)")

    print("\n--- Final Cleaned Label Distribution ---")
    counts = df["Label"].value_counts()
    pcts = df["Label"].value_counts(normalize=True) * 100
    label_summary = pd.DataFrame({"Count": counts, "Percentage (%)": pcts.round(4)})
    print(label_summary.to_string())

    # -------------------------------------------------------------
    # 8. Save cleaned dataset to Parquet
    # -------------------------------------------------------------
    print("\n[8/8] Saving cleaned dataset to Parquet...")
    df.to_parquet(output_path, engine="pyarrow", index=False)
    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"Saved: {output_path} ({file_size_mb:.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    clean_dataset()
