"""
Step 1: Load and merge the 8 CICIDS2017 CSV files.

Why this matters for viva:
Real-world network traffic is captured over different days and attack scenarios.
Merging all raw captures gives us a complete, unbiased representation of network
traffic before we perform data cleaning and split the data into training and test sets.
"""

import sys
from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """Return the absolute path of the nids-project directory."""
    return Path(__file__).resolve().parent.parent


def load_and_merge_data() -> pd.DataFrame:
    project_root = get_project_root()
    data_dir = project_root / "data"
    output_parquet_path = data_dir / "merged_raw.parquet"

    csv_files = sorted(list(data_dir.glob("*.csv")))
    num_files = len(csv_files)

    print("=" * 60)
    print(f"STEP 1: Loading & Merging CICIDS2017 CSV Files")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"CSV files found: {num_files}")

    if num_files == 0:
        print(f"\n[ERROR] No CSV files found in: {data_dir}")
        print("Please place the 8 CICIDS2017 CSV files in the 'data/' folder and rerun.")
        sys.exit(1)

    print("\nReading files:")
    dfs = []
    for idx, filepath in enumerate(csv_files, 1):
        filename = filepath.name
        print(f"  [{idx}/{num_files}] Reading {filename}...", end=" ", flush=True)

        # CICIDS2017 CSVs may have encoding variations
        try:
            df_temp = pd.read_csv(filepath, low_memory=False, encoding="utf-8")
        except UnicodeDecodeError:
            df_temp = pd.read_csv(filepath, low_memory=False, encoding="latin1")

        print(f"Done ({df_temp.shape[0]:,} rows, {df_temp.shape[1]} cols)")
        dfs.append(df_temp)

    print("\nConcatenating DataFrames...")
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)

    # Clean whitespace on column names to reliably identify the label column
    merged_df.columns = merged_df.columns.str.strip()

    total_rows, total_cols = merged_df.shape
    mem_usage_mb = merged_df.memory_usage(deep=True).sum() / (1024 ** 2)
    mem_usage_gb = mem_usage_mb / 1024

    print("\n" + "=" * 60)
    print("DATASET MERGE SUMMARY")
    print("=" * 60)
    print(f"Number of files merged:  {num_files}")
    print(f"Total shape:              {total_rows:,} rows, {total_cols} columns")
    print(f"Column count:             {total_cols}")
    print(f"Total memory usage:       {mem_usage_mb:.2f} MB ({mem_usage_gb:.2f} GB)")

    label_col = "Label" if "Label" in merged_df.columns else None
    if label_col:
        print("\n--- Label Value Counts ---")
        counts = merged_df[label_col].value_counts(dropna=False)
        pcts = merged_df[label_col].value_counts(normalize=True, dropna=False) * 100
        label_summary = pd.DataFrame({"Count": counts, "Percentage (%)": pcts.round(3)})
        print(label_summary.to_string())
    else:
        print("\n[WARNING] 'Label' column not detected in merged columns.")

    # Save to Parquet format for fast loading in subsequent steps
    print(f"\nSaving merged raw dataset to Parquet: {output_parquet_path}...")
    merged_df.to_parquet(output_parquet_path, engine="pyarrow", index=False)
    file_size_mb = output_parquet_path.stat().st_size / (1024 ** 2)
    print(f"Successfully saved merged_raw.parquet ({file_size_mb:.2f} MB)!")
    print("=" * 60)

    return merged_df


if __name__ == "__main__":
    load_and_merge_data()
