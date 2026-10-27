"""
Step 4: Boruta Feature Selection on the Training Set.

Why this matters for viva:
Boruta selects all all-relevant features by comparing original feature importances against
statistically randomized 'shadow features'. Performing this on a class-capped training sample
ensures that rare attack signatures are preserved, prevents dominant classes from biasing feature selection,
and strictly avoids data leakage by keeping the test set completely unseen.
"""

import json
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# Ensure standard UTF-8 console output for Windows PowerShell
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_project_root() -> Path:
    """Return the absolute path of the nids-project directory."""
    return Path(__file__).resolve().parent.parent


def select_features():
    project_root = get_project_root()
    train_path = project_root / "data" / "train.parquet"
    models_dir = project_root / "models"
    outputs_dir = project_root / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    selected_features_json = models_dir / "selected_features.json"
    ranking_csv_path = outputs_dir / "boruta_ranking.csv"
    plot_path = outputs_dir / "feature_importance.png"

    print("=" * 70)
    print("STEP 4: BORUTA FEATURE SELECTION (TRAINING SET ONLY)")
    print("=" * 70)
    print(f"Reading training data from: {train_path}...")

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    start_time = time.time()
    df_train = pd.read_parquet(train_path, engine="pyarrow")
    print(f"Loaded training data: {df_train.shape[0]:,} rows, {df_train.shape[1]} columns\n")

    # -------------------------------------------------------------
    # 1. Class-Capped Sampling (max 5,000 rows per class)
    # -------------------------------------------------------------
    print("[1/5] Building class-capped training sample (at most 5,000 rows per class)...")
    sample_dfs = []
    classes = sorted(df_train["Label"].unique())
    for cls_name in classes:
        group = df_train[df_train["Label"] == cls_name]
        n_sample = min(len(group), 5000)
        sample_dfs.append(group.sample(n=n_sample, random_state=42))

    sample_df = pd.concat(sample_dfs, axis=0, ignore_index=True)
    
    print("\nSample rows per class:")
    sample_counts = sample_df["Label"].value_counts().loc[classes]
    for cls_name, cnt in sample_counts.items():
        print(f"  - {cls_name:<28}: {cnt:>5,} rows")
    print(f"\nTotal sampled rows: {len(sample_df):,} (Expected ~44,112)")
    assert len(sample_df) == 44112, f"Expected 44,112 rows but got {len(sample_df)}"

    # -------------------------------------------------------------
    # 2. Prepare X and y for Boruta
    # -------------------------------------------------------------
    feature_cols = [c for c in sample_df.columns if c not in ["Label", "y_binary", "y_multi15", "y_multi13"]]
    print(f"\n[2/5] Preparing feature matrix X ({len(feature_cols)} features) and target y (y_multi15)...")
    X = sample_df[feature_cols].values
    y = sample_df["y_multi15"].values

    # -------------------------------------------------------------
    # 3. Configure Estimator and BorutaPy Selector
    # -------------------------------------------------------------
    print("\n[3/5] Initializing RandomForestClassifier and BorutaPy...")
    print("  Estimator: RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=7, random_state=42)")
    print("  Selector:  BorutaPy(n_estimators='auto', max_iter=60, perc=100, alpha=0.05, random_state=42, verbose=2)")
    print("\nStarting Boruta fitting process...")
    fit_start_time = time.time()

    rf = RandomForestClassifier(
        n_jobs=-1,
        class_weight="balanced",
        max_depth=7,
        random_state=42
    )

    feat_selector = BorutaPy(
        rf,
        n_estimators="auto",
        max_iter=60,
        perc=100,
        alpha=0.05,
        random_state=42,
        verbose=2
    )

    feat_selector.fit(X, y)
    fit_duration = time.time() - fit_start_time
    print(f"\nBoruta fitting completed in: {fit_duration:.2f} seconds ({fit_duration / 60:.2f} minutes)")

    # -------------------------------------------------------------
    # 4. Feature Selection Analysis & Artifact Saving
    # -------------------------------------------------------------
    print("\n[4/5] Analyzing feature selection results...")
    feature_cols_arr = np.array(feature_cols)

    confirmed_features = feature_cols_arr[feat_selector.support_].tolist()
    tentative_features = feature_cols_arr[feat_selector.support_weak_].tolist()
    rejected_mask = ~(feat_selector.support_ | feat_selector.support_weak_)
    rejected_features = feature_cols_arr[rejected_mask].tolist()
    selected_features = confirmed_features + tentative_features

    num_confirmed = len(confirmed_features)
    num_tentative = len(tentative_features)
    num_rejected = len(rejected_features)
    total_features = num_confirmed + num_tentative + num_rejected

    print("\n" + "=" * 50)
    print("BORUTA FEATURE SELECTION SUMMARY")
    print("=" * 50)
    print(f"Confirmed Features: {num_confirmed}")
    print(f"Tentative Features: {num_tentative}")
    print(f"Rejected Features:  {num_rejected}")
    print(f"Total Evaluated:    {total_features} (Assertion: {total_features == 69})")
    print(f"Total Selected:     {len(selected_features)} (Confirmed + Tentative)")
    print("=" * 50)

    assert total_features == 69, f"Feature count mismatch: {total_features} != 69"

    print("\nRejected features list:")
    for rf_name in rejected_features:
        print(f"  - {rf_name}")

    # Save to models/selected_features.json
    selected_dict = {
        "confirmed": confirmed_features,
        "tentative": tentative_features,
        "rejected": rejected_features,
        "selected": selected_features
    }
    with open(selected_features_json, "w", encoding="utf-8") as f:
        json.dump(selected_dict, f, indent=2)
    print(f"\nSaved feature sets to: {selected_features_json}")

    # Save ranking CSV
    status_list = []
    for i in range(len(feature_cols)):
        if feat_selector.support_[i]:
            status_list.append("Confirmed")
        elif feat_selector.support_weak_[i]:
            status_list.append("Tentative")
        else:
            status_list.append("Rejected")

    ranking_df = pd.DataFrame({
        "feature": feature_cols,
        "Boruta ranking": feat_selector.ranking_,
        "status": status_list
    }).sort_values(by=["Boruta ranking", "feature"])
    ranking_df.to_csv(ranking_csv_path, index=False)
    print(f"Saved ranking CSV to: {ranking_csv_path}")

    # -------------------------------------------------------------
    # 5. Feature Importance of Selected Features & Plotting
    # -------------------------------------------------------------
    print("\n[5/5] Fitting fresh RandomForest on selected features to compute importance...")
    rf_selected = RandomForestClassifier(
        n_jobs=-1,
        class_weight="balanced",
        max_depth=7,
        random_state=42
    )
    rf_selected.fit(sample_df[selected_features], y)
    importances = rf_selected.feature_importances_

    imp_df = pd.DataFrame({
        "feature": selected_features,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    # Plot top 25 horizontal bar chart
    top25 = imp_df.head(25).sort_values(by="importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top25["feature"], top25["importance"], color="#2b5c8f", edgecolor="black", height=0.65)
    plt.xlabel("Random Forest Feature Importance (Gini Impurity MDI)", fontsize=11)
    plt.ylabel("Selected Feature", fontsize=11)
    plt.title("Top 25 Features Selected by Boruta (Sample Importance)", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved top 25 feature importance chart to: {plot_path}")

    print("\n--- Top 15 Features by Importance ---")
    top15 = imp_df.head(15).copy()
    top15["importance"] = top15["importance"].round(5)
    top15.index = range(1, 16)
    print(top15.to_string())

    total_elapsed = time.time() - start_time
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)
    print("\n" + "=" * 70)
    print(f"STEP 4 COMPLETE | Total Elapsed Time: {minutes}m {seconds}s ({total_elapsed:.2f}s)")
    print("=" * 70)


if __name__ == "__main__":
    select_features()
