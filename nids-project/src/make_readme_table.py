"""
make_readme_table.py: Generates Markdown results table from outputs/final_results.csv.
Pure Python formatting (no external dependencies such as tabulate required).
"""

from pathlib import Path
import pandas as pd


def generate_markdown_table() -> str:
    possible_paths = [
        Path("outputs/final_results.csv"),
        Path("nids-project/outputs/final_results.csv"),
        Path(__file__).resolve().parent.parent / "outputs" / "final_results.csv",
        Path(__file__).resolve().parent.parent / "nids-project" / "outputs" / "final_results.csv",
    ]
    csv_path = None
    for p in possible_paths:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError("Could not locate outputs/final_results.csv")

    df = pd.read_csv(csv_path)

    display_cols = [
        "Model",
        "15-Class Acc",
        "15-Class Macro F1",
        "13-Class Macro F1",
        "Attack Precision",
        "Attack Recall",
        "Attack F1",
        "Benign FPR (%)",
        "FP Count",
        "FN Count",
        "ROC-AUC",
        "Average Precision",
    ]

    df_subset = df[display_cols].copy()

    df_subset["15-Class Acc"] = df_subset["15-Class Acc"].apply(lambda v: f"{v:.5f}")
    df_subset["15-Class Macro F1"] = df_subset["15-Class Macro F1"].apply(lambda v: f"{v:.5f}")
    df_subset["13-Class Macro F1"] = df_subset["13-Class Macro F1"].apply(lambda v: f"{v:.5f}")
    df_subset["Attack Precision"] = df_subset["Attack Precision"].apply(lambda v: f"{v:.5f}")
    df_subset["Attack Recall"] = df_subset["Attack Recall"].apply(lambda v: f"{v:.5f}")
    df_subset["Attack F1"] = df_subset["Attack F1"].apply(lambda v: f"{v:.5f}")
    df_subset["Benign FPR (%)"] = df_subset["Benign FPR (%)"].apply(lambda v: f"{v:.4f}%")
    df_subset["FP Count"] = df_subset["FP Count"].apply(lambda v: f"{int(v):,}")
    df_subset["FN Count"] = df_subset["FN Count"].apply(lambda v: f"{int(v):,}")
    df_subset["ROC-AUC"] = df_subset["ROC-AUC"].apply(lambda v: f"{v:.5f}")
    df_subset["Average Precision"] = df_subset["Average Precision"].apply(lambda v: f"{v:.5f}")

    # Build pure markdown table without tabulate dependency
    headers = list(df_subset.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df_subset.iterrows():
        lines.append("| " + " | ".join(str(val) for val in row) + " |")

    return "\n".join(lines)


if __name__ == "__main__":
    table = generate_markdown_table()
    print(table)
