# Network Intrusion Detection Using Machine Learning on CICIDS2017

An end-to-end network intrusion detection system (NIDS) benchmarked on the Canadian Institute for Cybersecurity CICIDS2017 dataset (~2.83M flows). The pipeline implements automated data cleaning and deduplication, label harmonization across 15 attack classes, stratified train/test partitioning, Boruta feature selection (65 selected features), standard scaling, majority class capping, minority class SMOTE, baseline model evaluation (Logistic Regression, Random Forest, XGBoost), an out-of-fold trained Stacked Ensemble with Logistic Regression meta-learner, systematic ablation analysis across multiple random seeds, and an interactive Streamlit detection dashboard.

---

## Dataset

> **Note:** Raw and preprocessed dataset files (`data/`) are **NOT included** in this repository due to file size constraints.

- **Official Source:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick (UNB): [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- **MachineLearningCVE CSV Files Required (8 files):**
  - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
  - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
  - `Friday-WorkingHours-Morning.pcap_ISCX.csv`
  - `Monday-WorkingHours.pcap_ISCX.csv`
  - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
  - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
  - `Tuesday-WorkingHours.pcap_ISCX.csv`
  - `Wednesday-workingHours.pcap_ISCX.csv`
- **Citation:**
  Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. In Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP 2018), pages 108–116.

To reproduce the pipeline from scratch, download the 8 `MachineLearningCVE` CSV files and place them into the `data/` directory.

---

## Folder Structure

```text
.
├── .gitignore
├── README.md
├── app.py
├── requirements.txt
├── requirements_frozen.txt
├── .streamlit/
│   └── config.toml
├── demo/
│   ├── demo_balanced.csv
│   ├── demo_realistic.csv
│   └── demo_unlabeled.csv
├── models/
│   ├── label_maps.json
│   ├── logistic_regression.joblib
│   ├── model_card.json
│   ├── random_forest.joblib
│   ├── scaler.joblib
│   ├── selected_features.json
│   ├── stacked_ids.joblib
│   └── xgboost.joblib
├── outputs/
│   ├── final_results.csv
│   ├── final_per_class.csv
│   ├── latency_final.csv
│   ├── permutation_importance.csv
│   ├── ablation_runs.csv
│   ├── ablation_summary.csv
│   ├── ablation_per_class.csv
│   ├── baseline_metrics.csv
│   ├── model_comparison.csv
│   └── report_figures/
│       ├── fig_ablation.png
│       ├── fig_class_distribution.png
│       ├── fig_confusion_stack.png
│       ├── fig_confusion_xgboost.png
│       ├── fig_per_class_f1.png
│       ├── fig_perm_importance.png
│       ├── fig_pr_binary.png
│       └── fig_roc_binary.png
├── src/
│   ├── load_data.py
│   ├── clean_data.py
│   ├── split_data.py
│   ├── select_features.py
│   ├── resample_data.py
│   ├── train_baselines.py
│   ├── stack_model.py
│   ├── train_stack.py
│   ├── diagnostics_ablation.py
│   ├── report_assets.py
│   ├── predict_utils.py
│   └── make_readme_table.py
└── tests/
    ├── __init__.py
    └── check_dashboard_logic.py
```

---

## Pipeline Execution Order

To run the complete data processing, feature selection, training, and evaluation pipeline from source, execute the scripts inside `src/` in the following sequence:

1. **Step 1 - Data Loading & Merging:**
   ```bash
   python src/load_data.py
   ```
2. **Step 2 - Cleaning & Deduplication:**
   ```bash
   python src/clean_data.py
   ```
3. **Step 3 - Stratified 80/20 Train/Test Split & Label Encoding:**
   ```bash
   python src/split_data.py
   ```
4. **Step 4 - Boruta Feature Selection (Training Set Only):**
   ```bash
   python src/select_features.py
   ```
5. **Step 5 - Feature Scaling, Majority Capping & Minority SMOTE:**
   ```bash
   python src/resample_data.py
   ```
6. **Step 6 - Baseline Models Training (LR, RF, XGBoost):**
   ```bash
   python src/train_baselines.py
   ```
7. **Step 7 - Stacking Ensemble (5-Fold Out-of-Fold CV & Meta-Learner):**
   ```bash
   python src/train_stack.py
   ```
8. **Step 8 - Diagnostics, Error Breakdown & Ablation Study:**
   ```bash
   python src/diagnostics_ablation.py
   ```
9. **Step 9 - Freeze Pipeline, Permutation Importance & Report Assets:**
   ```bash
   python src/report_assets.py
   ```
10. **Step 10 - Verify Dashboard Inference Logic:**
    ```bash
    python tests/check_dashboard_logic.py
    ```

---

## Interactive Streamlit Dashboard

To launch the web-based interactive intrusion detection dashboard:

```bash
python -m streamlit run app.py
```

The application provides:
- Model toggling between **Stacked Ensemble** (default) and **XGBoost**.
- One-click loading of bundled demo flows from `demo/` or custom CSV upload.
- Metric cards for analyzed flows, attack percentage, quarantined invalid flows, latency, and throughput.
- Filterable predictions table with confidence slider and CSV export.
- Dynamic evaluation with row-normalized confusion matrix heatmaps and per-class breakdowns whenever ground-truth labels are present.

---

## Benchmark Results

The following evaluation metrics were evaluated on the held-out test set ($N = 499,596$ flows) and generated directly by `src/make_readme_table.py` from `outputs/final_results.csv`:

| Model | 15-Class Acc | 15-Class Macro F1 | 13-Class Macro F1 | Attack Precision | Attack Recall | Attack F1 | Benign FPR (%) | FP Count | FN Count | ROC-AUC | Average Precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.93512 | 0.54071 | 0.62217 | 0.72970 | 0.99574 | 0.84221 | 7.5775% | 31,405 | 363 | 0.99366 | 0.96811 |
| Random Forest | 0.99807 | 0.86217 | 0.93113 | 0.99156 | 0.99945 | 0.99549 | 0.1747% | 724 | 47 | 0.99995 | 0.99969 |
| XGBoost | 0.99839 | 0.86609 | 0.94263 | 0.99295 | 0.99982 | 0.99638 | 0.1457% | 604 | 15 | 0.99997 | 0.99985 |
| Stacked Ensemble | 0.99847 | 0.87276 | 0.94853 | 0.99324 | 0.99972 | 0.99647 | 0.1397% | 579 | 24 | 0.99995 | 0.99974 |

---

## Security Notice on `.joblib` Artifacts

> **Security Warning:** The model files in `models/*.joblib` are serialized using Python's `joblib` (pickle-based) mechanism. Loading untrusted joblib files can lead to arbitrary code execution. Only load `.joblib` files from trusted sources. Furthermore, these artifacts are strictly coupled to the exact Python and package versions pinned in `requirements_frozen.txt`.

---

## Project Limitations

- **Random (Not Time-Based) Split:** The train/test split was performed using stratified random sampling across capture days rather than strict temporal partitioning. Performance on future network traffic under temporal concept drift may be lower than reported.
- **Extreme Minority Classes:** Classes with only 2–7 test flows (`Heartbleed`: 2 flows, `Infiltration`: 7 flows, `Web Attack - Sql Injection`: 4 flows) have wide statistical confidence intervals and unreliable score estimates.
- **Destination Port Influence:** Permutation feature importance confirms that `Destination Port` is the single most influential feature ($0.3021$ drop in Macro F1 when permuted); the models rely heavily on standard service port bindings.
- **Ensemble vs. XGBoost Significance:** In paired bootstrap statistical significance testing, the Stacked Ensemble ($0.87276$ Macro F1) did not show a statistically significant difference compared to standalone XGBoost ($0.86609$ Macro F1).
- **SMOTE and Boruta Gains:** Systematic ablation across 3 random seeds demonstrated that neither SMOTE synthetic oversampling nor Boruta feature reduction provided measurable gains over training directly on raw distributions.
- **Web Attack Confusion:** High cross-class confusion occurs between `Web Attack - XSS` and `Web Attack - Brute Force` ($77.69\%$ of XSS test flows are misclassified as Web Brute Force).
- **Throughput Measurement:** Inference latency and throughput benchmarks were evaluated on pre-extracted statistical tabular flow vectors in batch and single-flow Python calls, not on live wire packets or raw network interfaces.
- **Academic Prototype:** This codebase is an academic prototype and research benchmark, not a hardened, real-time enterprise Intrusion Prevention System (IPS).

---

## Project Team & Supervision

**Department of Artificial Intelligence, Anurag University**

- **Kakarlapudi Vaishnavi** (23EG106B24)
- **Mandali Durga Dedeepya** (23EG106B35)
- **Pannala Nikitha** (23EG106B47)

**Project Guide:**
- **Mr. G. Vamshi Krishna**, Anurag University
