"""
predict_utils.py: Core Inference and Evaluation Logic for NIDS Dashboard.

Pure logic module (no Streamlit dependencies):
- load_artifacts: loads scaler, feature list, label maps, XGBoost, and StackedIDS.
- prepare_features: cleans, validates, checks for missing/NaN/inf values, returns X.
- predict: transforms features via scaler, runs predict_proba, returns predictions.
- evaluate: computes accuracy, macro F1, per-class metrics, and confusion matrix.
"""

from pathlib import Path
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Project paths setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import StackedIDS before joblib.load to guarantee unpickling success
try:
    from src.stack_model import StackedIDS
except ImportError:
    from stack_model import StackedIDS

# Alias module in sys.modules to handle both pickle naming conventions
import stack_model
sys.modules["src.stack_model"] = stack_model

# Global artifact cache
_GLOBAL_ARTIFACTS: Optional[Dict[str, Any]] = None


class PredictionResult(dict):
    """
    Flexible prediction container supporting both dict and attribute access,
    plus convenient export to pandas DataFrame.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "predicted_class_id": self["pred_class_id"],
            "predicted_class": self["pred_class_name"],
            "confidence": np.round(self["confidence"], 5),
            "attack_score": np.round(self["attack_score"], 5),
            "is_attack": self["is_attack"],
        })
        return df


class EvaluationResult(dict):
    """Container for evaluation metrics supporting dict and attribute access."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)


def load_artifacts(models_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load scaler, feature list, label maps, XGBoost, and StackedIDS models.
    Asserts structural integrity of loaded components.
    """
    global _GLOBAL_ARTIFACTS

    if models_dir is None:
        models_dir = PROJECT_ROOT / "models"
    else:
        models_dir = Path(models_dir)

    scaler_path = models_dir / "scaler.joblib"
    features_path = models_dir / "selected_features.json"
    label_maps_path = models_dir / "label_maps.json"
    xgb_path = models_dir / "xgboost.joblib"
    stack_path = models_dir / "stacked_ids.joblib"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Selected features not found at: {features_path}")
    if not label_maps_path.exists():
        raise FileNotFoundError(f"Label maps not found at: {label_maps_path}")
    if not xgb_path.exists():
        raise FileNotFoundError(f"XGBoost model not found at: {xgb_path}")
    if not stack_path.exists():
        raise FileNotFoundError(f"StackedIDS model not found at: {stack_path}")

    # 1. Load scaler and features
    scaler = joblib.load(scaler_path)
    with open(features_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)["selected"]

    # 2. Load label mappings
    with open(label_maps_path, "r", encoding="utf-8") as f:
        label_maps = json.load(f)

    # 3. Load models
    xgb = joblib.load(xgb_path)
    stack = joblib.load(stack_path)

    # Assertions required by specification
    assert scaler.n_features_in_ == 65, (
        f"Scaler n_features_in_ mismatch: expected 65, found {scaler.n_features_in_}"
    )
    if hasattr(scaler, "feature_names_in_"):
        assert list(scaler.feature_names_in_) == selected_features, (
            "Scaler feature_names_in_ does not match selected_features in order"
        )
    assert np.array_equal(xgb.classes_, np.arange(15)), (
        f"XGBoost classes_ unexpected: {xgb.classes_}"
    )
    assert np.array_equal(stack.classes_, np.arange(15)), (
        f"StackedIDS classes_ unexpected: {stack.classes_}"
    )

    # Build convenience lookups
    class_to_id = label_maps["multi15"]["class_to_id"]
    id_to_class = {int(k): v for k, v in label_maps["multi15"]["id_to_class"].items()}
    class_names_15 = [id_to_class[i] for i in range(15)]
    benign_id = class_to_id["BENIGN"]

    artifacts = {
        "scaler": scaler,
        "selected_features": selected_features,
        "label_maps": label_maps,
        "xgboost": xgb,
        "stacked_ids": stack,
        "class_to_id": class_to_id,
        "id_to_class": id_to_class,
        "class_names_15": class_names_15,
        "benign_id": benign_id,
    }

    _GLOBAL_ARTIFACTS = artifacts
    return artifacts


def get_artifacts() -> Dict[str, Any]:
    """Retrieve currently loaded artifacts or load them from default path."""
    global _GLOBAL_ARTIFACTS
    if _GLOBAL_ARTIFACTS is None:
        _GLOBAL_ARTIFACTS = load_artifacts()
    return _GLOBAL_ARTIFACTS


def prepare_features(
    df: pd.DataFrame,
    selected_features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare raw features for inference:
    - Strips whitespace from column names.
    - Validates presence of all required features (raises clear ValueError if missing).
    - Ignores extra columns (Label column is optional).
    - Coerces required features to numeric and maps +/-inf to NaN.
    - Identifies rows with any NaN/inf in required features: returns them in skipped_df.
    - Returns X as float32 in exact feature order, plus the kept row positions.

    Returns:
        X: np.ndarray of shape (N_kept, 65) float32
        kept_positions: np.ndarray of shape (N_kept,) with 0-indexed row positions
        skipped_df: pd.DataFrame of skipped rows with 'skip_reason' column
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input must be a pandas DataFrame, got {type(df)}")

    if selected_features is None:
        selected_features = get_artifacts()["selected_features"]

    # 1. Clean column names
    clean_cols_map = {c: str(c).strip() for c in df.columns}
    df_clean = df.rename(columns=clean_cols_map)

    # 2. Check for missing required features
    missing = [f for f in selected_features if f not in df_clean.columns]
    if missing:
        raise ValueError(
            f"Input data is missing {len(missing)} required feature(s): {missing}"
        )

    # 3. Extract required features in exact order
    feat_df = df_clean[selected_features].copy()

    # 4. Convert to numeric and replace +/-inf with NaN
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # 5. Identify valid vs skipped rows
    nan_mask = feat_df.isna().any(axis=1)
    kept_mask = ~nan_mask
    kept_positions = np.where(kept_mask.to_numpy())[0]
    skipped_positions = np.where(nan_mask.to_numpy())[0]

    # 6. Build skipped frame with reason
    skipped_df = df.iloc[skipped_positions].copy()
    if len(skipped_df) > 0:
        nan_cols_series = feat_df.iloc[skipped_positions].apply(
            lambda row: [col for col in selected_features if pd.isna(row[col])],
            axis=1,
        )
        skipped_df["skip_reason"] = nan_cols_series.apply(
            lambda cols: f"Contains NaN or Inf in required feature(s): {', '.join(cols[:3])}{'...' if len(cols) > 3 else ''}"
        )
    else:
        skipped_df["skip_reason"] = pd.Series(dtype="object")

    # 7. Build X array as float32
    X = feat_df.iloc[kept_positions].values.astype(np.float32)

    return X, kept_positions, skipped_df


def predict(
    model_name: Any,
    X: np.ndarray,
    artifacts: Optional[Dict[str, Any]] = None,
) -> PredictionResult:
    """
    Scale X with the saved scaler, run model.predict_proba, and return predictions.

    Args:
        model_name: "Stacked Ensemble", "XGBoost", or model object
        X: numpy array of shape (N, 65)
        artifacts: loaded artifacts dictionary (optional)

    Returns:
        PredictionResult dict with keys:
            - pred_class_id: class index (0-14)
            - pred_class_name: class string name
            - confidence: max predicted probability
            - attack_score: 1.0 - P(BENIGN)
            - is_attack: True if predicted class != BENIGN
            - probabilities: full probability distribution (N, 15)
            - elapsed_time: seconds taken for inference
    """
    if artifacts is None:
        artifacts = get_artifacts()

    # Resolve model
    if isinstance(model_name, str):
        normalized_name = model_name.strip().lower().replace(" ", "").replace("_", "")
        if normalized_name in ["stackedensemble", "stackedids", "stack"]:
            model = artifacts["stacked_ids"]
        elif normalized_name in ["xgboost", "xgb"]:
            model = artifacts["xgboost"]
        else:
            raise ValueError(
                f"Unknown model name '{model_name}'. Choose 'Stacked Ensemble' or 'XGBoost'."
            )
    else:
        model = model_name

    scaler = artifacts["scaler"]
    id_to_class = artifacts["id_to_class"]
    benign_id = artifacts["benign_id"]

    if len(X) == 0:
        return PredictionResult(
            pred_class_id=np.array([], dtype=int),
            pred_class_name=np.array([], dtype=object),
            confidence=np.array([], dtype=float),
            attack_score=np.array([], dtype=float),
            is_attack=np.array([], dtype=bool),
            probabilities=np.empty((0, 15), dtype=float),
            elapsed_time=0.0,
            # Aliases
            pred_id=np.array([], dtype=int),
            pred_name=np.array([], dtype=object),
            predicted_class_id=np.array([], dtype=int),
            predicted_class=np.array([], dtype=object),
        )

    # Transform and predict
    X_scaled = scaler.transform(X)
    t0 = time.perf_counter()
    proba = model.predict_proba(X_scaled)
    elapsed_time = time.perf_counter() - t0

    pred_class_id = np.argmax(proba, axis=1)
    pred_class_name = np.array([id_to_class[i] for i in pred_class_id], dtype=object)
    confidence = np.max(proba, axis=1)
    attack_score = 1.0 - proba[:, benign_id]
    is_attack = pred_class_id != benign_id

    result = PredictionResult(
        pred_class_id=pred_class_id,
        pred_class_name=pred_class_name,
        confidence=confidence,
        attack_score=attack_score,
        is_attack=is_attack,
        probabilities=proba,
        elapsed_time=elapsed_time,
        # Aliases for convenience
        pred_id=pred_class_id,
        pred_name=pred_class_name,
        predicted_class_id=pred_class_id,
        predicted_class=pred_class_name,
    )
    return result


def evaluate(
    y_true_names: Union[List[str], np.ndarray, pd.Series],
    y_pred_names: Union[List[str], np.ndarray, pd.Series],
    class_names: Optional[List[str]] = None,
) -> EvaluationResult:
    """
    Compute accuracy, macro F1, per-class metrics table, and 15x15 confusion matrix.
    """
    if class_names is None:
        class_names = get_artifacts()["class_names_15"]

    y_true_arr = np.asarray(y_true_names)
    y_pred_arr = np.asarray(y_pred_names)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=class_names)

    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    records = []
    for c in class_names:
        row_dict = report.get(c, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        records.append({
            "Class": c,
            "Precision": round(float(row_dict["precision"]), 5),
            "Recall": round(float(row_dict["recall"]), 5),
            "F1": round(float(row_dict["f1-score"]), 5),
            "Support": int(row_dict["support"]),
        })

    per_class_df = pd.DataFrame(records)

    eval_result = EvaluationResult(
        accuracy=acc,
        macro_f1=macro_f1,
        per_class_table=per_class_df,
        per_class_df=per_class_df,
        confusion_matrix=cm,
    )
    return eval_result
