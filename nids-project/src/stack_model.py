"""
Stacking Ensemble Model Definition for Network Intrusion Detection (CICIDS2017).

Holds:
- base_models: [Logistic Regression, Random Forest, XGBoost]
- meta_model: Logistic Regression
- classes_: array of class integers [0..14]
- feature_names: list of 65 selected features
"""

from typing import List, Sequence
import numpy as np


class StackedIDS:
    """
    Stacked Ensemble Classifier for Network Intrusion Detection.
    Base models: [Logistic Regression, Random Forest, XGBoost] in fixed order.
    Meta learner: Logistic Regression fitted on concatenated out-of-fold probabilities.
    """

    def __init__(
        self,
        base_models: List,
        meta_model,
        classes_: Sequence,
        feature_names: List[str]
    ):
        self.base_models = base_models
        self.meta_model = meta_model
        self.classes_ = np.array(classes_)
        self.feature_names = list(feature_names)

    def _get_feature_array(self, X) -> np.ndarray:
        if hasattr(X, "columns"):
            return X[self.feature_names].values
        return np.asarray(X)

    def _construct_meta_features(self, X) -> np.ndarray:
        X_arr = self._get_feature_array(X)
        proba_list = []
        for model in self.base_models:
            proba_list.append(model.predict_proba(X_arr))
        # Concatenate 3 * 15 = 45 probability columns
        return np.hstack(proba_list)

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities from the meta-learner."""
        meta_features = self._construct_meta_features(X)
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X) -> np.ndarray:
        """Return class predictions from the meta-learner."""
        meta_features = self._construct_meta_features(X)
        return self.meta_model.predict(meta_features)
