from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true: Any, y_pred: Any, y_probs: Any | None = None) -> dict:
    """Calcula métricas de clasificación binarias (accuracy, precision, recall, f1, roc_auc opc.)."""
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

    if y_probs is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_probs)
        except Exception:
            metrics["roc_auc"] = None

    return metrics
