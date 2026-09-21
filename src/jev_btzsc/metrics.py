"""Pilot metrics. Primary: per-dataset macro-F1, then equal-weight average."""

from __future__ import annotations

from typing import Any, Sequence, cast

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(
    predictions: Sequence[int],
    references: Sequence[int],
    *,
    n_classes: int,
    zero_division: float = 0.0,
) -> dict[str, float]:
    labels = list(range(n_classes))
    preds = np.asarray(predictions, dtype=np.int64)
    refs = np.asarray(references, dtype=np.int64)
    zero_division_any = cast(Any, zero_division)
    return {
        "macro_f1": float(
            f1_score(
                refs,
                preds,
                average="macro",
                labels=labels,
                zero_division=zero_division_any,
            )
        ),
        "accuracy": float(accuracy_score(refs, preds)),
        "macro_precision": float(
            precision_score(
                refs,
                preds,
                average="macro",
                labels=labels,
                zero_division=zero_division_any,
            )
        ),
        "macro_recall": float(
            recall_score(
                refs,
                preds,
                average="macro",
                labels=labels,
                zero_division=zero_division_any,
            )
        ),
    }


def equal_weight_mean(per_dataset: dict[str, dict[str, float]], key: str) -> float:
    values = [metrics[key] for metrics in per_dataset.values() if key in metrics]
    if not values:
        return 0.0
    return float(np.mean(values))
