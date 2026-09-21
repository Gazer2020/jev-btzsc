"""BTZSC-current-valid scoring: eval only examples with exactly one entailment=1."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from jev_btzsc.data import GroupedExample, Validity
from jev_btzsc.metrics import compute_metrics
from jev_btzsc.protocol import CURRENT_REVISION_TAG, CURRENT_VALID_TAG, PAPER_CLASS_COUNTS


def latency_percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def score_valid(
    examples: Sequence[GroupedExample],
    predictions: Mapping[int, Mapping[str, Any]],
    *,
    n_classes: int,
    dataset: str,
) -> dict[str, Any]:
    n_raw = len(examples)
    n_valid = n_oos = n_anomaly = 0
    preds: list[int] = []
    refs: list[int] = []
    api_failures: list[dict[str, Any]] = []
    latencies: list[float] = []
    tokens = 0
    cost = 0.0
    models: dict[str, int] = {}
    usable_calls = 0

    for example in examples:
        if example.validity == "valid":
            n_valid += 1
        elif example.validity == "oos":
            n_oos += 1
        else:
            n_anomaly += 1

        row = predictions.get(example.grouped_index)
        if row is None:
            if example.validity == "valid":
                api_failures.append(
                    {"grouped_index": example.grouped_index, "error": "missing_prediction"}
                )
            continue
        if row.get("latency_ms") is not None:
            latencies.append(float(row["latency_ms"]))
        if row.get("input_tokens"):
            tokens += int(row["input_tokens"])
        if row.get("cost_usd"):
            cost += float(row["cost_usd"])
        model = row.get("resolved_model")
        if model:
            models[str(model)] = models.get(str(model), 0) + 1
        if row.get("ok"):
            usable_calls += 1
        else:
            api_failures.append(
                {
                    "grouped_index": example.grouped_index,
                    "error": row.get("error"),
                    "retries": row.get("retries"),
                }
            )
            continue
        if example.validity != "valid":
            continue
        predicted = row.get("predicted_index")
        if predicted is None or example.reference_index is None:
            api_failures.append(
                {"grouped_index": example.grouped_index, "error": "unusable_valid_row"}
            )
            continue
        preds.append(int(predicted))
        refs.append(int(example.reference_index))

    metrics = (
        compute_metrics(preds, refs, n_classes=n_classes)
        if preds
        else {
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
        }
    )
    paper = PAPER_CLASS_COUNTS.get(dataset)
    class_mismatch = paper is not None and n_classes != paper
    return {
        "dataset": dataset,
        "scoring": CURRENT_VALID_TAG,
        "revision_tag": CURRENT_REVISION_TAG if class_mismatch else "paper-table1",
        "n_classes_observed": n_classes,
        "n_classes_paper": paper,
        "n_raw": n_raw,
        "n_valid": n_valid,
        "n_oos": n_oos,
        "n_anomaly": n_anomaly,
        "n_scored": len(preds),
        "n_api_failures": len(api_failures),
        "usable_calls": usable_calls,
        "accuracy_valid": metrics["accuracy"],
        "macro_f1_valid": metrics["macro_f1"],
        "macro_precision_valid": metrics["macro_precision"],
        "macro_recall_valid": metrics["macro_recall"],
        "metrics_valid": metrics,
        "input_tokens": tokens,
        "cost_usd": cost,
        "latency_ms_p50": latency_percentile(latencies, 50),
        "latency_ms_p95": latency_percentile(latencies, 95),
        "resolved_models": models,
        "api_failures": api_failures,
    }


def examples_by_validity(examples: Iterable[GroupedExample]) -> dict[Validity, int]:
    counts: dict[Validity, int] = {"valid": 0, "oos": 0, "anomaly": 0}
    for example in examples:
        counts[example.validity] += 1
    return counts
