"""BTZSC-current-valid scoring. No API calls."""

from jev_btzsc.data import group_paired_rows
from jev_btzsc.protocol import CURRENT_VALID_TAG
from jev_btzsc.scoring import score_valid


def test_score_valid_excludes_oos_and_anomaly() -> None:
    grouped = group_paired_rows(
        name="banking77",
        texts=["a", "a", "b", "b", "c", "c"],
        hypotheses=["x", "y", "x", "y", "x", "y"],
        binary_labels=[1, 0, 0, 0, 1, 1],
        label_texts=["x", "y", "x", "y", "x", "y"],
        n_classes=2,
    )
    predictions = {
        0: {"ok": True, "predicted_index": 0, "input_tokens": 10, "cost_usd": 0.001, "latency_ms": 100.0, "resolved_model": "jev-1.13.0"},
        1: {"ok": True, "predicted_index": 1, "input_tokens": 10, "cost_usd": 0.001, "latency_ms": 110.0, "resolved_model": "jev-1.13.0"},
        2: {"ok": True, "predicted_index": 0, "input_tokens": 10, "cost_usd": 0.001, "latency_ms": 120.0, "resolved_model": "jev-1.13.0"},
    }
    report = score_valid(grouped.examples, predictions, n_classes=2, dataset="banking77")
    assert report["scoring"] == CURRENT_VALID_TAG
    assert report["n_raw"] == 3
    assert report["n_valid"] == 1
    assert report["n_oos"] == 1
    assert report["n_anomaly"] == 1
    assert report["n_scored"] == 1
    assert report["accuracy_valid"] == 1.0
    assert report["revision_tag"] == "BTZSC-current"
