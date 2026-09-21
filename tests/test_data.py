"""Paired-row grouping and harness seed=0 sampling. No Hugging Face / Jev calls."""

from jev_btzsc.data import group_paired_rows, sample_indices, select_pilot_examples
from jev_btzsc.metrics import compute_metrics, equal_weight_mean


def _amazon_rows() -> dict:
    texts = ["good phone", "good phone", "bad case", "bad case", "ok", "ok"]
    hyps = [
        "The overall sentiment within the Amazon product review is positive",
        "The overall sentiment within the Amazon product review is negative",
        "The overall sentiment within the Amazon product review is positive",
        "The overall sentiment within the Amazon product review is negative",
        "The overall sentiment within the Amazon product review is positive",
        "The overall sentiment within the Amazon product review is negative",
    ]
    labels = [1, 0, 0, 1, 1, 0]
    label_texts = ["positive", "negative", "positive", "negative", "positive", "negative"]
    return {
        "texts": texts,
        "hypotheses": hyps,
        "binary_labels": labels,
        "label_texts": label_texts,
    }


def test_group_paired_rows_uses_first_text_block() -> None:
    rows = _amazon_rows()
    grouped = group_paired_rows(name="amazonpolarity", **rows)
    assert grouped.n_classes == 2
    assert grouped.first_text_n_classes == 2
    assert len(grouped.examples) == 3
    assert grouped.examples[0].reference_index == 0
    assert grouped.examples[1].reference_index == 1
    assert grouped.n_no_positive == 0
    assert grouped.verbalizers[0].endswith("positive")


def test_no_positive_candidate_is_oos_not_class_zero() -> None:
    grouped = group_paired_rows(
        name="banking77",
        texts=["oos", "oos"],
        hypotheses=["intent a", "intent b"],
        binary_labels=[0, 0],
        label_texts=["a", "b"],
        n_classes=2,
    )
    assert grouped.n_oos == 1
    assert grouped.n_anomaly == 0
    assert grouped.examples[0].reference_index is None
    assert grouped.examples[0].validity == "oos"


def test_multi_positive_is_anomaly() -> None:
    grouped = group_paired_rows(
        name="banking77",
        texts=["x", "x"],
        hypotheses=["intent a", "intent b"],
        binary_labels=[1, 1],
        label_texts=["a", "b"],
        n_classes=2,
    )
    assert grouped.n_anomaly == 1
    assert grouped.n_oos == 0
    assert grouped.examples[0].validity == "anomaly"
    assert grouped.examples[0].reference_index is None


def test_sample_indices_match_harness_seed_zero() -> None:
    first = sample_indices(20, 5, seed=0)
    second = sample_indices(20, 5, seed=0)
    assert first == second
    assert first[:2] == [0, 1]
    assert len(first) == 5
    assert len(set(first)) == 5


def test_select_pilot_examples_is_deterministic() -> None:
    rows = _amazon_rows()
    grouped = group_paired_rows(name="amazonpolarity", **rows)
    a = select_pilot_examples(grouped, 3)
    b = select_pilot_examples(grouped, 3)
    assert [ex.grouped_index for ex in a] == [ex.grouped_index for ex in b]


def test_macro_f1_equal_weight_average() -> None:
    metrics_a = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0], n_classes=2)
    metrics_b = compute_metrics([0, 0, 0, 0], [0, 1, 0, 1], n_classes=2)
    overall = equal_weight_mean(
        {"a": metrics_a, "b": metrics_b},
        "macro_f1",
    )
    assert 0.0 <= metrics_a["macro_f1"] <= 1.0
    assert overall == (metrics_a["macro_f1"] + metrics_b["macro_f1"]) / 2
