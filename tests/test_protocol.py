"""Tests for protocol helpers. No network, no Jev calls."""

from jev_btzsc.protocol import (
    BTZSC_DATASETS,
    PAPER_CLASS_COUNTS,
    PILOT_DATASETS,
    PILOT_TOTAL_CALLS,
    option_id,
    option_index,
    token_cost_usd,
)


def test_option_ids_are_non_semantic_and_stable() -> None:
    assert option_id(0) == "L000"
    assert option_id(76) == "L076"
    assert option_index("L000") == 0
    assert option_index("L076") == 76


def test_twenty_two_datasets_and_banking77_paper_count() -> None:
    assert len(BTZSC_DATASETS) == 22
    assert len(PAPER_CLASS_COUNTS) == 22
    assert set(BTZSC_DATASETS) == set(PAPER_CLASS_COUNTS)
    assert PAPER_CLASS_COUNTS["banking77"] == 77


def test_pilot_is_four_hundred_calls() -> None:
    assert PILOT_TOTAL_CALLS == 400
    assert [spec.name for spec in PILOT_DATASETS] == [
        "amazonpolarity",
        "agnews",
        "banking77",
        "emotiondair",
    ]


def test_token_cost_uses_official_list_price() -> None:
    # $0.042 / 1M input tokens
    assert token_cost_usd(1_000_000) == 0.042
    assert token_cost_usd(None) is None
