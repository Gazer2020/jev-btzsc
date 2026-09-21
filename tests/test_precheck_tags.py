"""Precheck tagging without downloading Hugging Face data."""

from jev_btzsc.protocol import CURRENT_REVISION_TAG, PAPER_ALIGNED_TAG, PAPER_CLASS_COUNTS


def test_revision_tags() -> None:
    assert PAPER_ALIGNED_TAG == "paper-table1"
    assert CURRENT_REVISION_TAG == "BTZSC-current"
    assert PAPER_CLASS_COUNTS["agnews"] == 4
    assert PAPER_CLASS_COUNTS["emotiondair"] == 6
    assert PAPER_CLASS_COUNTS["amazonpolarity"] == 2
