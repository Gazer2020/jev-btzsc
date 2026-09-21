"""Adapter request shape. Does not call the TypeSafe API."""

from jev_btzsc.adapter import build_choice
from jev_btzsc.protocol import GLOBAL_INSTRUCTIONS, option_id


def test_choice_uses_l_ids_and_frozen_instructions() -> None:
    verbalizers = (
        "The overall sentiment within the Amazon product review is positive",
        "The overall sentiment within the Amazon product review is negative",
    )
    choice = build_choice(verbalizers)
    payload = choice.model_dump() if hasattr(choice, "model_dump") else dict(choice)
    criteria = payload.get("criteria") or getattr(choice, "criteria")
    instructions = payload.get("instructions") or getattr(choice, "instructions")
    assert instructions == GLOBAL_INSTRUCTIONS
    assert list(criteria.keys()) == [option_id(0), option_id(1)]
    assert criteria[option_id(0)] == verbalizers[0]
    assert criteria[option_id(1)] == verbalizers[1]
