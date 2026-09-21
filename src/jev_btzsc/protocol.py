"""Pinned protocol constants for the Phase A BTZSC × Jev pilot.

Frozen before any Jev inference. Do not tune the global instruction from
test errors. Do not use BTZSC train/dev for training, hyperparameter search,
or model selection.
"""

from __future__ import annotations

from dataclasses import dataclass

HARNESS_REPO = "https://github.com/IliasAarab/btzsc"
HARNESS_COMMIT = "4eac6a9561cb0a600ba65d94c62c3ce7f3e89547"
HARNESS_SAMPLING_SEED = 0

HF_DATASET_REPO = "btzsc/btzsc"
HF_DATASET_REVISION = "fef2a2ac62b69c58670047dddf045c53d7c3cb5e"

# Official TypeSafe System One HTTP API. Do not point this at unofficial proxies.
TYPESAFE_API_BASE_URL = "https://api.typesafe.ai"
JEV_MODEL_DEFAULT = "jev-latest"
QUESTION_NAME = "label"

# TypeSafe list price as of the Jev launch post (output tokens are free).
INPUT_USD_PER_MILLION_TOKENS = 0.042
PILOT_BUDGET_USD = 0.10
FULL_HARD_STOP_USD = 4.50
FULL_TARGET_USD = 3.00
CURRENT_VALID_TAG = "BTZSC-current-valid"

PAPER_ALIGNED_TAG = "paper-table1"
CURRENT_REVISION_TAG = "BTZSC-current"

# One global instruction for every dataset. Do not specialize per task.
GLOBAL_INSTRUCTIONS = (
    "Assign the input text to exactly one class. Each option description is "
    "the official BTZSC label verbalizer for that class. Choose the single "
    "best match. Do not invent classes."
)

BTZSC_DATASETS: tuple[str, ...] = (
    "amazonpolarity",
    "imdb",
    "appreviews",
    "yelpreviews",
    "rottentomatoes",
    "financialphrasebank",
    "emotiondair",
    "empathetic",
    "banking77",
    "biasframes_intent",
    "massive",
    "agnews",
    "yahootopics",
    "trueteacher",
    "manifesto",
    "capsotu",
    "biasframes_offensive",
    "biasframes_sex",
    "wikitoxic_toxicaggregated",
    "wikitoxic_obscene",
    "wikitoxic_threat",
    "wikitoxic_insult",
)

TASK_GROUPS: dict[str, tuple[str, ...]] = {
    "sentiment": (
        "amazonpolarity",
        "imdb",
        "appreviews",
        "yelpreviews",
        "rottentomatoes",
        "financialphrasebank",
    ),
    "emotion": ("emotiondair", "empathetic"),
    "intent": ("banking77", "biasframes_intent", "massive"),
    "topic": (
        "agnews",
        "yahootopics",
        "trueteacher",
        "manifesto",
        "capsotu",
        "biasframes_offensive",
        "biasframes_sex",
        "wikitoxic_toxicaggregated",
        "wikitoxic_obscene",
        "wikitoxic_threat",
        "wikitoxic_insult",
    ),
}

DATASET_TASK: dict[str, str] = {
    name: task for task, names in TASK_GROUPS.items() for name in names
}

DATASET_DOMAIN: dict[str, str] = {
    "amazonpolarity": "e-commerce",
    "imdb": "movies",
    "appreviews": "apps",
    "yelpreviews": "local-business",
    "rottentomatoes": "movies",
    "financialphrasebank": "finance",
    "emotiondair": "social-media",
    "empathetic": "dialogue",
    "banking77": "banking",
    "biasframes_intent": "social-media",
    "massive": "assistant",
    "agnews": "news",
    "yahootopics": "qa-forum",
    "trueteacher": "education",
    "manifesto": "politics",
    "capsotu": "politics",
    "biasframes_offensive": "social-media",
    "biasframes_sex": "social-media",
    "wikitoxic_toxicaggregated": "wikipedia",
    "wikitoxic_obscene": "wikipedia",
    "wikitoxic_threat": "wikipedia",
    "wikitoxic_insult": "wikipedia",
}

# ICLR 2026 BTZSC Table 1 class counts. Banking77 must be 77 for paper alignment.
PAPER_CLASS_COUNTS: dict[str, int] = {
    "amazonpolarity": 2,
    "imdb": 2,
    "appreviews": 2,
    "yelpreviews": 2,
    "rottentomatoes": 2,
    "financialphrasebank": 3,
    "emotiondair": 6,
    "empathetic": 32,
    "banking77": 77,
    "biasframes_intent": 2,
    "massive": 59,
    "agnews": 4,
    "yahootopics": 10,
    "trueteacher": 2,
    "manifesto": 56,
    "capsotu": 21,
    "biasframes_offensive": 2,
    "biasframes_sex": 2,
    "wikitoxic_toxicaggregated": 2,
    "wikitoxic_obscene": 2,
    "wikitoxic_threat": 2,
    "wikitoxic_insult": 2,
}

ICLR_TABLE2_BASELINES: tuple[tuple[str, float, float], ...] = (
    ("ModernBERT-large (base)", 0.27, 0.35),
    ("BART-large-MNLI", 0.51, 0.53),
    ("DeBERTa-v3-large-nli-triplet", 0.60, 0.62),
    ("GTE-large-en-v1.5", 0.62, 0.64),
    ("Qwen3-Reranker-0.6B", 0.61, 0.64),
    ("Qwen3-Reranker-8B", 0.72, 0.76),
    ("Qwen3-4B", 0.65, 0.70),
    ("Qwen3-8B", 0.66, 0.71),
    ("Mistral-Nemo-Instruct-2407", 0.67, 0.71),
)


@dataclass(frozen=True)
class PilotDatasetSpec:
    name: str
    task: str
    paper_classes: int
    n_samples: int


PILOT_DATASETS: tuple[PilotDatasetSpec, ...] = (
    PilotDatasetSpec("amazonpolarity", "sentiment", 2, 100),
    PilotDatasetSpec("agnews", "topic", 4, 100),
    PilotDatasetSpec("banking77", "intent", 77, 100),
    PilotDatasetSpec("emotiondair", "emotion", 6, 100),
)

PILOT_TOTAL_CALLS = sum(spec.n_samples for spec in PILOT_DATASETS)

FULL_RUN_DATASETS: tuple[str, ...] = (
    "banking77",
    "massive",
    "empathetic",
    "emotiondair",
    "agnews",
    "financialphrasebank",
)


def option_id(index: int) -> str:
    """Stable non-semantic Choice option id (L000, L001, …)."""
    if index < 0:
        raise ValueError(f"option index must be >= 0, got {index}")
    return f"L{index:03d}"


def option_index(option: str) -> int:
    if not option.startswith("L") or len(option) != 4 or not option[1:].isdigit():
        raise ValueError(f"invalid option id: {option!r}")
    return int(option[1:])


def token_cost_usd(input_tokens: int | None) -> float | None:
    if input_tokens is None:
        return None
    return input_tokens * INPUT_USD_PER_MILLION_TOKENS / 1_000_000
