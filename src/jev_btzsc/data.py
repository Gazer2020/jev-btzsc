"""BTZSC paired-row reconstruction, pinned to the official harness behavior.

Grouping follows IliasAarab/btzsc@4eac6a9 `src/btzsc/data.py`: consecutive
(text, hypothesis, binary-entailment) rows are rebuilt into multiclass samples.
This module does not depend on the `btzsc` PyPI package (that package pulls
torch / transformers / Gradio, which the pilot does not need).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from datasets import concatenate_datasets, load_dataset

from jev_btzsc.protocol import (
    DATASET_DOMAIN,
    DATASET_TASK,
    HARNESS_SAMPLING_SEED,
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
)


@dataclass(frozen=True)
class GroupedExample:
    grouped_index: int
    text: str
    verbalizers: tuple[str, ...]
    label_texts: tuple[str, ...]
    binary_labels: tuple[int, ...]
    reference_index: int | None
    text_sha256: str


@dataclass(frozen=True)
class GroupedDataset:
    name: str
    task: str
    domain: str
    n_classes: int
    verbalizers: tuple[str, ...]
    label_texts: tuple[str, ...]
    examples: tuple[GroupedExample, ...]
    n_rows: int
    n_no_positive: int
    harness_pattern_n_classes: int
    first_text_n_classes: int


def harness_n_classes_from_binary(labels: Sequence[int]) -> int:
    """Official harness `_get_n_classes`: first repeat of the opening binary label."""
    if not labels:
        return 0
    first = int(labels[0])
    for i in range(1, len(labels)):
        if int(labels[i]) == first:
            return i
    return len(labels)


def n_classes_from_first_text(texts: Sequence[str]) -> int:
    """Rows sharing the first text form one multiclass example."""
    if not texts:
        return 0
    first = texts[0]
    count = 0
    for text in texts:
        if text != first:
            break
        count += 1
    return count


def _normalize_entailment(value: object) -> int:
    if isinstance(value, str):
        if value in {"1", "entailment"}:
            return 1
        if value in {"0", "not_entailment"}:
            return 0
    return int(value)


def group_paired_rows(
    *,
    name: str,
    texts: Sequence[str],
    hypotheses: Sequence[str],
    binary_labels: Sequence[int],
    label_texts: Sequence[str] | None = None,
    n_classes: int | None = None,
) -> GroupedDataset:
    import hashlib

    if not (len(texts) == len(hypotheses) == len(binary_labels)):
        raise ValueError("text / hypothesis / labels columns must be the same length")
    if label_texts is not None and len(label_texts) != len(texts):
        raise ValueError("label_text column length must match texts")

    labels = [_normalize_entailment(v) for v in binary_labels]
    pattern_n = harness_n_classes_from_binary(labels)
    text_n = n_classes_from_first_text(texts)
    resolved_n = n_classes if n_classes is not None else text_n
    if resolved_n <= 0:
        raise ValueError(f"{name}: could not infer class count")

    n_rows = len(texts)
    n_groups = n_rows // resolved_n
    examples: list[GroupedExample] = []
    n_no_positive = 0
    verbalizers: tuple[str, ...] | None = None
    first_label_texts: tuple[str, ...] | None = None

    for grouped_index in range(n_groups):
        offset = grouped_index * resolved_n
        sample_verbalizers = tuple(hypotheses[offset + j] for j in range(resolved_n))
        sample_label_texts = tuple(
            (label_texts[offset + j] if label_texts is not None else hypotheses[offset + j])
            for j in range(resolved_n)
        )
        sample_binary = tuple(labels[offset + j] for j in range(resolved_n))
        if verbalizers is None:
            verbalizers = sample_verbalizers
            first_label_texts = sample_label_texts
        positives = [j for j, flag in enumerate(sample_binary) if flag == 1]
        if len(positives) != 1:
            n_no_positive += 1
            reference_index = None
        else:
            reference_index = positives[0]
        text = texts[offset]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        examples.append(
            GroupedExample(
                grouped_index=grouped_index,
                text=text,
                verbalizers=sample_verbalizers,
                label_texts=sample_label_texts,
                binary_labels=sample_binary,
                reference_index=reference_index,
                text_sha256=digest,
            )
        )

    return GroupedDataset(
        name=name,
        task=DATASET_TASK[name],
        domain=DATASET_DOMAIN[name],
        n_classes=resolved_n,
        verbalizers=verbalizers or (),
        label_texts=first_label_texts or (),
        examples=tuple(examples),
        n_rows=n_rows,
        n_no_positive=n_no_positive,
        harness_pattern_n_classes=pattern_n,
        first_text_n_classes=text_n,
    )


def _apply_yahootopics_harness_quirk(ds):  # noqa: ANN001
    """Keep the official loader's Yahoo Topics filter (first 20 rows + drop BF)."""
    first_20 = ds.select(range(min(20, len(ds))))
    filtered = ds.filter(lambda sample: sample["label_text"] != "Business & Finance")
    return concatenate_datasets([first_20, filtered])


def load_raw_split(name: str, *, cache_dir: str | None = None):
    """Load the pinned Hugging Face test split. Does not call Jev."""
    ds = load_dataset(
        HF_DATASET_REPO,
        name=name,
        split="test",
        revision=HF_DATASET_REVISION,
        cache_dir=cache_dir,
    )
    if name == "yahootopics":
        ds = _apply_yahootopics_harness_quirk(ds)
    return ds


def load_grouped_dataset(name: str, *, cache_dir: str | None = None) -> GroupedDataset:
    ds = load_raw_split(name, cache_dir=cache_dir)
    label_texts = ds["label_text"] if "label_text" in ds.column_names else None
    return group_paired_rows(
        name=name,
        texts=ds["text"],
        hypotheses=ds["hypothesis"],
        binary_labels=ds["labels"],
        label_texts=label_texts,
    )


def inspect_dataset(
    name: str,
    *,
    cache_dir: str | None = None,
    head_rows: int = 1024,
) -> GroupedDataset:
    """Class-count / verbalizer inspect from a prefix. Avoids grouping million-row splits."""
    ds = load_raw_split(name, cache_dir=cache_dir)
    n_rows = len(ds)
    head = ds.select(range(min(head_rows, n_rows)))
    label_texts = head["label_text"] if "label_text" in head.column_names else None
    grouped = group_paired_rows(
        name=name,
        texts=head["text"],
        hypotheses=head["hypothesis"],
        binary_labels=head["labels"],
        label_texts=label_texts,
    )
    return GroupedDataset(
        name=grouped.name,
        task=grouped.task,
        domain=grouped.domain,
        n_classes=grouped.n_classes,
        verbalizers=grouped.verbalizers,
        label_texts=grouped.label_texts,
        examples=grouped.examples,
        n_rows=n_rows,
        n_no_positive=grouped.n_no_positive,
        harness_pattern_n_classes=grouped.harness_pattern_n_classes,
        first_text_n_classes=grouped.first_text_n_classes,
    )


def sample_indices(n_groups: int, k: int, *, seed: int = HARNESS_SAMPLING_SEED) -> list[int]:
    """Official harness `max_samples` index selection (seed=0, keep groups 0 and 1)."""
    if k <= 0:
        return []
    if n_groups <= 0:
        raise ValueError("dataset has no grouped examples")
    take = min(k, n_groups)
    if take == 1:
        return [0]
    if take == n_groups:
        return list(range(n_groups))
    rng = random.Random(seed)
    remaining = rng.sample(range(2, n_groups), k=take - 2)
    return [0, 1, *remaining]


def select_pilot_examples(dataset: GroupedDataset, k: int) -> list[GroupedExample]:
    indices = sample_indices(len(dataset.examples), k)
    return [dataset.examples[i] for i in indices]


def references_array(examples: Sequence[GroupedExample]) -> np.ndarray:
    values = []
    for example in examples:
        if example.reference_index is None:
            values.append(-1)
        else:
            values.append(example.reference_index)
    return np.array(values, dtype=np.int64)
