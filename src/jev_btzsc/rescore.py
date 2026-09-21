"""Rescore existing Jev predictions with BTZSC-current-valid (no extra API calls)."""

from __future__ import annotations

import json
from pathlib import Path

from jev_btzsc.data import load_grouped_dataset, select_pilot_examples
from jev_btzsc.io import write_json
from jev_btzsc.protocol import CURRENT_VALID_TAG
from jev_btzsc.scoring import score_valid


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rescore_pilot_banking77(
    *,
    predictions_path: Path,
    sample_manifest_path: Path | None = None,
    cache_dir: str | None = None,
) -> dict:
    grouped = load_grouped_dataset("banking77", cache_dir=cache_dir)
    selected = {ex.grouped_index: ex for ex in select_pilot_examples(grouped, 100)}
    if sample_manifest_path and sample_manifest_path.exists():
        manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
        ids = [row["grouped_index"] for row in manifest["datasets"]["banking77"]["examples"]]
        selected = {i: grouped.examples[i] for i in ids}

    preds = {
        row["grouped_index"]: row
        for row in _load_jsonl(predictions_path)
        if row.get("dataset") == "banking77"
    }
    examples = [selected[i] for i in selected]
    report = score_valid(examples, preds, n_classes=grouped.n_classes, dataset="banking77")
    report["tag"] = CURRENT_VALID_TAG
    report["n_predictions"] = len(preds)
    report["source_predictions"] = str(predictions_path)
    return report
