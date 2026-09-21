"""Phase A 400-call Jev pilot runner. Does not train or tune prompts."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from statistics import median

from jev_btzsc.adapter import classify_text, make_client, record_to_dict
from jev_btzsc.data import load_grouped_dataset, select_pilot_examples
from jev_btzsc.io import write_json
from jev_btzsc.metrics import compute_metrics, equal_weight_mean
from jev_btzsc.precheck import run_precheck
from jev_btzsc.protocol import (
    CURRENT_REVISION_TAG,
    GLOBAL_INSTRUCTIONS,
    HF_DATASET_REVISION,
    JEV_MODEL_DEFAULT,
    PILOT_BUDGET_USD,
    PILOT_DATASETS,
    PILOT_TOTAL_CALLS,
    option_id,
)


def _model_name() -> str:
    return os.environ.get("TYPESAFE_DEFAULT_MODEL", JEV_MODEL_DEFAULT).strip() or JEV_MODEL_DEFAULT


def run_pilot(
    *,
    output_dir: Path,
    cache_dir: str | None = None,
    accept_btzsc_current: bool = False,
    dry_run: bool = False,
) -> dict:
    precheck_dir = output_dir / "precheck"
    precheck = run_precheck(
        output_dir=precheck_dir,
        cache_dir=cache_dir,
        datasets=None,
    )
    if not precheck["banking77_is_77"] and not accept_btzsc_current:
        raise RuntimeError(
            "Banking77 is not 77 classes on the pinned HF revision "
            f"{HF_DATASET_REVISION}. Tag={precheck['revision_tag']}. "
            "Refuse Table 2 comparison. Re-run with --accept-btzsc-current only "
            "to execute a BTZSC-current validity check, not an apples-to-apples "
            "ICLR Table 2 run."
        )

    sample_manifest: dict = {
        "seed": 0,
        "hf_revision": HF_DATASET_REVISION,
        "revision_tag": precheck["revision_tag"],
        "global_instructions": GLOBAL_INSTRUCTIONS,
        "model_requested": _model_name(),
        "datasets": {},
    }
    planned_calls: list[tuple[str, object]] = []
    grouped_by_name = {}
    for spec in PILOT_DATASETS:
        grouped = load_grouped_dataset(spec.name, cache_dir=cache_dir)
        grouped_by_name[spec.name] = grouped
        chosen = select_pilot_examples(grouped, spec.n_samples)
        sample_manifest["datasets"][spec.name] = {
            "task": spec.task,
            "n_classes": grouped.n_classes,
            "paper_classes": spec.paper_classes,
            "n_selected": len(chosen),
            "n_no_positive_in_selection": sum(1 for ex in chosen if ex.reference_index is None),
            "option_ids": [option_id(i) for i in range(grouped.n_classes)],
            "verbalizers": list(grouped.verbalizers),
            "examples": [
                {
                    "grouped_index": ex.grouped_index,
                    "text_sha256": ex.text_sha256,
                    "reference_index": ex.reference_index,
                    "reference_option_id": (
                        option_id(ex.reference_index) if ex.reference_index is not None else None
                    ),
                }
                for ex in chosen
            ],
        }
        for example in chosen:
            planned_calls.append((spec.name, example))

    write_json(output_dir / "sample_manifest.json", sample_manifest)

    if dry_run:
        summary = {
            "status": "dry_run",
            "n_planned_calls": len(planned_calls),
            "revision_tag": precheck["revision_tag"],
            "precheck": precheck,
        }
        write_json(output_dir / "summary.json", summary)
        return summary

    if len(planned_calls) != PILOT_TOTAL_CALLS:
        raise RuntimeError(
            f"expected {PILOT_TOTAL_CALLS} pilot calls, built {len(planned_calls)}"
        )

    records = []
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    with make_client() as client, predictions_path.open("w", encoding="utf-8") as handle:
        for dataset_name, example in planned_calls:
            grouped = grouped_by_name[dataset_name]
            record = classify_text(
                client,
                dataset=dataset_name,
                grouped_index=example.grouped_index,
                text=example.text,
                verbalizers=grouped.verbalizers,
                model=_model_name(),
            )
            row = record_to_dict(record)
            row["reference_index"] = example.reference_index
            row["text_sha256"] = example.text_sha256
            records.append((dataset_name, example, record))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_dataset: dict[str, dict] = {}
    resolved_models: Counter[str] = Counter()
    total_tokens = 0
    total_cost = 0.0
    usable = 0
    latencies: list[float] = []

    for spec in PILOT_DATASETS:
        subset = [(ex, rec) for name, ex, rec in records if name == spec.name]
        ok_pairs = [
            (ex, rec)
            for ex, rec in subset
            if rec.ok and rec.predicted_index is not None and ex.reference_index is not None
        ]
        preds = [rec.predicted_index for _, rec in ok_pairs]
        refs = [ex.reference_index for ex, _ in ok_pairs]
        metrics = (
            compute_metrics(preds, refs, n_classes=grouped_by_name[spec.name].n_classes)
            if ok_pairs
            else {
                "macro_f1": 0.0,
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
            }
        )
        ds_tokens = sum(rec.input_tokens or 0 for _, rec in subset)
        ds_cost = sum(rec.cost_usd or 0.0 for _, rec in subset)
        ds_usable = sum(1 for _, rec in subset if rec.ok)
        usable += ds_usable
        total_tokens += ds_tokens
        total_cost += ds_cost
        latencies.extend(rec.latency_ms for _, rec in subset)
        for _, rec in subset:
            if rec.resolved_model:
                resolved_models[rec.resolved_model] += 1
        per_dataset[spec.name] = {
            "n": len(subset),
            "usable": ds_usable,
            "n_classes": grouped_by_name[spec.name].n_classes,
            "metrics": metrics,
            "input_tokens": ds_tokens,
            "cost_usd": ds_cost,
            "failures": [
                {
                    "grouped_index": rec.grouped_index,
                    "error": rec.error,
                    "retries": rec.retries,
                }
                for _, rec in subset
                if not rec.ok
            ],
        }

    metric_only = {name: row["metrics"] for name, row in per_dataset.items()}
    n_models = len(resolved_models)
    go = (
        usable == PILOT_TOTAL_CALLS
        and n_models == 1
        and total_cost <= PILOT_BUDGET_USD
        and all(
            grouped_by_name[spec.name].n_classes == spec.paper_classes
            or precheck["revision_tag"] == CURRENT_REVISION_TAG
            for spec in PILOT_DATASETS
        )
    )
    summary = {
        "status": "complete",
        "go": go,
        "revision_tag": precheck["revision_tag"],
        "paper_aligned": precheck["paper_aligned"],
        "n_calls": len(planned_calls),
        "usable": usable,
        "resolved_models": dict(resolved_models),
        "model_version_stable": n_models == 1,
        "input_tokens": total_tokens,
        "cost_usd": total_cost,
        "budget_usd": PILOT_BUDGET_USD,
        "cost_within_budget": total_cost <= PILOT_BUDGET_USD,
        "latency_ms_p50": float(median(latencies)) if latencies else None,
        "overall": {
            "macro_f1": equal_weight_mean(metric_only, "macro_f1"),
            "accuracy": equal_weight_mean(metric_only, "accuracy"),
            "macro_precision": equal_weight_mean(metric_only, "macro_precision"),
            "macro_recall": equal_weight_mean(metric_only, "macro_recall"),
        },
        "per_dataset": per_dataset,
        "note": (
            "Pilot go/no-go is experimental validity, not F1. Do not compare "
            "to ICLR Table 2 unless revision_tag is paper-table1."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
