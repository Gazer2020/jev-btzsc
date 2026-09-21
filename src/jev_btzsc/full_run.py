"""Full-split Jev run for six BTZSC datasets with a hard cost stop."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path

from jev_btzsc.adapter import classify_text_async, make_async_client, record_to_dict
from jev_btzsc.data import load_grouped_dataset
from jev_btzsc.io import write_json
from jev_btzsc.protocol import (
    CURRENT_REVISION_TAG,
    CURRENT_VALID_TAG,
    FULL_HARD_STOP_USD,
    FULL_RUN_DATASETS,
    JEV_MODEL_DEFAULT,
    PAPER_CLASS_COUNTS,
    option_id,
)
from jev_btzsc.scoring import score_valid


def _model_name() -> str:
    return os.environ.get("TYPESAFE_DEFAULT_MODEL", JEV_MODEL_DEFAULT).strip() or JEV_MODEL_DEFAULT


class Budget:
    def __init__(self, hard_stop: float = FULL_HARD_STOP_USD) -> None:
        self.hard_stop = hard_stop
        self.spent = 0.0
        self.stopped = False
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            if self.spent >= self.hard_stop:
                self.stopped = True
                return False
            return True

    async def add(self, cost: float | None) -> None:
        async with self._lock:
            self.spent += float(cost or 0.0)
            if self.spent >= self.hard_stop:
                self.stopped = True


class RateLimiter:
    def __init__(self, per_second: float = 16.0) -> None:
        self.interval = 1.0 / per_second
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.interval
        if wait > 0:
            await asyncio.sleep(wait)


def _mirror(src: Path, dest: Path | None) -> None:
    if dest is None:
        return
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if path.is_file():
            target = dest / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _existing_spend(output_dir: Path) -> float:
    total = 0.0
    for path in output_dir.glob("*/predictions.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                total += float(row.get("cost_usd") or 0.0)
    return total


def _load_done_ids(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add(int(row["grouped_index"]))
    return done


def _load_pred_map(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["grouped_index"])] = row
    return rows


async def _run_dataset(
    name: str,
    *,
    client,
    budget: Budget,
    limiter: RateLimiter,
    output_dir: Path,
    cache_dir: str | None,
    model: str,
) -> dict:
    grouped = load_grouped_dataset(name, cache_dir=cache_dir)
    ds_dir = output_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    pred_path = ds_dir / "predictions.jsonl"
    done = _load_done_ids(pred_path)
    remaining = [ex for ex in grouped.examples if ex.grouped_index not in done]
    n_valid = len(grouped.examples) - grouped.n_oos - grouped.n_anomaly
    print(
        f"{name}: {len(grouped.examples)} grouped "
        f"(valid={n_valid}, oos={grouped.n_oos}, anomaly={grouped.n_anomaly}); "
        f"{len(remaining)} left to call; spend=${budget.spent:.4f}",
        flush=True,
    )

    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(32)
    finished = 0
    skipped_budget = 0
    handle = pred_path.open("a", encoding="utf-8")

    async def one(example):
        nonlocal finished, skipped_budget
        async with sem:
            if not await budget.allow():
                skipped_budget += 1
                return None
            await limiter.acquire()
            record = await classify_text_async(
                client,
                dataset=name,
                grouped_index=example.grouped_index,
                text=example.text,
                verbalizers=grouped.verbalizers,
                model=model,
            )
            await budget.add(record.cost_usd)
            row = record_to_dict(record)
            row["reference_index"] = example.reference_index
            row["validity"] = example.validity
            row["n_positives"] = example.n_positives
            row["text_sha256"] = example.text_sha256
            line = json.dumps(row, ensure_ascii=False) + "\n"
            async with write_lock:
                handle.write(line)
                finished += 1
                if finished % 25 == 0:
                    handle.flush()
            if finished % 200 == 0:
                print(
                    f"  {name}: {finished}/{len(remaining)} new calls, "
                    f"spend=${budget.spent:.4f}",
                    flush=True,
                )
            return record

    try:
        if remaining:
            await asyncio.gather(*(one(example) for example in remaining))
    finally:
        handle.flush()
        handle.close()

    preds = _load_pred_map(pred_path)
    report = score_valid(
        grouped.examples,
        preds,
        n_classes=grouped.n_classes,
        dataset=name,
    )
    report["model_requested"] = model
    report["n_calls_this_session"] = finished
    report["n_skipped_budget"] = skipped_budget
    report["budget_stopped"] = budget.stopped
    report["cumulative_spend_usd"] = budget.spent
    report["option_ids"] = [option_id(i) for i in range(grouped.n_classes)]
    report["verbalizers"] = list(grouped.verbalizers)
    if name == "banking77" or report["revision_tag"] == CURRENT_REVISION_TAG:
        report["class_count_tag"] = CURRENT_REVISION_TAG
    write_json(ds_dir / "summary.json", report)
    print(
        f"{name} done: n_raw={report['n_raw']} n_valid={report['n_valid']} "
        f"n_oos={report['n_oos']} n_anomaly={report['n_anomaly']} "
        f"acc={report['accuracy_valid']:.4f} f1={report['macro_f1_valid']:.4f} "
        f"cost=${report['cost_usd']:.4f} tokens={report['input_tokens']}",
        flush=True,
    )
    return report


async def _run_all(
    *,
    output_dir: Path,
    cache_dir: str | None,
    hard_stop: float,
    store_dir: Path | None,
) -> dict:
    model = _model_name()
    budget = Budget(hard_stop=hard_stop)
    limiter = RateLimiter(per_second=16.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    budget.spent = _existing_spend(output_dir)
    if budget.spent:
        print(f"resuming with recorded spend ${budget.spent:.4f}", flush=True)

    per_dataset: dict[str, dict] = {}
    stopped_reason = None
    async with make_async_client() as client:
        for name in FULL_RUN_DATASETS:
            if not await budget.allow():
                stopped_reason = "hard_stop"
                print(f"hard stop before {name}: spend=${budget.spent:.4f}", flush=True)
                break
            report = await _run_dataset(
                name,
                client=client,
                budget=budget,
                limiter=limiter,
                output_dir=output_dir,
                cache_dir=cache_dir,
                model=model,
            )
            per_dataset[name] = {k: v for k, v in report.items() if k != "verbalizers"}
            write_json(
                output_dir / "summary.json",
                {
                    "status": "in_progress",
                    "scoring": CURRENT_VALID_TAG,
                    "cost_usd": budget.spent,
                    "hard_stop_usd": hard_stop,
                    "per_dataset": per_dataset,
                },
            )
            _mirror(output_dir, store_dir)
            if budget.stopped:
                stopped_reason = "hard_stop"
                print(f"hard stop after {name}: spend=${budget.spent:.4f}", flush=True)
                break

    total_tokens = sum(int(row.get("input_tokens") or 0) for row in per_dataset.values())
    equal_f1 = (
        sum(row["macro_f1_valid"] for row in per_dataset.values()) / len(per_dataset)
        if per_dataset
        else 0.0
    )
    equal_acc = (
        sum(row["accuracy_valid"] for row in per_dataset.values()) / len(per_dataset)
        if per_dataset
        else 0.0
    )
    summary = {
        "status": "complete" if len(per_dataset) == len(FULL_RUN_DATASETS) and not budget.stopped else "stopped",
        "stopped_reason": stopped_reason,
        "scoring": CURRENT_VALID_TAG,
        "datasets_requested": list(FULL_RUN_DATASETS),
        "datasets_completed": list(per_dataset),
        "model_requested": model,
        "hard_stop_usd": hard_stop,
        "cost_usd": budget.spent,
        "input_tokens": total_tokens,
        "equal_weight_macro_f1_valid": equal_f1,
        "equal_weight_accuracy_valid": equal_acc,
        "paper_class_counts": {name: PAPER_CLASS_COUNTS[name] for name in FULL_RUN_DATASETS},
        "per_dataset": per_dataset,
        "note": (
            "Formal metrics use BTZSC-current-valid (exactly one entailment=1). "
            "Banking77 remains tagged BTZSC-current (72 vs paper 77). "
            "Do not claim ICLR Table 2 comparability."
        ),
    }
    write_json(output_dir / "summary.json", summary)
    _mirror(output_dir, store_dir)
    return summary


def run_full(
    *,
    output_dir: Path,
    cache_dir: str | None = None,
    hard_stop: float = FULL_HARD_STOP_USD,
    store_dir: Path | None = None,
) -> dict:
    return asyncio.run(
        _run_all(
            output_dir=output_dir,
            cache_dir=cache_dir,
            hard_stop=hard_stop,
            store_dir=store_dir,
        )
    )
