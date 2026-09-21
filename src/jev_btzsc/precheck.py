"""Pin check: 22 BTZSC configs, class counts, verbalizer manifest + SHA-256."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from datasets import get_dataset_config_names

from jev_btzsc.data import inspect_dataset
from jev_btzsc.io import write_canonical_json, write_json
from jev_btzsc.protocol import (
    BTZSC_DATASETS,
    CURRENT_REVISION_TAG,
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
    HARNESS_COMMIT,
    HARNESS_REPO,
    PAPER_ALIGNED_TAG,
    PAPER_CLASS_COUNTS,
    PILOT_DATASETS,
    option_id,
)


@dataclass(frozen=True)
class DatasetCheck:
    name: str
    task: str
    domain: str
    paper_classes: int
    observed_classes: int
    unique_verbalizers: int
    unique_label_texts: int
    harness_pattern_n_classes: int
    first_text_n_classes: int
    n_grouped_examples: int
    n_rows: int
    n_no_positive: int
    paper_aligned: bool
    verbalizers: list[str]
    option_ids: list[str]


def _check_one(name: str, *, cache_dir: str | None) -> DatasetCheck:
    grouped = inspect_dataset(name, cache_dir=cache_dir)
    unique_verbalizers = len(set(grouped.verbalizers))
    unique_label_texts = len(set(grouped.label_texts))
    paper = PAPER_CLASS_COUNTS[name]
    aligned = grouped.n_classes == paper and unique_verbalizers == paper
    n_grouped = grouped.n_rows // grouped.n_classes if grouped.n_classes else 0
    return DatasetCheck(
        name=name,
        task=grouped.task,
        domain=grouped.domain,
        paper_classes=paper,
        observed_classes=grouped.n_classes,
        unique_verbalizers=unique_verbalizers,
        unique_label_texts=unique_label_texts,
        harness_pattern_n_classes=grouped.harness_pattern_n_classes,
        first_text_n_classes=grouped.first_text_n_classes,
        n_grouped_examples=n_grouped,
        n_rows=grouped.n_rows,
        n_no_positive=grouped.n_no_positive,
        paper_aligned=aligned,
        verbalizers=list(grouped.verbalizers),
        option_ids=[option_id(i) for i in range(grouped.n_classes)],
    )


def list_hf_configs(*, cache_dir: str | None = None) -> list[str]:
    return list(get_dataset_config_names(HF_DATASET_REPO, revision=HF_DATASET_REVISION))


def run_precheck(
    *,
    output_dir: Path,
    cache_dir: str | None = None,
    datasets: tuple[str, ...] | None = None,
) -> dict:
    names = datasets or BTZSC_DATASETS
    hf_configs = list_hf_configs(cache_dir=cache_dir)
    missing = [name for name in BTZSC_DATASETS if name not in hf_configs]
    extra_note = sorted(set(hf_configs) - set(BTZSC_DATASETS))

    checks = [_check_one(name, cache_dir=cache_dir) for name in names]
    mismatches = [asdict(check) for check in checks if not check.paper_aligned]
    banking = next((c for c in checks if c.name == "banking77"), None)
    banking77_ok = banking is not None and banking.observed_classes == 77 and banking.paper_aligned
    all_aligned = not missing and not mismatches and banking77_ok
    revision_tag = PAPER_ALIGNED_TAG if all_aligned else CURRENT_REVISION_TAG

    verbalizer_manifest = {
        "hf_repo": HF_DATASET_REPO,
        "hf_revision": HF_DATASET_REVISION,
        "harness_repo": HARNESS_REPO,
        "harness_commit": HARNESS_COMMIT,
        "revision_tag": revision_tag,
        "datasets": {
            check.name: {
                "task": check.task,
                "option_ids": check.option_ids,
                "verbalizers": check.verbalizers,
            }
            for check in checks
        },
    }
    digest = write_canonical_json(output_dir / "verbalizer_manifest.json", verbalizer_manifest)
    (output_dir / "verbalizer_manifest.sha256").write_text(digest + "\n", encoding="utf-8")

    report = {
        "hf_repo": HF_DATASET_REPO,
        "hf_revision": HF_DATASET_REVISION,
        "harness_repo": HARNESS_REPO,
        "harness_commit": HARNESS_COMMIT,
        "revision_tag": revision_tag,
        "paper_aligned": all_aligned,
        "banking77_is_77": banking77_ok,
        "expected_dataset_count": 22,
        "observed_base_dataset_count": len([c for c in hf_configs if c in BTZSC_DATASETS]),
        "missing_datasets": missing,
        "non_base_hf_configs": extra_note,
        "mismatches": [
            {
                "name": row["name"],
                "paper_classes": row["paper_classes"],
                "observed_classes": row["observed_classes"],
                "unique_verbalizers": row["unique_verbalizers"],
            }
            for row in mismatches
        ],
        "datasets": [asdict(check) for check in checks],
        "verbalizer_manifest_sha256": digest,
        "pilot_datasets": [spec.name for spec in PILOT_DATASETS],
    }
    write_json(output_dir / "precheck.json", report)
    return report
