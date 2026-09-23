# Jev × BTZSC

Strict zero-shot text classification of [BTZSC](https://arxiv.org/abs/2603.11991) with TypeSafe [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). One Choice call per test example; frozen protocol; no training or few-shot.

Phase A: 400-call pilot. Then full test splits on six datasets.

## Protocol

- `state` = raw input text
- one `Choice` = the full candidate label set
- option ids `L000`, `L001`, …; descriptions are the official BTZSC verbalizers
- one global instruction for every dataset

| Source | Pin |
|---|---|
| Hugging Face `btzsc/btzsc` | `fef2a2ac62b69c58670047dddf045c53d7c3cb5e` |
| Harness grouping | [IliasAarab/btzsc](https://github.com/IliasAarab/btzsc) `@4eac6a9561cb0a600ba65d94c62c3ce7f3e89547` |
| Pilot sampling | harness `seed=0` (keep groups 0 and 1, sample the rest) |
| Jev API | `https://api.typesafe.ai` via `typesafe-sdk` |
| Model | `jev-latest` → `jev-1.13.0` on every call below |

Verbalizer manifest SHA-256: `18fd45eab99dd3b398abc4fc30a2615b1b63b7d876450066b49cf2955492259c`.

This Hugging Face revision has **72** Banking77 classes (paper: 77). Precheck exits 2 unless you pass `--accept-btzsc-current`.

**Scoring (`BTZSC-current-valid`):** count `entailment=1` per grouped sample. Exactly one → valid (metrics). Zero → OOS, exclude. More than one → anomaly, exclude. Macro-F1 is over the observed class set.

## Setup

Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
cp .env.example .env
```

Set `TYPESAFE_API_KEY` in `.env` ([console.typesafe.ai/keys](https://console.typesafe.ai/keys)). Do not commit `.env`. Optional: `TYPESAFE_DEFAULT_MODEL` (default `jev-latest`), `HF_TOKEN` if Hub rate-limits.

Cost accounting: **$0.042 / 1M input tokens**, output free.

```bash
uv run jev-btzsc env
uv run pytest
```

## Commands

```bash
uv run jev-btzsc precheck --output artifacts/precheck
uv run jev-btzsc pilot --dry-run --output artifacts/pilot
uv run jev-btzsc pilot --accept-btzsc-current --output artifacts/pilot
uv run jev-btzsc rescore-banking77 \
  --predictions artifacts/pilot/predictions.jsonl \
  --manifest artifacts/pilot/sample_manifest.json \
  --output artifacts/pilot/banking77-valid.json
uv run jev-btzsc full --output artifacts/full-run --hard-stop 4.50
```

Pilot datasets: amazonpolarity, agnews, banking77, emotiondair × 100. Full order: banking77 → massive → empathetic → emotiondair → agnews → financialphrasebank. `artifacts/` is gitignored.

## Results

Resolved model **jev-1.13.0**. API failures 0. Anomalies 0.

### Pilot Banking77

100-example slice, rescored from existing predictions. Macro-F1 uses all 72 observed classes (many have no support in the 87 valid rows).

| | |
|---|---:|
| N_raw | 100 |
| N_valid | 87 |
| N_OOS | 13 |
| N_anomaly | 0 |
| accuracy (valid) | 0.770 |
| macro-F1 (valid) | 0.536 |
| resolved model | jev-1.13.0 (100/100) |
| input tokens | 223,436 |
| cost | $0.00938 |

### Six-dataset full splits

| dataset | classes obs (paper) | N_raw | N_valid | N_OOS | acc (valid) | macro-F1 (valid) | tokens | USD | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 72 (77) | 3080 | 2880 | 200 | 0.818 | 0.812 | 6,876,490 | 0.289 | 152 | 904 |
| massive | 59 (59) | 2974 | 2805 | 169 | 0.847 | 0.805 | 5,726,214 | 0.241 | 148 | 230 |
| empathetic | 32 (32) | 2542 | 2542 | 0 | 0.543 | 0.528 | 3,105,436 | 0.130 | 143 | 213 |
| emotiondair | 6 (6) | 2000 | 2000 | 0 | 0.590 | 0.519 | 930,323 | 0.039 | 147 | 220 |
| agnews | 4 (4) | 7600 | 7600 | 0 | 0.881 | 0.880 | 3,452,959 | 0.145 | 143 | 217 |
| financialphrasebank | 3 (3) | 690 | 690 | 0 | 0.871 | 0.864 | 293,533 | 0.012 | 153 | 275 |

| | |
|---|---:|
| calls | 18,886 |
| input tokens | 20,384,955 |
| cost | **$0.856** |
| hard stop | $4.50 (not reached) |
| equal-weight acc (valid) | 0.758 |
| equal-weight macro-F1 (valid) | 0.735 |

## Input–prediction pairs

[`results/full/<dataset>.jsonl`](results/full/) (18,886) and [`results/pilot/<dataset>.jsonl`](results/pilot/) (400): input text, Jev Choice (`L000…` + verbalizer), gold label when the sample is valid.

## References

1. Aarab, I. *BTZSC*. ICLR 2026. https://arxiv.org/abs/2603.11991
2. Official harness: https://github.com/IliasAarab/btzsc
3. Dataset: https://huggingface.co/datasets/btzsc/btzsc
4. TypeSafe Jev: https://docs.typesafe.ai
5. TypeSafe Python SDK: https://docs.typesafe.ai/sdk/python
