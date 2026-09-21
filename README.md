# Jev × BTZSC

Strict zero-shot text classification of [BTZSC](https://arxiv.org/abs/2603.11991) with TypeSafe [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (official System One API). This repo is a small harness: one Jev Choice call per original test example, frozen protocol, no training, no few-shot, no prompt tuning from test errors.

It ships the Phase A 400-call pilot and a six-dataset full-split run. It is **not** a 22-dataset leaderboard submission.

## Protocol

Adapter (frozen):

- `state` = the raw input text
- one `Choice` = the full candidate label set
- option ids are non-semantic (`L000`, `L001`, …); descriptions are the official BTZSC verbalizers unchanged
- one global instruction for every dataset
- one Jev call per original test example

| Source | Pin |
|---|---|
| Hugging Face dataset `btzsc/btzsc` | `fef2a2ac62b69c58670047dddf045c53d7c3cb5e` |
| Official harness grouping | [IliasAarab/btzsc](https://github.com/IliasAarab/btzsc) `@4eac6a9561cb0a600ba65d94c62c3ce7f3e89547` |
| Sampling (pilot) | harness `seed=0` (keep grouped examples 0 and 1, sample the rest) |
| Jev HTTP API | `https://api.typesafe.ai` via official `typesafe-sdk` |
| Default model | `jev-latest` (resolved to `jev-1.13.0` on every call in the runs below) |

Verbalizer manifest SHA-256: `18fd45eab99dd3b398abc4fc30a2615b1b63b7d876450066b49cf2955492259c`.

This Hugging Face revision exposes **72** Banking77 hypotheses rather than the paper’s **77**. Precheck fails unless Banking77 is 77. Inference then refuses unless you pass `--accept-btzsc-current`. Those runs are tagged **`BTZSC-current`** and are **not ICLR 2026 Table 2 comparable**.

**`BTZSC-current-valid` scoring** (used in the tables below): for each grouped sample, count candidates with `entailment=1`. Exactly one → **valid** (formal eval). Zero → **OOS**, exclude. More than one → **anomaly**, exclude. Macro-F1 is over the observed class set.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
cp .env.example .env
```

Put the official TypeSafe key in `.env`. Never commit `.env`.

```bash
uv run jev-btzsc env          # shows which keys are set; never prints secret values
uv run pytest                 # offline unit tests; no Jev calls
```

### Required official API keys

Inference **requires** a TypeSafe Jev key. This project does not invent keys and does not use unofficial proxies.

| Provider | Env var | Required? |
|---|---|---|
| **TypeSafe AI** (Jev / System One) | `TYPESAFE_API_KEY` | **Yes**, to run `pilot` or `full`. Create at [console.typesafe.ai/keys](https://console.typesafe.ai/keys). The official SDK reads this name. |
| TypeSafe AI | `TYPESAFE_DEFAULT_MODEL` | No. Default `jev-latest`. Pin e.g. `jev-1.13.0` for a frozen resolved-model string. |
| **Hugging Face** | `HF_TOKEN` | **No**. `btzsc/btzsc` is public. Set only if anonymous downloads are rate-limited. |

List price used for cost accounting: **$0.042 / 1M input tokens**, output free.

## Commands

Precheck downloads the pinned BTZSC configs and does **not** call Jev:

```bash
uv run jev-btzsc precheck --output artifacts/precheck
```

Exit code 2 means class counts do not match ICLR Table 1 (Banking77 must be 77).

Phase A pilot (400 calls: amazonpolarity, agnews, banking77, emotiondair × 100):

```bash
uv run jev-btzsc pilot --dry-run --output artifacts/pilot
uv run jev-btzsc pilot --accept-btzsc-current --output artifacts/pilot
```

Rescore the pilot Banking77 slice from existing predictions (no extra Jev calls):

```bash
uv run jev-btzsc rescore-banking77 \
  --predictions artifacts/pilot/predictions.jsonl \
  --manifest artifacts/pilot/sample_manifest.json \
  --output artifacts/pilot/banking77-valid.json
```

Full test splits on six datasets (order: banking77 → massive → empathetic → emotiondair → agnews → financialphrasebank). Hard stop $4.50:

```bash
uv run jev-btzsc full --output artifacts/full-run --hard-stop 4.50
```

Artifacts under `artifacts/` are gitignored (full probability vectors and other dumps). Published input–prediction pairs are under [`results/`](results/).

## Results

All numbers below are **`BTZSC-current-valid`**. Banking77 is 72 classes vs paper 77. Do not treat these as ICLR Table 2 scores. Resolved model: **jev-1.13.0**. API failures: **0**. Anomalies: **0**.

### Pilot Banking77 valid rescore

No extra Jev calls. Predictions from the Phase A 100-example Banking77 slice.

| | |
|---|---:|
| N_raw | 100 |
| N_valid | 87 |
| N_OOS | 13 |
| N_anomaly | 0 |
| accuracy (valid) | 0.770 |
| macro-F1 (valid) | 0.536 |
| resolved model | jev-1.13.0 (100/100) |
| input tokens (this slice) | 223,436 |
| cost (this slice) | $0.00938 |

Macro-F1 uses all 72 observed classes (many have no support in the 87 valid rows), so it is lower than accuracy.

### Six-dataset full splits

Hard stop $4.50 **not hit**. Hosted p50 ~145 ms is not a China→US-adjusted speed claim.

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

Each line is one original input text and Jev’s predicted class (Choice id `L000…` plus the official BTZSC verbalizer). Gold label is included when the grouped sample has a single entailment. Probability vectors are not published.

- [`results/full/<dataset>.jsonl`](results/full/) — full test splits (18,886 examples)
- [`results/pilot/<dataset>.jsonl`](results/pilot/) — Phase A 100-example slices (400 examples)

Joined from stored predictions via `grouped_index`, checked against `text_sha256` on the pinned Hugging Face revision. No extra Jev calls.

## Notes

- No BTZSC train/dev for training, hyperparameter search, or model selection.
- Groups with no positive entailment candidate are **not** silently assigned to class 0.
- Public benchmarks may appear in pretraining data; this is strict *benchmark* zero-shot, not a contamination-free generalization claim.

## References

1. Aarab, I. *BTZSC*. ICLR 2026. https://arxiv.org/abs/2603.11991
2. Official harness: https://github.com/IliasAarab/btzsc
3. Dataset: https://huggingface.co/datasets/btzsc/btzsc
4. TypeSafe Jev: https://docs.typesafe.ai
5. TypeSafe Python SDK: https://docs.typesafe.ai/sdk/python
