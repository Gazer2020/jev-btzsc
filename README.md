# Jev × BTZSC Phase A pilot

Strict zero-shot text classification of [BTZSC](https://arxiv.org/abs/2603.11991) test examples with TypeSafe [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). This repository implements **Phase A only**: 400 official Jev Choice calls on four datasets (100 examples each). It does not train, does not use few-shot examples, and does not tune the prompt from test errors.

Full 22-dataset evaluation is out of scope here.

## What the pilot does

| Dataset | Task | Paper classes | n |
|---|---|---:|---:|
| `amazonpolarity` | sentiment | 2 | 100 |
| `agnews` | topic | 4 | 100 |
| `banking77` | intent | 77 | 100 |
| `emotiondair` | emotion | 6 | 100 |

Adapter (frozen):

- `state` = the raw input text
- one `Choice` = the full candidate label set
- option ids are non-semantic (`L000`, `L001`, …); descriptions are the official BTZSC verbalizers unchanged
- one global instruction for every dataset
- one Jev call per original test example

Go/no-go is experimental validity, not leaderboard score: class counts, label mapping, 400/400 usable outputs, stable resolved model, cost within $0.10. Do not compare to ICLR 2026 Table 2 unless precheck tags the pin as `paper-table1`. If Banking77 is not 77 classes, the run is `BTZSC-current`.

## Pins

| Source | Pin |
|---|---|
| Hugging Face dataset `btzsc/btzsc` | `fef2a2ac62b69c58670047dddf045c53d7c3cb5e` |
| Official harness grouping | [IliasAarab/btzsc](https://github.com/IliasAarab/btzsc) `@4eac6a9561cb0a600ba65d94c62c3ce7f3e89547` |
| Sampling | harness `seed=0` (keep grouped examples 0 and 1, sample the rest) |
| Jev HTTP API | `https://api.typesafe.ai` via official `typesafe-sdk` |

The community Jev protocol noted that this Hugging Face revision can expose **72** Banking77 hypotheses rather than the paper’s **77**. Precheck fails if Banking77 ≠ 77. Inference then refuses unless you pass `--accept-btzsc-current` (results are not Table 2 comparable).

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

## Required official API keys

Inference **requires** a TypeSafe Jev key. This project does not invent keys and does not use unofficial proxies.

| Provider | Env var | Where | Required? |
|---|---|---|---|
| **TypeSafe AI** (Jev / System One) | `TYPESAFE_API_KEY` | `.env` (see `.env.example`) or the process environment. Create at [console.typesafe.ai/keys](https://console.typesafe.ai/keys). The official SDK reads this name. | **Yes**, to run `pilot` |
| TypeSafe AI | `TYPESAFE_DEFAULT_MODEL` | `.env` or environment. Default `jev-latest`. Pin e.g. `jev-1.13.0` if you need a frozen resolved-model string. | No |
| **Hugging Face** | `HF_TOKEN` | `.env` or environment. Create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). | **No**. `btzsc/btzsc` is public. Set only if anonymous downloads are rate-limited. |

List price used for cost accounting: **$0.042 / 1M input tokens**, output free. Pilot budget $0.10 (expected roughly $0.004–$0.034).

## Commands

Precheck downloads the pinned BTZSC configs and does **not** call Jev:

```bash
uv run jev-btzsc precheck --output artifacts/precheck
# or only the four pilot datasets:
uv run jev-btzsc precheck --pilot-only --output artifacts/precheck
```

That writes `verbalizer_manifest.json`, `verbalizer_manifest.sha256`, and `precheck.json`. Exit code 2 means class counts do not match ICLR Table 1 (Banking77 must be 77).

Dry-run builds the 400-example sample manifest without Jev:

```bash
uv run jev-btzsc pilot --dry-run --output artifacts/pilot
```

Run the 400-call experiment (needs `TYPESAFE_API_KEY`):

```bash
uv run jev-btzsc pilot --output artifacts/pilot
```

If precheck tagged `BTZSC-current` and you still want a validity check:

```bash
uv run jev-btzsc pilot --accept-btzsc-current --output artifacts/pilot
```

Artifacts under `artifacts/` (gitignored):

- `precheck/verbalizer_manifest.json` + `.sha256`
- `pilot/sample_manifest.json`
- `pilot/predictions.jsonl` — top choice, full probabilities, resolved model, tokens, cost, latency, retries, failures
- `pilot/summary.json` — per-dataset macro-F1 / accuracy / precision / recall and the equal-weight average

## Protocol notes

- No BTZSC train/dev for training, hyperparameter search, or model selection.
- Groups with no positive entailment candidate are **not** silently assigned to class 0.
- China→US latency is not a model-speed claim. Public benchmarks may appear in pretraining data; this is strict *benchmark* zero-shot, not a contamination-free generalization claim.

## References

1. Aarab, I. *BTZSC*. ICLR 2026. https://arxiv.org/abs/2603.11991
2. Official harness: https://github.com/IliasAarab/btzsc
3. Dataset: https://huggingface.co/datasets/btzsc/btzsc
4. TypeSafe Jev: https://docs.typesafe.ai
5. TypeSafe Python SDK: https://docs.typesafe.ai/sdk/python
