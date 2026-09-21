"""CLI for the BTZSC × Jev Phase A pilot. Inference is opt-in via `pilot`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from jev_btzsc.protocol import (
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
    JEV_MODEL_DEFAULT,
    PILOT_BUDGET_USD,
    PILOT_TOTAL_CALLS,
    TYPESAFE_API_BASE_URL,
)


def _load_env() -> None:
    load_dotenv(Path.cwd() / ".env")


def _cache_dir() -> str | None:
    return os.environ.get("BTZSC_CACHE_DIR") or None


def cmd_env(_args: argparse.Namespace) -> int:
    type_safe = bool(os.environ.get("TYPESAFE_API_KEY", "").strip())
    hf = bool(os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip())
    model = os.environ.get("TYPESAFE_DEFAULT_MODEL", JEV_MODEL_DEFAULT).strip() or JEV_MODEL_DEFAULT
    print("Jev BTZSC Phase A — environment")
    print(f"  TypeSafe API:     {TYPESAFE_API_BASE_URL}")
    print(f"  TYPESAFE_API_KEY: {'set' if type_safe else 'MISSING (required to run pilot)'}")
    print(f"  model:            {model}")
    print(f"  HF dataset:       {HF_DATASET_REPO}@{HF_DATASET_REVISION}")
    print(f"  HF_TOKEN:         {'set' if hf else 'unset (optional; dataset is public)'}")
    print(f"  planned calls:    {PILOT_TOTAL_CALLS}")
    print(f"  pilot budget:     ${PILOT_BUDGET_USD:.2f}")
    print()
    print("Required official key to run inference:")
    print("  provider: TypeSafe AI (Jev / System One)")
    print("  env var:  TYPESAFE_API_KEY")
    print("  where:    .env (see .env.example) or the process environment")
    print("  console:  https://console.typesafe.ai/keys")
    print()
    print("Optional official token:")
    print("  provider: Hugging Face")
    print("  env var:  HF_TOKEN")
    print("  where:    .env or the process environment")
    print("  console:  https://huggingface.co/settings/tokens")
    print("  why:      public dataset; only if anonymous downloads are rate-limited")
    return 0 if type_safe else 1


def cmd_precheck(args: argparse.Namespace) -> int:
    from jev_btzsc.precheck import run_precheck
    from jev_btzsc.protocol import BTZSC_DATASETS, PILOT_DATASETS

    names = tuple(spec.name for spec in PILOT_DATASETS) if args.pilot_only else BTZSC_DATASETS
    report = run_precheck(
        output_dir=Path(args.output),
        cache_dir=_cache_dir(),
        datasets=names,
    )
    print(f"revision_tag:     {report['revision_tag']}")
    print(f"paper_aligned:    {report['paper_aligned']}")
    print(f"banking77_is_77:  {report['banking77_is_77']}")
    print(f"manifest sha256:  {report['verbalizer_manifest_sha256']}")
    print(f"wrote:            {args.output}")
    if report["mismatches"]:
        print("class-count mismatches vs ICLR Table 1:")
        for row in report["mismatches"]:
            print(
                f"  {row['name']}: paper={row['paper_classes']} "
                f"observed={row['observed_classes']} unique_verbalizers={row['unique_verbalizers']}"
            )
    if not report["banking77_is_77"]:
        return 2
    return 0 if report["paper_aligned"] else 2


def cmd_pilot(args: argparse.Namespace) -> int:
    from jev_btzsc.pilot import run_pilot

    try:
        summary = run_pilot(
            output_dir=Path(args.output),
            cache_dir=_cache_dir(),
            accept_btzsc_current=args.accept_btzsc_current,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"status: {summary.get('status')}")
    if summary.get("status") == "complete":
        print(f"go:     {summary.get('go')}")
        print(f"usable: {summary.get('usable')}/{summary.get('n_calls')}")
        print(f"cost:   ${summary.get('cost_usd'):.6f}")
        print(f"models: {summary.get('resolved_models')}")
    print(f"wrote:  {args.output}")
    if summary.get("status") == "complete" and not summary.get("go"):
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-btzsc",
        description="Phase A 400-call Jev pilot on BTZSC (strict zero-shot).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    env_p = sub.add_parser("env", help="Show required official API keys (never prints secret values).")
    env_p.set_defaults(func=cmd_env)

    pre_p = sub.add_parser("precheck", help="Pin-check datasets and export verbalizer manifest. No Jev calls.")
    pre_p.add_argument("--output", default="artifacts/precheck")
    pre_p.add_argument(
        "--pilot-only",
        action="store_true",
        help="Check only the four pilot datasets (still requires Banking77 == 77).",
    )
    pre_p.set_defaults(func=cmd_precheck)

    pilot_p = sub.add_parser("pilot", help="Run the 400 Jev Choice calls. Requires TYPESAFE_API_KEY.")
    pilot_p.add_argument("--output", default="artifacts/pilot")
    pilot_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the sample manifest only. Does not call Jev.",
    )
    pilot_p.add_argument(
        "--accept-btzsc-current",
        action="store_true",
        help="Allow a run tagged BTZSC-current when Banking77 is not 77 classes. Not Table 2 comparable.",
    )
    pilot_p.set_defaults(func=cmd_pilot)
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
