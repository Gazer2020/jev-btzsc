"""Official TypeSafe Jev Choice adapter.

Calls POST https://api.typesafe.ai/v1/systemone via the official `typesafe-sdk`.
Does not use unofficial proxies or gateway model IDs.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy, TypeSafeClient
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeError,
    TypeSafeRateLimitError,
)

from jev_btzsc.protocol import (
    GLOBAL_INSTRUCTIONS,
    JEV_MODEL_DEFAULT,
    QUESTION_NAME,
    TYPESAFE_API_BASE_URL,
    option_id,
    option_index,
    token_cost_usd,
)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}
MAX_ATTEMPTS = 4


@dataclass
class JevCallRecord:
    dataset: str
    grouped_index: int
    option_ids: list[str]
    top_choice: str | None
    predicted_index: int | None
    probabilities: dict[str, float] | None
    confidence: float | None
    resolved_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_ms: float
    attempts: int
    retries: int
    ok: bool
    error: str | None


def build_choice(verbalizers: tuple[str, ...] | list[str]) -> Choice:
    criteria = {option_id(i): verbalizer for i, verbalizer in enumerate(verbalizers)}
    return Choice(instructions=GLOBAL_INSTRUCTIONS, criteria=criteria)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (TypeSafeAPIConnectionError, TypeSafeRateLimitError)):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and status in RETRYABLE_STATUS:
        return True
    return False


def _probabilities_ok(probabilities: Mapping[str, float], option_ids: list[str]) -> bool:
    if set(probabilities) != set(option_ids):
        return False
    total = 0.0
    for value in probabilities.values():
        if not isinstance(value, (int, float)) or value != value:  # NaN
            return False
        if value < 0.0 or value > 1.0:
            return False
        total += float(value)
    return abs(total - 1.0) <= 0.02


def _parse_choice_response(
    *,
    dataset: str,
    grouped_index: int,
    option_ids: list[str],
    response: Any,
    latency_ms: float,
    attempts: int,
) -> JevCallRecord:
    answer = response.choices[QUESTION_NAME]
    probabilities = {str(k): float(v) for k, v in dict(answer.probabilities).items()}
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
    output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
    resolved_model = getattr(response, "model", None)
    top = str(answer.choice)
    usable = top in option_ids and _probabilities_ok(probabilities, option_ids)
    predicted = option_index(top) if usable else None
    return JevCallRecord(
        dataset=dataset,
        grouped_index=grouped_index,
        option_ids=option_ids,
        top_choice=top if usable else None,
        predicted_index=predicted,
        probabilities=probabilities if usable else None,
        confidence=float(answer.confidence) if usable else None,
        resolved_model=str(resolved_model) if resolved_model else None,
        input_tokens=int(input_tokens) if input_tokens is not None else None,
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        cost_usd=token_cost_usd(int(input_tokens) if input_tokens is not None else None),
        latency_ms=latency_ms,
        attempts=attempts,
        retries=attempts - 1,
        ok=usable,
        error=None if usable else "invalid_choice_payload",
    )


def _failed_record(
    *,
    dataset: str,
    grouped_index: int,
    option_ids: list[str],
    latency_ms: float,
    attempts: int,
    error: str | None,
) -> JevCallRecord:
    return JevCallRecord(
        dataset=dataset,
        grouped_index=grouped_index,
        option_ids=option_ids,
        top_choice=None,
        predicted_index=None,
        probabilities=None,
        confidence=None,
        resolved_model=None,
        input_tokens=None,
        output_tokens=None,
        cost_usd=None,
        latency_ms=latency_ms,
        attempts=attempts,
        retries=max(attempts - 1, 0),
        ok=False,
        error=error or "unknown_error",
    )


def classify_text(
    client: TypeSafeClient,
    *,
    dataset: str,
    grouped_index: int,
    text: str,
    verbalizers: tuple[str, ...] | list[str],
    model: str,
) -> JevCallRecord:
    option_ids = [option_id(i) for i in range(len(verbalizers))]
    question = build_choice(verbalizers)
    attempts = 0
    last_error: str | None = None
    started = time.perf_counter()

    while attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            response = client.system_one(
                state=text,
                questions={QUESTION_NAME: question},
                model=model,
                retry=RetryPolicy(max_retries=0),
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            return _parse_choice_response(
                dataset=dataset,
                grouped_index=grouped_index,
                option_ids=option_ids,
                response=response,
                latency_ms=latency_ms,
                attempts=attempts,
            )
        except (TypeSafeAPIError, TypeSafeAPIConnectionError, TypeSafeError, KeyError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempts >= MAX_ATTEMPTS or not _is_retryable(exc):
                break
            time.sleep(min(2 ** (attempts - 1) * 0.5, 8.0))

    return _failed_record(
        dataset=dataset,
        grouped_index=grouped_index,
        option_ids=option_ids,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        attempts=attempts,
        error=last_error,
    )


async def classify_text_async(
    client: AsyncTypeSafeClient,
    *,
    dataset: str,
    grouped_index: int,
    text: str,
    verbalizers: tuple[str, ...] | list[str],
    model: str,
) -> JevCallRecord:
    import asyncio

    option_ids = [option_id(i) for i in range(len(verbalizers))]
    question = build_choice(verbalizers)
    attempts = 0
    last_error: str | None = None
    started = time.perf_counter()

    while attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            response = await client.system_one(
                state=text,
                questions={QUESTION_NAME: question},
                model=model,
                retry=RetryPolicy(max_retries=0),
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            return _parse_choice_response(
                dataset=dataset,
                grouped_index=grouped_index,
                option_ids=option_ids,
                response=response,
                latency_ms=latency_ms,
                attempts=attempts,
            )
        except (TypeSafeAPIError, TypeSafeAPIConnectionError, TypeSafeError, KeyError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempts >= MAX_ATTEMPTS or not _is_retryable(exc):
                break
            await asyncio.sleep(min(2 ** (attempts - 1) * 0.5, 8.0))

    return _failed_record(
        dataset=dataset,
        grouped_index=grouped_index,
        option_ids=option_ids,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        attempts=attempts,
        error=last_error,
    )


def make_async_client() -> AsyncTypeSafeClient:
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TYPESAFE_API_KEY is not set. Copy .env.example to .env and add an "
            "official key from https://console.typesafe.ai/keys"
        )
    model = os.environ.get("TYPESAFE_DEFAULT_MODEL", JEV_MODEL_DEFAULT).strip() or JEV_MODEL_DEFAULT
    return AsyncTypeSafeClient(
        api_key=api_key,
        model=model,
        base_url=TYPESAFE_API_BASE_URL,
    )


def make_client() -> TypeSafeClient:
    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "TYPESAFE_API_KEY is not set. Copy .env.example to .env and add an "
            "official key from https://console.typesafe.ai/keys"
        )
    model = os.environ.get("TYPESAFE_DEFAULT_MODEL", JEV_MODEL_DEFAULT).strip() or JEV_MODEL_DEFAULT
    return TypeSafeClient(
        api_key=api_key,
        model=model,
        base_url=TYPESAFE_API_BASE_URL,
    )


def record_to_dict(record: JevCallRecord) -> dict[str, Any]:
    return asdict(record)
