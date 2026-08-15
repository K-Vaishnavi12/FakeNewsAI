"""NVIDIA OpenAI-compatible API client.

Responsibilities:
* Call the NVIDIA endpoint with a timeout and never raise into the pipeline.
* Extract JSON robustly, even if the model wraps it in Markdown fences or adds
  prose around it.
* Coerce the parsed object into `FinalAnalysis`, replacing any out-of-vocabulary
  value with a safe default instead of failing.
* Return `(None, reason)` on empty or unusable responses so the caller can fall
  back to the deterministic analysis.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.config import Settings, get_settings
from app.schemas import (
    ClaimBreakdownItem,
    FinalAnalysis,
    MLAssessment,
    SourceAssessmentItem,
)

FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

VALID_VERDICTS = {"Likely Real", "Likely Fake", "Needs Verification"}
VALID_ML_PREDICTIONS = {"REAL", "FAKE", "UNKNOWN"}
VALID_CLAIM_STATUS = {
    "SUPPORTED", "CONTRADICTED", "PARTIALLY_SUPPORTED", "UNVERIFIED",
}
VALID_RELATIONS = {
    "SUPPORTS", "CONTRADICTS", "PARTIALLY_SUPPORTS", "UNRELATED", "UNKNOWN",
}
VALID_EVIDENCE_STATUS = {
    "RELEVANT", "WEAK", "CONTRADICTORY", "UNRELATED", "UNKNOWN",
}
VALID_AGREEMENT = {"HIGH", "MEDIUM", "LOW", "NONE"}


# --- JSON recovery ----------------------------------------------------------


def _balanced_json_slice(text: str) -> str | None:
    """Return the first balanced `{...}` block, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def extract_json(raw: str) -> dict | None:
    """Best-effort extraction of a JSON object from a model response."""
    if not raw or not raw.strip():
        return None

    candidates: list[str] = []
    fenced = FENCE_PATTERN.search(raw)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(raw.strip())
    sliced = _balanced_json_slice(raw)
    if sliced:
        candidates.append(sliced)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# --- Coercion ---------------------------------------------------------------


def _as_str(value: Any, default: str = "") -> str:
    return value.strip() if isinstance(value, str) else default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(0, min(100, int(value)))
    if isinstance(value, str):
        digits = re.search(r"\d+", value)
        if digits:
            return max(0, min(100, int(digits.group())))
    return default


def _as_enum(value: Any, allowed: set[str], default: str) -> str:
    if isinstance(value, str) and value.strip() in allowed:
        return value.strip()
    if isinstance(value, str):
        upper = value.strip().upper()
        for option in allowed:
            if option.upper() == upper:
                return option
    return default


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [v.strip() for v in value if isinstance(v, str) and v.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def coerce_analysis(payload: dict) -> FinalAnalysis:
    """Turn an arbitrary model payload into a valid `FinalAnalysis`."""
    ml_raw = payload.get("ml_assessment")
    ml_raw = ml_raw if isinstance(ml_raw, dict) else {}

    claim_items: list[ClaimBreakdownItem] = []
    for item in payload.get("claim_breakdown") or []:
        if not isinstance(item, dict):
            continue
        claim_items.append(
            ClaimBreakdownItem(
                claim_id=_as_str(item.get("claim_id")),
                claim_text=_as_str(item.get("claim_text")),
                status=_as_enum(
                    item.get("status"), VALID_CLAIM_STATUS, "UNVERIFIED"
                ),  # type: ignore[arg-type]
                explanation=_as_str(item.get("explanation")),
                source_ids=_as_str_list(item.get("source_ids")),
            )
        )

    source_items: list[SourceAssessmentItem] = []
    for item in payload.get("source_assessment") or []:
        if not isinstance(item, dict):
            continue
        source_items.append(
            SourceAssessmentItem(
                source_id=_as_str(item.get("source_id")),
                publisher=_as_str(item.get("publisher"), "Not provided by the source."),
                title=_as_str(item.get("title"), "Not provided by the source."),
                url=_as_str(item.get("url"), "Not provided by the source."),
                relation=_as_enum(
                    item.get("relation"), VALID_RELATIONS, "UNKNOWN"
                ),  # type: ignore[arg-type]
                evidence_status=_as_enum(
                    item.get("evidence_status"), VALID_EVIDENCE_STATUS, "UNKNOWN"
                ),  # type: ignore[arg-type]
                reason=_as_str(item.get("reason")),
                used_in_final_answer=bool(item.get("used_in_final_answer", False)),
            )
        )

    return FinalAnalysis(
        verdict=_as_enum(
            payload.get("verdict"), VALID_VERDICTS, "Needs Verification"
        ),  # type: ignore[arg-type]
        confidence=_as_int(payload.get("confidence"), 0),
        headline_summary=_as_str(payload.get("headline_summary")),
        plain_language_explanation=_as_str(
            payload.get("plain_language_explanation")
        ),
        ml_assessment=MLAssessment(
            model_name=_as_str(ml_raw.get("model_name")),
            prediction=_as_enum(
                ml_raw.get("prediction"), VALID_ML_PREDICTIONS, "UNKNOWN"
            ),  # type: ignore[arg-type]
            confidence=_as_int(ml_raw.get("confidence"), 0),
            interpretation="ML output is a signal, not proof.",
        ),
        claim_breakdown=claim_items,
        source_assessment=source_items,
        source_agreement=_as_enum(
            payload.get("source_agreement"), VALID_AGREEMENT, "NONE"
        ),  # type: ignore[arg-type]
        recommended_action=_as_str(payload.get("recommended_action")),
        limitations=_as_str_list(payload.get("limitations")),
        generated_by="NVIDIA_LLM",
    )


# --- Client -----------------------------------------------------------------


class NvidiaClient:
    """Thin wrapper over the OpenAI-compatible NVIDIA endpoint."""

    def __init__(self, settings: Settings | None = None, client: Any = None) -> None:
        self.settings = settings or get_settings()
        self._client = client

    @property
    def configured(self) -> bool:
        return self.settings.nvidia_configured

    def _get_client(self):
        if self._client is not None:
            return self._client
        from openai import OpenAI  # imported lazily so tests need no network

        self._client = OpenAI(
            base_url=self.settings.nvidia_base_url,
            api_key=self.settings.nvidia_api_key,
            timeout=self.settings.nvidia_timeout_seconds,
        )
        return self._client

    def complete(self, system_prompt: str, user_message: str) -> tuple[str | None, str | None]:
        """Return `(raw_text, error)`. Never raises."""
        if not self.configured:
            return None, (
                "The AI explanation service is not configured, so a rule-based "
                "explanation was generated instead."
            )

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.settings.nvidia_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.settings.nvidia_temperature,
                max_tokens=self.settings.nvidia_max_tokens,
                response_format={"type": "json_object"},
            )
        except TypeError:
            # Some models reject `response_format`; retry without it.
            try:
                response = self._client.chat.completions.create(
                    model=self.settings.nvidia_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.settings.nvidia_temperature,
                    max_tokens=self.settings.nvidia_max_tokens,
                )
            except Exception:  # noqa: BLE001
                return None, (
                    "The AI explanation service could not be reached, so a "
                    "rule-based explanation was generated instead."
                )
        except Exception:  # noqa: BLE001 - network, auth, rate limit, timeout
            return None, (
                "The AI explanation service could not be reached or returned an "
                "error, so a rule-based explanation was generated instead."
            )

        try:
            choices = response.choices
            if not choices:
                return None, (
                    "The AI explanation service returned an empty response, so a "
                    "rule-based explanation was generated instead."
                )
            content = choices[0].message.content
        except (AttributeError, IndexError, KeyError):
            return None, (
                "The AI explanation service returned an unreadable response, so "
                "a rule-based explanation was generated instead."
            )

        if not content or not str(content).strip():
            return None, (
                "The AI explanation service returned an empty response, so a "
                "rule-based explanation was generated instead."
            )
        return str(content), None

    def analyze(
        self, system_prompt: str, user_message: str
    ) -> tuple[FinalAnalysis | None, str | None]:
        """Full round trip: call, extract JSON, coerce. Never raises."""
        raw, error = self.complete(system_prompt, user_message)
        if raw is None:
            return None, error

        payload = extract_json(raw)
        if payload is None:
            return None, (
                "The AI explanation service returned malformed JSON, so a "
                "rule-based explanation was generated instead."
            )

        try:
            return coerce_analysis(payload), None
        except Exception:  # noqa: BLE001
            return None, (
                "The AI explanation could not be validated against the required "
                "format, so a rule-based explanation was generated instead."
            )
