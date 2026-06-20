# core/ai_engine.py

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from config import (
    AI_FAILURE_COOLDOWN_SECONDS,
    AI_NETWORK_CONNECT_TIMEOUT_SECONDS,
    AI_PREFERRED_BACKEND,
    AI_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_ENABLED,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    OPENROUTER_MODEL,
    OPENROUTER_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
)


LOGGER = logging.getLogger(__name__)

_last_backend = "not_used"
_openrouter_retry_after = 0.0
_openrouter_last_error = ""


class AIBackendUnavailable(RuntimeError):
    """Raised when an AI provider cannot answer the request."""


def openrouter_is_configured() -> bool:
    """
    Return True only when both the OpenRouter key and model are available.
    """

    return bool(OPENROUTER_API_KEY and OPENROUTER_MODEL)


def get_last_backend() -> str:
    """
    Return the provider that generated the most recent response.
    """

    return _last_backend


def get_ai_status() -> dict[str, Any]:
    """
    Return safe AI runtime information without exposing API keys.
    """

    return {
        "preferred_backend": AI_PREFERRED_BACKEND,
        "last_backend": _last_backend,
        "openrouter_configured": openrouter_is_configured(),
        "openrouter_temporarily_skipped": (
            time.monotonic() < _openrouter_retry_after
        ),
        "ollama_enabled": OLLAMA_ENABLED,
        "ollama_model": OLLAMA_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
    }


def _clean_text(value: Any) -> str:
    """
    Convert common API content formats into a plain text response.
    """

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts: list[str] = []

        for item in value:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")

                if isinstance(text, str):
                    parts.append(text)

        return " ".join(parts).strip()

    if value is None:
        return ""

    return str(value).strip()


def _get_error_detail(response: requests.Response) -> str:
    """
    Extract a safe human-readable API error without exposing sensitive data.
    """

    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:300] or "No error details returned."

    if isinstance(payload, dict):
        error = payload.get("error")

        if isinstance(error, dict):
            return str(
                error.get("message")
                or error.get("code")
                or "Unknown provider error."
            )

        if isinstance(error, str):
            return error

        message = payload.get("message")

        if isinstance(message, str):
            return message

    return "No error details returned."


def _call_openrouter(prompt: str) -> str:
    """
    Ask the online OpenRouter model for a response.
    """

    if not openrouter_is_configured():
        raise AIBackendUnavailable(
            "OpenRouter is not configured. Add OPENROUTER_API_KEY and "
            "OPENROUTER_MODEL to .env."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": AI_TEMPERATURE,
        "max_tokens": 450,
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=(
                AI_NETWORK_CONNECT_TIMEOUT_SECONDS,
                OPENROUTER_TIMEOUT_SECONDS,
            ),
        )
    except requests.RequestException as error:
        raise AIBackendUnavailable(
            f"OpenRouter connection failed: {error}"
        ) from error

    if response.status_code in {401, 403}:
        raise AIBackendUnavailable(
            "OpenRouter rejected the API key. Add a valid key to .env."
        )

    if not response.ok:
        detail = _get_error_detail(response)

        raise AIBackendUnavailable(
            f"OpenRouter returned HTTP {response.status_code}: {detail}"
        )

    try:
        payload = response.json()
        choices = payload.get("choices", [])
        message = choices[0].get("message", {})
        answer = _clean_text(message.get("content"))
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise AIBackendUnavailable(
            "OpenRouter returned an unreadable response."
        ) from error

    if not answer:
        raise AIBackendUnavailable(
            "OpenRouter returned an empty response."
        )

    return answer


def _call_ollama(prompt: str) -> str:
    """
    Ask the locally running Ollama model for an offline response.
    """

    if not OLLAMA_ENABLED:
        raise AIBackendUnavailable(
            "Ollama fallback is disabled in .env."
        )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "options": {
            "temperature": AI_TEMPERATURE,
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=(
                AI_NETWORK_CONNECT_TIMEOUT_SECONDS,
                OLLAMA_TIMEOUT_SECONDS,
            ),
        )
    except requests.RequestException as error:
        raise AIBackendUnavailable(
            "Ollama is not running. Start it with: ollama serve"
        ) from error

    if not response.ok:
        detail = _get_error_detail(response)

        if "model" in detail.lower() and "not found" in detail.lower():
            raise AIBackendUnavailable(
                f"Ollama model '{OLLAMA_MODEL}' is not installed. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )

        raise AIBackendUnavailable(
            f"Ollama returned HTTP {response.status_code}: {detail}"
        )

    try:
        payload = response.json()
        message = payload.get("message", {})
        answer = _clean_text(message.get("content"))
    except (AttributeError, TypeError, ValueError) as error:
        raise AIBackendUnavailable(
            "Ollama returned an unreadable response."
        ) from error

    if not answer:
        raise AIBackendUnavailable(
            "Ollama returned an empty response."
        )

    return answer


def _get_backend_order() -> tuple[str, str]:
    """
    Decide which provider Jarvis should try first.
    """

    if AI_PREFERRED_BACKEND == "ollama":
        return ("ollama", "openrouter")

    return ("openrouter", "ollama")


def _mark_openrouter_unavailable(reason: str) -> None:
    """
    Avoid repeatedly calling a failed OpenRouter key for a short period.
    """

    global _openrouter_retry_after
    global _openrouter_last_error

    cooldown_seconds = AI_FAILURE_COOLDOWN_SECONDS

    if "rejected the api key" in reason.lower():
        cooldown_seconds = max(cooldown_seconds, 3600)

    _openrouter_retry_after = time.monotonic() + cooldown_seconds
    _openrouter_last_error = reason

    LOGGER.warning(
        "OpenRouter fallback activated: %s",
        reason,
    )


def _offline_message() -> str:
    """
    Give a clear response when no AI provider is currently available.
    """

    return (
        "Boss, online AI is unavailable and the local Ollama model is not "
        "ready. Start Ollama, then run: "
        f"ollama pull {OLLAMA_MODEL}. "
        "After that, Jarvis will work offline."
    )


def get_ai_response(prompt: str) -> str:
    """
    Return an AI response using OpenRouter or local Ollama.

    Order:
    1. Preferred provider
    2. Backup provider
    3. Clear offline recovery message
    """

    global _last_backend

    cleaned_prompt = str(prompt).strip()

    if not cleaned_prompt:
        _last_backend = "none"

        return "Boss, I did not hear a question."

    failures: list[str] = []

    for backend in _get_backend_order():
        if backend == "openrouter":
            if time.monotonic() < _openrouter_retry_after:
                failures.append(
                    "OpenRouter is temporarily skipped after a recent error."
                )
                continue

            try:
                answer = _call_openrouter(cleaned_prompt)
                _last_backend = "openrouter"

                return answer

            except AIBackendUnavailable as error:
                reason = str(error)
                failures.append(reason)
                _mark_openrouter_unavailable(reason)

        elif backend == "ollama":
            try:
                answer = _call_ollama(cleaned_prompt)
                _last_backend = "ollama"

                return answer

            except AIBackendUnavailable as error:
                failures.append(str(error))
                LOGGER.warning("Ollama fallback unavailable: %s", error)

    _last_backend = "offline_fallback"

    if failures:
        LOGGER.warning(
            "Jarvis could not reach any AI backend: %s",
            " | ".join(failures),
        )

    return _offline_message()