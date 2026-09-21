"""Minimal DeepSeek Chat Completions adapter for reproducible experiments."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from financial_router.llm_router import ModelReply, ModelRequest


Transport = Callable[[str, Mapping[str, str], bytes, float], Mapping[str, Any]]


class DeepSeekConfigurationError(ValueError):
    """Raised when the experiment cannot be configured safely."""


class DeepSeekCallError(RuntimeError):
    """Raised when a provider call fails after the configured attempts."""

    def __init__(self, message: str, *, attempts: int):
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True)
class DeepSeekConfig:
    model: str
    temperature: float | None
    timeout_seconds: float
    max_retries: int
    max_tokens: int = 256
    thinking_enabled: bool = False
    base_url: str = "https://api.deepseek.com"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise DeepSeekConfigurationError("model must not be empty")
        if self.thinking_enabled and self.temperature is not None:
            raise DeepSeekConfigurationError(
                "temperature must be omitted because thinking mode ignores it"
            )
        if not self.thinking_enabled and self.temperature is None:
            raise DeepSeekConfigurationError(
                "temperature must be explicit when thinking mode is disabled"
            )
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise DeepSeekConfigurationError("temperature must be between 0 and 2")
        if self.timeout_seconds <= 0:
            raise DeepSeekConfigurationError("timeout_seconds must be positive")
        if self.max_retries != 0:
            raise DeepSeekConfigurationError(
                "max_retries must be 0 because provider errors are returned directly"
            )
        if self.max_tokens <= 0:
            raise DeepSeekConfigurationError("max_tokens must be positive")
        if not self.base_url.startswith("https://"):
            raise DeepSeekConfigurationError("base_url must use HTTPS")


def _urllib_transport(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=body,
        headers=dict(headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("provider response must be a JSON object")
    return value


class DeepSeekChatModel:
    def __init__(
        self,
        config: DeepSeekConfig,
        *,
        transport: Transport | None = None,
        environ: Mapping[str, str] | None = None,
    ):
        source = os.environ if environ is None else environ
        api_key = source.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise DeepSeekConfigurationError(
                "DEEPSEEK_API_KEY must be provided through the environment"
            )
        self._config = config
        self._api_key = api_key
        self._transport = transport or _urllib_transport

    def complete(self, request: ModelRequest) -> ModelReply:
        config = self._config
        url = f"{config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(request.user_payload, ensure_ascii=False),
                },
            ],
            "response_format": {"type": "json_object"},
            "thinking": {
                "type": "enabled" if config.thinking_enabled else "disabled"
            },
            "max_tokens": config.max_tokens,
            "stream": False,
        }
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        try:
            value = self._transport(url, headers, body, config.timeout_seconds)
        except urllib.error.HTTPError as error:
            raise DeepSeekCallError(
                f"provider HTTP request failed with status {error.code}",
                attempts=1,
            ) from None
        except Exception as error:
            raise DeepSeekCallError(
                f"provider transport failed: {type(error).__name__}",
                attempts=1,
            ) from None

        try:
            return self._parse_reply(value)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise DeepSeekCallError(
                f"invalid provider response: {type(error).__name__}",
                attempts=1,
            ) from None

    @staticmethod
    def _parse_reply(value: Mapping[str, Any]) -> ModelReply:
        choices = value["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("choices must be a non-empty array")
        content = choices[0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        usage = value.get("usage", {})
        if not isinstance(usage, Mapping):
            raise ValueError("usage must be an object")
        model = value.get("model")
        fingerprint = value.get("system_fingerprint")
        return ModelReply(
            content=content,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            model=model if isinstance(model, str) else None,
            system_fingerprint=fingerprint if isinstance(fingerprint, str) else None,
        )


__all__ = [
    "DeepSeekCallError",
    "DeepSeekChatModel",
    "DeepSeekConfig",
    "DeepSeekConfigurationError",
]
