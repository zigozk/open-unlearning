from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("***REDACTED***" if "key" in key.lower() or "authorization" in key.lower() else redact_secrets(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def extract_response_text(response: dict[str, Any]) -> str:
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content["text"]
            if content.get("type") == "refusal":
                raise ValueError(f"API refusal: {content.get('refusal', '')}")
    raise ValueError("Responses API payload contains no output_text")


@dataclass
class ResponsesProvider:
    model: str
    schema_name: str
    schema: dict[str, Any]
    base_url: str = "https://api.openai.com"
    timeout: int = 120
    max_retries: int = 6

    @classmethod
    def from_env(cls, *, model_env: str, schema_name: str, schema: dict[str, Any]) -> "ResponsesProvider":
        model = os.environ.get(model_env)
        if not model:
            raise RuntimeError(f"{model_env} must name a fixed, available model snapshot")
        return cls(
            model=model,
            schema_name=schema_name,
            schema=schema,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/"),
            timeout=int(os.environ.get("ATOMIC_TOFU_TIMEOUT", "120")),
        )

    def request(self, *, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")
        body = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "text": {"format": {
                "type": "json_schema",
                "name": self.schema_name,
                "strict": True,
                "schema": self.schema,
            }},
            "store": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = json.load(response)
                parsed = json.loads(extract_response_text(raw))
                return parsed, {
                    "provider": "openai-responses",
                    "model": raw.get("model", self.model),
                    "response_id": raw.get("id"),
                    "usage": raw.get("usage", {}),
                    "request": redact_secrets(body),
                    "raw_response": redact_secrets(raw),
                }
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                retryable = not isinstance(error, urllib.error.HTTPError) or error.code in {408, 409, 429, 500, 502, 503, 504}
                if attempt + 1 == self.max_retries or not retryable:
                    raise
                time.sleep(min(30.0, (2**attempt) + random.random()))
        raise AssertionError("unreachable")


def batch_request_line(custom_id: str, provider: ResponsesProvider, system: str, user: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": provider.model,
            "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "text": {"format": {"type": "json_schema", "name": provider.schema_name, "strict": True, "schema": provider.schema}},
            "store": False,
        },
    }
