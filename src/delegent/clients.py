"""Provider-neutral actor and retrieval interfaces plus HTTP implementations."""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ActorResult:
    text: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class SearchHit:
    docid: str
    rank: int
    score: float
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "docid": self.docid,
            "rank": self.rank,
            "score": self.score,
            "text": self.text,
        }


@runtime_checkable
class Actor(Protocol):
    def generate(self, prompt: str) -> ActorResult: ...


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, *, hits: int) -> list[SearchHit]: ...


def _read_api_key(env_name: str | None) -> str | None:
    if env_name is None:
        return None
    if not env_name or not env_name.replace("_", "").isalnum():
        raise ValueError("credential environment-variable name is invalid")
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(
            f"required credential environment variable is unset: {env_name}"
        )
    return value


def _headers(*, api_key_env: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = _read_api_key(api_key_env)
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _validate_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    return base_url.rstrip("/")


@dataclass
class OpenAICompatibleActor:
    """Chat-completions actor with no provider SDK and no embedded credential."""

    base_url: str
    model: str
    api_key_env: str | None = None
    max_tokens: int = 192
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        self.base_url = _validate_base_url(self.base_url)
        if not self.model:
            raise ValueError("model must be non-empty")
        if self.max_tokens < 1 or self.timeout_seconds <= 0:
            raise ValueError("max_tokens and timeout_seconds must be positive")

    def generate(self, prompt: str) -> ActorResult:
        completion_path = (
            "/chat/completions"
            if self.base_url.endswith("/v1")
            else "/v1/chat/completions"
        )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + completion_path,
            data=payload,
            method="POST",
            headers=_headers(api_key_env=self.api_key_env),
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.loads(response.read().decode("utf-8"))
        elapsed = time.monotonic() - started
        if not isinstance(value, dict):
            raise TypeError("actor returned non-object JSON")
        if value.get("model") not in (None, self.model):
            raise RuntimeError(
                "actor response model identity does not match the request"
            )
        choices = value.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            raise RuntimeError("actor response has no completion choice")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise TypeError("actor response has no text content")
        text = message["content"].strip()
        if not text:
            raise RuntimeError("actor returned empty text")
        usage = value.get("usage") or {}
        if not isinstance(usage, dict):
            raise TypeError("actor usage must be an object")
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        generated_tokens = int(usage.get("completion_tokens", 0) or 0)
        if min(prompt_tokens, generated_tokens) < 0:
            raise RuntimeError("actor returned negative token usage")
        return ActorResult(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            elapsed_seconds=elapsed,
        )


def _normalize_document(document: object) -> str:
    if document is None:
        return ""
    if isinstance(document, str):
        return document
    if isinstance(document, dict):
        for key in ("text", "contents", "content", "segment", "body"):
            value = document.get(key)
            if isinstance(value, str):
                return value
        return " ".join(
            str(value) for value in document.values() if isinstance(value, str)
        )
    if isinstance(document, list):
        return " ".join(_normalize_document(item) for item in document)
    return str(document)


@dataclass
class HttpRetriever:
    """HTTP client for the Pyserini REST ``/v1/{index}/search`` contract."""

    base_url: str
    index: str
    api_key_env: str | None = None
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        self.base_url = _validate_base_url(self.base_url)
        if not self.index:
            raise ValueError("index must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    def search(self, query: str, *, hits: int) -> list[SearchHit]:
        normalized_query = " ".join(str(query).split())
        if not normalized_query:
            raise ValueError("query must be non-empty")
        if hits < 1:
            raise ValueError("hits must be positive")
        path = f"/v1/{urllib.parse.quote(self.index, safe='')}/search"
        parameters = urllib.parse.urlencode({"query": normalized_query, "hits": hits})
        request = urllib.request.Request(
            f"{self.base_url}{path}?{parameters}",
            method="GET",
            headers=_headers(api_key_env=self.api_key_env),
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("candidates"), list):
            raise TypeError("retrieval response must contain a candidates list")
        result: list[SearchHit] = []
        for position, candidate in enumerate(value["candidates"], start=1):
            if not isinstance(candidate, dict):
                raise TypeError("retrieval candidate must be an object")
            docid = str(candidate.get("docid") or "")
            if not docid:
                raise RuntimeError("retrieval candidate has no docid")
            result.append(
                SearchHit(
                    docid=docid,
                    rank=int(candidate.get("rank", position) or position),
                    score=float(candidate.get("score", 0.0) or 0.0),
                    text=_normalize_document(
                        candidate.get("doc", candidate.get("text"))
                    ),
                )
            )
        if len(result) > hits:
            raise RuntimeError("retrieval service returned more hits than requested")
        return result
