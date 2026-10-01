"""The LLM judge used by the intruder, autointerp and matching evaluations.

Any OpenAI-compatible `/chat/completions` endpoint: a local `vllm serve` (the default, as in the
paper) or a hosted provider serving the same model. The paper's judge is Llama-3.3-70B-Instruct.

    --judge-base-url http://127.0.0.1:8010/v1                       local vLLM (no key)
    --judge-base-url https://openrouter.ai/api/v1 \\
        --judge-model meta-llama/llama-3.3-70b-instruct \\
        --judge-api-key-env OPENROUTER_API_KEY                      hosted
"""

import argparse
import json
import os
from typing import Any, override

import httpx
from param_decomp.base_config import BaseConfig
from param_decomp.log import logger
from param_decomp_lab.autointerp.providers import (
    ChatResponse,
    LLMProvider,
    RetryableAPIError,
    _parse_retry_after_header,
)

DEFAULT_JUDGE_BASE_URL = "http://127.0.0.1:8010/v1"
DEFAULT_JUDGE_MODEL = "unsloth/Llama-3.3-70B-Instruct"


class JudgeConfig(BaseConfig):
    """Where the judge lives and how hard to drive it."""

    base_url: str = DEFAULT_JUDGE_BASE_URL
    model: str = DEFAULT_JUDGE_MODEL
    api_key_env: str | None = None
    """Environment variable holding the API key; `None` for a local server that needs none."""
    structured: bool = False
    """`json_schema` response format when True; `json_object` otherwise (vLLM and most hosts)."""
    max_concurrent: int = 64
    max_requests_per_minute: int = 1_000_000

    def api_key(self) -> str:
        if self.api_key_env is None:
            return "EMPTY"
        key = os.environ.get(self.api_key_env)
        assert key, f"{self.api_key_env} is not set"
        return key


class OpenAICompatProvider(LLMProvider):
    """`param_decomp_lab`'s `LLMProvider` over an OpenAI-compatible `/chat/completions` endpoint."""

    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self.model = cfg.model
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url, headers={"Authorization": f"Bearer {cfg.api_key()}"}
        )

    @override
    async def chat(
        self, prompt: str, max_tokens: int, response_schema: dict[str, Any], timeout_ms: int
    ) -> ChatResponse:
        if self.cfg.structured:
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {**response_schema, "additionalProperties": False},
                    "strict": True,
                },
            }
        else:
            response_format = {"type": "json_object"}
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": response_format,
        }
        try:
            resp = await self._client.post("/chat/completions", json=body, timeout=timeout_ms / 1000)
        except httpx.TransportError as e:
            raise RetryableAPIError(str(e)) from e
        if resp.status_code in (408, 429, 500, 502, 503, 504):
            raise RetryableAPIError(
                f"HTTP {resp.status_code}: {resp.text[:200]}",
                retry_after=_parse_retry_after_header(resp),
            )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RetryableAPIError(data["error"].get("message", str(data["error"])))
        choice = data["choices"][0]
        content = choice["message"]["content"]
        assert isinstance(content, str)
        json.loads(content)
        if choice.get("finish_reason") == "length":
            logger.warning(f"judge response truncated at {max_tokens} tokens")
        usage = data["usage"]
        return ChatResponse(
            content=content,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
        )

    @override
    async def get_pricing(self) -> tuple[float, float]:
        # Cost accounting is not tracked; `--cost-limit-usd` is therefore inert.
        return (0.0, 0.0)

    @override
    async def close(self) -> None:
        await self._client.aclose()


def add_judge_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL,
                    help="OpenAI-compatible base url of the judge")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                    help="model id the judge endpoint serves")
    ap.add_argument("--judge-api-key-env", default=None,
                    help="environment variable holding the judge's API key (omit for local vLLM)")
    ap.add_argument("--judge-concurrency", type=int, default=64,
                    help="in-flight requests against the judge")


def judge_config_from_args(args: argparse.Namespace, *, structured: bool = False) -> JudgeConfig:
    return JudgeConfig(
        base_url=args.judge_base_url,
        model=args.judge_model,
        api_key_env=args.judge_api_key_env,
        structured=structured,
        max_concurrent=args.judge_concurrency,
    )
