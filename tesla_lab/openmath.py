r"""OpenMath-Nemotron client: the math consultant.

Self-hosted (see deploy/openmath/) behind vLLM's OpenAI-compatible server. Nemotron 3 Super on
Token Factory stays the lead; this model only derives equations.

Prompt format and settings come from NVIDIA's model card and NeMo-Skills eval docs
(checked 2026-10-13):
  * no system prompt; one user turn:
    "Solve the following math problem. Make sure to put the answer (and only answer) inside \boxed{}.\n\n<problem>"
    (https://huggingface.co/nvidia/OpenMath-Nemotron-7B, NeMo-Skills prompt config generic/math)
  * temperature 0.6 (NeMo-Skills CoT eval: ++inference.temperature=0.6); they allow up to 32,768
    new tokens for competition maths. Our problems are short, so we cap at OPENMATH_MAX_TOKENS (12000);
    the model always thinks first (<think>...</think>, seen on the real 1.5B model), so don't cap it low.
  * top_p 0.95 is our choice (the card and the eval docs do not state one).
  * the card warns the model is not instruction-tuned for general chat: give it maths only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from .config import Settings

PROMPT = "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{{}}.\n\n{problem}"


@dataclass
class Completion:
    text: str  # full text, any <think>...</think> included
    reasoning: str = ""  # separate reasoning field, when vLLM runs with a reasoning parser
    latency_s: float = 0.0
    completion_tokens: int | None = None
    model: str = ""


class MathModel(Protocol):
    model: str

    def solve(self, problem: str) -> Completion: ...


class OpenMathClient:
    r"""Thin wrapper over the OpenAI SDK pointed at your vLLM server."""

    def __init__(self, settings: Settings, client=None):
        if not settings.openmath_base_url and client is None:
            raise RuntimeError("Set OPENMATH_BASE_URL (your vLLM server, e.g. http://localhost:8000/v1)")
        self.s = settings
        self.model = settings.openmath_model
        if client is None:
            from openai import OpenAI  # lazy: mock mode needs no SDK

            client = OpenAI(base_url=settings.openmath_base_url, api_key=settings.openmath_api_key or "EMPTY",
                            timeout=settings.openmath_timeout_s, max_retries=1)
        self.client = client

    def solve(self, problem: str) -> Completion:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": PROMPT.format(problem=problem)}],
            temperature=self.s.openmath_temperature,
            top_p=0.95,
            max_tokens=self.s.openmath_max_tokens,
        )
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        return Completion(text=msg.content or "", reasoning=reasoning or "",
                          latency_s=round(time.perf_counter() - t0, 2),
                          completion_tokens=getattr(usage, "completion_tokens", None), model=self.model)

    def ping(self) -> list[str]:
        r"""Model ids the server lists (GET /v1/models)."""
        return [m.id for m in self.client.models.list().data]
