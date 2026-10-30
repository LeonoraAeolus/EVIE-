"""Nemotron on Nebius Token Factory (OpenAI-compatible API).

Verified 2026-10-12 at https://docs.tokenfactory.nebius.com/quickstart :
  base_url = https://api.tokenfactory.nebius.com/v1/   auth = Bearer $NEBIUS_API_KEY
Structured output (response_format json_schema / json_object):
  https://docs.tokenfactory.nebius.com/ai-models-inference/json
Model ids: see config.py; run scripts/check_nebius.py to list what your key can call.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .config import Settings


def _strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.strip()


def parse_json(text: str) -> Any:
    """Robustly pull a JSON object out of a model reply (fences, thinking, chatter)."""
    t = _strip_reasoning(text)
    m = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.S)
    if m:
        t = m.group(1)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            return json.loads(t[start : end + 1])
        raise


class NemotronClient:
    def __init__(self, settings: Settings):
        from openai import OpenAI  # imported lazily so mock mode needs no SDK

        if not settings.nebius_api_key:
            raise RuntimeError("Set NEBIUS_API_KEY for live mode")
        self.s = settings
        self.client = OpenAI(base_url=settings.nebius_base_url, api_key=settings.nebius_api_key)

    def json_call(self, system: str, user: str, schema: dict | None = None, model: str | None = None,
                  temperature: float = 0.2, max_tokens: int = 4000) -> Any:
        model = model or self.s.model
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        attempts: list[dict] = []
        if schema:
            attempts.append({"type": "json_schema", "json_schema": {"name": "out", "schema": schema}})
        attempts += [{"type": "json_object"}, None]
        last_err: Exception | None = None
        for rf in attempts:
            try:
                kw: dict = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                if rf:
                    kw["response_format"] = rf
                resp = self.client.chat.completions.create(**kw)
                return parse_json(resp.choices[0].message.content or "")
            except Exception as e:  # unsupported response_format, bad JSON, ... try the next way
                last_err = e
        raise RuntimeError(f"Nemotron did not return usable JSON: {last_err}")

    def image(self, prompt: str, out_path: str) -> str:
        """Render via Token Factory's images endpoint (example in
        https://docs.tokenfactory.nebius.com/api-reference/examples/image-generation).
        The model id is NOT verified as currently served; set IMAGE_MODEL."""
        import base64

        resp = self.client.images.generate(
            model=self.s.image_model, prompt=prompt, response_format="b64_json",
            extra_body={"response_extension": "png", "width": 1024, "height": 768, "num_inference_steps": 4},
        )
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(resp.data[0].b64_json))
        return out_path
