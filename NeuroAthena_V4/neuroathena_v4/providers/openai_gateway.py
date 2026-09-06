"""Single boundary around the OpenAI Responses API."""

from __future__ import annotations

from pydantic import BaseModel

from ..credentials import load_openai_api_key


def _client():
    """Construct an authenticated client without exposing its credential."""
    from openai import OpenAI

    return OpenAI(api_key=load_openai_api_key())


def parse_structured(model: str, system: str, prompt: str, schema: type[BaseModel]) -> BaseModel:
    """Return a response validated against the supplied Pydantic schema."""
    response = _client().responses.parse(
        model=model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        text_format=schema,
    )
    if response.output_parsed is None:
        raise RuntimeError("OpenAI returned no structured output")
    return response.output_parsed


def create_text(model: str, system: str, prompt: str) -> str:
    """Return final prose from OpenAI and reject an empty response."""
    response = _client().responses.create(
        model=model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    if not response.output_text:
        raise RuntimeError("OpenAI returned empty text")
    return response.output_text
