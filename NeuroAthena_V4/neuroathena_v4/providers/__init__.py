"""Gateways to external model and knowledge providers."""

from .openai_gateway import create_text, parse_structured

__all__ = ["create_text", "parse_structured"]
