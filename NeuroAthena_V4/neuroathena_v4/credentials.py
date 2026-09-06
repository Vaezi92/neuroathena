"""Local credential loading without logging or serializing secrets."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_API_KEY_FILE = PROJECT_ROOT / "api.txt"


def load_openai_api_key() -> str:
    """Prefer the environment; otherwise read the local, Git-ignored key file."""
    environment_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    if not DEFAULT_API_KEY_FILE.is_file():
        raise RuntimeError(
            "OpenAI API key is unavailable. Set OPENAI_API_KEY or create the local Git-ignored api.txt file."
        )
    key = DEFAULT_API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("api.txt is empty")
    if not key.startswith("sk-"):
        raise RuntimeError("api.txt does not contain a recognized OpenAI API-key format")
    return key


def openai_api_key_available() -> bool:
    """Return whether an environment or local-file credential is available."""
    return bool(os.environ.get("OPENAI_API_KEY", "").strip()) or DEFAULT_API_KEY_FILE.is_file()
