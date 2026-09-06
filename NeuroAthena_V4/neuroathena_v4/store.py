"""Atomic JSON snapshots and append-only events."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def _json(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


class RunStore:
    """Persist an append-only audit log and latest atomic state checkpoint."""
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events = self.run_dir / "events.jsonl"

    def append(self, event_type: str, payload: Any) -> None:
        """Append one UTC-timestamped, JSON-serializable audit event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "payload": _json(payload),
        }
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    def checkpoint(self, state: Any) -> Path:
        """Atomically replace state.json with the latest complete run state."""
        destination = self.run_dir / "state.json"
        temporary = self.run_dir / "state.json.tmp"
        temporary.write_text(json.dumps(_json(state), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
        return destination
