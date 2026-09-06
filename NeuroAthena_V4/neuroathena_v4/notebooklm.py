"""Read-only NotebookLM access using the V3 notebooklm-py MCP transport."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


READ_ONLY_TOOLS = frozenset({"server_info", "notebook_list", "source_list", "chat_ask"})


def _payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for block in getattr(result, "content", []):
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("NotebookLM MCP returned no structured JSON payload")


async def _ask(notebook_id: str, question: str, profile: str, attempts: int) -> dict[str, Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    bundled = Path(sys.executable).parent / "notebooklm-mcp"
    executable = shutil.which("notebooklm-mcp") or (str(bundled) if bundled.is_file() else None)
    if executable is None:
        raise RuntimeError("notebooklm-mcp is not installed in the active environment")
    environment = os.environ.copy()
    environment["NOTEBOOKLM_MCP_STRICT_IDS"] = "1"
    parameters = StdioServerParameters(
        command=executable,
        args=["--profile", profile, "--log-level", "ERROR"],
        env=environment,
    )
    last_error: Exception | None = None
    with open(os.devnull, "w", encoding="utf-8") as errlog:
        async with stdio_client(parameters, errlog=errlog) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                available = {tool.name for tool in (await session.list_tools()).tools}
                if "chat_ask" not in available:
                    raise RuntimeError("NotebookLM MCP does not expose chat_ask")
                for attempt in range(attempts):
                    try:
                        result = await session.call_tool(
                            "chat_ask",
                            arguments={
                                "notebook": notebook_id,
                                "question": question,
                                "references": "full",
                                "history": 0,
                                "suggest_followups": False,
                            },
                        )
                        if getattr(result, "isError", False):
                            message = " ".join(str(getattr(block, "text", "")) for block in getattr(result, "content", []))
                            raise RuntimeError(message.strip() or "NotebookLM MCP returned an error")
                        return _payload(result)
                    except Exception as error:
                        last_error = error
                        if attempt + 1 < attempts:
                            await asyncio.sleep(2**attempt)
    raise RuntimeError(f"NotebookLM request failed: {last_error}")


def ask_notebook(notebook_id: str, question: str, *, profile: str = "default", attempts: int = 2) -> dict[str, Any]:
    """Ask one specialist notebook. This method never modifies notebooks or sources."""
    return asyncio.run(_ask(notebook_id, question, profile, attempts))


def notebook_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Retain the answer and source-level reference fields for provenance."""
    references = payload.get("references") or payload.get("citations") or payload.get("sources") or []
    return {
        "answer": payload.get("answer", ""),
        "references": references,
        "notebook_id": payload.get("notebook_id"),
        "conversation_id": payload.get("conversation_id"),
    }
