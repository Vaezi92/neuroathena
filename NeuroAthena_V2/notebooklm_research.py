"""Read-only NotebookLM evidence retrieval through a local MCP subprocess."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from models import LiteratureRecord, SearchQuery


READ_ONLY_TOOLS = frozenset({"server_info", "notebook_list", "source_list", "chat_ask"})


@dataclass(frozen=True)
class NotebookLMEvidence:
    records: list[LiteratureRecord]
    responses: list[dict[str, Any]]


def _payload_from_result(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for block in getattr(result, "content", []):
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("NotebookLM MCP returned no structured JSON payload")


def records_from_payload(query: SearchQuery, notebook: str, payload: dict[str, Any]) -> list[LiteratureRecord]:
    """Accept documented and backward-compatible citation/reference shapes."""
    references = payload.get("references") or payload.get("citations") or payload.get("sources") or []
    records: list[LiteratureRecord] = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        nested_source = reference.get("source") if isinstance(reference.get("source"), dict) else {}
        source_id = str(reference.get("source_id") or reference.get("id") or nested_source.get("id") or "").strip() or None
        title = str(reference.get("source_title") or reference.get("title") or nested_source.get("title") or "").strip()
        excerpt = str(reference.get("cited_text") or reference.get("excerpt") or reference.get("text") or "").strip() or None
        if not title or not source_id:
            continue
        records.append(
            LiteratureRecord(
                record_id=f"literature-{uuid.uuid4().hex[:10]}",
                query_id=query.query_id,
                title=title,
                url=f"notebooklm://{notebook}/{source_id}",
                provider="notebooklm",
                source_id=source_id,
                excerpt=excerpt,
                peer_reviewed=None,
            )
        )
    return records


def _research_prompt(query: SearchQuery) -> str:
    return (
        "Use only the sources already selected in this notebook. Treat this as an independent evidence search. "
        f"Search intent: {query.intent}. Scientific question: {query.query}. "
        "Return a cautious answer grounded in citations. State explicitly when the notebook lacks evidence. "
        "Do not infer biological causality and do not use knowledge outside the notebook."
    )


async def _retrieve(notebook: str, profile: str, queries: list[SearchQuery]) -> NotebookLMEvidence:
    bundled_executable = Path(sys.executable).parent / "notebooklm-mcp"
    executable = shutil.which("notebooklm-mcp") or (str(bundled_executable) if bundled_executable.is_file() else None)
    if executable is None:
        raise RuntimeError("notebooklm-mcp is not installed in the active environment")
    environment = os.environ.copy()
    environment["NOTEBOOKLM_MCP_STRICT_IDS"] = "1"
    parameters = StdioServerParameters(
        command=executable,
        args=["--profile", profile, "--log-level", "ERROR"],
        env=environment,
    )
    records: list[LiteratureRecord] = []
    responses: list[dict[str, Any]] = []
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            available = {tool.name for tool in (await session.list_tools()).tools}
            if "chat_ask" not in available:
                raise RuntimeError("NotebookLM MCP does not expose chat_ask")
            for query in queries:
                result = await session.call_tool(
                    "chat_ask",
                    arguments={
                        "notebook": notebook,
                        "question": _research_prompt(query),
                        "references": "full",
                        "history": 0,
                        "suggest_followups": False,
                    },
                )
                if getattr(result, "isError", False):
                    raise RuntimeError(f"NotebookLM query failed for intent {query.intent}")
                payload = _payload_from_result(result)
                query_records = records_from_payload(query, notebook, payload)
                records.extend(query_records)
                responses.append(
                    {
                        "query_id": query.query_id,
                        "claim_id": query.claim_id,
                        "intent": query.intent,
                        "notebook": notebook,
                        "answer": payload.get("answer", ""),
                        "notebook_id": payload.get("notebook_id"),
                        "conversation_id": payload.get("conversation_id"),
                        "citation_count": len(query_records),
                        "provider": "notebooklm-py-mcp",
                    }
                )
    return NotebookLMEvidence(records=records, responses=responses)


def retrieve_notebooklm(notebook: str, profile: str, queries: Iterable[SearchQuery]) -> NotebookLMEvidence:
    """Run the async MCP client from a synchronous LangGraph node."""
    return asyncio.run(_retrieve(notebook, profile, list(queries)))
