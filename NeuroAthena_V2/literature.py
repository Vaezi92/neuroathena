"""Conservative Europe PMC literature retrieval with explicit search intent."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
import uuid
from typing import Iterable

from models import LiteratureRecord, SearchQuery


EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def build_queries(claim_id: str, interpretation: str) -> list[SearchQuery]:
    base = " ".join(interpretation.split())[:350]
    return [
        SearchQuery(query_id=f"query-{uuid.uuid4().hex[:10]}", claim_id=claim_id, intent="support", query=f"{base} mouse brain MRI tracer"),
        SearchQuery(query_id=f"query-{uuid.uuid4().hex[:10]}", claim_id=claim_id, intent="contradiction", query=f"{base} conflicting evidence mouse brain"),
        SearchQuery(query_id=f"query-{uuid.uuid4().hex[:10]}", claim_id=claim_id, intent="alternative", query=f"{base} alternative explanation diffusion partial volume"),
        SearchQuery(query_id=f"query-{uuid.uuid4().hex[:10]}", claim_id=claim_id, intent="limitation", query=f"MR-AIV inferred velocity permeability methodological limitations"),
    ]


def search_europe_pmc(query: SearchQuery, page_size: int = 3, timeout: float = 15.0) -> list[LiteratureRecord]:
    params = urllib.parse.urlencode({"query": query.query, "format": "json", "pageSize": page_size})
    request = urllib.request.Request(f"{EUROPE_PMC}?{params}", headers={"User-Agent": "NeuroAthena/2.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    records: list[LiteratureRecord] = []
    for result in payload.get("resultList", {}).get("result", []):
        doi = result.get("doi")
        identifier = result.get("pmcid") or result.get("pmid") or doi
        url = f"https://europepmc.org/article/{'PMC' if result.get('pmcid') else 'MED'}/{identifier}" if identifier else EUROPE_PMC
        records.append(
            LiteratureRecord(
                record_id=f"literature-{uuid.uuid4().hex[:10]}",
                query_id=query.query_id,
                title=result.get("title") or "Untitled record",
                authors=result.get("authorString") or "",
                year=int(result["pubYear"]) if str(result.get("pubYear", "")).isdigit() else None,
                journal=result.get("journalTitle") or "",
                doi=doi,
                url=url,
                peer_reviewed=None,
                provider="europe_pmc",
            )
        )
    return records

def retrieve_all(queries: Iterable[SearchQuery], online: bool) -> list[LiteratureRecord]:
    if not online:
        return []
    records: list[LiteratureRecord] = []
    for query in queries:
        try:
            records.extend(search_europe_pmc(query))
        except (OSError, TimeoutError, ValueError, json.JSONDecodeError):
            continue
    return records
