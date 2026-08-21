"""Regenerate a V3 HTML report from an existing report.json without external calls."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from models import NeuroAthenaState
from reporting import render_html_report


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("report_json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.report_json.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    state = NeuroAthenaState.model_validate(payload)
    output = args.output.resolve() if args.output else source.with_name("report_rebuilt.html")
    render_html_report(state, output)
    print(f"Rebuilt report: {output}")


if __name__ == "__main__":
    main()
