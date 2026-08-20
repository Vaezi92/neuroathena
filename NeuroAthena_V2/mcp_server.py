"""MR-AIV Analysis MCP server. Run with: python mcp_server.py"""

from __future__ import annotations

import os
from pathlib import Path

from analysis import AnalysisService


def build_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install V2 requirements first: python -m pip install -r requirements.txt") from exc

    source = Path(os.environ["NEUROATHENA_MAT_FILE"])
    run_id = os.getenv("NEUROATHENA_RUN_ID", "mcp-run")
    artifact_dir = Path(os.getenv("NEUROATHENA_ARTIFACT_DIR", Path(__file__).parent / "artifacts" / run_id))
    service = AnalysisService(source, run_id, artifact_dir)
    server = FastMCP("MR-AIV Analysis")

    @server.tool(name="mraiv.get_metadata")
    def metadata() -> dict:
        return service.execute("mraiv.get_metadata", {}).model_dump()

    @server.tool(name="mraiv.get_roi_statistics")
    def roi_statistics(roi: dict) -> dict:
        return service.execute("mraiv.get_roi_statistics", {"roi": roi}).model_dump()

    @server.tool(name="mraiv.plot_velocity_slice")
    def velocity_slice() -> dict:
        return service.execute("mraiv.plot_velocity_slice", {}).model_dump()

    @server.tool(name="mraiv.plot_permeability_slice")
    def permeability_slice() -> dict:
        return service.execute("mraiv.plot_permeability_slice", {}).model_dump()

    @server.tool(name="mraiv.plot_distribution")
    def distribution(field: str) -> dict:
        return service.execute("mraiv.plot_distribution", {"field": field}).model_dump()

    @server.tool(name="mraiv.compare_rois")
    def roi_comparison(first: dict, second: dict) -> dict:
        return service.execute("mraiv.compare_rois", {"first": first, "second": second}).model_dump()

    @server.tool(name="mraiv.retrieve_artifact")
    def retrieve_artifact(artifact_id: str) -> dict:
        return service.execute("mraiv.retrieve_artifact", {"artifact_id": artifact_id}).model_dump()

    return server


if __name__ == "__main__":
    build_server().run(transport="stdio")
