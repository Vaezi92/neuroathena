"""Provider-facing import for the read-only NotebookLM MCP adapter."""

from ..notebooklm import READ_ONLY_TOOLS, ask_notebook, notebook_context

__all__ = ["READ_ONLY_TOOLS", "ask_notebook", "notebook_context"]
