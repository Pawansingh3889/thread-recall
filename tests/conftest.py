"""Stub fastmcp when it is not installed, so mcp_server imports and its tool
functions stay plain callables for isolation tests. Only stubs when absent."""
import importlib.machinery
import importlib.util
import os
import sys
import types

# The MCP server builds a store at import; keep that off disk during tests.
os.environ.setdefault("THREAD_RECALL_DB", ":memory:")


if importlib.util.find_spec("fastmcp") is None:
    fastmcp = types.ModuleType("fastmcp")

    class FastMCP:
        def __init__(self, *a, **k):
            pass

        def tool(self, fn=None, *a, **k):
            # Supports both @mcp.tool and @mcp.tool(...); returns the plain fn.
            if fn is None:
                return lambda f: f
            return fn

        def run(self, *a, **k):
            pass

    fastmcp.FastMCP = FastMCP
    fastmcp.__spec__ = importlib.machinery.ModuleSpec("fastmcp", loader=None)
    sys.modules["fastmcp"] = fastmcp
