"""thread-recall-mcp: expose governed agent memory as MCP tools.

An agent calls ``remember`` to store a turn and ``recall`` to retrieve relevant
past turns, instead of holding unbounded raw history in its context. The
governance is on the write side: PII is masked before anything is stored (via
pii-veil when installed), so the long-term memory never retains raw PII, and
every write can be mirrored into a tamper-evident audit ledger. Memory is
thread-scoped, so one thread never recalls another's turns.

Configuration (environment):
  THREAD_RECALL_DB        SQLite path for the store (default: logs/recall-mem.db)
  THREAD_RECALL_MASK      mask PII on write (default on; set 0 to disable)
  THREAD_RECALL_AUDIT     mirror writes to agent-blackbox (default off; set 1 on)
  THREAD_RECALL_EMBED     hashing (default) | ollama
  THREAD_RECALL_OLLAMA_*  HOST / MODEL overrides for the Ollama embedder
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any

from fastmcp import FastMCP

from thread_recall.store import Memory

_TOKEN = re.compile(r"[a-z0-9]+")
_ON = {"1", "true", "yes", "on"}


def _hash_embed(text: str, dim: int = 256) -> list[float]:
    """Deterministic, offline bag-of-words embedding (no model, no network)."""
    vec = [0.0] * dim
    for tok in _TOKEN.findall(text.lower()):
        vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _embed(text: str) -> list[float]:
    if os.environ.get("THREAD_RECALL_EMBED", "hashing").strip().lower() == "ollama":
        import json
        import urllib.request

        host = os.environ.get("THREAD_RECALL_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        model = os.environ.get("THREAD_RECALL_OLLAMA_MODEL", "nomic-embed-text")
        req = urllib.request.Request(
            f"{host}/api/embeddings",
            data=json.dumps({"model": model, "prompt": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return list(json.loads(resp.read())["embedding"])
    return _hash_embed(text)


def _build_memory() -> Memory:
    return Memory(
        os.environ.get("THREAD_RECALL_DB", "logs/recall-mem.db"),
        mask=os.environ.get("THREAD_RECALL_MASK", "1").strip().lower() in _ON,
        audit=os.environ.get("THREAD_RECALL_AUDIT", "0").strip().lower() in _ON,
    )


mcp = FastMCP("thread-recall")
_mem = _build_memory()


@mcp.tool
def remember(thread_id: str, content: str, role: str = "user") -> dict[str, Any]:
    """Store one turn of a thread's memory. PII is masked before storage.

    Args:
        thread_id: The conversation/thread this turn belongs to (the access scope).
        content: The text to remember. Masked on write if masking is enabled.
        role: Who said it (user / assistant / system).

    Returns:
        The stored turn's id. The content held in memory is the masked form.
    """
    turn_id = _mem.remember(thread_id, role, content, embedding=_embed(content))
    return {"id": turn_id, "thread_id": thread_id, "stored": True}


@mcp.tool
def recall(thread_id: str, query: str, k: int = 5) -> dict[str, Any]:
    """Recall the turns most relevant to a query, scoped to one thread.

    Semantic nearest-neighbour over the thread's masked memory. Only this
    thread's turns are searched; nothing leaks across threads.
    """
    hits = _mem.search(thread_id, _embed(query), k)
    if not hits:  # nothing embedded yet -> fall back to recency
        hits = _mem.recent(thread_id, k)
    return {"count": len(hits), "results": [t.as_dict() for t in hits]}


@mcp.tool
def recent(thread_id: str, k: int = 10) -> dict[str, Any]:
    """The last k turns of a thread, in chronological order."""
    hits = _mem.recent(thread_id, k)
    return {"count": len(hits), "results": [t.as_dict() for t in hits]}


@mcp.tool
def forget(thread_id: str) -> dict[str, Any]:
    """Delete a thread's entire memory. Returns the number of turns removed."""
    return {"removed": _mem.forget(thread_id), "thread_id": thread_id}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
