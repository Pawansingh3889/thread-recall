"""thread-recall-mcp: expose governed agent memory as MCP tools.

An agent calls ``remember`` to store a turn and ``recall`` to retrieve relevant
past turns, instead of holding unbounded raw history in its context. The
governance is on the write side: PII is masked before anything is stored (via
pii-veil when installed), so the long-term memory never retains raw PII, and
every write can be mirrored into a tamper-evident audit ledger.

Isolation: every thread is namespaced by an ``actor`` -- the authenticated
principal. A caller can only reach threads under its own actor, so a raw
``thread_id`` is no longer a key into anyone else's memory (no cross-thread
read, poison, or mass-delete). The governed gateway sets ``actor`` from the
caller's token and does not let the client override it; standalone, set
``THREAD_RECALL_ACTOR`` (or pass ``actor``), else all callers share one
namespace. The caller always sees its own plain ``thread_id`` back.

Configuration (environment):
  THREAD_RECALL_DB        SQLite path for the store (default: logs/recall-mem.db)
  THREAD_RECALL_MASK      mask PII on write (default on; set 0 to disable)
  THREAD_RECALL_AUDIT     mirror writes to agent-blackbox (default off; set 1 on)
  THREAD_RECALL_ACTOR     default principal when no actor is passed (default: shared)
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
_SEP = "\x1f"  # unit separator: partitions the store by actor, unseen by callers


def _principal(actor: str | None) -> str:
    """The authenticated identity a thread belongs to. The gateway supplies it."""
    return (actor or os.environ.get("THREAD_RECALL_ACTOR", "shared")).strip() or "shared"


def _key(actor: str | None, thread_id: str) -> str:
    """Namespace a thread by its actor so ids cannot collide across principals."""
    return f"{_principal(actor)}{_SEP}{thread_id}"


def _plain(namespaced: str) -> str:
    """Strip the actor namespace so the caller sees the thread_id it passed in."""
    return namespaced.split(_SEP, 1)[1] if _SEP in namespaced else namespaced


def _view(turn: dict) -> dict:
    turn["thread_id"] = _plain(turn.get("thread_id", ""))
    return turn


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


@mcp.tool()
def remember(thread_id: str, content: str, role: str = "user",
             actor: str | None = None) -> dict[str, Any]:
    """Store one turn of a thread's memory. PII is masked before storage.

    Args:
        thread_id: The conversation/thread this turn belongs to.
        content: The text to remember. Masked on write if masking is enabled.
        role: Who said it (user / assistant / system).
        actor: Authenticated principal owning the thread. Set by the gateway;
            do not rely on client-supplied values for isolation.

    Returns:
        The stored turn's id. The content held in memory is the masked form.
    """
    turn_id = _mem.remember(_key(actor, thread_id), role, content, embedding=_embed(content))
    return {"id": turn_id, "thread_id": thread_id, "stored": True}


@mcp.tool()
def recall(thread_id: str, query: str, k: int = 5,
           actor: str | None = None) -> dict[str, Any]:
    """Recall the turns most relevant to a query, scoped to one thread.

    Semantic nearest-neighbour over the actor's own thread. Only this actor's
    copy of the thread is searched; another principal's memory is unreachable.
    """
    key = _key(actor, thread_id)
    hits = _mem.search(key, _embed(query), k)
    if not hits:  # nothing embedded yet -> fall back to recency
        hits = _mem.recent(key, k)
    return {"count": len(hits), "results": [_view(t.as_dict()) for t in hits]}


@mcp.tool()
def recent(thread_id: str, k: int = 10, actor: str | None = None) -> dict[str, Any]:
    """The last k turns of the actor's thread, in chronological order."""
    hits = _mem.recent(_key(actor, thread_id), k)
    return {"count": len(hits), "results": [_view(t.as_dict()) for t in hits]}


@mcp.tool()
def forget(thread_id: str, actor: str | None = None) -> dict[str, Any]:
    """Delete the actor's copy of a thread. Returns the number of turns removed.

    Only the calling principal's namespace is touched; a caller cannot wipe
    another actor's thread even by naming the same thread_id.
    """
    return {"removed": _mem.forget(_key(actor, thread_id)), "thread_id": thread_id}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
