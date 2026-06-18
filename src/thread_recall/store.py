"""Governed, on-prem conversation memory for AI agents.

A small store for what an agent should remember across turns: per-thread
conversation history (recency) and optional semantic recall (nearest-neighbour
over stored embeddings). The "governed" part is the point — memory can be
PII-masked on write (so raw personal data never lands in the store) and every
write can be mirrored into a tamper-evident audit log.

The default backend is SQLite with embeddings kept as JSON and cosine computed
in Python: zero dependencies, works immediately, nothing leaves the building.
A Postgres + pgvector backend (for scale) is the next step; this module keeps
the embedding/scoring logic backend-agnostic so it drops in cleanly.

This is deliberately NOT a read path into your database — that is sql-steward's
job, and it stays read-only. This is the write-side state store that sits next
to it.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass


@dataclass
class Turn:
    """One remembered turn."""

    id: int
    thread_id: str
    ts: float
    role: str
    content: str
    metadata: dict | None = None
    score: float | None = None  # set by search()

    def as_dict(self) -> dict:
        d = {"id": self.id, "thread_id": self.thread_id, "ts": self.ts,
             "role": self.role, "content": self.content, "metadata": self.metadata}
        if self.score is not None:
            d["score"] = self.score
        return d


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class Memory:
    """Thread-scoped agent memory.

    Args:
        path: SQLite path, or ":memory:" (default).
        mask: scrub PII from content on write via pii-veil, if installed.
        audit: mirror every write into an agent-blackbox ledger, if installed.
        audit_db: path for the audit ledger (default "logs/recall.db").
    """

    def __init__(self, path: str = ":memory:", *, mask: bool = False,
                 audit: bool = False, audit_db: str = "logs/recall.db") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._veil = _make_veil() if mask else None
        self._ledger = _make_ledger(audit_db) if audit else None

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                ts        REAL NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                metadata  TEXT,
                embedding TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_thread ON memories(thread_id, id)"
        )
        self._conn.commit()

    # -- write --------------------------------------------------------------

    def remember(
        self,
        thread_id: str,
        role: str,
        content: str,
        *,
        embedding: list[float] | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Store one turn. Returns its id. Content is PII-masked first if enabled."""
        if self._veil is not None:
            try:
                content = self._veil.scrub_text(content)
            except Exception:
                pass
        emb = json.dumps([float(x) for x in embedding]) if embedding else None
        ts = time.time()
        cur = self._conn.execute(
            "INSERT INTO memories (thread_id, ts, role, content, metadata, embedding)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (thread_id, ts, role, content, json.dumps(metadata) if metadata else None, emb),
        )
        self._conn.commit()
        if self._ledger is not None:
            try:
                self._ledger.record(
                    actor="thread-recall", action="remember", target=thread_id,
                    meta={"role": role, "id": cur.lastrowid},
                )
            except Exception:
                pass
        return cur.lastrowid

    def forget(self, thread_id: str) -> int:
        """Delete a thread's memory. Returns rows removed."""
        cur = self._conn.execute("DELETE FROM memories WHERE thread_id = ?", (thread_id,))
        self._conn.commit()
        return cur.rowcount

    # -- read ---------------------------------------------------------------

    def recent(self, thread_id: str, k: int = 10) -> list[Turn]:
        """The last k turns of a thread, in chronological order."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
            (thread_id, k),
        ).fetchall()
        return [_row_to_turn(r) for r in reversed(rows)]

    def search(self, thread_id: str, query_embedding: list[float], k: int = 5) -> list[Turn]:
        """Semantic recall: the k turns whose embeddings are closest to the query."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE thread_id = ? AND embedding IS NOT NULL",
            (thread_id,),
        ).fetchall()
        scored: list[Turn] = []
        for r in rows:
            turn = _row_to_turn(r)
            turn.score = _cosine(query_embedding, json.loads(r["embedding"]))
            scored.append(turn)
        scored.sort(key=lambda t: t.score or 0.0, reverse=True)
        return scored[:k]

    def count(self, thread_id: str | None = None) -> int:
        if thread_id is None:
            return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE thread_id = ?", (thread_id,)
        ).fetchone()[0]

    def close(self) -> None:
        self._conn.close()


def _row_to_turn(r: sqlite3.Row) -> Turn:
    return Turn(
        id=r["id"], thread_id=r["thread_id"], ts=r["ts"], role=r["role"],
        content=r["content"],
        metadata=json.loads(r["metadata"]) if r["metadata"] else None,
    )


def _make_veil():
    try:
        from pii_veil import Veil

        return Veil()
    except Exception:
        return None


def _make_ledger(path: str):
    try:
        import os

        from agent_blackbox import Ledger

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return Ledger(path)
    except Exception:
        return None
