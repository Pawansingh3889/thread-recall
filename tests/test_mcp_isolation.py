"""Cross-actor isolation for the MCP tools.

Proves a raw thread_id is no longer a key into another principal's memory:
one actor cannot read, recall, or wipe another's thread, even with the same
thread_id string. The real tool functions run against an in-memory store.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import thread_recall.mcp_server as srv
from thread_recall.store import Memory


def _fresh_store(monkeypatch):
    mem = Memory(":memory:", mask=False)
    monkeypatch.setattr(srv, "_mem", mem)
    return mem


def test_recall_is_scoped_to_the_actor(monkeypatch):
    _fresh_store(monkeypatch)
    srv.remember("t1", "alice private note", role="user", actor="alice")

    # Same thread_id, different actor -> sees nothing.
    bob = srv.recall("t1", "note", k=5, actor="bob")
    assert bob["count"] == 0

    alice = srv.recall("t1", "note", k=5, actor="alice")
    assert alice["count"] == 1
    # Caller sees its own plain thread_id, not the internal namespaced key.
    assert alice["results"][0]["thread_id"] == "t1"
    assert "alice private note" in alice["results"][0]["content"]


def test_forget_cannot_wipe_another_actors_thread(monkeypatch):
    _fresh_store(monkeypatch)
    srv.remember("shared-name", "keep me", role="user", actor="alice")

    # Bob tries to wipe the same thread_id -> removes nothing of Alice's.
    assert srv.forget("shared-name", actor="bob")["removed"] == 0
    assert srv.recent("shared-name", k=5, actor="alice")["count"] == 1

    # Alice can wipe her own.
    assert srv.forget("shared-name", actor="alice")["removed"] == 1
    assert srv.recent("shared-name", k=5, actor="alice")["count"] == 0


def test_default_actor_groups_unscoped_callers(monkeypatch):
    _fresh_store(monkeypatch)
    # No actor -> the "shared" namespace; still round-trips for standalone use.
    srv.remember("t9", "standalone turn", role="user")
    got = srv.recent("t9", k=5)
    assert got["count"] == 1
    assert got["results"][0]["thread_id"] == "t9"
