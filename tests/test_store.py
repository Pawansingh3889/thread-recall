"""Tests for the governed agent-memory store (SQLite backend)."""
from thread_recall.store import Memory, _cosine


def test_remember_and_recent_chronological():
    mem = Memory(":memory:")
    mem.remember("t1", "user", "first")
    mem.remember("t1", "assistant", "second")
    mem.remember("t1", "user", "third")
    turns = mem.recent("t1")
    assert [t.content for t in turns] == ["first", "second", "third"]
    assert [t.role for t in turns] == ["user", "assistant", "user"]


def test_recent_respects_k_and_keeps_latest():
    mem = Memory(":memory:")
    for i in range(5):
        mem.remember("t1", "user", f"msg{i}")
    turns = mem.recent("t1", k=2)
    assert [t.content for t in turns] == ["msg3", "msg4"]


def test_threads_are_isolated():
    mem = Memory(":memory:")
    mem.remember("a", "user", "for a")
    mem.remember("b", "user", "for b")
    assert [t.content for t in mem.recent("a")] == ["for a"]
    assert mem.count("a") == 1
    assert mem.count() == 2


def test_semantic_search_returns_nearest():
    mem = Memory(":memory:")
    mem.remember("t1", "user", "revenue question", embedding=[1.0, 0.0, 0.0])
    mem.remember("t1", "assistant", "revenue answer", embedding=[0.9, 0.1, 0.0])
    mem.remember("t1", "user", "weather", embedding=[0.0, 0.0, 1.0])
    hits = mem.search("t1", [1.0, 0.0, 0.0], k=2)
    assert [h.content for h in hits] == ["revenue question", "revenue answer"]
    assert hits[0].score >= hits[1].score
    # the unrelated turn is not in the top-2
    assert "weather" not in [h.content for h in hits]


def test_search_ignores_turns_without_embeddings():
    mem = Memory(":memory:")
    mem.remember("t1", "user", "no embedding")
    mem.remember("t1", "user", "has embedding", embedding=[1.0, 0.0])
    hits = mem.search("t1", [1.0, 0.0], k=5)
    assert [h.content for h in hits] == ["has embedding"]


def test_forget_clears_thread():
    mem = Memory(":memory:")
    mem.remember("t1", "user", "x")
    mem.remember("t1", "user", "y")
    removed = mem.forget("t1")
    assert removed == 2
    assert mem.recent("t1") == []


def test_metadata_round_trips():
    mem = Memory(":memory:")
    mem.remember("t1", "user", "x", metadata={"tool": "get_metric", "rows": 3})
    t = mem.recent("t1")[0]
    assert t.metadata == {"tool": "get_metric", "rows": 3}


def test_mask_true_is_graceful_without_pii_veil():
    # pii-veil is optional; with it absent, mask=True must not crash and stores as-is.
    mem = Memory(":memory:", mask=True)
    mem.remember("t1", "user", "email me at a@b.com")
    assert mem.count("t1") == 1


def test_cosine_basics():
    assert _cosine([1, 0], [1, 0]) == 1.0
    assert _cosine([1, 0], [0, 1]) == 0.0
    assert _cosine([], [1]) == 0.0
