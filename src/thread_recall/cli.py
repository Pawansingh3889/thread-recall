"""Command line: a quick zero-config demo of governed agent memory."""
from __future__ import annotations

import argparse

from thread_recall import __version__
from thread_recall.store import Memory


def cmd_demo(_args) -> int:
    mem = Memory(":memory:")
    thread = "demo-conversation"

    # toy 3-dim embeddings just to show semantic recall without a model
    mem.remember(thread, "user", "How much MRR did the pro plan make?", embedding=[1, 0, 0])
    mem.remember(thread, "assistant", "Pro plan MRR was 297.", embedding=[0.9, 0.1, 0])
    mem.remember(thread, "user", "What's the weather like?", embedding=[0, 0, 1])

    print("thread-recall demo  (in-memory SQLite, no model)\n")
    print("Recent turns (chronological):")
    for t in mem.recent(thread):
        print(f"  [{t.role}] {t.content}")

    print("\nSemantic recall for a revenue-ish query [1,0,0] (top 2):")
    for t in mem.search(thread, [1, 0, 0], k=2):
        print(f"  {t.score:.2f}  [{t.role}] {t.content}")

    print("\nWith mask=True (pii-veil, if installed) content is scrubbed before")
    print("storage; with audit=True every write lands in an agent-blackbox ledger.")
    mem.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="thread-recall", description=__doc__)
    parser.add_argument("--version", action="version", version=f"thread-recall {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="run a zero-config in-memory demo")
    args = parser.parse_args()
    if args.cmd == "demo":
        raise SystemExit(cmd_demo(args))
    parser.print_help()


if __name__ == "__main__":
    main()
