"""thread-recall: governed, on-prem conversation memory for AI agents.

Per-thread history (recency) and semantic recall (nearest-neighbour over
embeddings), with optional PII masking on write and tamper-evident audit.
Part of the Governed Agent Stack.
"""
from thread_recall.store import Memory, Turn

__version__ = "0.1.0"
__all__ = ["Memory", "Turn", "__version__"]
