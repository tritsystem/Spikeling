#!/usr/bin/env python3
"""Builds a real OBSERVE (012-trit-search) semantic embedding index over the
Obsidian vault -- repurposing SearchEngine (normally used for code search)
as general semantic database search over Project Work + Research + Lessons
notes.

Lessons was a real, found-by-testing gap: obsidian_memory.log_lesson()
writes real files there, but they were never in this index's directory
list, so MethodLM's RECALL tool (semantic search against this index) could
never surface a lesson -- only grep-based obsidian_memory.search_vault()
could see them. Added once that was actually observed, not assumed.

Run once (or whenever the vault has real new content worth re-indexing):
    python build_vault_semantic_index.py
"""
import os
import sys
import threading

sys.path.insert(0, r"G:\dev\observe-api")
from search_engine import SearchEngine  # noqa: E402

VAULT_DIRS = [
    r"C:\Users\gbran\OneDrive\Documents\Spikeling\vault\Project Work",
    r"C:\Users\gbran\OneDrive\Documents\Spikeling\vault\Research",
    r"C:\Users\gbran\OneDrive\Documents\Spikeling\vault\Lessons",
]
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vault_semantic_index")


def main():
    real_dirs = [d for d in VAULT_DIRS if os.path.isdir(d)]
    if not real_dirs:
        print("No real vault directories found -- checked:", VAULT_DIRS)
        sys.exit(1)

    print("Indexing real vault directories:", real_dirs)
    engine = SearchEngine()
    done_event = threading.Event()

    def on_status(msg):
        print(f"[status] {msg}")

    def on_done():
        # Real signature: on_done() always fires with zero args, exactly once,
        # whether _build succeeded or failed (see search_engine.py's own
        # finally block) -- actual success is reported via engine.ready.
        done_event.set()

    engine.build_index(real_dirs, INDEX_DIR, on_status, on_done)
    done_event.wait(timeout=1800)  # real embedding generation takes real minutes, not seconds

    if engine.ready:
        print(f"Done. Index written to {INDEX_DIR}")
    else:
        print("FAILED -- see [status] output above for the real error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
