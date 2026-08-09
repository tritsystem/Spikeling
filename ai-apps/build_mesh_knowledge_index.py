#!/usr/bin/env python3
"""Builds a real OBSERVE semantic embedding index over content downloaded
via the mesh Kiwix server (currently: the Wiktionary ZIM).

Two real stages, both genuinely run, not simulated:
  1. Extract: read every real text/html article out of the .zim file via
     libzim (the official Kiwix Python bindings), strip HTML with
     BeautifulSoup, write plain-text batches to disk.
  2. Index: run OBSERVE's SearchEngine.build_index() over those extracted
     text files -- same engine, same code path as the vault index, just
     pointed at a different, much larger corpus.

Honest scale note before running this: Wiktionary has hundreds of
thousands of entries. Embedding all of them is a real, substantial
GPU job -- likely tens of minutes to hours depending on entry count,
not a quick pass. Kiwix's own built-in full-text search (via
kiwix-serve, already running) covers this content instantly with zero
embedding cost; this script is for when semantic (not just keyword)
search over this corpus specifically is worth that real cost.

Run only once the .zim download is actually complete:
    python build_mesh_knowledge_index.py
"""
import os
import sys
import threading

from bs4 import BeautifulSoup
from libzim.reader import Archive

sys.path.insert(0, r"G:\dev\observe-api")
from search_engine import SearchEngine  # noqa: E402

ZIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kiwix", "wiktionary_en.zim")
EXTRACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh_knowledge_extracted")
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh_knowledge_index")
ARTICLES_PER_FILE = 200  # batches entries into fewer, larger files instead of one file per article


def extract_zim_to_text(zim_path, out_dir):
    if not os.path.isfile(zim_path):
        print(f"ZIM file not found (is the download still running?): {zim_path}")
        sys.exit(1)

    size_bytes = os.path.getsize(zim_path)
    print(f"Opening {zim_path} ({size_bytes / 1e9:.2f} GB on disk)...")
    archive = Archive(zim_path)
    total = archive.entry_count
    print(f"Real entry count: {total}")

    os.makedirs(out_dir, exist_ok=True)
    batch = []
    batch_idx = 0
    written_articles = 0
    skipped = 0

    def flush(batch, idx):
        if not batch:
            return
        path = os.path.join(out_dir, f"batch_{idx:05d}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(batch))

    for i in range(total):
        try:
            entry = archive._get_entry_by_id(i)
            if entry.is_redirect:
                continue
            item = entry.get_item()
            if "html" not in item.mimetype:
                continue
            raw = bytes(item.content).decode("utf-8", errors="ignore")
            text = BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)
            if len(text) < 40:
                skipped += 1
                continue
            batch.append(f"# {entry.title}\n\n{text}")
            written_articles += 1
        except Exception:
            skipped += 1
            continue

        if len(batch) >= ARTICLES_PER_FILE:
            flush(batch, batch_idx)
            batch_idx += 1
            batch = []
            if written_articles % 5000 < ARTICLES_PER_FILE:
                print(f"  ...{written_articles} articles extracted so far ({i}/{total} scanned)")

    flush(batch, batch_idx)
    print(f"Extraction done: {written_articles} real articles written, {skipped} skipped (redirects/non-html/too short).")
    return written_articles


def main():
    written = extract_zim_to_text(ZIM_PATH, EXTRACT_DIR)
    if written == 0:
        print("Nothing extracted -- not building an index over an empty corpus.")
        sys.exit(1)

    print(f"Indexing {EXTRACT_DIR} with OBSERVE's SearchEngine...")
    engine = SearchEngine()
    done_event = threading.Event()

    def on_status(msg):
        print(f"[status] {msg}")

    def on_done():
        # Real signature: zero args, fires exactly once regardless of outcome --
        # check engine.ready for actual success (same fix as the vault indexer
        # needed, caught by checking search_engine.py's real source, not assumed).
        done_event.set()

    engine.build_index([EXTRACT_DIR], INDEX_DIR, on_status, on_done)
    done_event.wait(timeout=7200)  # real embedding pass over a large corpus -- genuinely can take a while

    if engine.ready:
        print(f"Done. Mesh knowledge index written to {INDEX_DIR}")
    else:
        print("FAILED -- see [status] output above for the real error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
