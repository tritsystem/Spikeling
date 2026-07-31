#!/usr/bin/env python
"""
project_assistant.py — general-purpose project-context tools, domain-
agnostic (electronics, automotive, tribe, scrapyard-sensor-project, or
anything else): search past vault notes, read an existing project
directory's real README/top-level structure, and import structured data
(CSV/JSON) someone already compiled elsewhere.

INDEPENDENT of electronics_assistant.py / automotive_obd.py -- zero
shared code, same "each capability works standalone" design used
throughout tonight's MCP additions. All three functions here are
read-only -- nothing writes, deletes, or executes anything.
"""
import csv
import json
import os
from pathlib import Path

VAULT_DIR = os.environ.get(
    "VOICE_VAULT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vault"))

_SKIP_ENTRIES = {".git", "node_modules", "__pycache__", ".venv", "venv", ".obsidian"}


def search_vault_notes(query: str, limit: int = 10) -> list:
    """Real text search over vault/Project Work + vault/Research's actual
    markdown files (filename OR content match, case-insensitive) -- the
    vault equivalent of the memory folder's own "research find" tool, so
    a new session on a topic can find what's already been logged instead
    of starting cold. Returns [{"path", "matched_on", "snippet", "mtime"}],
    most-recently-modified first."""
    search_dirs = [
        os.path.join(VAULT_DIR, "Project Work"),
        os.path.join(VAULT_DIR, "Research"),
    ]
    query_lower = query.lower()
    results = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(d, fname)
            matched_on = "filename" if query_lower in fname.lower() else None
            snippet = ""
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except OSError:
                continue
            if query_lower in content.lower():
                matched_on = matched_on or "content"
                idx = content.lower().find(query_lower)
                snippet = content[max(0, idx - 80):idx + 120].replace("\n", " ")
            if matched_on:
                results.append({
                    "path": fpath,
                    "matched_on": matched_on,
                    "snippet": snippet,
                    "mtime": os.path.getmtime(fpath),
                })
    results.sort(key=lambda r: -r["mtime"])
    return results[:limit]


def read_project_directory(path: str, max_entries: int = 30) -> dict:
    """Real, read-only snapshot of an existing project directory: README
    content (if present) and the TOP-LEVEL entries only (files + immediate
    subdirectory names, not a full recursive walk -- keeps this fast and
    bounded regardless of how large the actual project is, e.g. never
    descends into node_modules/.git even if not explicitly skipped by
    name). Raises if the path doesn't exist or isn't a directory -- never
    fabricates a summary of something that isn't there."""
    p = Path(path)
    if not p.is_dir():
        raise ValueError(f"'{path}' is not a real, existing directory.")

    readme_content = None
    for candidate in ("README.md", "readme.md", "Readme.md"):
        rp = p / candidate
        if rp.is_file():
            readme_content = rp.read_text(encoding="utf-8", errors="ignore")[:5000]
            break

    entries = []
    for entry in p.iterdir():
        if entry.name.startswith(".") or entry.name in _SKIP_ENTRIES:
            continue
        is_dir = entry.is_dir()
        entries.append({
            "name": entry.name + ("/" if is_dir else ""),
            "is_dir": is_dir,
            "mtime": entry.stat().st_mtime,
        })
    entries.sort(key=lambda e: -e["mtime"])

    return {"path": str(p), "readme": readme_content, "entries": entries[:max_entries]}


def import_structured_data(path: str) -> dict:
    """Real CSV/JSON import -- reads an existing parts list/spec file
    someone already compiled elsewhere. Returns {"format": "csv"|"json",
    "data": ...}. Raises on an unsupported extension or a real parse
    error rather than guessing at malformed data."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"'{path}' is not a real, existing file.")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        with open(p, "r", encoding="utf-8", errors="ignore", newline="") as f:
            rows = list(csv.DictReader(f))
        return {"format": "csv", "data": rows}
    elif suffix == ".json":
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        return {"format": "json", "data": data}
    else:
        raise ValueError(f"Unsupported file type '{suffix}' -- only .csv and .json are handled.")
