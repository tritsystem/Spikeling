#!/usr/bin/env python
"""
electronics_assistant.py — real infrastructure for an electronics
salvage/inventor assistant: capture a real photo (reusing capture_frame.
py's proven C922 pipeline), then log identification/research/next-steps
into the SAME Obsidian vault convention voice_commands.py's log_to_vault()
already uses (vault/Project Work, dated filename, frontmatter, embedded
image via ![[name]]) -- not reimplemented from scratch, matched exactly
so these notes sit alongside every other project's real work log.

WHAT THIS FILE DOES NOT DO: identify or research parts itself. There is
no local vision/classifier model here on purpose -- the scrapyard-sensor-
project's own README already states the ceiling explicitly: an earlier
attempt to compress a vision classifier to run on-device (ternary-
quantized MobileNetV3) failed hard, twice (78.8%->4.9% accuracy). That
failure was specific to squeezing a model down for constrained on-device
inference. This is a different mechanism entirely: Claude looks at the
real captured photo directly (full model, can web-search a real part
number) and reasons about it in the conversation -- the IDENTIFICATION
and RESEARCH are done by Claude in the loop, this module just handles
the real capture + the real, correctly-formatted vault write.

    from electronics_assistant import capture_part_photo, log_electronics_note
"""
import datetime
import os
import re
import shutil
import sys

import cv2

VAULT_DIR = os.environ.get(
    "VOICE_VAULT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vault"))
VAULT_PROJECT_WORK_DIR = os.path.join(VAULT_DIR, "Project Work")
VAULT_ATTACH_DIR = os.path.join(VAULT_DIR, "attachments")

DEVICE_INDEX = 0


def capture_part_photo(label: str, out_dir: str = "electronics_captures") -> str:
    """Real webcam capture (same C922 pipeline as capture_frame.py),
    saved under a labeled, timestamped filename so multiple photos in one
    salvage/build session don't overwrite each other."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40] or "part"
    path = os.path.join(out_dir, f"{stamp}_{slug}.jpg")

    cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at device index {DEVICE_INDEX}.")
    for _ in range(3):   # warm-up reads, same as capture_frame.py
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Webcam opened but frame read failed.")
    cv2.imwrite(path, frame)
    return path


def log_electronics_note(title: str, photo_paths: list, body_markdown: str,
                          status: str = "logged", kind: str = "electronics-salvage") -> str:
    """Writes one real markdown note into vault/Project Work, matching
    voice_commands.py's log_to_vault() convention exactly (frontmatter,
    dated filename, embedded images via ![[name]]) so this sits alongside
    every other project's real work log, not a separate parallel system.
    `body_markdown` is whatever Claude actually determined this session
    (identification, research findings, next steps) -- this function only
    handles real file I/O, never invents content.

    Best-effort like the original: a logging failure should never crash
    a real salvage/build session in progress."""
    try:
        os.makedirs(VAULT_PROJECT_WORK_DIR, exist_ok=True)
        os.makedirs(VAULT_ATTACH_DIR, exist_ok=True)
        ts = datetime.datetime.now()
        stamp = ts.strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "untitled"
        note_path = os.path.join(VAULT_PROJECT_WORK_DIR, f"{stamp}_{slug}.md")

        lines = [
            "---",
            f"date: {ts.isoformat()}",
            f"kind: {kind}",
            f"status: {status}",
            "---",
            "",
            f"# {title}",
            "",
        ]
        for photo_path in photo_paths:
            if photo_path and os.path.exists(photo_path):
                att_name = f"{stamp}_{os.path.basename(photo_path)}"
                att_dest = os.path.join(VAULT_ATTACH_DIR, att_name)
                shutil.copy2(photo_path, att_dest)
                lines += [f"![[{att_name}]]", ""]
        lines += [body_markdown.strip(), ""]

        with open(note_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return note_path
    except Exception as e:
        print(f"(vault logging failed, non-fatal: {e})", flush=True)
        return ""


def capture_step_photo(session_label: str, step_num: int, out_dir: str = "electronics_captures") -> str:
    """Same real C922 capture as capture_part_photo(), but named by
    session + step number so a multi-step dismantling/construction
    session's photos stay in clear sequential order instead of colliding
    or being ordered only by timestamp."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", session_label.lower()).strip("-")[:40] or "session"
    path = os.path.join(out_dir, f"{stamp}_{slug}_step{step_num:02d}.jpg")

    cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at device index {DEVICE_INDEX}.")
    for _ in range(3):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Webcam opened but frame read failed.")
    cv2.imwrite(path, frame)
    return path


def log_dismantling_session(title: str, steps: list, status: str = "in-progress",
                             kind: str = "electronics-dismantling") -> str:
    """Writes ONE consolidated vault note for a whole dismantling/
    construction session -- `steps` is a list of {"photo": path,
    "guidance": text} dicts, one per real step Claude actually looked at
    and gave guidance for during the session. Each step gets its own
    "## Step N" heading with the photo embedded above the guidance text,
    so the note reads as a real, browsable step-by-step record in
    Obsidian, not a single static snapshot.

    status is meant to be updated across calls for the SAME session
    (e.g. "in-progress" while working, "complete" on the final call) --
    each call writes a NEW timestamped note rather than editing a prior
    one, matching every other note in this vault (a session's full
    history stays as separate real entries, not silently overwritten)."""
    try:
        os.makedirs(VAULT_PROJECT_WORK_DIR, exist_ok=True)
        os.makedirs(VAULT_ATTACH_DIR, exist_ok=True)
        ts = datetime.datetime.now()
        stamp = ts.strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "untitled"
        note_path = os.path.join(VAULT_PROJECT_WORK_DIR, f"{stamp}_{slug}.md")

        lines = [
            "---",
            f"date: {ts.isoformat()}",
            f"kind: {kind}",
            f"status: {status}",
            f"steps: {len(steps)}",
            "---",
            "",
            f"# {title}",
            "",
        ]
        for i, step in enumerate(steps, start=1):
            photo_path = step.get("photo")
            guidance = step.get("guidance", "")
            lines.append(f"## Step {i}")
            lines.append("")
            if photo_path and os.path.exists(photo_path):
                att_name = f"{stamp}_{os.path.basename(photo_path)}"
                att_dest = os.path.join(VAULT_ATTACH_DIR, att_name)
                shutil.copy2(photo_path, att_dest)
                lines += [f"![[{att_name}]]", ""]
            lines += [guidance.strip(), ""]

        with open(note_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return note_path
    except Exception as e:
        print(f"(vault logging failed, non-fatal: {e})", flush=True)
        return ""


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "test_part"
    p = capture_part_photo(label)
    print(f"captured: {p}")
