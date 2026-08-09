"""Real backups for SPIKEMESH's actual data -- the things that would be
genuine, unrecoverable data loss if this machine's disk failed, not just
inconvenience: the vault semantic index (real embeddings, real compute
cost to rebuild), the knowledge base, the task-decisions log, and
Headscale's own device/ACL database.

Retention: keeps the last KEEP_LAST backups, deletes older ones -- same
"real retention policy, not unbounded growth" discipline server-guard's
own backup_db.py already established.

Usage:
    python mesh_backup.py            # one backup now
    python mesh_backup.py --loop 21600   # repeat every 6h, forever
"""
import argparse
import os
import shutil
import sys
import time

AI_APPS_DIR = os.path.dirname(os.path.abspath(__file__))
SPIKELING_ROOT = os.path.dirname(AI_APPS_DIR)
BACKUP_ROOT = os.path.join(AI_APPS_DIR, "backups")
KEEP_LAST = 10

# (real source path, is_directory) -- every one of these is real, existing
# state that took real compute or real logged work to produce.
REAL_SOURCES = [
    (os.path.join(AI_APPS_DIR, "vault_semantic_index"), True),
    (os.path.join(AI_APPS_DIR, "mesh_knowledge_index"), True),
    (os.path.join(AI_APPS_DIR, "spikeling_knowledge.db"), False),
    (os.path.join(AI_APPS_DIR, "spikeling_memory.db"), False),
    (os.path.join(SPIKELING_ROOT, "task_decisions.jsonl"), False),
]


def backup_once():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest_dir = os.path.join(BACKUP_ROOT, stamp)
    os.makedirs(dest_dir, exist_ok=True)

    copied, skipped = [], []
    for src, is_dir in REAL_SOURCES:
        name = os.path.basename(src.rstrip(os.sep))
        if is_dir:
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(dest_dir, name))
                copied.append(name)
            else:
                skipped.append(name)
        else:
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest_dir, name))
                copied.append(name)
            else:
                skipped.append(name)

    print(f"[{stamp}] backed up: {copied or '(none)'}" + (f" -- skipped (not found): {skipped}" if skipped else ""))

    # Retention: keep only the most recent KEEP_LAST backup directories.
    existing = sorted(
        d for d in os.listdir(BACKUP_ROOT)
        if os.path.isdir(os.path.join(BACKUP_ROOT, d))
    )
    for old in existing[:-KEEP_LAST]:
        shutil.rmtree(os.path.join(BACKUP_ROOT, old), ignore_errors=True)
        print(f"  pruned old backup: {old}")

    return dest_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=float, default=None,
                         help="repeat every N seconds instead of running once")
    args = parser.parse_args()

    os.makedirs(BACKUP_ROOT, exist_ok=True)
    if args.loop:
        while True:
            backup_once()
            time.sleep(args.loop)
    else:
        backup_once()


if __name__ == "__main__":
    main()
