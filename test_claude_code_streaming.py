#!/usr/bin/env python
"""
test_claude_code_streaming.py — do_claude_code()'s live progress relay and
--resume session continuity (2026-07-28 upgrade: "interact fully with claude
remotely" via the Discord bot).

Before this: do_claude_code() ran a single blocking subprocess.run() and
stayed completely silent for the whole task (up to 300-900s), then dumped one
final summary -- "fire a task and wait blind," not real interaction. This
verifies the fix actually works end to end against the REAL claude CLI (not
mocked): tool-use events post to the response sink AS THEY HAPPEN, and a
follow-up call with resume=True genuinely remembers the prior turn instead of
starting blind.

    python test_claude_code_streaming.py

Costs real API usage (a handful of trivial one-line exchanges) -- run
deliberately, not in a tight loop.
"""
import time
import voice_commands as vc

_pass = 0
_fail = 0


def check(label, ok):
    global _pass, _fail
    if ok:
        _pass += 1
        print(f"  ok    {label}")
    else:
        _fail += 1
        print(f"  FAIL  {label}")


def main():
    posted = []

    def fake_sink(text):
        posted.append(text)

    vc.set_response_sink(fake_sink)
    vc._suppress_tts = True   # don't actually speak during this test

    print("=" * 60)
    print("  CLAUDE CODE STREAMING -- live relay + session continuity")
    print("=" * 60)

    print("\n-- Call 1: fresh task --")
    posted.clear()
    summary1 = vc.do_claude_code(
        task="Read the file discord_bot.py and reply with exactly: read done",
        project_dir=r"C:\Users\gbran\OneDrive\Documents\Spikeling",
        tools=["Read"], timeout=60)
    check("a tool_use event was relayed live (not just at the end)",
          any("Read" in p for p in posted))
    check("the final summary reflects the actual completed task",
          bool(summary1) and "read done" in summary1.lower())

    print("\n-- Call 2: resume=True, same project_dir --")
    posted.clear()
    summary2 = vc.do_claude_code(
        task="What file did you just read in the previous message? Reply with exactly its filename, nothing else.",
        project_dir=r"C:\Users\gbran\OneDrive\Documents\Spikeling",
        tools=["Read"], timeout=60, resume=True)
    check("a resumed call genuinely remembers the prior turn's context",
          bool(summary2) and "discord_bot.py" in summary2)

    print("\n-- Call 3: resume=True but a DIFFERENT project_dir --")
    vc.do_claude_code(
        task="Reply with exactly: fresh check",
        project_dir=r"C:\Users\gbran\OneDrive\Documents\horde-beta-version-1",
        tools=["Read"], timeout=60, resume=True)
    check("switching project_dir tracks the NEW project, not a stale resume target",
          vc._last_claude_session["project_dir"] == r"C:\Users\gbran\OneDrive\Documents\horde-beta-version-1")

    print("\n" + "-" * 42)
    print(f"  {_pass} passed, {_fail} FAILED")
    print("-" * 42 + "\n")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
