#!/usr/bin/env python
"""
kernel_milestone_pipeline.py -- dispatches ONE spikeling-os kernel milestone
through real local `claude` CLI subprocesses (voice_commands.do_claude_code),
so the actual implementation/debugging work happens OUTSIDE any interactive
Claude Code session -- cheap, disposable local processes instead of eating
an expensive conversation's own context/token budget.

WHY THIS EXISTS, NOT THE GENERIC spiking_orchestrator.py SpikingPipeline:
SpikingPipeline's SNN routing exists to skip specialists a TRIVIAL task
doesn't need (a rename doesn't need PreRegister/TestWriter -- see that
file's own demo). Every spikeling-os milestone is real, non-trivial systems
work: Implementer and Reviewer both fire every time, so there's no routing
decision to make. More importantly, as of this writing SpikingPipeline's
fixed Implementer/Reviewer neurons always use CLAUDE_CODE_TOOLS/REVIEW_TOOLS
(NO Bash) regardless of project -- verified by reading _run_real_agent()
directly, not assumed. That's fatal for kernel work: this project's own
vault has two lessons that say so directly --
  - Lessons/research-pipeline-execution.md: "an agent that cannot execute
    can only propose, never measure" -- Bash is required to cargo build.
  - Lessons/process-f5-gate.md: peer review that only READS files missed
    two real regressions that only broke at runtime -- "not done until a
    human F5s it." For a kernel, the equivalent gate is a real QEMU boot,
    and unlike Godot's F5 that CAN be automated into the reviewer's own
    step (RESEARCH_REVIEW_TOOLS gives it Bash), which is exactly what this
    script does.

HARD RULES, stated explicitly to every agent this script calls (the shared
preamble's own "never commit" rule is written for chat-triggered single-file
edits, not repeated here by default, so this script restates it for every
call rather than assuming it carries over):
  - NEVER `git commit`, never `git push`. Leave the diff in the working
    tree. This script prints the diff/log locations; a human (or the
    orchestrating Claude Code session, reading the real diff and the real
    QEMU boot log itself, same discipline as every milestone 1-44 already
    committed) decides whether to commit.
  - Implementer gets RESEARCH_CODE_TOOLS (Bash + edit) -- it must actually
    run `cargo build` and boot the kernel in QEMU (capturing real serial
    output to a file) before claiming anything works.
  - Reviewer gets RESEARCH_REVIEW_TOOLS (Bash, NO edit) -- it independently
    re-runs `cargo build` + its own fresh QEMU boot and compares against
    the Implementer's claimed evidence. It cannot edit code to make its own
    check pass. A claim it cannot reproduce is a real finding, not noise.
  - Corrector (RESEARCH_CODE_TOOLS again) only fires if the Reviewer's own
    output reports a real problem -- same result-driven-not-scheduled
    pattern spiking_orchestrator.py already uses for its own Corrector.

Each stage's FULL transcript-relevant output also gets written to
milestones/<N>_<stage>.log in the target repo, since do_claude_code()'s
returned string is deliberately short (matches the shared preamble's
"2-4 sentences" summary convention) -- the log files are the real
evidence a human/orchestrating session should actually read, not the
short spoken-style summary alone.

Usage:
    python kernel_milestone_pipeline.py 45 "one-paragraph scope description"
    python kernel_milestone_pipeline.py 45 "..." --dry-run   # no API calls, prints the framed tasks
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voice_commands as vc  # noqa: E402

# BUG FIX (found live, 2026-08-14): do_claude_code()'s progress relay
# (_relay_claude_event) calls vc.say() for every real tool call the agent
# makes. vc._suppress_tts only gets set True inside process_text() for the
# discord/webui input paths -- this script calls do_claude_code() directly,
# bypassing that, so it defaults to vc._suppress_tts = False and every tool
# call gets spoken aloud through the real pyttsx3/SAPI voice on this
# machine. Confirmed the hard way: a live milestone run had to be killed
# mid-Implementer-stage because it was audibly narrating every Read/Bash
# call. Suppress it explicitly, once, at import time -- this script has no
# legitimate reason to ever want audio.
vc._suppress_tts = True

# Parallel-dispatch override: kernel_milestone_snn_dispatcher.py launches
# this script once per git-worktree-isolated milestone track, each needing
# its OWN target directory (not the shared main checkout every other
# invocation of this script uses) -- an env var rather than a CLI flag so
# ordinary single-milestone usage (`python kernel_milestone_pipeline.py 45
# "..."`) stays completely unchanged.
REPO_DIR = os.environ.get("SPIKELING_OS_OVERRIDE_DIR") or vc.PROJECTS.get("spikeling_os")
if not REPO_DIR or not os.path.isdir(REPO_DIR):
    print(f"spikeling_os target dir not found ({REPO_DIR!r}) -- check "
          f"voice_commands.PROJECTS or SPIKELING_OS_OVERRIDE_DIR.", file=sys.stderr)
    sys.exit(1)

MILESTONE_LOG_DIR = os.path.join(REPO_DIR, "milestones")

# The exact discipline every real milestone in this repo's own README
# already follows -- restated here so an agent that has never seen the
# README (or skims it) still gets the load-bearing rules directly in its
# own task text, not just implied by "match the existing style."
KERNEL_DISCIPLINE = """
--- SPIKELING-OS MILESTONE DISCIPLINE (read this fully before starting) ---
This is a from-scratch x86_64 kernel (Rust, #![no_std]). Read README.md's
"Status" section first -- especially the last 2-3 milestones -- to match
this project's own established real discipline:

1. Real verification, not assumed. "It compiles" is not "it works." Boot
   the kernel in QEMU (`cargo run -- bios`, piping stdout to a real log
   file) and read the REAL serial output. A self-test that prints PASS/FAIL
   is the strongest evidence -- if this milestone doesn't have one yet, add
   a boot-time, non-interactive self-test for it (see process::self_test_*
   functions for the existing pattern), because interactive shell-command
   testing via QEMU's sendkey has been repeatedly unreliable in this
   environment (see the README's own milestone 39/40 notes on this).
2. When something doesn't work, find the REAL root cause via real evidence
   (a QEMU `-d int` trace via the SPIKELING_QEMU_TRACE=1 env var this
   runner already supports, a serial log, an actual error message) --
   never guess-and-check multiple unverified hypotheses and claim the one
   that happened to compile is the fix. If you eliminate a hypothesis,
   say so plainly (this repo's own README does this constantly -- "ruled
   out X, evidence: Y").
3. Honest scope-cuts over silent gaps. If full POSIX semantics for
   something aren't achievable in this milestone, implement a real,
   working subset and say exactly what's deferred and why (again: read
   how e.g. Milestone 40's pipe() or Milestone 42's setpgid() document
   their own disclosed scope-cuts -- match that exact tone and honesty).
4. NEVER claim something is verified from reading the code alone. Every
   "verified" claim in your final summary must correspond to a real
   command you ran and a real output you saw THIS session.
5. Do NOT git commit or git push. Leave every change in the working tree.
   Do not touch git state at all beyond `git diff`/`git log` for your own
   reference.
6. Write your FULL technical account (not just a short summary) to the
   exact file path given to you in this task -- include the real commands
   you ran and their real output/log excerpts. This file is what actually
   gets reviewed; your returned chat summary is secondary.
"""

IMPLEMENTER_FRAME = """You are implementing Milestone {n} of the spikeling-os
kernel: {scope}

{discipline}
Write your full technical account to: {report_path}
Follow the exact structure the README's own "Status" checklist entries use
(what you built, what you verified and how, any real bugs found+fixed, any
honest scope-cuts) -- your account will likely become the actual new README
checklist entry for this milestone if the reviewer confirms it, so write it
in that voice directly.

If, after real effort, you cannot get a genuinely working, verified result
for this milestone's full scope, that is a valid honest outcome -- write
exactly what you tried, what real evidence you gathered, and what's still
broken/open, in {report_path}, the same way this repo's own README
documents an unresolved bug (see the pre-fix Milestone 41 entry as a
model) rather than claiming success it doesn't have.
"""

REVIEWER_FRAME = """You are independently verifying spikeling-os Milestone {n}:
{scope}

The implementer's account is here -- READ IT FIRST: {report_path}

{discipline}
Your job is specifically to CHECK, not re-implement: re-run `cargo build`
yourself, fresh, and boot the kernel in QEMU yourself, fresh (do not reuse
any log file the implementer already produced -- generate your own). Compare
what you actually observe against every concrete claim in the implementer's
account (build succeeds, specific self-test PASS lines, specific serial-log
text). You have Bash but NOT Edit/Write -- you cannot fix anything you find,
only report it precisely.

Write your verification account to: {verify_path}
State explicitly, for each claim you checked: REPRODUCED or NOT REPRODUCED
(with the real command + real output that shows which), and an overall
verdict: VERIFIED, VERIFIED WITH ISSUES, or NOT VERIFIED. If NOT REPRODUCED
on anything, that is the single most important line in your report -- do
not bury it.
"""

CORRECTOR_FRAME = """The reviewer found real issues with spikeling-os
Milestone {n} ({scope}) that need fixing. Reviewer's account: {verify_path}
Implementer's original account: {report_path}

{discipline}
Fix exactly what the reviewer flagged. Do not touch anything else. After
fixing, re-run the same real verification the reviewer did (fresh cargo
build + fresh QEMU boot) and confirm the specific issues are actually
resolved -- append a "## Correction" section to {report_path} with what you
changed and the real evidence it now works.
"""


def _report_paths(n: int):
    os.makedirs(MILESTONE_LOG_DIR, exist_ok=True)
    return (
        os.path.join(MILESTONE_LOG_DIR, f"m{n}_implement.md"),
        os.path.join(MILESTONE_LOG_DIR, f"m{n}_review.md"),
    )


def _log_vault(n: int, scope: str, stage: str, output: str) -> None:
    try:
        notes_dir = os.path.join(vc.VAULT_DIR, "Project Work")
        vc.log_to_vault(
            "agent_task",
            f"[spikeling_os milestone {n}] {scope} ({stage})",
            output=output,
            status=f"kernel_milestone_pipeline:{stage}",
            notes_dir=notes_dir,
        )
    except Exception as e:
        print(f"vault logging failed (non-fatal): {e}", flush=True)


def run_milestone(n: int, scope: str, dry_run: bool = True, timeout: int = 1800) -> dict:
    report_path, verify_path = _report_paths(n)
    result = {"milestone": n, "scope": scope, "stages": []}

    implementer_task = IMPLEMENTER_FRAME.format(
        n=n, scope=scope, discipline=KERNEL_DISCIPLINE, report_path=report_path,
    )
    reviewer_task = REVIEWER_FRAME.format(
        n=n, scope=scope, discipline=KERNEL_DISCIPLINE,
        report_path=report_path, verify_path=verify_path,
    )

    if dry_run:
        print("=" * 70)
        print(f"DRY RUN -- milestone {n}: {scope}")
        print("=" * 70)
        print("\n--- IMPLEMENTER TASK ---\n" + implementer_task)
        print("\n--- REVIEWER TASK (would run after) ---\n" + reviewer_task)
        result["dry_run"] = True
        return result

    print(f"[milestone {n}] stage 1/2: Implementer (RESEARCH_CODE_TOOLS, timeout={timeout}s)...", flush=True)
    t0 = time.time()
    impl_out = vc.do_claude_code(
        task=implementer_task, project_dir=REPO_DIR,
        tools=vc.RESEARCH_CODE_TOOLS, timeout=timeout,
    )
    result["stages"].append({"stage": "implement", "seconds": time.time() - t0, "summary": impl_out})
    _log_vault(n, scope, "implement", impl_out or "(no summary returned)")
    print(f"[milestone {n}] Implementer summary: {impl_out}\n", flush=True)

    print(f"[milestone {n}] stage 2/2: Reviewer (RESEARCH_REVIEW_TOOLS, timeout={timeout}s)...", flush=True)
    t0 = time.time()
    review_out = vc.do_claude_code(
        task=reviewer_task, project_dir=REPO_DIR,
        tools=vc.RESEARCH_REVIEW_TOOLS, timeout=timeout,
    )
    result["stages"].append({"stage": "review", "seconds": time.time() - t0, "summary": review_out})
    _log_vault(n, scope, "review", review_out or "(no summary returned)")
    print(f"[milestone {n}] Reviewer summary: {review_out}\n", flush=True)

    needs_correction = bool(review_out) and (
        "NOT REPRODUCED" in review_out.upper() or "NOT VERIFIED" in review_out.upper()
    )
    result["needs_correction"] = needs_correction

    if needs_correction:
        print(f"[milestone {n}] Reviewer flagged real issues -- running Corrector...", flush=True)
        corrector_task = CORRECTOR_FRAME.format(
            n=n, scope=scope, discipline=KERNEL_DISCIPLINE,
            report_path=report_path, verify_path=verify_path,
        )
        t0 = time.time()
        corr_out = vc.do_claude_code(
            task=corrector_task, project_dir=REPO_DIR,
            tools=vc.RESEARCH_CODE_TOOLS, timeout=timeout,
        )
        result["stages"].append({"stage": "correct", "seconds": time.time() - t0, "summary": corr_out})
        _log_vault(n, scope, "correct", corr_out or "(no summary returned)")
        print(f"[milestone {n}] Corrector summary: {corr_out}\n", flush=True)

    result["report_path"] = report_path
    result["verify_path"] = verify_path
    print(f"[milestone {n}] done. Read {report_path} and {verify_path} before trusting/committing.")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("milestone", type=int)
    ap.add_argument("scope", help="one-paragraph scope description for this milestone")
    ap.add_argument("--real", action="store_true", help="actually call Claude (spends tokens); default is --dry-run")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    run_milestone(args.milestone, args.scope, dry_run=not args.real, timeout=args.timeout)


if __name__ == "__main__":
    main()
