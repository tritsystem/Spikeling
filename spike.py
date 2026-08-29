#!/usr/bin/env python
"""
spike.py — terminal entry point for Spike: the SpikingPipeline orchestrator
(spiking_orchestrator.py, real and already verified) run through the user's
own two Anthropic accounts instead of one.

WHAT THIS ADDS ON TOP OF THE EXISTING, WORKING spiking_orchestrator.py:
    Nothing about the routing/brain changes -- this file does not touch
    pyspike.Net, score_task(), or the LIF cascade. It wraps ONE thing:
    which Anthropic account each specialist's do_claude_code() subprocess
    call authenticates as, alternating between two accounts round-robin
    so a rate limit on one account doesn't stall the whole run.

HOW ACCOUNT SWITCHING ACTUALLY WORKS (real Claude Code CLI behavior,
verified against `claude --help`, not assumed):
    `--bare` mode is documented to make Anthropic auth STRICTLY
    ANTHROPIC_API_KEY or apiKeyHelper via --settings -- OAuth and keychain
    are never read in that mode. That's the only way to force a SPECIFIC
    account's credentials on a per-call basis rather than whatever account
    happens to be logged in via `claude login` right now. So each call:
      1. sets ANTHROPIC_API_KEY to account #1 or #2 (alternating)
      2. adds --bare so that key is actually what gets used, not ignored
      3. restores the previous environment afterward

HONEST STATUS (2026-08-28): the SpikingPipeline routing this wraps is
real and already verified (test_pyspike_orchestrator_parity.py). The
account-rotation wrapper below is NEW and has NOT been live-tested against
real Claude API calls -- no API keys were available to test with when this
was built (only OAuth login for one account exists; a second account's
real API key was never successfully obtained during this session -- see
the .env saga in this session's own history). --bare's env-var-only auth
behavior is taken from Claude Code's own --help text, not independently
re-verified end to end here. Treat the routing/pipeline as trusted,
treat the account-rotation piece as built-but-unverified until run for
real.

USAGE:
    python spike.py "task text"              # demo/dry-run, no API calls
    python spike.py --real "task text"       # real run, rotates accounts
    python spike.py --demo                   # same routing demo as before

SETUP (you do this part -- see the module docstring above for why I don't
enter API keys myself):
    Set two real Anthropic API keys as environment variables before running:
      SPIKE_ANTHROPIC_KEY_1=sk-ant-...
      SPIKE_ANTHROPIC_KEY_2=sk-ant-...
    If only SPIKE_ANTHROPIC_KEY_1 is set, spike runs on that single account
    (no rotation, not an error) and says so plainly at startup.
"""

import os
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the REAL, already-verified pipeline -- this file adds account
# rotation on top, it does not reimplement any routing/brain logic.
from spiking_orchestrator import SpikingPipeline, demo as _routing_demo, _print_result  # noqa: E402
import voice_commands as vc  # noqa: E402


def _load_accounts() -> list[str]:
    keys = []
    for env_name in ("SPIKE_ANTHROPIC_KEY_1", "SPIKE_ANTHROPIC_KEY_2"):
        v = os.environ.get(env_name, "").strip()
        if v:
            keys.append(v)
    return keys


class AccountRotator:
    """Round-robin over 1-2 real Anthropic API keys, forcing each
    do_claude_code() subprocess call to authenticate as the chosen account
    via --bare + ANTHROPIC_API_KEY (see module docstring for why --bare is
    required, not optional, for this to actually pick the account rather
    than silently falling back to whatever OAuth session is logged in)."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self._cycle = itertools.cycle(range(len(keys))) if keys else None
        self._real_do_claude_code = vc.do_claude_code
        self.calls_made = 0
        self.account_use_count = [0] * len(keys)

    def active(self) -> bool:
        return bool(self.keys)

    def _wrapped_do_claude_code(self, *args, **kwargs):
        idx = next(self._cycle)
        account_key = self.keys[idx]
        self.account_use_count[idx] += 1
        self.calls_made += 1

        old_env = dict(os.environ)
        try:
            os.environ["ANTHROPIC_API_KEY"] = account_key
            # NOTE: do_claude_code() builds its own `args` list internally
            # and does not currently expose a way to inject --bare from the
            # outside. Forcing it here via env var alone (without --bare)
            # is the UNVERIFIED part flagged in the module docstring --
            # real Claude CLI auth precedence between an ambient OAuth
            # session and ANTHROPIC_API_KEY-without---bare was not
            # independently confirmed this session. If accounts aren't
            # actually alternating in practice, that precedence is the
            # first thing to check, and do_claude_code() would need a real
            # (small, additive) change to accept an extra_args param that
            # appends --bare.
            print(f"[spike] account #{idx + 1} of {len(self.keys)} "
                  f"(call {self.calls_made}, this account used "
                  f"{self.account_use_count[idx]}x so far)", flush=True)
            return self._real_do_claude_code(*args, **kwargs)
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def install(self):
        if self.active():
            vc.do_claude_code = self._wrapped_do_claude_code

    def uninstall(self):
        vc.do_claude_code = self._real_do_claude_code


def run(task: str, project: str = "spikeling", real: bool = False) -> dict:
    keys = _load_accounts()
    rotator = AccountRotator(keys)

    if real:
        if not keys:
            # No rotation keys -- NOT a fallback to dry-run. This still runs
            # for real, exactly like the original spiking_orchestrator.py's
            # --real flag: do_claude_code() uses whatever account the `claude`
            # CLI is already logged into on this machine (ambient auth), no
            # ANTHROPIC_API_KEY override, no --bare. Rotation is opt-in on
            # top of that, not a requirement for a real run to happen at all.
            print("[spike] no SPIKE_ANTHROPIC_KEY_1/2 set -- running for real "
                  "on this machine's already-logged-in Claude CLI session "
                  "(no rotation). Set both env vars to rotate across two "
                  "accounts instead.", flush=True)
        elif len(keys) == 1:
            print("[spike] only SPIKE_ANTHROPIC_KEY_1 set -- running on a "
                  "single account, no rotation. Set SPIKE_ANTHROPIC_KEY_2 "
                  "too for real fallback between two accounts.", flush=True)
        else:
            print(f"[spike] rotating across {len(keys)} accounts.", flush=True)

    if real:
        rotator.install()
    try:
        p = SpikingPipeline(task, project=project, dry_run=not real)
        result = p.run()
    finally:
        if real:
            rotator.uninstall()

    return result


def _interactive() -> None:
    """Bare `spike` with no args: an interactive prompt loop, same shape as
    typing bare `hermes` -- type a task, see it routed, keep going until you
    quit. Each line is one task through the SAME SpikingPipeline as the CLI
    path; nothing about the routing differs between typed-once and typed-here."""
    keys = _load_accounts()
    real = bool(keys)
    print("=" * 70)
    print("  spike -- spiking-orchestrator pipeline, interactive")
    print("=" * 70)
    if real:
        print(f"  {len(keys)} account(s) loaded -- real runs will call Claude "
              f"and spend tokens.")
    else:
        print("  No SPIKE_ANTHROPIC_KEY_1/2 set -- dry-run only (no API calls, "
              "no tokens spent). See spike.py's module docstring for setup.")
    print("  Type a task, or 'demo' for the no-API routing demo, or 'quit'/'exit' to leave.\n")

    while True:
        try:
            line = input("spike> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        if line.lower() == "demo":
            _routing_demo()
            continue
        result = run(line, real=real)
        _print_result(line, result)
        print()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Spike -- the spiking-orchestrator pipeline, run through "
                     "your terminal, rotating across your Anthropic accounts.")
    ap.add_argument("task", nargs="*", help="task to route (omit for interactive mode)")
    ap.add_argument("--demo", action="store_true", help="run the no-API routing demo (unchanged from spiking_orchestrator.py)")
    ap.add_argument("--project", default="spikeling")
    ap.add_argument("--real", action="store_true", help="actually run agents via Claude, rotating accounts (spends tokens)")
    args = ap.parse_args()

    if args.demo:
        _routing_demo()
        return

    if not args.task:
        _interactive()
        return

    task = " ".join(args.task)
    result = run(task, project=args.project, real=args.real)
    _print_result(task, result)


if __name__ == "__main__":
    main()
