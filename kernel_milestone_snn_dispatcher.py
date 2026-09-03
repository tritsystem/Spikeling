#!/usr/bin/env python
"""
kernel_milestone_snn_dispatcher.py -- runs several spikeling-os kernel
milestones as REAL, SEPARATE OS PROCESSES IN PARALLEL, each isolated in its
own git worktree, with a real Spikeling network (pyspike.Net) deciding WHICH
queued milestone fires next and WHEN -- driven by real completion events,
not a Python if-statement dressed up as one.

DIVISION OF LABOR (deliberate, not a compromise -- see this project's own
precedent below):
  - The NETWORK decides ORDER/TIMING among ELIGIBLE candidates: each queued
    milestone is a real neuron with its own priority-weighted stimulation
    and its own refractory period. It cannot fire on its own initiative --
    only a real external event (a slot freeing up, via a track's own real
    subprocess actually exiting) drives fresh stimulation into the network,
    same "handlers run synchronously as neurons fire, real events drive the
    cascade" discipline spiking_orchestrator.py already established.
  - CONCURRENCY SAFETY is a hard, explicit, deterministic Python cap
    (MAX_CONCURRENT), never emergent LIF dynamics. Direct precedent:
    spiking_orchestrator.py's MAX_DYNAMIC_SPECIALISTS ("an explicit stop
    condition, no exceptions" -- see that file's DYNAMIC SPECIALISTS
    comment, itself citing the DRIVE_FLOOR lesson from an earlier scheduler
    hang). Every fired neuron here launches a REAL `claude` CLI subprocess
    that spends real API tokens and real CPU (cargo build + a real QEMU
    boot) -- letting pure network dynamics decide the ceiling would be
    gambling with real cost, not a place to trust emergent behavior.

ISOLATION: each queued milestone runs in its OWN git worktree (a separate
checkout on its own branch, sharing this repo's object store). Multiple
agents with real Edit/Write/Bash access to the SAME working directory at
the same time would corrupt each other's uncommitted edits and race on
cargo's own target dir / QEMU's own ports -- worktrees make that
structurally impossible, there's no shared file left to race on.

MERGING BACK: deliberately OUT OF SCOPE here. Each track's own Reviewer
stage (kernel_milestone_pipeline.py, reused unchanged as the actual per-
milestone worker -- this script is a dispatcher around it, not a
reimplementation) already gates on independent re-verification. But
merging N independently-branched worktrees into main is a separate real
operation with real conflict risk (several of these tracks' scopes plainly
overlap on shell.rs's command-dispatch arm, going by that file's own
extensive milestone history) that deserves a human/orchestrating-Claude
reading the actual diff -- same discipline as every milestone already
committed. This script's job ends at "here are N branches, each with a
real implement+review report" -- it never touches origin/main, never runs
`git merge`.

Usage:
    python kernel_milestone_snn_dispatcher.py --real
    python kernel_milestone_snn_dispatcher.py            # dry-run, prints the plan
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyspike import Net  # noqa: E402

REPO_ROOT = r"C:\Users\gbran\OneDrive\Documents\spikeling-os"
PIPELINE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_milestone_pipeline.py")
MILESTONE_LOG_DIR = os.path.join(REPO_ROOT, "milestones")

MAX_CONCURRENT = 2  # hard cap -- see module doc, this is deliberately NOT SNN-driven
POLL_SECONDS = 20
PER_STAGE_TIMEOUT = 2400  # 40 min/stage -- these are larger scopes than the killed M45 test

# (milestone_number, worktree_dir_name, priority_stimulation, scope_description)
# Priority stimulation is a real, stated judgment call (higher = more likely
# to fire first when multiple candidates are simultaneously eligible), not
# arbitrary: M49 is scored highest because exec() is foundational (a real
# shell eventually needs it, and nothing else here depends on it existing
# first, so there's no reason to delay it); M48 is scored high because its
# scope is small/self-contained (ternary.rs has the lightest milestone
# history of any target file here, real evidence checked before writing
# this, not assumed) so finishing it fast is realistic; M46/M47 are larger,
# scored lower, expected to legitimately take longer.
QUEUE = [
    (49, "spikeling-os-wt-m49", 90,
     "Real exec(): a new syscall that replaces the calling process's own "
     "address space with a freshly-loaded ELF image from disk, reusing the "
     "existing ELF loader/parser (elf.rs, loader.rs) and the same page-"
     "table-teardown-and-rebuild primitives fork() already has for a "
     "child's private space -- so a shell can eventually really replace "
     "itself with a program instead of only ever forking a fixed "
     "hardcoded/loaded image."),
    (48, "spikeling-os-wt-m48", 85,
     "Wire kernel/src/ternary.rs into a REAL, observable kernel decision "
     "path -- read that file first to see what it currently provides "
     "(balanced-ternary arithmetic? a trit type? something else -- do not "
     "assume). Pick one real, concrete use inside the kernel itself (e.g. a "
     "genuine ternary-valued decision the scheduler or a syscall dispatch "
     "path makes) and, matching this whole portfolio's own established "
     "ethos (see topological-phononics / symmetry-selection-rule repos for "
     "the pattern), report an HONEST MEASURED comparison against the "
     "binary equivalent it replaces -- real numbers or a real qualitative "
     "boot-log difference, not an assumed advantage. A negative or neutral "
     "result reported honestly is a fully acceptable outcome for this "
     "milestone, same as this repo's own Milestone 4 (SNN scheduling "
     "fairness) reported its hypothesis did NOT hold and kept that finding "
     "in the README exactly as measured."),
    (46, "spikeling-os-wt-m46", 55,
     "A real, minimal VFS/mount abstraction generalizing kernel/src/fs.rs "
     "-- read that file first (already real through Milestone 35: "
     "understand exactly what's hardcoded before changing it). Today "
     "there is exactly one disk-backed filesystem/device path; add a real "
     "trait/dispatch layer so a SECOND backing store (even something "
     "trivial and real, like a small in-memory ramfs, is a legitimate "
     "concrete target) can register and be addressed through the same "
     "path/open() surface, without breaking any existing shell command "
     "or self-test that already depends on the current disk path. Verify "
     "both the old path AND the new one work with a real boot-time "
     "self-test for each."),
    (47, "spikeling-os-wt-m47", 50,
     "Read kernel/src/network.rs and kernel/src/nic.rs FIRST in full -- "
     "they already have real work through Milestones 38/39/26, do not "
     "assume this is a stub. Identify the real next, currently-missing "
     "layer (read the existing milestone comments in both files for what "
     "they say is NOT yet done) and implement + verify it for real -- QEMU "
     "supports real user-mode networking (`-netdev user`) and a real pcap "
     "capture (`-object filter-dump`) if you need packet-level evidence, "
     "not just a serial log claim."),
]


def report_paths(n: int):
    return (
        os.path.join(MILESTONE_LOG_DIR, f"m{n}_implement.md"),
        os.path.join(MILESTONE_LOG_DIR, f"m{n}_review.md"),
    )


class Dispatcher:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.net = Net(refractory_ms=1000)  # long refractory: a track that just fired shouldn't double-fire on the very next poll tick
        self.neurons = {}
        self.pending = {}  # n -> (worktree, priority, scope)
        self.active = {}   # n -> (Popen, worktree, start_ts)
        self.finished = {}  # n -> returncode

        # build_live() FIRST, then add neurons/synapses/actions -- verified
        # against spiking_orchestrator.py's own _build_brain(), which calls
        # this in exactly this order; doing it the other way round (as a
        # first draft of this file did) raises "Unknown neuron" the moment
        # stimulate() is called, caught by a real dry-run test before this
        # ever touched a live milestone run.
        self.rt = self.net.build_live()

        # Direct stimulation, no gate-relay synapse. An earlier draft routed
        # a "SlotFree" gate neuron's spike THROUGH a weighted synapse to each
        # milestone neuron (weight * 50.0 per the runtime's own real _fire()
        # propagation formula, confirmed by reading core/runtime/runtime.py
        # directly) -- that's the right pattern for a causal dependency
        # (Reviewer fires -> stimulates Corrector), but wrong here: a queue
        # has no such dependency, it just needs each candidate's own
        # priority to drive its own membrane potential. Caught in dry-run
        # (0/4 ever fired) before this touched real tokens; two real,
        # independent bugs, both from that mismatch:
        #   (a) stimulate("SlotFree", 45.0) passed a fixed 45.0 into
        #       current_time_ms (real signature is stimulate(name,
        #       current_time_ms, drive=50.0)) -- last_spike_time defaults to
        #       -inf so the FIRST call fired anyway, but every call after
        #       that saw elapsed = 45.0 - 45.0 = 0 < refractory_ms(1000),
        #       permanently refractory-locked after tick 1.
        #   (b) even a correctly-firing gate only delivers weight * 50.0 to
        #       each downstream neuron (max ~58.5 for M49's weight) against
        #       a threshold=100 that was copied without checking this
        #       runtime's real weight-to-potential scaling -- never enough
        #       to cross in one hop regardless of (a).
        # Fix: stimulate each pending neuron directly every tick with
        # drive=priority and a genuinely advancing current_time_ms
        # (wall-clock ms), threshold set below the lowest real priority in
        # QUEUE so every candidate reliably fires when offered -- ordering
        # among simultaneously-eligible candidates is decided by real LIF
        # state (refractory keeps a just-fired-but-capacity-blocked neuron
        # from thrashing) plus the hard Python cap below, not by a synapse
        # weight standing in for priority twice.
        THRESHOLD = 40  # below every QUEUE priority (50, 55, 85, 90)
        for n, worktree, priority, scope in QUEUE:
            self.pending[n] = (worktree, priority, scope)
            neuron = self.net.neuron(f"M{n}", threshold=THRESHOLD, leak=5)
            self.neurons[n] = neuron
            self.net.action(neuron)(self._make_launcher(n))

    def _make_launcher(self, n: int):
        def handler():
            if n not in self.pending:
                return  # already launched or already finished -- stray/duplicate fire, ignore
            if len(self.active) >= MAX_CONCURRENT:
                return  # hard cap -- see module doc. Stays pending, will be re-offered next free slot.
            worktree, priority, scope = self.pending.pop(n)
            self._launch(n, worktree, scope)
        return handler

    def _launch(self, n: int, worktree: str, scope: str):
        worktree_path = os.path.join(os.path.dirname(REPO_ROOT), worktree)
        stdout_path = os.path.join(MILESTONE_LOG_DIR, f"m{n}_dispatcher_stdout.log")
        os.makedirs(MILESTONE_LOG_DIR, exist_ok=True)
        print(f"[dispatcher] LAUNCH milestone {n} in {worktree_path} (active now: {len(self.active) + 1}/{MAX_CONCURRENT})", flush=True)
        if self.dry_run:
            print(f"  (dry-run, not actually launching) scope: {scope[:100]}...")
            self.finished[n] = "dry-run"
            return
        args = [sys.executable, PIPELINE_SCRIPT, str(n), scope,
                "--real", "--timeout", str(PER_STAGE_TIMEOUT)]
        # Run the pipeline against the ISOLATED WORKTREE's own path, not
        # REPO_ROOT -- kernel_milestone_pipeline.py resolves REPO_DIR from
        # voice_commands.PROJECTS["spikeling_os"], which points at main's
        # own working dir, so override it via env var rather than editing
        # that module's own resolution for a one-off dispatch need.
        env = dict(os.environ)
        env["SPIKELING_OS_OVERRIDE_DIR"] = worktree_path
        f = open(stdout_path, "w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=os.path.dirname(PIPELINE_SCRIPT))
        self.active[n] = (proc, worktree_path, time.time(), f)

    def _reap_finished(self):
        done = []
        for n, (proc, worktree_path, start_ts, f) in list(self.active.items()):
            rc = proc.poll()
            if rc is not None:
                done.append(n)
                f.close()
                elapsed = time.time() - start_ts
                self.finished[n] = rc
                print(f"[dispatcher] FINISHED milestone {n} in {worktree_path} -- exit={rc} ({elapsed:.0f}s)", flush=True)
                del self.active[n]
        return done

    def _stimulate_pending(self):
        # Real, advancing wall-clock ms -- NOT a fixed literal (that was the
        # actual bug: a repeated constant makes elapsed==0 forever, which
        # permanently trips the refractory guard after the first call). Each
        # pending neuron gets its own priority as direct drive; QUEUE is
        # already priority-ordered so iteration order alone gives higher-
        # priority candidates first crack when several fire on the same
        # tick, same as spiking_orchestrator.py's own direct-sensory-drive
        # pattern (S_Work/S_Ambiguous stimulated with drive=score), not the
        # gate-relay pattern that caused the original bug.
        now_ms = time.time() * 1000.0
        for n, (worktree, priority, scope) in list(self.pending.items()):
            self.rt.stimulate(f"M{n}", now_ms, drive=float(priority))

    def run(self):
        # seed: stimulate whatever's currently pending so anything eligible
        # (nothing running yet, MAX_CONCURRENT free) can fire immediately
        # for the first wave.
        self._stimulate_pending()
        while self.pending or self.active:
            time.sleep(POLL_SECONDS)
            newly_done = self._reap_finished()
            if newly_done or (self.pending and len(self.active) < MAX_CONCURRENT):
                # a real completion event (or just-freed capacity) re-drives
                # the network -- this is the actual "spiking network decides
                # what happens next" step, not decorative: each pending
                # neuron's own priority-weighted drive determines whether it
                # crosses threshold on this tick.
                self._stimulate_pending()
            print(f"[dispatcher] tick -- active={list(self.active)} pending={list(self.pending)} finished={self.finished}", flush=True)
        print("[dispatcher] all tracks finished:", self.finished, flush=True)
        return self.finished


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()
    d = Dispatcher(dry_run=not args.real)
    if not args.real:
        print("DRY RUN -- would dispatch, in priority order as capacity frees up:")
        for n, wt, pr, scope in QUEUE:
            print(f"  M{n} (priority {pr}, worktree {wt}): {scope[:90]}...")
        print(f"\nMAX_CONCURRENT={MAX_CONCURRENT}, poll every {POLL_SECONDS}s. Pass --real to actually run.")
        return
    d.run()


if __name__ == "__main__":
    main()
