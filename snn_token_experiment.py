"""
Does SNN agent-routing actually save tokens vs a fixed full pipeline?

Measured, not asserted. For each task:
  - FIXED  = run every agent in the roster once (the non-SNN pipeline shape)
  - SNN    = run only the agents spiking_orchestrator.SpikingPipeline fires

Each agent invocation is a real `claude -p ... --output-format json` call with
that agent's framed prompt + the target repo's code as context. We record
usage.input_tokens + usage.output_tokens (agent-attributable; the CLI's own
system prompt / tools / CLAUDE.md are served from cache and reported
separately, so they do NOT inflate these numbers) and total_cost_usd.

Target repo: doorcam (github.com/tritsystem/doorcam). No files are edited --
this measures the token cost of the routing decision, not the code change.

    python snn_token_experiment.py            # full run
    python snn_token_experiment.py --dry      # print routing + prompt sizes, no API
"""

import json
import subprocess
import sys
import time
import os

from spiking_orchestrator import SpikingPipeline, score_task

REPO = r"C:\Users\gbran\OneDrive\Documents\doorcam"
CLAUDE = os.path.expanduser(r"~\AppData\Roaming\npm\claude.cmd")
MODEL = "claude-haiku-4-5"
COST_CAP_USD = 4.0
DRY = "--dry" in sys.argv

ROSTER = ["Clarifier", "PreRegister", "Implementer", "TestWriter",
          "Reviewer", "Corrector", "VaultLogger"]

TASKS = [
    ("typo",      "Fix the typo in the module docstring of calibrate_roi.py"),
    ("port-flag", "Add a --port command-line flag to main.py so the MJPEG server port is not hardcoded"),
    ("add-tests", "Add unit tests for the bbox_in_roi function in detection/roi.py"),
    ("ambiguous", "make the alerting better"),
    ("complex",   "Add a rate-limited email alerter alongside the Discord one, "
                  "wire it into config and main, add tests, and update the README"),
]

# per-role framing, mirrored from spiking_orchestrator._agent_task
def frame(role, task):
    return {
        "Clarifier":   f"You are the Clarifier. If this task is ambiguous, ask ONE blocking question and STOP. Otherwise say PROCEED. Task: {task}",
        "PreRegister": f"Before any edit, state ONE falsifiable claim about what this change will do: {task}",
        "Implementer": f"Implement this change. Show the edited code. Task: {task}",
        "TestWriter":  f"Add or adjust tests for: {task}",
        "Reviewer":    f"Peer-review the change for: {task}. Read-only. Call out overclaiming and any bug.",
        "Corrector":   f"The reviewer flagged issues with the change for: {task}. Apply the corrections; show the fixed code.",
        "VaultLogger": f"Write a 4-6 line ledger entry summarising the work done for: {task}",
    }[role]


def repo_context():
    parts = []
    for rel in ["main.py", "config.py", "calibrate_roi.py",
                "detection/roi.py", "alerts/discord_alerter.py"]:
        p = os.path.join(REPO, rel)
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                parts.append(f"# ===== {rel} =====\n{f.read()}")
    return "\n\n".join(parts)


CTX = repo_context()


def run_agent(role, task):
    """One real claude -p call. Returns (in_tok, out_tok, cost_usd)."""
    prompt = (f"{frame(role, task)}\n\n"
              f"Work only from the code below -- do NOT use any tools, "
              f"just respond with your {role} output.\n\n{CTX}")
    if DRY:
        return (len(prompt) // 4, 0, 0.0)  # rough token estimate
    out = subprocess.run(
        [CLAUDE, "-p", "--output-format", "json", "--model", MODEL],
        input=prompt, capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"claude call failed rc={out.returncode}: {out.stderr[:400]}")
    d = json.loads(out.stdout)
    u = d["usage"]
    return (u["input_tokens"], u["output_tokens"], d.get("total_cost_usd", 0.0))


def main():
    results = []
    cache = {}          # (task_id, role) -> (in, out, cost)  -- pay once, use in both configs
    cum_cost = 0.0

    for tid, task in TASKS:
        snn_fired = SpikingPipeline(task, project="doorcam", dry_run=True).run()["fired"]
        fixed = ROSTER[:]                      # full roster, once each
        needed = list(dict.fromkeys(fixed + snn_fired))

        for role in needed:
            key = (tid, role)
            if key in cache:
                continue
            if not DRY and cum_cost > COST_CAP_USD:
                print(f"!! cost cap ${COST_CAP_USD} hit -- stopping early")
                _report(results); return
            it, ot, c = run_agent(role, task)
            cache[key] = (it, ot, c)
            cum_cost += c
            print(f"  [{tid:9}] {role:12} in={it:6} out={ot:5} ${c:.4f}  (cum ${cum_cost:.3f})", flush=True)

        def tally(agent_list):
            ti = to = 0
            for r in agent_list:
                i, o, _ = cache[(tid, r)]
                ti += i; to += o
            return ti, to

        f_in, f_out = tally(fixed)
        s_in, s_out = tally(snn_fired)
        results.append(dict(
            task=tid, prompt=task,
            fixed_agents=len(fixed), snn_agents=len(snn_fired),
            snn_fired=snn_fired,
            fixed_tok=f_in + f_out, snn_tok=s_in + s_out,
            fixed_in=f_in, fixed_out=f_out, snn_in=s_in, snn_out=s_out,
            saved_tok=(f_in + f_out) - (s_in + s_out),
            saved_pct=round(100 * (1 - (s_in + s_out) / max(f_in + f_out, 1)), 1),
        ))

    _report(results, cum_cost)


def _report(results, cum_cost=0.0):
    print("\n" + "=" * 92)
    print(f"{'task':10} {'fixed_ag':>8} {'snn_ag':>7} {'fixed_tok':>10} {'snn_tok':>9} {'saved':>9} {'saved%':>7}")
    print("-" * 92)
    F = S = 0
    for r in results:
        F += r["fixed_tok"]; S += r["snn_tok"]
        print(f"{r['task']:10} {r['fixed_agents']:>8} {r['snn_agents']:>7} "
              f"{r['fixed_tok']:>10} {r['snn_tok']:>9} {r['saved_tok']:>9} {r['saved_pct']:>6}%")
    print("-" * 92)
    tot_pct = round(100 * (1 - S / max(F, 1)), 1)
    print(f"{'TOTAL':10} {'':>8} {'':>7} {F:>10} {S:>9} {F - S:>9} {tot_pct:>6}%")
    print(f"\nexperiment API spend: ${cum_cost:.2f}   model: {MODEL}   repo: doorcam")
    json.dump({"model": MODEL, "cost_usd": cum_cost, "results": results},
              open("snn_token_experiment_results.json", "w"), indent=1)
    print("wrote snn_token_experiment_results.json")


if __name__ == "__main__":
    main()
