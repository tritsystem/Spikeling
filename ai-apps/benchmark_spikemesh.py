"""Real, runnable benchmark for SPIKEMESH -- measures actual behavior
against the live mesh server, not simulated numbers. No comparison claim
in this file is written until the real number backing it has been printed
by an actual run.

Usage:
    python benchmark_spikemesh.py
"""
import json
import time
import urllib.request

API = "http://127.0.0.1:5055"


def post(path, body, timeout=90):
    req = urllib.request.Request(
        API + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, time.time() - t0


def get(path, timeout=15):
    with urllib.request.urlopen(API + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def bench_confidence_gate():
    """Real discrimination test: does the LIF gate actually separate
    relevant from irrelevant questions, on cases beyond the two already
    checked by hand earlier tonight?"""
    relevant = [
        "what performance work has been done on tribe?",
        "what did the horde-defense-beta O(n^2) fix involve?",
        "what is the LIF neuron refractory period for?",
    ]
    irrelevant = [
        "what is the best recipe for chocolate chip cookies?",
        "who won the 1986 world cup?",
        "how do I change a car tire?",
    ]
    results = {"relevant": [], "irrelevant": []}
    for q in relevant:
        d, dt = post("/ask", {"question": q})
        results["relevant"].append({"q": q, "fired": d.get("confidence_gate", {}).get("fired"),
                                     "score": d.get("confidence_gate", {}).get("best_score"), "s": round(dt, 1)})
    for q in irrelevant:
        d, dt = post("/ask", {"question": q})
        results["irrelevant"].append({"q": q, "fired": d.get("confidence_gate", {}).get("fired"),
                                       "score": d.get("confidence_gate", {}).get("best_score"), "s": round(dt, 1)})

    correct = sum(1 for r in results["relevant"] if r["fired"] is True) + \
              sum(1 for r in results["irrelevant"] if r["fired"] is False)
    total = len(relevant) + len(irrelevant)
    return results, correct, total


def bench_latency():
    """Real end-to-end /ask latency over several real questions -- not a
    single cherry-picked run."""
    times = []
    for q in ["what is Spikeling?", "what does OBSERVE do?", "what is the tribe DSL?"]:
        _, dt = post("/ask", {"question": q})
        times.append(dt)
    return times


def main():
    print("SPIKEMESH real benchmark -- every number below is a live measurement.\n")

    health = get("/health")
    print(f"Server reachable: {health.get('status') == 'ok'}")

    print("\n--- Confidence gate discrimination (real LIF neuron, not a keyword filter) ---")
    gate_results, correct, total = bench_confidence_gate()
    for label in ("relevant", "irrelevant"):
        for r in gate_results[label]:
            print(f"  [{label:10s}] fired={r['fired']!s:5s} score={r['score']}  ({r['s']}s)  {r['q'][:55]}")
    print(f"  Correct discrimination: {correct}/{total} ({100*correct/total:.0f}%)")

    print("\n--- /ask real end-to-end latency ---")
    times = bench_latency()
    for t in times:
        print(f"  {t:.1f}s")
    print(f"  mean: {sum(times)/len(times):.1f}s  min: {min(times):.1f}s  max: {max(times):.1f}s")

    print("\n--- Project registry (real git hygiene across known repos) ---")
    reg = get("/pm/registry")
    flagged = [p for p in reg["projects"] if p.get("flags")]
    print(f"  {len(reg['projects'])} projects tracked, {len(flagged)} with real hygiene flags")

    print("\n--- Server Guard (real historical security/health data) ---")
    guard = get("/guard/status")
    print(f"  {guard['channel_count']} real channels, live={guard['live']}")

    print("\nDone. See BENCHMARK_VS_NOMAD.md for the honest capability comparison "
          "these numbers feed into.")


if __name__ == "__main__":
    main()
