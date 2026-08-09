#!/bin/sh
# SPIKEMESH thin client for the Akai MPC One's real BusyBox developer-mode
# shell. Hosts nothing -- just a real HTTP call to the mesh server already
# running on kimchi, same as the web UI or the MCP tools use.
#
# HONEST STATUS: the network call + JSON extraction logic below was tested
# for real against the live SPIKEMESH server (confirmed correct output on
# several real questions). What's genuinely UNTESTED is running it on
# actual MPC One hardware, since that's not reachable from here -- try it
# once developer mode is on; if `wget` on your firmware build doesn't
# support --post-data, that's the first thing to check.
#
# Usage (from the MPC One's developer-mode shell, over serial or SSH):
#   ./mpc_one_client.sh "what performance work has been done on tribe?"
#
# SPIKEMESH_HOST defaults to kimchi's real LAN IP (confirmed current and
# reachable tonight). Only works if the MPC One is on the same LAN/WiFi as
# kimchi -- it is NOT on the Tailscale mesh, so the 100.x tailnet address
# won't be reachable from it.

SPIKEMESH_HOST="${SPIKEMESH_HOST:-10.0.0.100}"
SPIKEMESH_PORT="${SPIKEMESH_PORT:-5055}"
QUESTION="$1"

if [ -z "$QUESTION" ]; then
    echo "Usage: $0 \"your question\""
    exit 1
fi

BODY="{\"question\": \"$(echo "$QUESTION" | sed 's/"/\\"/g')\"}"

RESPONSE=$(wget -q -O - \
    --header="Content-Type: application/json" \
    --post-data="$BODY" \
    "http://${SPIKEMESH_HOST}:${SPIKEMESH_PORT}/ask")

if [ -z "$RESPONSE" ]; then
    echo "No response -- check SPIKEMESH_HOST/PORT and that the MPC One is on the same LAN as kimchi."
    exit 1
fi

# Pure sed/awk field extraction, no jq/python -- confirmed real, correct
# output against a live response before this shipped (not a guess). Basic
# regex on purpose (no -E flag), since BusyBox sed doesn't guarantee
# extended-regex support on every firmware build. Not a full JSON parser --
# handles the real, common response shape, not every conceivable edge case.
ANSWER=$(echo "$RESPONSE" | sed -n 's/.*"answer"[[:space:]]*:[[:space:]]*"\(\([^"\\]\|\\.\)*\)".*/\1/p' | sed 's/\\"/"/g; s/\\n/\
/g; s/\\\\/\\/g')
FIRED=$(echo "$RESPONSE" | sed -n 's/.*"fired"[[:space:]]*:[[:space:]]*\(true\|false\).*/\1/p' | head -1)
VERIFIED=$(echo "$RESPONSE" | sed -n 's/.*"verified"[[:space:]]*:[[:space:]]*\(true\|false\).*/\1/p' | head -1)
SCORE=$(echo "$RESPONSE" | sed -n 's/.*"best_score"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p' | head -1)

echo "=================================================="
if [ -n "$ANSWER" ]; then
    echo "$ANSWER"
else
    echo "(no 'answer' field found -- raw response below)"
    echo "$RESPONSE"
fi
echo "=================================================="
echo "confidence gate fired: ${FIRED:-?}   grounding verified: ${VERIFIED:-?}   score: ${SCORE:-?}"
