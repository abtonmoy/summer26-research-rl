#!/usr/bin/env bash
# Chain the remaining heavy sims sequentially. Each writes its own .mat.
# Order: simS6 (deterministic, ~50min) -> simS7 (~108min) -> simS1 (~30min) ->
# simS4 (~80min). Each prints its own progress to stdout.

set -euo pipefail
PY="${PY:-C:/Python313/python.exe}"
cd "$(dirname "$0")/.."

echo "=== START $(date '+%H:%M:%S') ==="

echo "--- [1/4] simS6 (hetero N=50,200) ---"
"$PY" -u -m sims.simS6_heterogeneity_Nswipe 2>&1
echo "simS6 done $(date '+%H:%M:%S')"

echo "--- [2/4] simS7 (partialcomm full re-run) ---"
"$PY" -u -m sims.simS7_partialcomm_performance 2>&1
echo "simS7 done $(date '+%H:%M:%S')"

echo "--- [3/4] simS1 (GA, medium) ---"
"$PY" -u -m sims.simS1_genetic_algorithm 2>&1
echo "simS1 done $(date '+%H:%M:%S')"

echo "--- [4/4] simS4 (response time, mesh=25) ---"
"$PY" -u -m sims.simS4_response_time 2>&1
echo "simS4 done $(date '+%H:%M:%S')"

echo "=== END $(date '+%H:%M:%S') ==="
