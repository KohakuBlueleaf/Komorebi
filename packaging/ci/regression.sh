#!/usr/bin/env bash
# The whole regression on the card machine (two P150 cards), from the repository's root:
# a CPU lane and a card lane, a line a part with its seconds, the total; exit 1 on any
# failure or on a total over LIMIT seconds. Usage: bash packaging/ci/regression.sh
# The card list is prebuilt first when its stamp is stale (after a code change, or not
# made off card); a warm prebuild counts in the total, a cold build's (it compiled
# kernels) does not.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 2
PY=${PY:-python}
LIMIT=${LIMIT:-300}
JOBS=${JOBS:-48}
LOGS="$ROOT/build/regression/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGS"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
FAILED=0
# the CPU lane: no card can be opened (off_card.sh: a real open stops at once, "No chips
# detected in the cluster"); a card lock's card list is not the mock's
OFF="$ROOT/packaging/ci/off_card.sh env -u TT_VISIBLE_DEVICES"

now() { date +%s.%N; }

# part NAME COMMAND...: runs a part, its output to its log; prints its line
part() {
    local name=$1 started code seconds
    shift
    started=$(now)
    "$@" >"$LOGS/$name.log" 2>&1
    code=$?
    seconds=$(echo "$(now) - $started" | bc)
    if [ "$code" -eq 0 ]; then
        printf 'PASS %6.1f s  %s\n' "$seconds" "$name"
    else
        printf 'FAIL %6.1f s  %s (exit %s; log %s)\n' "$seconds" "$name" "$code" \
            "$LOGS/$name.log"
    fi
    return "$code"
}

# The card list's stamp: fresh when its prebuild ran off card (it says so itself), every
# line was prebuilt with this code, and the shared opens were built for this list.
stale=$("$PY" -c 'from komorebi.tools import card_check as tool
print("" if tool.fresh_off_card(".") else "stale")')
started=$(now)
if [ -n "$stale" ]; then
    part prebuild $OFF "$PY" -m komorebi.tools.card_check --prebuild --jobs "$JOBS" ||
        { echo "the card list did not build; nothing else run"; exit 1; }
    cold=$(grep "^cold build:" "$LOGS/prebuild.log")
    if [ -n "$cold" ]; then
        echo "$cold"
        started=$(now)
    fi
fi
# the CPU lane, all at once
part host_check $OFF "$PY" -m komorebi.tools.host_check &
cpu_host=$!
part pytest_cpu $OFF "$PY" -m pytest -q -n "$JOBS" -m "not tree and not gpu" \
    tests/core tests/projects tests/ref &
cpu_tests=$!
part pytest_tree $OFF "$PY" -m pytest -q -n 8 -m tree tests/core &
cpu_tree=$!
# the card lane: card 0's list beside the card tests on card 1 and then card 1's
# Tensix dispatcher lines (while card 0's list still runs), then cards 0 and 1
part card_check_card0 "$PY" -m komorebi.tools.card_check --cards 1 &
card0=$!
{
    part hardware_card1 env KMR_CARD=1 TT_VISIBLE_DEVICES=1 "$PY" -m pytest -q \
        tests/hardware
    code=$?
    part tensix_card1 "$PY" -m komorebi.tools.card_check --cards tensix || code=1
    exit "$code"
} &
card1=$!
for job in $card0 $card1; do
    wait "$job" || FAILED=1
done
part card_check_cards01 "$PY" -m komorebi.tools.card_check --cards 2 || FAILED=1
for job in $cpu_host $cpu_tests $cpu_tree; do
    wait "$job" || FAILED=1
done
total=$(echo "$(now) - $started" | bc)
over=$(echo "$total > $LIMIT" | bc)
[ "$over" -eq 1 ] && FAILED=1
verdict=$([ "$FAILED" -eq 0 ] && echo passed || echo FAILED)
printf 'total %.1f s of %s s: %s; logs %s\n' "$total" "$LIMIT" "$verdict" "$LOGS"
exit "$FAILED"
