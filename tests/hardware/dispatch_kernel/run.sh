#!/usr/bin/env bash
# run.sh <tag> <dispatch_card.py args...>: one card job under a hang watcher, an external
# tool of the machine these tests were written on (not part of this repository).
#   KMR_CARD_ENV  script to source for the tt-metal environment; it must set $PY
#   HANGWATCH     the hang watcher (takes the card lock, kills on limit or stall)
#   LOCK_DIR      directory of the lab's stop / reset lock files
#   CARD          card index (default 1), LIMIT seconds (default 120)
#   STALL         seconds without a new log line (default LIMIT / 3)
#   OP_TIMEOUT    TT_METAL_OPERATION_TIMEOUT_SECONDS (default 30: three times the
#                 longest job, at least 20)
set -uo pipefail
tag=${1:?usage: run.sh <tag> <dispatch_card.py args...>}
shift
: "${KMR_CARD_ENV:?}" "${HANGWATCH:?}" "${LOCK_DIR:?}"
here=$(cd "$(dirname "$0")" && pwd)
logs="$here/../../../.logs"
mkdir -p "$logs"
# shellcheck disable=SC1090
source "$KMR_CARD_ENV"
for f in bench.stop reset.pending mesh.request; do
    if [[ -e $LOCK_DIR/$f ]]; then
        echo "blocked: $f present"
        exit 3
    fi
done
limit=${LIMIT:-120}
stall=${STALL:-$((limit / 3))}
if ((stall * 3 > limit)); then
    echo "STALL must be at most a third of LIMIT"
    exit 2
fi
export TT_METAL_OPERATION_TIMEOUT_SECONDS=${OP_TIMEOUT:-30}
echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
cd "$logs"
python3 "$HANGWATCH" --tag "dk-$tag" --cards "${CARD:-1}" --limit "$limit" \
    --stall "$stall" --wait 600 --log "$logs/dk-$tag.log" -- \
    "$PY" "$here/dispatch_card.py" "$@"
rc=$?
echo "rc=$rc"
grep -E '^\{|Traceback|Error|FATAL|kernel ' "$logs/dk-$tag.log" | tail -20
exit $rc
