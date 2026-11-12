#!/usr/bin/env bash
# run.sh <tag> <dispatch_card.py args...>: one card job under a hang watcher, an external
# tool of the machine these tests were written on (not part of this repository).
#   KMR_CARD_ENV  script to source for the tt-metal environment; it must set $PY
#   HANGWATCH     the hang watcher (takes the card lock, kills on limit or stall)
#   LOCK_DIR      directory of the lab's stop / reset lock files
#   CARD          card index (default 1), LIMIT seconds (default 120)
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
echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
cd "$logs"
python3 "$HANGWATCH" --tag "dk-$tag" --cards "${CARD:-1}" --limit "${LIMIT:-120}" \
    --stall "${LIMIT:-120}" --wait 600 --log "$logs/dk-$tag.log" -- \
    "$PY" "$here/dispatch_card.py" "$@"
rc=$?
echo "rc=$rc"
grep -E '^\{|Traceback|Error|FATAL|kernel ' "$logs/dk-$tag.log" | tail -20
exit $rc
