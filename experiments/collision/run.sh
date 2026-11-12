#!/usr/bin/env bash
# run.sh <tag> <collision.py args...>: one card job under the hang watcher.
#   COLLISION_ENV  script to source for the tt-metal environment; it must set $PY
#   HANGWATCH      the hang watcher (takes the card lock, kills on limit or stall)
#   LOCK_DIR       directory of the host's stop / reset lock files
#   CARD           card index (default 1), LIMIT seconds (default 100)
set -uo pipefail
tag=${1:?usage: run.sh <tag> <collision.py args...>}
shift
: "${COLLISION_ENV:?}" "${HANGWATCH:?}" "${LOCK_DIR:?}"
here=$(cd "$(dirname "$0")" && pwd)
logs="$here/../../.logs"
mkdir -p "$logs"
# shellcheck disable=SC1090
source "$COLLISION_ENV"
for f in bench.stop reset.pending mesh.request; do
    if [[ -e $LOCK_DIR/$f ]]; then
        echo "blocked: $f present"
        exit 3
    fi
done
echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
cd "$logs"
python3 "$HANGWATCH" --tag "collision-$tag" --cards "${CARD:-1}" --limit "${LIMIT:-100}" \
    --stall "${LIMIT:-100}" --wait 600 --log "$logs/collision-$tag.log" -- \
    "$PY" "$here/collision.py" "$@"
rc=$?
echo "rc=$rc"
grep -E '^\{|^x280 done|Traceback|Error|FATAL' "$logs/collision-$tag.log" | tail -20
exit $rc
