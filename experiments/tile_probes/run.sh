#!/usr/bin/env bash
# run.sh <tag> <probes.py args...>: one card job under the hang watcher.
#   TP_ENV      script to source for the tt-metal environment; it must set $PY
#   HANGWATCH   the hang watcher (takes the card lock, kills on limit or stall)
#   LOCK_DIR    directory of the lab's stop / reset lock files
#   CARD        card index (default 0), LIMIT seconds (default 60)
set -uo pipefail
tag=${1:?usage: run.sh <tag> <probes.py args...>}
shift
: "${TP_ENV:?}" "${HANGWATCH:?}" "${LOCK_DIR:?}"
here=$(cd "$(dirname "$0")" && pwd)
logs="$here/../../.logs"
mkdir -p "$logs"
# shellcheck disable=SC1090
source "$TP_ENV"
for f in bench.stop reset.pending mesh.request; do
    if [[ -e $LOCK_DIR/$f ]]; then
        echo "blocked: $f present"
        exit 3
    fi
done
echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
cd "$logs"
python3 "$HANGWATCH" --tag "tp-$tag" --cards "${CARD:-0}" --limit "${LIMIT:-60}" \
    --stall "${LIMIT:-60}" --wait 600 --log "$logs/tp-$tag.log" -- \
    "$PY" "$here/probes.py" "$@"
rc=$?
echo "rc=$rc"
grep -E '^\{|^kernel done|Traceback|Error|error:|FATAL' "$logs/tp-$tag.log" | tail -20
exit $rc
