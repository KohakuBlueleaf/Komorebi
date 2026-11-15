#!/usr/bin/env bash
# run.sh <tag> <probe.py args...>: one card job under the hang watcher.
#   TL_ENV      script to source for the tt-metal environment; it must set $PY
#   HANGWATCH   the hang watcher (takes the card lock, kills on limit or stall)
#   LOCK_DIR    directory of the lab's stop / reset lock files
#   CARD        card index (default 0); LIMIT and STALL seconds (default 120, 40)
set -uo pipefail
tag=${1:?usage: run.sh <tag> <probe.py args...>}
shift
: "${TL_ENV:?}" "${HANGWATCH:?}" "${LOCK_DIR:?}"
here=$(cd "$(dirname "$0")" && pwd)
logs="$here/../../../.logs"
mkdir -p "$logs"
# shellcheck disable=SC1090
source "$TL_ENV"
for f in bench.stop reset.pending mesh.request; do
    if [[ -e $LOCK_DIR/$f ]]; then
        echo "blocked: $f present"
        exit 3
    fi
done
echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
export TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-20}
cd "$here/../../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 "$HANGWATCH" --tag "tl-$tag" --cards "${CARD:-0}" --limit "${LIMIT:-120}" \
    --stall "${STALL:-40}" --wait 600 --log "$logs/tl-$tag.log" -- \
    "$PY" "$here/probe.py" "$@"
rc=$?
echo "rc=$rc"
grep -E '^\{|Traceback|Error|error:|FATAL' "$logs/tl-$tag.log" | tail -20
exit $rc
