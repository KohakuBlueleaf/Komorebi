#!/usr/bin/env bash
# off_card.sh COMMAND...: runs COMMAND where no Tenstorrent device can be opened, for the
# regression's lanes that run on the stand-in card: in a mount namespace of its own whose
# /dev/tenstorrent and /sys/class/tenstorrent (the card types: none, as with no card)
# are empty read-only folders, as this user (a user namespace maps the
# same ids). A real open then finds no device and tt-metal stops it at once ("No chips
# detected in the cluster"); the stand-in card opens no device and is not changed.
# Refused when the namespaces cannot be made: the lane never runs unguarded.
set -u
[ $# -gt 0 ] || { echo "usage: off_card.sh COMMAND..." >&2; exit 2; }
uid=$(id -u) gid=$(id -g)
exec unshare --user --map-root-user --mount sh -c '
mount -t tmpfs -o ro,size=4k off-card /dev/tenstorrent &&
    mount -t tmpfs -o ro,size=4k off-card /sys/class/tenstorrent || {
    echo "off_card.sh: the card could not be hidden; the lane is not run" >&2
    exit 2
}
uid=$1 gid=$2
shift 2
exec unshare --user --map-user="$uid" --map-group="$gid" -- "$@"' off-card "$uid" "$gid" "$@"
