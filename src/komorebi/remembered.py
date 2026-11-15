"""The kernel buffer a script's loads needed, remembered in the build cache so that its
next run opens the card at that size instead of opening it again at its first load:
by program (its kernels' sources, defines, compile args) and by script (docs/reference/
host.md#the-kernel-buffer)."""

import hashlib
import json
import os
import sys
from pathlib import Path

from komorebi import _env, kernel_buffer, native
from komorebi.program import Program

FILE = "kernel_buffer.json"
HEADROOM_LEAST = 8 << 10  # a session that cannot reopen opens with an eighth more


def headroom(size):
    """The bytes a session that cannot reopen (islands, a mesh) opens with beyond a
    remembered `size`: an eighth of it, at least HEADROOM_LEAST, 64-byte aligned."""
    return -(-max(size // 8, HEADROOM_LEAST) // 64) * 64


def script_key(shape):
    """The running script: its resolved path and the session's `shape` ("card",
    "islands 2"); None for code with no file (python -c, a notebook) and for a runner
    installed in Python's prefix (pytest), whose entry every run of it would share."""
    path = getattr(sys.modules.get("__main__"), "__file__", None)
    if path is None:
        return None
    path = Path(path).resolve()
    prefixes = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    if any(path.is_relative_to(prefix) for prefix in prefixes):
        return None
    return f"{path}|{shape}"


def program_key(program, shape):
    """A program's key: its kernels' source bytes, defines and compile args, the
    package's native kernels, the session's `shape`."""
    if not isinstance(program, Program):  # a list or plan: the script's entry alone
        return None
    digest = hashlib.sha256(f"{program.name}|{shape}".encode())
    try:
        sources = program.sources()
    except Exception:  # noqa: BLE001  (no sources of its own)
        return None
    for kernel, path in sorted(sources.items(), key=lambda item: str(item[0])):
        digest.update(f"{kernel}|".encode() + Path(path).read_bytes())
    digest.update(repr(sorted(program.defines.items())).encode())
    digest.update(repr(getattr(program, "compile_args", None)).encode())
    digest.update(_env.code_key().encode())
    return digest.hexdigest()[:24]


def _path():
    return native.cache_directory() / FILE


_unread = []  # the file was found unreadable once (logged once)


def read(log=None):
    """{"programs": {key: bytes}, "scripts": {key: bytes}}: no entry when the file is
    missing, corrupt or half written (two cards' processes write it at once)."""
    try:
        made = json.loads(_path().read_text())
        made = made if isinstance(made, dict) else {}
    except FileNotFoundError:
        made = {}
    except (OSError, ValueError) as error:
        if log is not None and not _unread:
            _unread.append(error)
            log(f"{_path()}: unreadable ({error}): no kernel buffer remembered")
        made = {}
    return {"programs": made.get("programs", {}), "scripts": made.get("scripts", {})}


def write(memory):
    """Writes the maps (a whole new file, renamed over the old one)."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{FILE}.{os.getpid()}")
    temporary.write_text(json.dumps(memory, indent=1, sort_keys=True))
    temporary.replace(path)


class Memory:
    """A session's remembered kernel buffer: the size it opens at, chosen from
    `expect` (a made Program) or the script's entry, and what its loads need;
    `default` names the size it opens at without one; `room`: it opens with headroom
    and its entry shrinks only below three quarters of that (it cannot reopen)."""

    def __init__(
        self, log, expect=None, shape="card", default="the runtime's", room=False
    ):
        self.log, self.shape, self.room = log, shape, room
        self.opened = None  # the size it opens at: the entry, with headroom if `room`
        self.script = script_key(shape)
        self.largest = 0  # the largest buffer a load of this run needed
        memory = read(log)
        self.size, why = None, None
        if expect is not None:
            key = program_key(expect, shape)
            self.size = memory["programs"].get(key)
            why = f"expect={expect.name}" if self.size else None
        if self.size is None and self.script is not None:
            self.size = memory["scripts"].get(self.script)
            why = "remembered for this script" if self.size else None
        if self.size is not None:
            self.size = kernel_buffer.need(self.size - kernel_buffer.MARGIN)
            extra = headroom(self.size) if room else 0
            self.opened = self.size + extra
            more = f": {self.size:,} B and {extra:,} B of headroom" if extra else ""
            log(f"kernel buffer {self.opened:,} B at the open ({why}{more})")
        elif self.script is None:
            log(
                f"kernel buffer: {default} at the open (no script file, or a runner "
                "such as pytest: none kept)"
            )
        else:
            log(f"kernel buffer: {default} at the open (nothing remembered)")

    def loaded(self, program, total):
        """A load of `program` whose kernels took `total` bytes: its program's entry,
        and the script's grown to it."""
        need = kernel_buffer.need(total)
        self.largest = max(self.largest, need)
        memory = read()
        key = program_key(program, self.shape)
        if key is not None:
            memory["programs"][key] = need
        if self.script is not None:
            memory["scripts"][self.script] = max(
                memory["scripts"].get(self.script, 0), need
            )
        write(memory)

    def wanted(self, total, smaller=False):
        """A load refused for kernels of `total` bytes (the session cannot follow):
        the script's entry grown to it (`smaller`: set to it, the L1 table failed),
        so that the next run opens right."""
        if self.script is None:
            return
        memory = read()
        need = kernel_buffer.need(total)
        before = memory["scripts"].get(self.script, 0)
        memory["scripts"][self.script] = need if smaller else max(before, need)
        write(memory)
        self.largest = need if smaller else max(self.largest, need)

    def closed(self):
        """At the session's close: the script's entry is this run's largest; with
        headroom, only once that is under three quarters of what it opened with."""
        if self.script is None or not self.largest:
            return
        memory = read()
        before = memory["scripts"].get(self.script, 0)
        keep = self.room and self.opened and self.largest * 4 >= self.opened * 3
        memory["scripts"][self.script] = (
            max(before, self.largest) if keep else self.largest
        )
        write(memory)
