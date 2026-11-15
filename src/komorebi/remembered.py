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

FILE = "kernel_buffer.json"


def script_key(shape):
    """The running script: its resolved path and the session's `shape` ("card",
    "islands 2"), or None for code with no file (python -c, a notebook)."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    return None if path is None else f"{Path(path).resolve()}|{shape}"


def program_key(program, shape):
    """A program's key: its kernels' source bytes, defines and compile args, the
    package's native kernels, the session's `shape`."""
    digest = hashlib.sha256(f"{program.name}|{shape}".encode())
    try:
        sources = program.sources()
    except Exception:  # noqa: BLE001  (a plan or a list: no sources of its own)
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
    `default` names the size it opens at without one."""

    def __init__(self, log, expect=None, shape="card", default="the runtime's"):
        self.log, self.shape = log, shape
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
            log(f"kernel buffer {self.size:,} B at the open ({why})")
        elif self.script is None:
            log(f"kernel buffer: {default} at the open (no script file: none kept)")
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
        """At the session's close: the script's entry is this run's largest."""
        if self.script is None or not self.largest:
            return
        memory = read()
        memory["scripts"][self.script] = self.largest
        write(memory)
