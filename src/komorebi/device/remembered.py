"""The kernel buffer a script's loads needed, remembered in the build cache so that its
next run opens the card at that size instead of opening it again at its first load:
by program (its kernels' sources, defines, compile args) and by script (docs/reference/
host.md#the-kernel-buffer)."""

import contextlib
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

from komorebi import _env
from komorebi.device import chips, kernel_buffer, mock, native

FILE = "kernel_buffer.json"
LOCK = f"{FILE}.lock"  # beside the file: held across every read-modify-write
HEADROOM_LEAST = 8 << 10  # a session that cannot reopen opens with an eighth more


def headroom(size):
    """The bytes a session that cannot reopen (islands, a mesh) opens with beyond a
    remembered `size`: an eighth of it, at least HEADROOM_LEAST, 64-byte aligned."""
    return -(-max(size // 8, HEADROOM_LEAST) // 64) * 64


def chip_shape(shape, arch):
    """A session's `shape` on a chip of `arch`: Blackhole's (and an unread one's) as
    it was, another's with its name (a script's kernels' sizes differ by chip)."""
    return shape if arch in ("blackhole", None) else f"{shape} {arch}"


def dispatcher_shape(shape, arch, asked):
    """A session's `shape` under the dispatcher `asked` (named by the session or the
    environment; None: the chip's own): another than the chip's default (X280 on a
    chip with one, else Tensix) is named in it, the default leaves the key as it was:
    the two kinds of session have different kernels in the buffer, an entry each."""
    default = "x280" if chips.chip(arch or "blackhole").x280 else "tensix"
    return shape if not asked or asked == default else f"{shape} {asked}"


def script_key(shape):
    """The running script: its resolved path, the session's `shape` ("card", "islands
    2") and the code's digest; None for code with no file (python -c, a notebook) and
    for a runner installed in Python's prefix (pytest), all its runs one entry."""
    path = getattr(sys.modules.get("__main__"), "__file__", None)
    if path is None:
        return None
    path = Path(path).resolve()
    prefixes = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    if any(path.is_relative_to(prefix) for prefix in prefixes):
        return None
    return f"{path}|{shape}|{code_digest()}"


def code_digest():
    """The package's Python and native headers' digest: a script's entry measured with
    other code (other kernels' sizes) is not read, and is no entry."""
    digest = hashlib.sha256(_env.code_key().encode())
    native = _env.include_directory()
    for one in sorted(native.rglob("*")) if native is not None else ():
        if one.is_file():
            digest.update(str(one.relative_to(native)).encode() + one.read_bytes())
    return digest.hexdigest()[:12]


def program_key(program, shape):
    """A program's key: its kernels' source bytes, defines and compile args, the
    package's native kernels, the session's `shape`."""
    if not callable(getattr(program, "sources", None)):  # a list or plan
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
    real = native.cache_directory() / FILE
    return mock.memory(real) or real


_unread = []  # the file was found unreadable once (logged once)


def read(log=None):
    """{"programs": {key: bytes}, "scripts": {key: bytes}, "prebuilt": [keys a prebuild
    seeded]}, and "writer_levels" {key: level} and "writer_near" {key: [writer bytes,
    region bytes]} when the file has them: no entry when the file is missing, corrupt
    or half written."""
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
    out = {
        "programs": made.get("programs", {}),
        "scripts": made.get("scripts", {}),
        "prebuilt": made.get("prebuilt", []),
    }
    for kept in ("writer_levels", "writer_near"):
        if kept in made:
            out[kept] = made[kept]
    return out


@contextlib.contextmanager
def locked():
    """The file's lock (an exclusive flock on LOCK beside it) held for a block that
    reads, changes and writes it: two runs closing at once keep both their entries.
    A prebuild's copy and a folder that cannot be written take no lock (write() says
    the second); readers take none, a write renames a whole file over the old."""
    path = _path().with_name(LOCK)
    handle = None
    if not mock.active() or mock.memory(native.cache_directory() / FILE) is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a")
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError:
            handle = None
    try:
        yield
    finally:
        if handle is not None:
            handle.close()  # closing releases the lock


def measured(memory, *keys):
    """`memory` with `keys` measured by this run (a card's run: no longer seeded)."""
    if not mock.active():
        memory["prebuilt"] = [key for key in memory["prebuilt"] if key not in keys]
    return memory


def seed(own, log=print):
    """The sizes a prebuild's loads measured (its copy `own`) written into the card's
    file, marked as a prebuild's: a prebuilt script's first card run opens right. An
    entry the card's file has at that size or larger stays (never lowered)."""
    if not own.is_file():
        return
    made = json.loads(own.read_text())
    with locked():
        real, changed = read(), []
        for kind in ("programs", "scripts", "writer_levels"):
            for key, size in made.get(kind, {}).items():
                if size > real.setdefault(kind, {}).get(key, 0):
                    real[kind][key] = size
                    changed.append(key)
        if changed:
            real["prebuilt"] = sorted(set(real["prebuilt"]) | set(changed))
            write(real)
    if changed:
        log(f"kmr {mock.COMMAND}: {len(changed)} kernel buffer size(s) seeded")


def write(memory):
    """Writes the maps (a whole new file, renamed over the old one); a prebuild
    writes its own copy (mock.memory), never the card's file. A file that cannot be
    written (a read-only cache folder) is said on stderr and nothing more: the
    memory never stops a load or a close."""
    if mock.active() and mock.memory(native.cache_directory() / FILE) is None:
        return
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{FILE}.{os.getpid()}")
        temporary.write_text(json.dumps(memory, indent=1, sort_keys=True))
        temporary.replace(path)
    except OSError as error:
        print(
            f"komorebi: {path}: not written ({error}): the kernel buffer this run "
            "needed is not remembered",
            file=sys.stderr,
        )


def writer_level(program, arch):
    """The writer level a build of `program` on a chip of `arch` took last (0: none
    kept): the level its next load builds at first (kernels.writer_level)."""
    key = program_key(program, f"writer {arch}")
    return read().get("writer_levels", {}).get(key, 0) if key is not None else 0


def keep_writer_level(program, arch, level):
    """`level`, the lowest writer level at which `program`'s writer fit, kept for its
    next load (a program the key does not name keeps none)."""
    key = program_key(program, f"writer {arch}")
    if key is None:
        return
    with locked():
        memory = read()
        levels = memory.setdefault("writer_levels", {})
        if levels.get(key, 0) != level:
            levels[key] = level
            write(memory)


def writer_near(program, arch):
    """The far build's (writer bytes, region bytes) when the last build of `program` on
    a chip of `arch` was without the far paths (the near build), else None: its next
    load builds near at first (a far build over at every level is not built again)."""
    key = program_key(program, f"writer {arch}")
    found = read().get("writer_near", {}).get(key) if key is not None else None
    return tuple(found) if found else None


def keep_writer_near(program, arch, far):
    """`far`, the far build's (writer bytes, region bytes) of a near build of
    `program`, or None (built with the far paths), kept for its next load."""
    key = program_key(program, f"writer {arch}")
    if key is None:
        return
    with locked():
        memory = read()
        near = memory.setdefault("writer_near", {})
        if near.get(key) != (list(far) if far else None):
            if far:
                near[key] = list(far)
            else:
                near.pop(key, None)
            write(memory)


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
            seeded = self.script in memory["prebuilt"]
            why = (
                f"{'seeded by a prebuild' if seeded else 'remembered'} for this script"
            )
            why = why if self.size else None
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
        key = program_key(program, self.shape)
        with locked():
            memory = read()
            if key is not None:
                memory["programs"][key] = need
            if self.script is not None:
                memory["scripts"][self.script] = max(
                    memory["scripts"].get(self.script, 0), need
                )
            write(measured(memory, key, self.script))

    def wanted(self, total, smaller=False):
        """A load refused for kernels of `total` bytes (the session cannot follow):
        the script's entry grown to it (`smaller`: set to it, the L1 table failed),
        so that the next run opens right."""
        if self.script is None:
            return
        need = kernel_buffer.need(total)
        with locked():
            memory = read()
            before = memory["scripts"].get(self.script, 0)
            memory["scripts"][self.script] = need if smaller else max(before, need)
            write(measured(memory, self.script))
        self.largest = need if smaller else max(self.largest, need)

    def closed(self):
        """At the session's close: the script's entry grown to this run's largest, never
        lowered but by a session with headroom whose largest is under three quarters
        of what it opened with (its entry is then that largest)."""
        if self.script is None or not self.largest:
            return
        lowers = self.room and self.opened and self.largest * 4 < self.opened * 3
        with locked():
            memory = read()
            before = memory["scripts"].get(self.script, 0)
            memory["scripts"][self.script] = (
                self.largest if lowers else max(before, self.largest)
            )
            write(measured(memory, self.script))
