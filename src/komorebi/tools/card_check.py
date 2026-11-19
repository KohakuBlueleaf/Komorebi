"""The fast card check: the examples' own exactness checks (card_checks.txt), card 0
opened once for its list, then a process for two cards; a line a check, the total.
Usage: python -m komorebi.tools.card_check [--prebuild] [--cards 1|2|tensix|both]
Run --prebuild first: the check refuses a list whose second prebuild compiled anything.
"""

import argparse
import concurrent.futures
import contextlib
import json
import os
import re
import runpy
import shlex
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from komorebi.bench import power
from komorebi.device import chips, kernel_buffer, mock, native, remembered
from komorebi.islands import session as island_session
from komorebi.program import overlay
from komorebi.session import dispatch, sizes
from komorebi.session import kernels as kernel_module
from komorebi.session import session as card_session
from komorebi.tools.child_env import child_environment

LIST = Path(__file__).with_name("card_checks.txt")
LOGS = Path("build") / "card_check"  # under the root: a log a check, the prebuild stamp
STAMP = "prebuilt.json"
SHAPES = {"1": (0,), "2": (0, 1), "tensix": (1,)}
ALONE = ("1", "tensix")  # a card opened once for the shape's list
DISPATCHERS = {"tensix": dispatch.TENSIX}  # a shape's dispatcher (else the default)
JSON_KEYS = ("ok", "exact")  # and every key ending in "_exact"
BUILT = re.compile(r"built: (\d+) found, (\d+) compiled")
PREBUILD_CODES = (0, 75, 76, 77)  # kmr prebuild's own exits (builder/prebuild.py)
KERNELS = re.compile(r"kernel buffer: (\d+) of (\d+) B")
AGAIN = "kmr prebuild: again"  # its second pass, at the size its first one kept


@dataclass(frozen=True)
class Check:
    """One line of the list: the cards it opens ("1" or "2"), the script from the
    root and its arguments."""

    cards: str
    script: str
    args: tuple

    @property
    def name(self):
        words = [Path(self.script).parent.name, Path(self.script).stem, *self.args]
        words = [self.cards, *words] if self.cards in DISPATCHERS else words
        return "_".join(words).replace("/", "-").replace(",", "-")[:120]

    @property
    def line(self):
        return shlex.join([self.cards, self.script, *self.args])


def read_list(path, cards=None):
    """The checks of `path`, those opening `cards` ("1", "2") or all."""
    checks = []
    for number, text in enumerate(Path(path).read_text().splitlines(), 1):
        words = shlex.split(text, comments=True)
        if not words:
            continue
        if words[0] not in SHAPES or len(words) < 2:
            raise SystemExit(f"{path}:{number}: '1' or '2', then a script: {text!r}")
        if cards is None or words[0] == cards:
            checks.append(Check(words[0], words[1], tuple(words[2:])))
    return checks


def shape_environment(cards, **extra):
    """A child's environment for a check of `cards`: its shape's dispatcher named."""
    env = child_environment(os.environ, **extra)
    if cards in DISPATCHERS:
        env[dispatch.VARIABLE] = DISPATCHERS[cards]
    return env


def stamp_path(root):
    return Path(root) / LOGS / STAMP


def prebuild_one(check, root, log_dir):
    """One check's `kmr prebuild` (once more when its own passes did not run twice):
    (check, the second pass's compiled count, the kernel buffer it opens, its exit,
    seconds, the first pass's compiled count)."""
    started = time.perf_counter()
    cards = ",".join(str(card) for card in SHAPES[check.cards])
    script = (Path(root) / check.script).resolve()
    command = [sys.executable, "-m", "komorebi.cli", "prebuild", "--cards", cards]
    command += [str(script), *check.args]
    outputs = []
    while True:
        done = subprocess.run(
            command,
            cwd=script.parent,
            env=shape_environment(check.cards),
            capture_output=True,
            text=True,
            check=False,
        )
        outputs.append(done.stdout + done.stderr)
        last = outputs[-1].split(AGAIN)[-1]  # its own second pass, or this run
        if AGAIN in outputs[-1] or len(outputs) > 1:
            break
    (log_dir / f"prebuild-{check.name}.log").write_text("\n".join(outputs))
    kernels = [
        max(int(opened), kernel_buffer.need(int(total)) + headroom)
        for total, opened in KERNELS.findall(last)
        for headroom in [remembered.headroom(kernel_buffer.need(int(total)))]
    ]
    seconds = time.perf_counter() - started
    first = compiled_in(outputs[0].split(AGAIN)[0])
    code = done.returncode
    return check, compiled_in(last), max(kernels, default=0), code, seconds, first


def prebuild(checks, root, log_dir, jobs, list_path):
    """Every check's prebuild, `jobs` at a time; the second pass must compile nothing.
    Writes the stamp the card run reads; returns the number of checks that failed."""
    failed, kept, cold, started = 0, read_stamp(root), 0, time.perf_counter()
    stamp = {"code": remembered.code_digest(), "checks": kept.get("checks", {})}
    for check in checks:
        stamp["checks"].pop(check.line, None)
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        futures = [pool.submit(prebuild_one, c, root, log_dir) for c in checks]
        for future in concurrent.futures.as_completed(futures):
            check, compiled, kernels, code, seconds, first = future.result()
            cold += first
            ok = compiled == 0 and code in PREBUILD_CODES
            failed += not ok
            if ok:
                stamp["checks"][check.line] = kernels
            print(
                f"{'PASS' if ok else 'FAIL'} {seconds:6.1f} s  prebuild {check.line}: "
                f"{compiled} compiled at the second pass, exit {code}",
                flush=True,
            )
    stamp_path(root).write_text(json.dumps(stamp, indent=1))
    stamp["shared"] = kept.get("shared", {})  # a shape's lines whose shared open built
    shapes = sorted({check.cards for check in checks})
    with concurrent.futures.ThreadPoolExecutor(len(shapes) or 1) as pool:
        passes = pool.map(lambda c: shared_pass(c, root, log_dir, list_path), shapes)
        for cards, (passed, first) in zip(shapes, passes, strict=True):
            lines = [check.line for check in checks if check.cards == cards]
            stamp["shared"][cards] = lines if passed else []
            failed += not passed
            cold += first
    stamp_path(root).write_text(json.dumps(stamp, indent=1))
    seconds = time.perf_counter() - started
    if cold:
        print(f"cold build: {cold} kernels, {seconds:.1f} s (not counted)")
    else:
        print(f"warm prebuild: {seconds:.1f} s, 0 compiled")
    return failed


def shared_pass(cards, root, log_dir, list_path):
    """The card run of `cards` on the mock, once more when the first compiled: its loads
    at the shared buffer's addresses. (True when the last built none, the first's
    count)."""
    env = shape_environment(cards, **{mock.ACTIVE: "1"})
    env[mock.MOCK] = str(mock.descriptor(SHAPES[cards]))
    own, real = log_dir / f"mock-memory-{cards}.json", native.cache_directory()
    real = real / remembered.FILE
    own.write_text(real.read_text() if real.is_file() else "{}")
    env[mock.MEMORY] = str(own)  # the card's remembered sizes: its opens are alike
    command = [sys.executable, "-m", "komorebi.tools.card_check", "--cards", cards]
    command += ["--root", str(root), "--list", str(list_path), "--mock-pass"]
    first, started, run = None, time.perf_counter(), 0
    while run < 2:
        run += 1
        done = subprocess.run(
            command, env=env, capture_output=True, text=True, check=False
        )
        lines = [ln for ln in done.stdout.splitlines() if ln.startswith("BUILT")]
        built = [line for line in lines if not line.startswith("BUILT 0 ")]
        if first is None:
            first = sum(int(line.split()[1]) for line in built)
        # each attempt's own file: a second pass never hides how the first ended
        attempt = log_dir / f"mock-pass-{cards}-{run}.log"
        attempt.write_text(done.stdout + done.stderr)
        if not built:
            break
    ok = bool(lines) and not built
    print(
        f"{'PASS' if ok else 'FAIL'} {time.perf_counter() - started:6.1f} s  the "
        f"shared open of cards {SHAPES[cards]} on the mock, pass {run}: "
        f"{len(built)} of {len(lines)} checks compiled"
        + ("" if lines else f"; {child_ended(done)} (log {attempt})"),
        flush=True,
    )
    for line in built:
        print(f"  {line}")
    return ok, first


def child_ended(done, tail=5):
    """How a child that printed no BUILT line ended: its return code (a signal's name
    when the code is negative) and the last `tail` lines of its stderr."""
    code = done.returncode
    if code < 0:
        try:
            how = f"killed by {signal.Signals(-code).name}"
        except ValueError:
            how = f"killed by signal {-code}"
    else:
        how = f"exit {code}"
    said = [line for line in done.stderr.splitlines() if line.strip()][-tail:]
    return how + ("; its stderr ended: " + " | ".join(said) if said else "; no stderr")


def compiled_in(text):
    """The kernels and parts a log's loads compiled ("built: N found, M compiled")."""
    return sum(int(compiled) for _, compiled in BUILT.findall(text))


def read_stamp(root):
    """The last prebuild's stamp ({"checks": {line: kernel buffer}, "shared": {cards:
    the lines of the list built at its shared open}}) if made with this code; else {}.
    """
    try:
        stamp = json.loads(stamp_path(root).read_text())
    except (OSError, ValueError):
        return {}
    return stamp if stamp.get("code") == remembered.code_digest() else {}


def open_size(checks, built):
    """The largest kernel buffer the checks' prebuilds opened, or their kernels and a
    mesh's headroom, so that one open serves every check; None: the runtime's."""
    largest = max((built[check.line] for check in checks), default=0)
    return largest if largest > sizes.KERNEL_BUFFER_DEFAULT else None


class Watch:
    """The process's opens, reopens and compiles while the checks run: the card's
    device or mesh is opened once and handed to every later session."""

    def __init__(self):
        self.compiled = 0
        self.refused = []  # a load that wanted the card opened at another buffer
        self.shared = None  # card 0's open device
        self.clock = None  # AICLK read at the last island session's close
        self.busy_mhz = None  # its cards' busy clock (chips), below it: throttled
        self.closing = False
        self.saved = []  # (object, attribute, value) to put back at the end

    def patch(self, owner, name, value):
        self.saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def restore(self):
        for owner, name, value in reversed(self.saved):
            setattr(owner, name, value)

    def built_files(self, original):
        def counted(recipes, started):
            found, compiled = original(recipes, started)
            self.compiled += compiled + overlay.PARTS_BUILT["compiled"]
            return found, compiled

        return counted

    def factory(self, original):
        """The module's Device or Islands: the first call opens, later ones get it (a
        load's kernels are checked against its real buffer, not the size asked)."""

        def made(*args):
            if self.shared is None:
                self.shared = original(*args)
            return self.shared

        return made

    def clocked(self, close):
        """An island session's close that reads its cards' AICLK first (while open)."""

        def closing(session):
            self.clock = clocks(session.cards)
            arch = session.sessions[0].device.arch()
            self.busy_mhz = chips.chip(arch).busy_mhz
            return close(session)

        return closing

    def reopened(self, *args, **kwargs):
        """A load that would open the card again at another kernel buffer: refused."""
        self.refused.append(args[1:2])
        raise RuntimeError(
            "card_check: a load asked to open the card again at another kernel "
            "buffer; the list's open size is too small for it"
        )

    def closed_once(self, original):
        """A close of the shared device or mesh is a no-op until the run ends."""

        def close(handle):
            if self.closing:
                original(handle)

        return close


def install(watch, module, shared):
    """The process's hooks: the native module's Device and Islands, the build count;
    put back by watch.restore()."""
    counted = watch.built_files(kernel_module.built_files)
    watch.patch(kernel_module, "built_files", counted)
    if shared:  # card 0: one device for every session; a mesh opens a check
        watch.patch(module, "Device", watch.factory(module.Device))
    watch.patch(kernel_buffer, "reopen", watch.reopened)
    sessions = island_session.IslandSession
    watch.patch(sessions, "_cards_reopen", watch.reopened)
    watch.patch(sessions, "close", watch.clocked(sessions.close))


def run_one(check, root, log_dir, watch):
    """Runs one check as `python script args` would, in this process; (passed, why,
    seconds, kernels and parts compiled). Its output (the runtime's too) goes to its
    log file."""
    script = (Path(root) / check.script).resolve()
    log_path = log_dir / f"{check.name}.log"
    before_modules, environ = set(sys.modules), dict(os.environ)
    saved = sys.argv, list(sys.path), os.getcwd()
    compiled, refused = watch.compiled, len(watch.refused)
    code, error = 0, None
    sys.stdout.flush()
    sys.stderr.flush()
    with open(log_path, "w") as log, redirected(log.fileno()):
        sys.argv = [str(script), *check.args]
        sys.path[:0] = [str(script.parent), str(Path(root).resolve())]
        os.chdir(script.parent)
        started = time.perf_counter()
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as stop:
            code = (
                stop.code if isinstance(stop.code, int) else int(stop.code is not None)
            )
        except BaseException as caught:  # noqa: BLE001 (a check's failure, logged)
            error = f"{type(caught).__name__}: {caught}"
            traceback.print_exc()
        seconds = time.perf_counter() - started
        sys.stdout.flush()
        sys.stderr.flush()
    sys.argv, sys.path[:] = saved[0], saved[1]
    os.chdir(saved[2])
    os.environ.clear()
    os.environ.update(environ)
    for name in set(sys.modules) - before_modules:
        if Path(getattr(sys.modules[name], "__file__", None) or "/").is_relative_to(
            script.parent
        ):
            del sys.modules[name]
    text = log_path.read_text(errors="replace")
    built = max(watch.compiled - compiled, compiled_in(text))
    why = f"{built} compiled" if built else verdict(code, error, text)
    if len(watch.refused) > refused or "opened again" in text:
        why = why or "the card was opened again (another kernel buffer)"
    return why is None, why, seconds, built


def verdict(code, error, text):
    """Why a check failed (its exit, an exception, a JSON key not true), or None."""
    if error is not None:
        return error
    if code:
        return f"exit {code}"
    lines = [line for line in text.splitlines() if line.startswith("{")]
    if not lines:
        return None
    try:
        out = json.loads(lines[-1])
    except ValueError:
        return None
    false = [key for key, value in flat(out) if not holds(value)]
    return f"not true: {', '.join(false)}" if false else None


def flat(out, prefix=""):
    """The (key path, value) pairs of a result whose key is ok, exact or *_exact."""
    pairs = []
    if isinstance(out, dict):
        for key, value in out.items():
            name = f"{prefix}{key}"
            if key in JSON_KEYS or str(key).endswith("_exact"):
                pairs.append((name, value))
            elif isinstance(value, (dict, list)):
                pairs += flat(value, f"{name}.")
    elif isinstance(out, list):
        for index, value in enumerate(out):
            pairs += flat(value, f"{prefix}{index}.")
    return pairs


def holds(value):
    """True when every leaf of `value` is true or nonzero (an exact count, a dict of
    exact flags)."""
    if isinstance(value, dict):
        return all(holds(item) for item in value.values())
    if isinstance(value, list):
        return all(holds(item) for item in value)
    return bool(value)


@contextlib.contextmanager
def redirected(fd):
    """File descriptors 1 and 2 (Python's and the runtime's output) to `fd`."""
    saved = os.dup(1), os.dup(2)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    try:
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])


def clocks(cards):
    """AICLK of each card in MHz, read now (while the card is open)."""
    return [power.aiclk_mhz(card) for card in cards]


def card_run(cards, root, log_dir, list_path, mock_pass=False):
    """The checks opening `cards` in this process: card 0 opened once at the list's
    largest kernel buffer, a mesh by each check; returns the number that failed.
    `mock_pass`: on the mock, a line a check with what its loads compiled."""
    stamp = read_stamp(root)
    checks, built = read_list(list_path, cards), stamp.get("checks", {})
    missing = [check.line for check in checks if check.line not in built]
    if (
        not missing
        and not mock_pass
        and stamp.get("shared", {}).get(cards) != [check.line for check in checks]
    ):
        missing = [f"the shared open of cards {SHAPES[cards]} on the mock"]
    if missing:
        print(
            f"REFUSED: not prebuilt (run --prebuild): {len(missing)} check(s), first:"
        )
        print(f"  {missing[0]}")
        return len(missing)
    shape, holder, opened = SHAPES[cards], None, 0.0
    module = native.load_module()
    watch = Watch()
    install(watch, module, shared=cards in ALONE)
    if cards in DISPATCHERS:  # every session of the checks takes it
        os.environ[dispatch.VARIABLE] = DISPATCHERS[cards]
    if cards in ALONE:  # its card opened once at the list's largest kernel buffer
        size = open_size(checks, built)
        holder = card_session.Session(card=shape[0], kernel_bytes=size)
        started = time.perf_counter()
        with open(log_dir / f"open-{cards}.log", "w") as log, redirected(log.fileno()):
            holder.open()
        opened = time.perf_counter() - started
        kind = type(watch.shared)
        watch.patch(kind, "close", watch.closed_once(kind.close))
        kernel = "the runtime's" if size is None else f"{size:,} B"
        said = f"kernel buffer {kernel}, AICLK {clocks(shape)}"
        print(f"open card {shape[0]}: {opened:.1f} s, {said}")
    failed, total, low = 0, opened, []
    for check in checks:
        watch.clock = None
        passed, why, seconds, built = run_one(check, root, log_dir, watch)
        total += seconds
        if mock_pass:  # the mock's reads give zeros: only the builds count
            print(f"BUILT {built} compiled {seconds:5.1f} s  {check.line}", flush=True)
            failed += built > 0
            continue
        clock = clocks(shape) if holder is not None else watch.clock
        busy = watch.busy_mhz
        if holder is not None:
            busy = chips.chip(holder.device.arch()).busy_mhz
        low += [mhz for mhz in clock or () if mhz is not None and mhz < (busy or 0)]
        failed += not passed
        said = "" if passed else f"  ({why}; log {log_dir / (check.name + '.log')})"
        print(
            f"{'PASS' if passed else 'FAIL'} {seconds:6.1f} s  {check.line}{said}  "
            f"AICLK {clock}"
        )
    watch.closing = True
    if holder is not None:
        with open(log_dir / f"close-{cards}.log", "w") as log, redirected(log.fileno()):
            holder.close()
    watch.restore()
    throttled = f", throttled: AICLK {min(low)} MHz" if low else ""
    print(
        f"cards {shape}: {len(checks) - failed} of {len(checks)} passed, "
        f"{total:.1f} s{throttled}"
    )
    return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", choices=("1", "2", "tensix", "both"), default="both")
    parser.add_argument("--root", default=".", help="the repository's root")
    parser.add_argument("--list", default=str(LIST), help="the checks' list")
    parser.add_argument("--prebuild", action="store_true", help="build, no card")
    parser.add_argument("--jobs", type=int, default=48, help="prebuilds at a time")
    parser.add_argument("--mock-pass", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    log_dir = root / LOGS
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"logs: {log_dir}")
    if args.prebuild:
        cards = None if args.cards == "both" else args.cards
        checks = read_list(args.list, cards)
        return 1 if prebuild(checks, root, log_dir, args.jobs, args.list) else 0
    if args.cards != "both":
        failed = card_run(args.cards, root, log_dir, args.list, args.mock_pass)
        return 1 if failed else 0
    started, code = time.perf_counter(), 0
    for cards in ("1", "2"):  # a process each: a card alone and a mesh never share one
        command = [sys.executable, "-m", "komorebi.tools.card_check", "--cards", cards]
        command += ["--root", str(root), "--list", args.list]
        env = child_environment(os.environ)
        code |= subprocess.run(command, env=env, check=False).returncode
    print(
        f"both: {time.perf_counter() - started:.1f} s, {'FAILED' if code else 'passed'}"
    )
    return 1 if code else 0


if __name__ == "__main__":
    sys.exit(main())
