"""Power sampling for benchmarks: whole-card power read from hwmon, a GPU from
nvidia-smi, the CPU package from an energy counter when the host lets a user read it.
`with PowerSampler(card) as s: ...` then `s.result` (and `s.result.phases` per mark).
Only reads sensors; opens no card and takes no lock.
"""

import contextlib
import os
import subprocess
import threading
import time
from bisect import bisect_left
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

SYS = Path("/sys")
NVIDIA_SMI = "nvidia-smi"
# The card's sensor changes at most every 0.1 s; sampling twice per update sees each.
DEFAULT_PERIOD_S = 0.05
# A section shorter than this many sensor updates reports no mean.
MIN_UPDATES = 5
IDLE_S = 1.0
# Consecutive failed reads before a sensor counts as lost (one bad read is not).
LOST_AFTER = 3
SMI_TIMEOUT_S = 5.0


class SensorLost(Exception):
    """A sensor could not be read (the card went away, the file is gone)."""


@dataclass(frozen=True)
class Summary:
    """Power over one stretch of time; the numbers are None when `too_short`."""

    seconds: float
    samples: int
    too_short: bool
    mean_w: float | None = None
    peak_w: float | None = None
    joules: float | None = None
    mean_temp_c: float | None = None
    limit_w: float | None = None
    idle_mean_w: float | None = None
    lost: bool = False
    note: str = ""
    phases: dict = field(default_factory=dict)

    def as_dict(self):
        """The summary as plain data (phases nested), for a JSON line."""
        return asdict(self)

    def format(self):
        """One line: time, mean (share of limit), peak, energy, idle, temperature."""
        head = f"{self.seconds:.2f} s, {self.samples} samples"
        if self.too_short:
            return f"{head}: {self.note}"
        text = f"{head}, mean {self.mean_w:.1f} W"
        if self.limit_w:
            text += (
                f" ({100 * self.mean_w / self.limit_w:.0f}% of {self.limit_w:.0f} W)"
            )
        text += f", peak {self.peak_w:.1f} W, {self.joules:.1f} J"
        if self.idle_mean_w is not None:
            text += f", idle {self.idle_mean_w:.1f} W"
        if self.mean_temp_c is not None:
            text += f", {self.mean_temp_c:.1f} C"
        return text + (f" ({self.note})" if self.note else "")


class Source:
    """A power sensor: `read()` gives (watts, temperature in C or None), or None when
    there is no value yet; it raises SensorLost when the sensor cannot be read."""

    name = "sensor"
    limit_w = None
    update_s = 0.1

    def read(self):
        raise NotImplementedError


def read_number(path):
    """The number in a sysfs file; SensorLost when it cannot be read."""
    try:
        return int(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise SensorLost(f"{path}: {error}") from error


class Card(Source):
    """One Tenstorrent card: its index, PCI address and hwmon directory."""

    update_s = 0.1

    def __init__(self, index, pci, hwmon, limit_w, update_s=None):
        self.index, self.pci, self.hwmon, self.limit_w = index, pci, hwmon, limit_w
        self.name = f"card {index} ({pci})"
        self.update_s = self.update_s if update_s is None else update_s

    def read(self):
        watts = read_number(self.hwmon / "power1_input") / 1e6
        try:
            temp = read_number(self.hwmon / "temp1_input") / 1e3
        except SensorLost:
            temp = None
        return watts, temp


def card_hwmon(device):
    """The hwmon directory named `blackhole` below the PCI device, else None."""
    for path in sorted((device / "hwmon").glob("hwmon*")):
        with contextlib.suppress(OSError):
            if (path / "name").read_text().strip() == "blackhole":
                return path
    return None


def cards(root=None):
    """The cards on this host by card index (the N of /dev/tenstorrent/N), found
    through the class entry's device link to the hwmon device of the same PCI device."""
    root = Path(SYS if root is None else root)
    found = []
    for entry in (root / "class" / "tenstorrent").glob("tenstorrent!*"):
        index = entry.name.partition("!")[2]
        device = (entry / "device").resolve()
        hwmon = card_hwmon(device)
        if index.isdigit() and hwmon is not None:
            limit = read_number(hwmon / "power1_max") / 1e6
            found.append(Card(int(index), device.name, hwmon, limit))
    return sorted(found, key=lambda card: card.index)


# A driver's JSON key for the clock its times were taken at (aiclk_mhz, clocked).
CLOCK_KEY = "aiclk_mhz"


def visible_card():
    """The card this process uses: the first of TT_VISIBLE_DEVICES, else 0."""
    return int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0] or 0)


def aiclk_mhz(card=None, root=None):
    """The Tensix clock (AICLK) of `card` (None: the first of TT_VISIBLE_DEVICES, else
    0) in MHz, from the driver's sysfs entry; None where it cannot be read. UMD raises
    it to the busy clock when a process opens the card and keeps it until that process
    exits; a reading before any open is the idle clock, not a session's."""
    if card is None:
        card = visible_card()
    path = Path(SYS if root is None else root) / "class" / "tenstorrent"
    try:
        return int(read_number(path / f"tenstorrent!{card}" / "tt_aiclk"))
    except SensorLost:
        return None


def clocked(result, card=None):
    """A driver's JSON result (a dict) with the clock its times were taken at under
    CLOCK_KEY, read now: call it in the process that ran the session."""
    if isinstance(result, dict) and CLOCK_KEY not in result:
        result[CLOCK_KEY] = aiclk_mhz(card)
    return result


class Gpu(Source):
    """One GPU, read through nvidia-smi (a process per sample, about 40 ms)."""

    update_s = 0.1

    def __init__(self, index, limit_w):
        self.index, self.limit_w = index, limit_w
        self.name = f"gpu {index}"

    def read(self):
        query = "--query-gpu=power.draw,temperature.gpu"
        out = smi([query, "--format=csv,noheader,nounits", "-i", str(self.index)])
        watts, _, temp = out.strip().partition(",")
        return number_or_none(watts), number_or_none(temp)


def number_or_none(text):
    """The float in `text`, or None for what nvidia-smi prints when it has none."""
    try:
        return float(text)
    except ValueError:
        return None


def smi(args):
    """The output of nvidia-smi with `args`; SensorLost when it does not run."""
    try:
        done = subprocess.run(
            [NVIDIA_SMI, *args],
            capture_output=True,
            text=True,
            timeout=SMI_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SensorLost(f"nvidia-smi: {error}") from error
    return done.stdout


def gpus():
    """The GPUs nvidia-smi lists with their power limits; empty when it is missing."""
    try:
        out = smi(["--query-gpu=index,power.limit", "--format=csv,noheader,nounits"])
    except SensorLost:
        return []
    found = []
    for line in out.splitlines():
        index, _, limit = line.partition(",")
        if index.strip().isdigit():
            found.append(Gpu(int(index), number_or_none(limit)))
    return found


class CpuPackage(Source):
    """The CPU package, from the energy counter of its powercap (RAPL) directory."""

    update_s = 0.1

    def __init__(self, path):
        self.path, self.name = Path(path), "cpu package"
        self.wrap = read_number(self.path / "max_energy_range_uj")
        self.last = None

    def read(self):
        energy, now = read_number(self.path / "energy_uj"), time.monotonic()
        last, self.last = self.last, (now, energy)
        if last is None or now <= last[0]:
            return None
        return ((energy - last[1]) % self.wrap) / 1e6 / (now - last[0]), None


def readable_package(path):
    """True for a powercap directory of a CPU package whose counter we may read."""
    try:
        named = (path / "name").read_text().startswith("package")
    except OSError:
        return False
    return (
        named
        and ":" not in path.name.partition(":")[2]
        and os.access(path / "energy_uj", os.R_OK)
    )


def cpu_package(root=None):
    """The first CPU package whose energy counter this user can read, else None."""
    root = Path(SYS if root is None else root)
    for path in sorted((root / "class" / "powercap").glob("intel-rapl:*")):
        if readable_package(path):
            return CpuPackage(path)
    return None


def value_at(points, at_time):
    """`points` (sorted (time, value)) interpolated at `at_time`, held constant beyond
    the ends."""
    times = [point[0] for point in points]
    i = bisect_left(times, at_time)
    if i == 0:
        return points[0][1]
    if i == len(points):
        return points[-1][1]
    (start, value_start), (end, value_end) = points[i - 1], points[i]
    return (
        value_end
        if end == start
        else value_start + (value_end - value_start) * (at_time - start) / (end - start)
    )


def integral(points, start, stop):
    """The trapezoid integral of `points` between `start` and `stop` (the ends
    interpolated)."""
    inner = [point for point in points if start < point[0] < stop]
    line = [(start, value_at(points, start)), *inner, (stop, value_at(points, stop))]
    return sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2
        for left, right in zip(line, line[1:])
    )


def too_short_note(covered, update_s, lost):
    """Why no numbers are reported, or "" when the stretch is long enough."""
    needed = MIN_UPDATES * update_s
    if covered >= needed:
        return ""
    base = f"shorter than {MIN_UPDATES} sensor updates ({needed:.2f} s)"
    return f"{base} and the sensor was lost" if lost else base


def summarize(samples, start, stop, update_s, limit_w=None, lost_at=None):
    """The Summary of `samples` ((time, watts, temperature)) between `start` and
    `stop`; when the sensor was lost after its last good sample at `lost_at`, only the
    time before it is covered."""
    end = stop if lost_at is None else min(stop, lost_at)
    covered = max(0.0, end - start)
    inside = [sample for sample in samples if start <= sample[0] <= end]
    lost = lost_at is not None and lost_at < stop
    note = too_short_note(covered, update_s, lost)
    if note or not inside:
        return Summary(
            stop - start, len(inside), True, limit_w=limit_w, lost=lost, note=note
        )
    power = [(sample[0], sample[1]) for sample in samples]
    temps = [(sample[0], sample[2]) for sample in samples if sample[2] is not None]
    joules = integral(power, start, end)
    temp = integral(temps, start, end) / covered if temps else None
    note = f"sensor lost after {covered:.2f} s" if lost else ""
    return Summary(
        stop - start,
        len(inside),
        False,
        mean_w=joules / covered,
        peak_w=max(sample[1] for sample in inside),
        joules=joules,
        mean_temp_c=temp,
        limit_w=limit_w,
        lost=lost,
        note=note,
    )


class PowerSampler:
    """Samples one source in a thread while the `with` block runs; on exit `result`
    is the block's Summary and `result.phases` has one per `mark(name)` (from the mark
    to the next mark or the end). `idle_s` seconds before the block give the idle mean.
    """

    def __init__(self, card, period_s=DEFAULT_PERIOD_S, idle_s=IDLE_S):
        self.source = self.resolve(card)
        self.period_s, self.idle_s = period_s, idle_s
        self.result = None
        self.samples, self.marks = [], []
        self.failures, self.last_good = 0, 0.0
        self.start = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    @staticmethod
    def resolve(card):
        """The source for a card index, or the source itself."""
        if isinstance(card, int):
            for found in cards():
                if found.index == card:
                    return found
            raise LookupError(f"no Tenstorrent card {card} with a power sensor")
        return card

    def take(self):
        """One sample; LOST_AFTER failed reads in a row end the sampling."""
        now = time.perf_counter()
        try:
            value = self.source.read()
        except SensorLost:
            with self.lock:
                self.failures += 1
            if self.failures >= LOST_AFTER:
                self.stop.set()
            return
        with self.lock:
            self.failures = 0
            if value is not None:
                self.samples.append((now, value[0], value[1]))
                self.last_good = now

    def run(self):
        """The sampling thread: one sample per period on a fixed schedule."""
        due = time.perf_counter() + self.period_s
        while not self.stop.wait(max(0.0, due - time.perf_counter())):
            self.take()
            due = max(due + self.period_s, time.perf_counter())

    def latest(self):
        """The last sample (t, watts, temperature), or None."""
        with self.lock:
            return self.samples[-1] if self.samples else None

    def mark(self, name):
        """Starts the phase `name`; it ends at the next mark or the end of the block."""
        if self.start is None or self.result is not None:
            raise RuntimeError("mark() is for use inside the with block")
        with self.lock:
            if any(name == known for _, known in self.marks):
                raise ValueError(f"phase {name!r} is already marked")
            self.marks.append((time.perf_counter(), name))

    def __enter__(self):
        self.take()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        self.stop.wait(self.idle_s)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        end = time.perf_counter()
        self.stop.set()
        self.thread.join()
        self.take()
        self.result = self.summary(end, self.last_good if self.failures else None)
        return False

    def summary(self, end, lost_at):
        """The section's Summary with its phases and the idle mean."""
        source, samples = self.source, self.samples
        whole = summarize(
            samples, self.start, end, source.update_s, source.limit_w, lost_at
        )
        idle = None
        if self.idle_s > 0:
            before = summarize(
                samples,
                self.start - self.idle_s,
                self.start,
                source.update_s,
                None,
                lost_at,
            )
            idle = before.mean_w
        bounds = [mark_time for mark_time, _ in self.marks[1:]] + [end]
        phases = {
            name: summarize(
                samples, mark_time, stop, source.update_s, source.limit_w, lost_at
            )
            for (mark_time, name), stop in zip(self.marks, bounds)
        }
        return replace(whole, idle_mean_w=idle, phases=phases)


def select(card, gpu, cpu):
    """(sources, notes): what the command line asks for; every card when nothing is."""
    chosen, notes = [], []
    if card is not None or not (gpu is not None or cpu):
        found = [source for source in cards() if card in (None, source.index)]
        chosen += found
        if not found:
            notes.append("no Tenstorrent card with a power sensor found")
    if gpu is not None:
        found = [source for source in gpus() if source.index == gpu]
        chosen += found
        if not found:
            notes.append(f"gpu {gpu}: not available (nvidia-smi)")
    if cpu:
        package = cpu_package()
        chosen += [package] if package else []
        if not package:
            notes.append("cpu package: not available (no readable energy counter)")
    return chosen, notes


def run_cli(args, out):
    """`kmr power`: prints each second's reading, then each source's summary."""
    sources, notes = select(args.card, args.gpu, args.cpu)
    for note in notes:
        print(note, file=out)
    if not sources:
        return 1
    with contextlib.ExitStack() as stack:
        samplers = [
            stack.enter_context(PowerSampler(source, args.period, idle_s=0))
            for source in sources
        ]
        begin = time.perf_counter()
        while (elapsed := time.perf_counter() - begin) < args.seconds:
            time.sleep(min(1.0, args.seconds - elapsed))
            for sampler in samplers:
                last = sampler.latest()
                if last:
                    temp = "" if last[2] is None else f", {last[2]:.1f} C"
                    print(f"{sampler.source.name}: {last[1]:.1f} W{temp}", file=out)
    for sampler in samplers:
        print(f"{sampler.source.name}: {sampler.result.format()}", file=out)
    return 0
