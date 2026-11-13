"""The two one-time steps before a card runs: fetching the RISC-V compiler the tree
names into the cache (`fetch_sfpi`), and priming a card's X280 memory (`prime`).

`python -m komorebi_runtime` and Komorebi's `kmr setup` call these.
"""

import hashlib
import os
import platform
import re
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import komorebi_runtime as kr

DEVICES = Path("/dev/tenstorrent")
# What the priming program prints: on a card it primed, and on one primed before (its
# open then returns a usable runtime, which the program asserts against).
PRIMED = "This chip now needs an ASIC reset"
ALREADY = "must not return a usable runtime"
# Opening the card compiles its firmware first; the priming itself takes seconds.
PRIME_TIMEOUT_S = 600


def sfpi_url(release, machine, distro):
    """(url, sha256) of the sfpi archive for `machine` and `distro`."""
    name = f"sfpi_{release['version']}_{machine}_{distro}.txz"
    url = f"{release['repo']}/releases/download/{release['version']}/{name}"
    return url, release[f"{machine}_{distro}_txz_hash"]


def host_distro():
    """The sfpi build for this host: "fedora" on RHEL-likes, else "debian"."""
    try:
        text = Path("/etc/os-release").read_text()
    except OSError:
        return "debian"
    return "fedora" if re.search(r"(?m)^ID(_LIKE)?=.*(fedora|rhel)", text) else "debian"


def fetch_sfpi():
    """Downloads and unpacks the sfpi release the tree names into the cache, unless it
    is there; returns its directory."""
    target = kr.sfpi_dir()
    if kr.sfpi_bin() is not None:
        return target
    url, sha256 = sfpi_url(kr.sfpi_release(), platform.machine(), host_distro())
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        archive = Path(tmp) / "sfpi.txz"
        digest = hashlib.sha256()
        with urllib.request.urlopen(url) as response, archive.open("wb") as out:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256:
            raise SystemExit(f"{url} does not match its recorded sha256")
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        (Path(tmp) / "sfpi").replace(target)
    return target


def default_card(environ=None):
    """The first card of TT_VISIBLE_DEVICES, else 0."""
    environ = os.environ if environ is None else environ
    first = environ.get("TT_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(first) if first.isdigit() else 0


def prime_environment(card, environ, base=None):
    """The priming program's environment: the runtime root of this package, the one
    card, the program's opt-in, slow dispatch unless set, and no TT_METAL_HOME."""
    env = {k: v for k, v in environ.items() if k != "TT_METAL_HOME"}
    env[kr.ROOT_VARIABLE] = str(kr.root(environ, base))
    env["TT_VISIBLE_DEVICES"] = str(card)
    env["X280_ALLOW_LIM_ECC_PRIME"] = "1"
    env.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    return env


def prime(card, sfpi_command, environ=None, base=None, devices=DEVICES, run=None):
    """Primes the X280 memory of `card` with the package's program; returns (lines,
    exit code). A card primed before is left as it is; `sfpi_command` is named when
    the compiler is missing."""
    environ = os.environ if environ is None else environ
    run = run or subprocess.run
    program = kr.prime_program(base)
    if not program.is_file():
        return [
            f"komorebi-runtime {kr.version(base)} has no priming program "
            "(0.81.1.3 and later have)"
        ], 1
    if not (devices / str(card)).exists():
        return [f"no card {card}: {devices / str(card)} does not exist"], 1
    if kr.sfpi_bin(environ, base) is None:
        return [f"the RISC-V compiler is not fetched: run `{sfpi_command}` first"], 1
    try:
        done = run(
            [str(program)],
            env=prime_environment(card, environ, base),
            capture_output=True,
            text=True,
            timeout=PRIME_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return [
            f"card {card}: the priming program did not end in {PRIME_TIMEOUT_S} s"
        ], 1
    output = done.stdout + done.stderr
    if done.returncode == 0 and PRIMED in output:
        return [
            f"card {card}: X280 memory primed (LIM ECC of every L2CPU tile)",
            f"reset the card once now: tt-smi -r {card} (the priming survives resets)",
        ], 0
    if ALREADY in output:
        return [f"card {card}: X280 memory already primed; nothing written"], 0
    lines = [line for line in output.splitlines() if line.strip()]
    keep = [line for line in lines if re.search(r"x280|L2CPU|LIM|rror", line)]
    return [
        f"card {card}: priming failed (exit status {done.returncode})",
        *(keep or lines)[-6:],
    ], 1
