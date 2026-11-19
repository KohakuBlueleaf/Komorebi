"""SERVICE_SLOT_MOST against the service overlay's reader part this chip builds, with or
without KM_CHECK_NOC_SOURCES: every part fits it; with --largest, it is the largest
part in whole KiB. kmr prebuild [--arch A] service_slot.py [--largest] (a prebuild
passes a script's exit after a load: a failed check raises)"""

import sys
from pathlib import Path

import komorebi as km
from komorebi.program import overlay

ROOT = Path(__file__).resolve().parents[3]
KERNELS = ROOT / "tutorial" / "43_overlays" / "kernels"
KIB = 1024


def service_part(session, defines):
    """The service overlay's slot (its reader part, whole lines) of a load built with
    `defines`."""
    program = km.Program(
        "service_slot",
        kernels=KERNELS,
        handlers=KERNELS / "handlers.h",
        defines=defines,
    )
    program.overlay("add", KERNELS / "add")
    session.load(program).stop()
    return program.overlays_built.service


def main():
    most = overlay.SERVICE_SLOT_MOST
    with km.Session() as session:
        parts = {
            name: service_part(session, defines)
            for name, defines in (
                ("plain", {}),
                ("checked", {"KM_CHECK_NOC_SOURCES": 1}),
            )
        }
    print(f"service reader part: {parts}, SERVICE_SLOT_MOST {most:,} B")
    over = {name: size for name, size in parts.items() if size > most}
    if over:
        raise RuntimeError(f"over SERVICE_SLOT_MOST ({most:,} B): {over}")
    largest = max(parts.values())
    if "--largest" in sys.argv and most != -(-largest // KIB) * KIB:
        raise RuntimeError(f"SERVICE_SLOT_MOST {most:,} B is not {largest:,} B in KiB")


main()
