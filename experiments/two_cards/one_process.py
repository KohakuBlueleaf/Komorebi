"""Two sessions in one process: opens a Session on card 0, then one on card 1, with
both cards visible, and prints one JSON line saying which opened and what the second
raised. Run with TT_VISIBLE_DEVICES=0,1. See README.md.
"""

import json
import os
import time

import komorebi as km

STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def main():
    report = {"visible_before": os.environ.get("TT_VISIBLE_DEVICES")}
    with km.Session(card=0, log=log) as first:
        log(f"card 0 open: grid {first.grid}")
        report["visible_after_card_0"] = os.environ.get("TT_VISIBLE_DEVICES")
        second = km.Session(card=1, log=log)
        try:
            second.open()
            log(f"card 1 open: grid {second.grid}")
            report["card_1"] = "opened"
        except Exception as error:  # noqa: BLE001 (the result is what it raised)
            report["card_1"] = f"{type(error).__name__}: {error}"
            log(f"card 1 not opened: {report['card_1']}")
        finally:
            second.close()
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
