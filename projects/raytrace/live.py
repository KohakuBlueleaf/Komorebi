"""The chained frame as a loop a host drives: the scene, the program and the frame's
chain made once; for each camera the constants page is written and the same chain
command submitted. The colour stream is read back, or with `--show` the 8-bit image
the cores write. Prints one JSON line; exits 1 unless `ok`."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from komorebi.bench import power  # noqa: E402
from projects.raster.card_thread import SHOW  # noqa: E402
from projects.raster.live import figures, show_table, table_check, untile  # noqa: E402
from projects.raytrace import card, frame, framegraph, packets  # noqa: E402
from reference import png, raytrace  # noqa: E402

F = np.float32
SIDE = packets.WIDTH
DEADLINE_S = 5.0  # per command, as frame.py's run_chain
SAME = ("kShowPageBytes", "kShowTableWords", "kShowAlpha", "kShowBucketBase")
SAME += ("kShowOne", "kShowBucketShift")  # projects/raster's show.h


def frame_arguments(args):
    """The arguments frame.prepare and frame.build read: the whole frame, tiles dealt
    by packet count, no plan."""
    return argparse.Namespace(
        scene=args.scene,
        treelets=args.treelets,
        sort=args.sort,
        tiles=None,
        balance=None,
        plan=None,
        slots=4,
    )


def show_table_checked():
    """SHOW's thresholds (projects/raster's table), after checking that the 8-bit
    rule's constants are raster's and the table gives png's bytes."""
    differing = [name for name in SAME if packets.K[name] != SHOW[name]]
    if differing or packets.K["kShowWidth"] != packets.WIDTH:
        raise ValueError(f"SHOW's constants differ from projects/raster's: {differing}")
    if packets.K["kShowBlockW"] != packets.BLOCK_W:
        raise ValueError("SHOW's packet width differs from packets.BLOCK_W")
    table = show_table()
    return table, table_check(table)


def show_record(built, display, table):
    """SHOW's record: every core's tiles from the colour stream into the display."""
    words = [0] * 10
    words[0] = built["directory_tensors"][0].address
    words[1] = built["tensors"]["colour"].address
    words[2], words[3] = display.address, table.address
    words[5] = built["tensors"]["desc"].address
    return packets.K["kHandlerShow"], words


def assemble(pages, packet_count, flat_pixels, image):
    """The colour stream's pages into the float image (pixels, 3), as frame.py puts
    them for its check."""
    rows = packets.ungroup(pages.view(np.uint32), frame.vectors("colour"), packet_count)
    image[flat_pixels] = frame.lanes_of(rows, "colour").T.view(F)
    return image


class Loop:
    """The frame built in `session`, the program loaded, LOAD run and the frame's
    chain made: a chain command a camera (`frame`). With `table` SHOW ends the chain
    and a frame reads the 8-bit display, else the colour stream. `report` gets the
    build's entries and times, `events` LOAD's event."""

    def __init__(self, session, arguments, prepared, table, report):
        self.session = session
        self.show = table is not None
        started = time.perf_counter()
        self.built = built = frame.build(arguments, session, prepared, report)
        self.workers, self.pixels = built["workers"], built["pixels"]
        self.packet_count = built["packet_count"]
        self.constants_tensor = built["constants_tensor"]
        self.colour_tensor = built["tensors"]["colour"]
        frame_commands = list(built["frame_commands"])
        sizes = built["sizes"]
        if self.show:
            self.display = card.tensor(
                session, np.zeros((packets.TILES, packets.K["kShowPageBytes"] // 4))
            )
            self.table_tensor = card.tensor(session, table.reshape(1, -1))
            frame_commands.append(show_record(built, self.display, self.table_tensor))
            sizes = dict(sizes or {}, show=packets.K["kShowScratchBytes"])
            self.display_pages = np.empty(
                (self.display.pages, self.display.page_bytes), np.uint8
            )
        session.device.finish()
        report["upload_s"] = round(time.perf_counter() - started, 3)
        self.pages = np.empty(
            (self.colour_tensor.pages, self.colour_tensor.page_bytes), np.uint8
        )
        started = time.perf_counter()
        self.run = run = session.load(
            card.program(len(prepared["image"]), sizes), deadline=DEADLINE_S
        )
        self.events = []
        try:
            load = built["records"][0]
            self.events.append(
                run.wait(run.submit(load[0], args=load[1]), timeout=10.0)
            )
            self.chain = card.frame_chain(run, frame_commands, self.workers)
        except BaseException:
            run.stop()
            raise
        report["load_s"] = round(time.perf_counter() - started, 3)
        report["chain_records"] = len(frame_commands)
        self.chain_records = len(frame_commands)

    def frame(self, view, out):
        """The chain command for `view` (a scene dict with its camera): with SHOW the
        8-bit display into `out` (SIDE, SIDE, 4), else the colour stream's pages read
        (`self.pages`). Returns the event and the times (s) began, constants written,
        commanded, read."""
        constants = packets.frame_constants(view).view(np.uint8)
        began = time.perf_counter()
        self.session.write(self.constants_tensor, constants)
        written = time.perf_counter()
        event = self.run.wait(self.run.submit(self.chain), timeout=10.0 + DEADLINE_S)
        commanded = time.perf_counter()
        if self.show:
            self.session.read(self.display, out=self.display_pages)
            read = time.perf_counter()
            untile(self.display_pages, out)
        else:
            self.session.read(self.colour_tensor, out=self.pages)
            read = time.perf_counter()
        return event, (began, written, commanded, read)

    def stop(self):
        """Stops the program; returns frame.py's Ran of the events so far."""
        self.run.stop()
        return card.Ran(self.events, self.run.result(), self.run.state(), 0.0)


def loop(args):
    arguments = frame_arguments(args)
    prepared = frame.prepare(arguments)
    scene, tree = prepared["scene"], prepared["tree"]
    views = framegraph.cameras(scene, args.frames)
    checked = sorted({0, args.frames // 2, args.frames - 1}) if args.check else []
    report = {
        "scene": args.scene,
        "frames": args.frames,
        "treelets_kib": args.treelets,
        "sort": args.sort,
        "path": "show" if args.show else "read",
    }
    table = None
    if args.show:
        table, report["table_differs"] = show_table_checked()
    kept = {}
    host, constants_ms, command_ms, read_ms, assembly_ms = [], [], [], [], []
    convert_ms, card_ms = [], []
    with card.open_session() as session:
        live = Loop(session, arguments, prepared, table, report)
        workers, pixels = live.workers, live.pixels
        packet_count, colour_tensor = live.packet_count, live.colour_tensor
        pages, events = live.pages, live.events
        if args.show:
            shown = np.empty((SIDE, SIDE, 4), dtype=np.uint8)
        flat_pixels = pixels.reshape(-1)
        image = np.zeros((SIDE * SIDE, 3), dtype=F)
        try:
            card.log(
                f"live: {packet_count} packets, a chain of "
                f"{live.chain_records}, {args.frames} cameras"
            )
            for index, view in enumerate(views):
                event, times = live.frame(view, shown if args.show else None)
                began, written, commanded, read = times
                if args.show:
                    assembled = converted = time.perf_counter()
                else:
                    assemble(pages, packet_count, flat_pixels, image)
                    assembled = time.perf_counter()
                    eight = png.to_srgb8(image.reshape(SIDE, SIDE, 3))
                    converted = time.perf_counter()
                events.append(event)
                host.append((assembled - began) * 1e3)
                constants_ms.append((written - began) * 1e3)
                command_ms.append((commanded - written) * 1e3)
                read_ms.append((read - commanded) * 1e3)
                assembly_ms.append((assembled - read) * 1e3)
                convert_ms.append((converted - assembled) * 1e3)
                card_ms.append(event.microseconds / 1e3)
                if index in checked and args.show:  # the colour too, after timing
                    session.read(colour_tensor, out=pages)
                    assemble(pages, packet_count, flat_pixels, image)
                    kept[index] = (image.copy(), shown[..., :3].copy())
                    report.setdefault("alpha_not_255", 0)
                    report["alpha_not_255"] += int((shown[..., 3] != 255).sum())
                elif index in checked:
                    kept[index] = (image.copy(), eight)
                if event.kind != "done":
                    card.log(f"frame {index}: {event.kind}")
                    break
        finally:
            ran = live.stop()
        status = card.check_run(ran, workers)
    report["frames_run"] = len(events) - 1
    report["host_ms"] = figures(host)
    report["fps"] = round(1000.0 / report["host_ms"]["median"], 2)
    report["constants_write_ms"] = figures(constants_ms)
    report["command_ms"] = figures(command_ms)
    report["read_ms"] = figures(read_ms)
    report["assembly_ms"] = figures(assembly_ms)
    if not args.show:
        report["convert_8bit_ms"] = figures(convert_ms)
    report["card_frame_ms"] = round(float(np.median(card_ms)), 4)
    report["cameras_differ"] = 0
    checks = {}
    for index in checked:
        if index not in kept:
            continue
        float_image, eight = kept[index]
        colour = float_image[flat_pixels]
        if index == 0:  # the scene's own camera: frame.py's stored reference
            card.log(f"check frame {index}: the scene's reference frame")
            against, want = frame.reference_frame(args.scene)
            want = want["image"].reshape(-1, 3)[flat_pixels]
        else:
            card.log(f"check frame {index}: numpy renders that camera's frame")
            against = "numpy"
            want = raytrace.render(views[index], tree, flat_pixels)["image"]
        card.log(f"check frame {index}: comparing (ties classified)")
        check = frame.image_check(
            views[index], tree, flat_pixels, colour, want, against
        )
        checks[index] = {
            key: check[key] for key in ("against", "differs", "ties", "first_pixels")
        }
        checks[index]["first_pixels"] = checks[index]["first_pixels"][:8]
        if args.show:  # the card's 8-bit image: png's rule on its colour; ties apart
            shown = eight.reshape(-1, 3)[flat_pixels]
            ties = (packets.bits(colour) != packets.bits(want)).any(axis=1)
            differ = (shown != png.to_srgb8(want)).any(axis=1)
            checks[index]["differs_8bit"] = int(differ.sum())
            checks[index]["differs_8bit_not_ties"] = int((differ & ~ties).sum())
            checks[index]["rule_differs"] = int(
                (shown != png.to_srgb8(colour)).any(axis=1).sum()
            )
        if args.pictures:
            folder = Path(args.pictures)
            folder.mkdir(parents=True, exist_ok=True)
            png.write_png(folder / f"{args.scene}_{index:04d}.png", eight)
    if len(kept) > 1:
        first, last = kept[min(kept)][0], kept[max(kept)][0]
        report["cameras_differ"] = int(frame.differ(first, last) > 0)
    report["checks"] = checks
    ok = card.ran_cleanly(status, workers) and report["frames_run"] == args.frames
    ok &= len(checks) == len(checked)
    ok &= all(check["differs"] == 0 for check in checks.values())
    if args.show:
        ok &= report["table_differs"] == 0 and report.get("alpha_not_255", 0) == 0
        ok &= all(
            check["differs_8bit_not_ties"] == 0 and check["rule_differs"] == 0
            for check in checks.values()
        )
    if len(checked) > 1:
        ok &= report["cameras_differ"] == 1
    report["ok"] = bool(ok)
    return report, status


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument(
        "--treelets", type=int, default=0, help="KiB: treelets from DRAM"
    )
    parser.add_argument("--sort", action="store_true", help="sort rays on each core")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument(
        "--show", action="store_true", help="SHOW on the card: the 8-bit display read"
    )
    parser.add_argument(
        "--check", action="store_true", help="frames 0, N/2, N-1 against numpy"
    )
    parser.add_argument("--pictures", default=None, help="PNG of the checked frames")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    if args.sort and not args.treelets:
        parser.error("--sort needs --treelets")
    report, status = loop(args)
    print(json.dumps(power.clocked(report)), flush=True)
    if args.json:
        with open(args.json, "w") as file:
            json.dump({**report, "status": status}, file, indent=1, default=str)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
