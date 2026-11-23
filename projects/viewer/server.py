"""A web viewer: the card renders, a browser steers. The main thread owns the card's
session and renders a frame for the latest pose the page posts; the frames go out as
a multipart JPEG stream, and every request carries the token. Needs Pillow.
python -m projects.viewer.server --host HOST --port PORT --token TOKEN [--size WxH]"""

import argparse
import collections
import functools
import hmac
import io
import json
import math
import signal
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from projects.viewer import renderers
from reference import lights as light_data

HERE = Path(__file__).resolve().parent
SCENES = (
    "tiny",
    "sponza_overview",
    "bistro_interior_overview",
    "bistro_exterior_overview",
    "breakfast_room_overview",
    "salle_de_bain_overview",
    "modern_hall_overview",
    "modern_living_room_overview",
)
OFFERS = renderers.offers(SCENES)  # {renderer: the scenes it is offered on}
KEEP_ALIVE_S = 1.0  # a frame a second when no pose comes
JPEG_QUALITY = 85
BOUNDARY = "frame"
TOKEN_HEADER = "X-Token"
PITCH_LIMIT = math.radians(89.0)
EXPOSURE_RANGE = (0.25, 8.0)  # the tone's exposure slider (logarithmic)
SATURATION_RANGE = (0.0, 2.0)
SIZE = (1280, 720)  # the frame's size at the start (--size)


def log(message):
    """A progress line, flushed."""
    print(f"[viewer {time.strftime('%H:%M:%S')}] {message}", flush=True)


# The camera: 16 numbers as reference/scenes.py lays them out: eye (0:3), right (3:6),
# up (6:9), forward (9:12), focal, centre x, centre y, near (12:16); y is up.


def pose_of(camera):
    """The pose of a camera: position, yaw (about y, from +z toward +x) and pitch
    (above the horizon), from its forward direction."""
    camera = np.asarray(camera, dtype=np.float64)
    fx, fy, fz = camera[9:12] / np.linalg.norm(camera[9:12])
    return {
        "position": [float(value) for value in camera[0:3]],
        "yaw": math.atan2(fx, fz),
        "pitch": math.asin(max(-1.0, min(1.0, fy))),
    }


def camera_of(own, pose):
    """The 16 numbers of `pose`: position and the orientation from yaw and pitch (right
    = y x forward, up = forward x right, as reference/scenes.py makes them); focal,
    centre and near the scene's own camera's."""
    yaw, pitch = pose["yaw"], pose["pitch"]
    forward = np.array(
        [
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
            math.cos(pitch) * math.cos(yaw),
        ]
    )
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    camera = np.array(own, dtype=np.float32)
    camera[0:3] = pose["position"]
    camera[3:6], camera[6:9], camera[9:12] = right, up, forward
    return camera


def parse_pose(body):
    """A pose from the page's JSON, its numbers finite and the pitch clamped; else
    ValueError."""
    position = [float(value) for value in body["position"]]
    yaw, pitch = float(body["yaw"]), float(body["pitch"])
    if len(position) != 3 or not all(map(math.isfinite, position + [yaw, pitch])):
        raise ValueError("a pose is 3 finite numbers, a yaw and a pitch")
    pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, pitch))
    return {"position": position, "yaw": yaw, "pitch": pitch}


def parse_sun(body):
    """A sun from the page's JSON: azimuth and elevation in degrees, finite, the
    elevation clamped to [-90, 90]; else ValueError."""
    azimuth, elevation = float(body["azimuth"]), float(body["elevation"])
    if not (math.isfinite(azimuth) and math.isfinite(elevation)):
        raise ValueError("a sun is a finite azimuth and elevation")
    return azimuth, max(-90.0, min(90.0, elevation))


def parse_tone(body):
    """A tone from the page's JSON: exposure and saturation finite, clamped to
    EXPOSURE_RANGE and SATURATION_RANGE, curve 0 (plain) or 1 (filmic); else
    ValueError."""
    exposure, saturation = float(body["exposure"]), float(body["saturation"])
    curve = body["curve"]
    if not (math.isfinite(exposure) and math.isfinite(saturation)):
        raise ValueError("a tone is a finite exposure and saturation")
    if curve not in (0, 1) or isinstance(curve, bool):
        raise ValueError("a tone curve is 0 (plain) or 1 (filmic)")
    exposure = max(EXPOSURE_RANGE[0], min(EXPOSURE_RANGE[1], exposure))
    saturation = max(SATURATION_RANGE[0], min(SATURATION_RANGE[1], saturation))
    return exposure, saturation, int(curve)


def parse_indirect(body):
    """The indirect switch from the page's JSON: `on`, true or false; else
    ValueError."""
    on = body["on"]
    if not isinstance(on, bool):
        raise ValueError("the indirect switch is true or false")
    return on


def parse_bounces(body):
    """The bounce passes from the page's JSON: `bounces`, a whole number 0 to 3
    (lights.bounce_count); else ValueError with the reason."""
    return light_data.bounce_count(body["bounces"])


def parse_size(body):
    """A frame size from the page's JSON: width and height, integers on the card's grid
    (renderers.check_size); else ValueError with the reason."""
    width, height = body["width"], body["height"]
    if not all(isinstance(n, int) and not isinstance(n, bool) for n in (width, height)):
        raise ValueError("a size is a whole width and height in pixels")
    return renderers.check_size((width, height))


def diagonal(scene):
    """The diagonal of the scene's box."""
    positions = np.asarray(scene["positions"], dtype=np.float64)
    return float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))


def jpeg(image):
    """The image's colour (no alpha) as JPEG bytes."""
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image[..., :3])).save(
        buffer, "JPEG", quality=JPEG_QUALITY
    )
    return buffer.getvalue()


class Viewer:
    """The shared state and the render loop (run on the main thread). `lock` guards
    everything; `changed` is notified on a new pose, a selection or the stop; `framed`
    on a new frame."""

    def __init__(self, renderer, scene, size=SIZE):
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.framed = threading.Condition(self.lock)
        self.selected = (renderer, scene)
        self.selection = 1  # serial of /select requests (the first: the command line)
        self.pose = None
        self.pose_serial = 0
        self.stopping = False
        self.status = "loading"
        self.loaded = (None, None)
        self.scene_serial = 0  # a new scene: the page takes the new first pose
        self.first_pose = None
        self.first_scene = None  # the scene first_pose is of
        self.diagonal = 1.0
        self.jpeg = None
        self.frame_number = 0
        self.card_ms = None
        self.card_ms_label = None
        self.frame_times = collections.deque()
        # the sun (raster-lit): the newest the page posted, not yet applied; its serial;
        # the sun the frames have, the scene's own, the last SUN pass's ms
        self.sun_pending = None
        self.sun_serial = 0
        self.sun = None
        self.sun_own = None
        self.sun_ms = None
        # the tone (raster-lit, raster-gi) and the indirect switch (raster-gi), as the
        # sun: the newest posted, its serial, what the frames have; the scene's tone
        self.tone_pending = None
        self.tone_serial = 0
        self.tone = None
        self.tone_own = None
        self.indirect_pending = None
        self.indirect_serial = 0
        self.indirect = None
        # the bounce passes (raster-gi), as the switch; the last field update's seconds
        self.bounces_pending = None
        self.bounces_serial = 0
        self.bounces = None
        self.field_s = None
        # the frame's size: asked (a load takes it), posted not applied and its serial,
        # the frames', the last change's ms, the last refusal's reason (memory) or None
        self.size_asked = size
        self.size_pending = None
        self.size_serial = 0
        self.size = None
        self.size_ms = None
        self.size_refused = None

    # Requests

    def set_pose(self, pose, scene_serial):
        """A pose from the page, for the scene it was made in."""
        with self.lock:
            if scene_serial != self.scene_serial:
                return False
            self.pose = pose
            self.pose_serial += 1
            self.changed.notify_all()
            return True

    def set_sun(self, sun, scene_serial):
        """A sun from the page, for the scene it was posted in: the newest waits."""
        with self.lock:
            if scene_serial != self.scene_serial or self.sun_own is None:
                return False
            self.sun_pending = sun
            self.sun_serial += 1
            self.changed.notify_all()
            return True

    def set_tone(self, tone, scene_serial):
        """A tone from the page, for the scene it was posted in: the newest waits."""
        with self.lock:
            if scene_serial != self.scene_serial or self.tone_own is None:
                return False
            self.tone_pending = tone
            self.tone_serial += 1
            self.changed.notify_all()
            return True

    def set_indirect(self, on, scene_serial):
        """The indirect switch from the page, for the scene it was posted in."""
        with self.lock:
            if scene_serial != self.scene_serial or self.indirect is None:
                return False
            self.indirect_pending = on
            self.indirect_serial += 1
            self.changed.notify_all()
            return True

    def set_bounces(self, bounces, scene_serial):
        """The bounce passes from the page (parse_bounces'), for the scene it was posted
        in: the newest waits; ValueError with the reason when the renderer has no
        indirect light."""
        with self.lock:
            if self.loaded[0] is None:
                raise ValueError("no renderer is loaded yet")
            if self.bounces is None:
                raise ValueError(f"{self.loaded[0]} has no indirect light")
            if scene_serial != self.scene_serial:
                return False
            self.bounces_pending = bounces
            self.bounces_serial += 1
            self.changed.notify_all()
            return True

    def set_size(self, size):
        """A frame size from the page (parse_size's): asked from now on (a load takes
        it), applied between frames; False, and not asked, when the loaded renderer
        renders one size only."""
        with self.lock:
            if (
                self.loaded[0] is not None
                and not renderers.RENDERERS[self.loaded[0]].sizes
            ):
                return False
            self.size_asked = size
            self.size_pending = size
            self.size_serial += 1
            self.changed.notify_all()
            return True

    def select(self, renderer, scene):
        with self.lock:
            self.selected = (renderer, scene)
            self.selection += 1
            self.changed.notify_all()

    def stop(self):
        with self.lock:
            self.stopping = True
            self.changed.notify_all()
            self.framed.notify_all()

    def state(self):
        with self.lock:
            now = time.monotonic()
            while self.frame_times and now - self.frame_times[0] > 1.0:
                self.frame_times.popleft()
            return {
                "renderer": self.loaded[0],
                "scene": self.loaded[1],
                "selected": list(self.selected),
                "status": self.status,
                "card_ms": None if self.card_ms is None else round(self.card_ms, 3),
                "card_ms_label": self.card_ms_label,
                "fps": len(self.frame_times),
                "frame": self.frame_number,
                "renderers": list(renderers.RENDERERS),
                "scenes": list(SCENES),
                "offers": OFFERS,
                "diagonal": self.diagonal,
                "scene_serial": self.scene_serial,
                "first_pose": self.first_pose,
                "sun": None if self.sun is None else list(self.sun),
                "sun_own": None if self.sun_own is None else list(self.sun_own),
                "sun_ms": None if self.sun_ms is None else round(self.sun_ms, 3),
                "tone": None if self.tone is None else list(self.tone),
                "tone_own": None if self.tone_own is None else list(self.tone_own),
                "indirect": self.indirect,
                "bounces": self.bounces,
                "field_s": None if self.field_s is None else round(self.field_s, 3),
                "size": None if self.size is None else list(self.size),
                "size_asked": list(self.size_asked),
                "size_ms": None if self.size_ms is None else round(self.size_ms, 3),
                "size_refused": self.size_refused,
                "sizes": (
                    None
                    if self.loaded[0] is None
                    else renderers.RENDERERS[self.loaded[0]].sizes
                ),
            }

    def next_jpeg(self, after, timeout):
        """The newest frame's (number, JPEG) once its number is past `after`, else
        (after, None) after `timeout` or at the stop."""
        with self.lock:
            self.framed.wait_for(
                lambda: self.stopping or self.frame_number > after, timeout
            )
            if self.frame_number > after and self.jpeg is not None:
                return self.frame_number, self.jpeg
            return after, None

    # The render thread

    def render_loop(self):
        renderer = None
        failed = 0  # the selection that failed: wait for a new one
        # (selection, pose, sun, tone, switch, size, passes serials) of the last frame
        rendered = None
        last_frame = 0.0
        try:
            while True:
                with self.lock:
                    due = functools.partial(self.due, failed, rendered, last_frame)
                    self.changed.wait_for(due, timeout=0.1)
                    if self.stopping:
                        return
                    if not due():
                        continue
                    selection, selected = self.selection, self.selected
                try:
                    if renderer is None or selected != self.loaded:
                        renderer = self.open(renderer, selected)
                    with self.lock:
                        pose_serial, pose = self.pose_serial, self.pose
                        own = pose == self.first_pose
                        sun_serial, sun = self.sun_serial, self.sun_pending
                        self.sun_pending = None
                        tone_serial, tone = self.tone_serial, self.tone_pending
                        self.tone_pending = None
                        switch_serial = self.indirect_serial
                        switch, self.indirect_pending = self.indirect_pending, None
                        bounces_serial = self.bounces_serial
                        bounces, self.bounces_pending = self.bounces_pending, None
                        size_serial = self.size_serial
                        size, self.size_pending = self.size_pending, None
                    refused = None
                    if size is not None and renderer.sizes and size != renderer.size:
                        # the frame's size, between frames: the grid word, or past the
                        # capacity a reload; refused for memory, the size as it was
                        try:
                            size_ms = renderer.set_size(size)
                        except renderers.SizeRefused as error:
                            with self.lock:
                                self.size_refused = str(error)
                                self.size_asked = renderer.size
                        else:
                            with self.lock:
                                self.size, self.size_ms = renderer.size, size_ms
                                self.size_refused = None
                    if sun is not None and renderer.sun_own is not None:
                        try:  # the SUN pass, between frames
                            sun_ms = renderer.set_sun(*sun)
                        except renderers.FrameRefused as error:
                            refused = f"refused: SUN {error.word}"
                        else:
                            with self.lock:
                                self.sun, self.sun_ms = renderer.sun, sun_ms
                    if tone is not None and renderer.tone_own is not None:
                        renderer.set_tone(*tone)  # LIGHT's table, between frames
                        with self.lock:
                            self.tone = renderer.tone
                    if switch is not None and renderer.indirect is not None:
                        renderer.set_indirect(switch)
                        with self.lock:
                            self.indirect = renderer.indirect
                    if bounces is not None and renderer.bounces is not None:
                        renderer.set_bounces(bounces)  # the field, between frames
                    if renderer.bounces is not None:  # its passes; a sun's update too
                        with self.lock:
                            self.bounces = renderer.bounces
                            self.field_s = renderer.field_s()
                    camera = (
                        renderer.scene["camera"]
                        if own
                        else camera_of(renderer.scene["camera"], pose)
                    )
                    try:
                        image, card_ms = renderer.frame(camera)
                    except renderers.FrameRefused as error:
                        with self.lock:
                            self.status = f"refused: {error.word}"
                    else:
                        encoded = jpeg(image)
                        with self.lock:
                            self.jpeg = encoded
                            self.card_ms = card_ms
                            self.card_ms_label = renderer.time_label
                            self.frame_number += 1
                            self.frame_times.append(time.monotonic())
                            self.status = refused or "running"
                            self.framed.notify_all()
                    rendered = (
                        selection,
                        pose_serial,
                        sun_serial,
                        tone_serial,
                        switch_serial,
                        size_serial,
                        bounces_serial,
                    )
                    last_frame = time.monotonic()
                except Exception as error:  # noqa: BLE001 - shown on the page, closed
                    traceback.print_exc()
                    with self.lock:
                        self.status = f"error: {type(error).__name__}: {error}"
                        self.loaded = (None, None)
                    failed = selection
                    renderer = self.close(renderer)
        finally:
            self.close(renderer)

    def due(self, failed, rendered, last_frame):
        """Under the lock: the stop, or a frame to render (a selection that has not
        failed, and a new selection or pose, or the keep-alive)."""
        if self.stopping:
            return True
        if self.selection == failed:
            return False
        serials = (
            self.selection,
            self.pose_serial,
            self.sun_serial,
            self.tone_serial,
            self.indirect_serial,
            self.size_serial,
            self.bounces_serial,
        )
        return serials != rendered or time.monotonic() - last_frame >= KEEP_ALIVE_S

    def open(self, renderer, selected):
        """Closes the current renderer and opens the selected one on its scene; the
        pose becomes the scene's own camera's when the scene is new."""
        renderer = self.close(renderer)
        with self.lock:
            self.status = "loading"
            self.loaded = (None, None)
            self.sun = self.sun_own = self.sun_ms = self.sun_pending = None
            self.tone = self.tone_own = self.tone_pending = None
            self.indirect = self.indirect_pending = None
            self.bounces = self.bounces_pending = self.field_s = None
            self.size = self.size_ms = self.size_pending = self.size_refused = None
            size = self.size_asked
        name, scene = selected
        log(f"open {name} on {scene}")
        renderer = renderers.RENDERERS[name]()
        try:
            renderer.open(scene, size)
        except BaseException:
            renderer.close()
            raise
        with self.lock:
            if scene != self.first_scene:
                self.first_pose = pose_of(renderer.scene["camera"])
                self.pose = self.first_pose
                self.diagonal = diagonal(renderer.scene)
                self.scene_serial += 1
            self.loaded = selected
            self.first_scene = scene
            self.sun = self.sun_own = renderer.sun_own  # a new load: the scene's sun
            self.tone = self.tone_own = renderer.tone_own  # and its tone
            self.indirect = renderer.indirect  # raster-gi: on
            self.bounces = renderer.bounces  # raster-gi: the scene's passes
            self.size = renderer.size  # the size asked, or the renderer's only one
        log(f"opened {name} on {scene} at {renderer.size[0]} x {renderer.size[1]}")
        return renderer

    def close(self, renderer):
        """Closes a renderer (its card session); returns None."""
        if renderer is not None:
            log(f"close {renderer.name}")
            try:
                renderer.close()
            except Exception:  # noqa: BLE001 - the close of a failed renderer
                traceback.print_exc()
        return None


MOST_BODY_BYTES = 1 << 16  # a pose or a selection is far smaller


def handler_class(viewer, token, page):
    """The request handler, bound to the viewer, the token and the page's bytes."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # quiet: 30 poses a second
            pass

        def allowed(self, given):
            if given is not None and hmac.compare_digest(given, token):
                return True
            self.reply(403, b"forbidden\n", "text/plain")
            return False

        def reply(self, code, body, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def reply_json(self, code, value):
            self.reply(code, json.dumps(value).encode(), "application/json")

        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            query = urllib.parse.parse_qs(url.query)
            if not self.allowed(query.get("token", [None])[0]):
                return
            if url.path == "/":
                self.reply(200, page, "text/html; charset=utf-8")
            elif url.path == "/state":
                self.reply_json(200, viewer.state())
            elif url.path == "/stream":
                self.stream()
            else:
                self.reply(404, b"not found\n", "text/plain")

        def do_POST(self):
            url = urllib.parse.urlsplit(self.path)
            if not self.allowed(self.headers.get(TOKEN_HEADER)):
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 <= length <= MOST_BODY_BYTES:
                    raise ValueError("a body of that size is not taken")
                body = json.loads(self.rfile.read(length) or b"{}")
                if url.path == "/pose":
                    taken = viewer.set_pose(
                        parse_pose(body), int(body.get("scene_serial", -1))
                    )
                    self.reply_json(200, {"taken": taken})
                elif url.path == "/sun":
                    taken = viewer.set_sun(
                        parse_sun(body), int(body.get("scene_serial", -1))
                    )
                    self.reply_json(200, {"taken": taken})
                elif url.path == "/tone":
                    taken = viewer.set_tone(
                        parse_tone(body), int(body.get("scene_serial", -1))
                    )
                    self.reply_json(200, {"taken": taken})
                elif url.path == "/indirect":
                    taken = viewer.set_indirect(
                        parse_indirect(body), int(body.get("scene_serial", -1))
                    )
                    self.reply_json(200, {"taken": taken})
                elif url.path == "/bounces":
                    taken = viewer.set_bounces(
                        parse_bounces(body), int(body.get("scene_serial", -1))
                    )
                    self.reply_json(200, {"taken": taken})
                elif url.path == "/size":
                    taken = viewer.set_size(parse_size(body))
                    if not taken:
                        raise ValueError(
                            f"{viewer.state()['renderer']} renders "
                            f"{renderers.SIDE} x {renderers.SIDE} only"
                        )
                    self.reply_json(200, {"taken": taken})
                elif url.path == "/select":
                    renderer, scene = body["renderer"], body["scene"]
                    if renderer not in renderers.RENDERERS or scene not in SCENES:
                        raise ValueError(f"no renderer {renderer} or scene {scene}")
                    if scene not in OFFERS[renderer]:
                        raise ValueError(f"{renderer} is not offered on {scene}")
                    viewer.select(renderer, scene)
                    self.reply_json(200, {"selected": [renderer, scene]})
                else:
                    self.reply(404, b"not found\n", "text/plain")
            except (ValueError, KeyError, TypeError) as error:
                self.reply_json(400, {"error": str(error)})

        def stream(self):
            self.close_connection = True
            self.send_response(200)
            self.send_header(
                "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            number = 0
            try:
                while not viewer.stopping:
                    number, data = viewer.next_jpeg(number, timeout=1.0)
                    if data is None:
                        continue
                    head = (
                        f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {len(data)}\r\n"
                        f"X-Frame: {number}\r\n\r\n"
                    )
                    self.wfile.write(head.encode() + data + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument(
        "--renderer", choices=list(renderers.RENDERERS), default="raster"
    )
    parser.add_argument("--scene", choices=list(SCENES), default="tiny")
    parser.add_argument(
        "--size",
        default=f"{SIZE[0]}x{SIZE[1]}",
        help="the frame's WxH at the start (the rasterizer's card grid)",
    )
    args = parser.parse_args()
    try:
        size = renderers.check_size(args.size.lower().split("x"))
    except ValueError as error:
        parser.error(f"--size {args.size}: {error}")
    page = (HERE / "page.html").read_bytes()
    if not renderers.RENDERERS[args.renderer].offered(args.scene):
        parser.error(f"{args.renderer} is not offered on {args.scene}")
    viewer = Viewer(args.renderer, args.scene, size)
    server = ThreadingHTTPServer(
        (args.host, args.port), handler_class(viewer, args.token, page)
    )
    server.daemon_threads = True
    for number in (signal.SIGINT, signal.SIGTERM):  # only the flag: no lock taken
        signal.signal(number, lambda *_: setattr(viewer, "stopping", True))
    serving = threading.Thread(target=server.serve_forever, name="http", daemon=True)
    serving.start()
    log(f"serving on http://{args.host}:{args.port}/?token=...")
    viewer.render_loop()  # on the main thread: it opens and owns the card's session
    log("stopping: renderer closed")
    viewer.stop()
    server.shutdown()
    server.server_close()
    log("stopped")
    sys.exit(0)


if __name__ == "__main__":
    main()
