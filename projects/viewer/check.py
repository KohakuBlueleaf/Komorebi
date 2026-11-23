"""The viewer checked without a browser: each renderer on each scene it is offered on
selected (400 on the others), 20 poses posted, the frames of /stream decoded (1024 x
1024, first and last differing), a lit renderer's sun moved (frame and /state change),
403 without the token. One JSON line, exit 1 unless `ok`. check --url URL --token T"""

import argparse
import io
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request

from PIL import Image

SIDE = 1024
# each renderer must offer Bistro exterior (2.83 M triangles, 1 GB of texture pages)
# but raster-gi, offered where the scene has an indirect file (Bistro exterior: none)
EXTERIOR = "bistro_exterior_overview"
NO_EXTERIOR = ("raster-gi",)
POSES = 20
POSE_GAP_S = 0.1
OPEN_TIMEOUT_S = 1800.0  # a scene's first open builds its tree, lists and kernels
AFTER_S = 2.0  # frames read after the last pose (and after the sun's move)
FOLLOW_S = 60.0  # the sun's move: at most this for /state to show it and a frame after
SUN_TURN = (
    90.0  # the sun's move: its azimuth turned by this, its elevation SUN_ELEVATION
)
SUN_ELEVATION = 40.0


def get(url, timeout=10.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read()


def post(url, body, token, timeout=10.0):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Token"] = token
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def status_of(call):
    """The HTTP status of a call, an error's included."""
    try:
        return call()[0]
    except urllib.error.HTTPError as error:
        return error.code


class Reader(threading.Thread):
    """Reads the multipart stream's JPEG parts until `stop`."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url, self.frames, self.numbers = url, [], []
        self.stop, self.error = threading.Event(), None

    def run(self):
        try:
            response = urllib.request.urlopen(self.url, timeout=5.0)
            with response:
                while not self.stop.is_set():
                    line = response.readline()
                    if not line:
                        break
                    if not line.startswith(b"--"):
                        continue
                    headers = {}
                    while True:
                        line = response.readline().strip()
                        if not line:
                            break
                        name, _, value = line.decode().partition(":")
                        headers[name.strip().lower()] = value.strip()
                    data = response.read(int(headers["content-length"]))
                    self.frames.append(data)
                    self.numbers.append(int(headers.get("x-frame", -1)))
        except Exception as error:  # noqa: BLE001 - a timeout after the stop ends it
            if not self.stop.is_set():
                self.error = f"{type(error).__name__}: {error}"


def wait_running(base, token, renderer, scene):
    """The state once the selection runs (or has failed), with the seconds taken."""
    started = time.monotonic()
    while time.monotonic() - started < OPEN_TIMEOUT_S:
        _, body = get(f"{base}/state?token={token}")
        state = json.loads(body)
        loaded = (state["renderer"], state["scene"]) == (renderer, scene)
        if state["status"].startswith("error"):
            return state, time.monotonic() - started
        if loaded and state["status"] == "running":
            return state, time.monotonic() - started
        time.sleep(0.5)
    return state, time.monotonic() - started


def path_pose(first, diagonal, index):
    """Pose `index` of the short path: forward along the first view, turning."""
    yaw, pitch = first["yaw"], first["pitch"]
    step = 0.005 * diagonal * index
    forward = [
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
        math.cos(pitch) * math.cos(yaw),
    ]
    return {
        "position": [p + step * f for p, f in zip(first["position"], forward)],
        "yaw": yaw + 0.01 * index,
        "pitch": pitch,
    }


def run_case(base, token, renderer, scene):
    case = {"renderer": renderer, "scene": scene}
    post(f"{base}/select", {"renderer": renderer, "scene": scene}, token)
    time.sleep(0.6)  # the render thread takes the selection
    state, seconds = wait_running(base, token, renderer, scene)
    case["open_s"] = round(seconds, 1)
    case["status"] = state["status"]
    if state["status"] != "running":
        case["ok"] = False
        return case
    reader = Reader(f"{base}/stream?token={token}")
    reader.start()
    card_ms, fps, statuses, taken = [], [], set(), 0
    for index in range(1, POSES + 1):
        pose = path_pose(state["first_pose"], state["diagonal"], index)
        pose["scene_serial"] = state["scene_serial"]
        taken += bool(post(f"{base}/pose", pose, token)[1]["taken"])
        time.sleep(POSE_GAP_S)
        now = json.loads(get(f"{base}/state?token={token}")[1])
        statuses.add(now["status"])
        if now["card_ms"] is not None:
            card_ms.append(now["card_ms"])
        fps.append(now["fps"])
    time.sleep(AFTER_S)
    sun = None
    if state.get("sun_own"):
        sun = sun_case(base, token, state, reader)
    reader.stop.set()
    reader.join(10.0)
    sizes = set()
    for data in reader.frames:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            sizes.add(image.size)
    differ = len(reader.frames) > 1 and (
        Image.open(io.BytesIO(reader.frames[0])).tobytes()
        != Image.open(io.BytesIO(reader.frames[-1])).tobytes()
    )
    case.update(
        poses_taken=taken,
        frames=len(reader.frames),
        frame_numbers=[reader.numbers[0], reader.numbers[-1]] if reader.numbers else [],
        sizes=sorted(list(size) for size in sizes),
        first_last_differ=bool(differ),
        card_ms_label=now["card_ms_label"],
        card_ms_median=sorted(card_ms)[len(card_ms) // 2] if card_ms else None,
        fps_max=max(fps) if fps else None,
        fps_median=sorted(fps)[len(fps) // 2] if fps else None,
        statuses=sorted(statuses),
        stream_error=reader.error,
    )
    if sun is not None:
        case["sun"] = sun
    case["ok"] = bool(
        taken == POSES
        and len(reader.frames) >= 2
        and sizes == {(SIDE, SIDE)}
        and differ
        and reader.error is None
        and (sun is None or sun["ok"])
    )
    return case


def sun_case(base, token, state, reader):
    """The sun moved (SUN_TURN, SUN_ELEVATION) after the poses: the newest frame before
    and after against each other, and the angles and the SUN pass's ms /state shows."""
    before = reader.frames[-1] if reader.frames else None
    count = len(reader.frames)
    azimuth = round(state["sun_own"][0] + SUN_TURN, 4)
    taken = post(
        f"{base}/sun",
        {
            "azimuth": azimuth,
            "elevation": SUN_ELEVATION,
            "scene_serial": state["scene_serial"],
        },
        token,
    )[1]["taken"]
    posted = time.monotonic()
    follow_s = None  # until /state shows the sun and a frame came after it
    while time.monotonic() - posted < FOLLOW_S:
        now = json.loads(get(f"{base}/state?token={token}")[1])
        shown = now["sun"] is not None and abs(now["sun"][0] - azimuth) < 1e-6
        if shown and follow_s is None:
            follow_s, count_shown = time.monotonic() - posted, len(reader.frames)
        if follow_s is not None and len(reader.frames) > count_shown:
            follow_s = time.monotonic() - posted
            break
        time.sleep(0.05)
    time.sleep(AFTER_S)
    now = json.loads(get(f"{base}/state?token={token}")[1])
    after = reader.frames[-1] if len(reader.frames) > count else None
    changed = (
        before is not None
        and after is not None
        and Image.open(io.BytesIO(before)).tobytes()
        != Image.open(io.BytesIO(after)).tobytes()
    )
    named = now["sun"] is not None and all(
        abs(got - want) < 1e-6
        for got, want in zip(now["sun"], (azimuth, SUN_ELEVATION))
    )
    return {
        "own": state["sun_own"],
        "posted": [azimuth, SUN_ELEVATION],
        "taken": taken,
        "state_sun": now["sun"],
        "sun_ms": now["sun_ms"],
        "follow_s": None if follow_s is None else round(follow_s, 2),
        "frames_after": len(reader.frames) - count,
        "frame_changed": bool(changed),
        "status": now["status"],
        "ok": bool(taken and changed and named and now["status"] == "running"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    base, token = args.url.rstrip("/"), args.token
    report = {
        "forbidden": {
            "state": status_of(lambda: get(f"{base}/state")),
            "state_wrong": status_of(lambda: get(f"{base}/state?token=wrong")),
            "page": status_of(lambda: get(f"{base}/")),
            "stream": status_of(lambda: get(f"{base}/stream")),
            "pose": status_of(lambda: post(f"{base}/pose", {}, None)),
            "select": status_of(lambda: post(f"{base}/select", {}, None)),
            "sun": status_of(lambda: post(f"{base}/sun", {}, None)),
        }
    }
    page_status, page = get(f"{base}/?token={token}")
    report["page"] = page_status == 200 and b"/stream" in page
    state = json.loads(get(f"{base}/state?token={token}")[1])
    offers = state["offers"]
    report["offers"] = offers
    report["exterior_offered"] = all(
        EXTERIOR in offers[name] for name in offers if name not in NO_EXTERIOR
    )
    report["refused"] = {  # a renderer selected on a scene it is not offered on: 400
        f"{renderer}/{scene}": status_of(
            lambda renderer=renderer, scene=scene: post(
                f"{base}/select", {"renderer": renderer, "scene": scene}, token
            )
        )
        for renderer in state["renderers"]
        for scene in state["scenes"]
        if scene not in offers[renderer]
    }
    report["cases"] = [
        run_case(base, token, renderer, scene)
        for renderer in state["renderers"]
        for scene in offers[renderer]
    ]
    report["ok"] = bool(
        all(code == 403 for code in report["forbidden"].values())
        and all(code == 400 for code in report["refused"].values())
        and offers.get("raster-textured")
        and offers.get("raster-lit")
        and report["page"]
        and report["exterior_offered"]
        and all(case["ok"] for case in report["cases"])
    )
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
