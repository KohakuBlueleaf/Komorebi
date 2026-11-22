# A web viewer: the card renders, a browser steers

A small web application around the two interactive loops of a Blackhole card: the
rasterizer's (`projects/raster/live.py --push`) and the ray tracer's
(`projects/raytrace/live.py --show`). A person opens the page from another machine,
moves the camera with keyboard and mouse, chooses the renderer and the scene, and sees
the frames the card renders.

The path of one frame: the browser posts a pose -> the server -> the card's loop (one
command a frame with the new camera) -> the card puts the 8-bit image in host memory ->
the server encodes it as JPEG and sends it down the page's stream.

| file | what it is |
|---|---|
| `server.py` | the web server (Python's `http.server` only) and the render loop, on the main thread, that owns the card's session |
| `renderers.py` | `Raster`, `RasterTextured` (`--shade`, offered on scenes with a materials file), `RasterLit` (`--light`, offered on scenes with a materials and a lighting file) and `Raytrace`: `open(scene)`, `frame(camera_words) -> (image, card_ms)`, `close()`, on the `Loop` class of each project's `live.py` |
| `page.html` | the page: the stream, the numbers, the selects and the controls; no external files |
| `check.py` | the check without a browser |

## Requirements

- What every card run of the projects needs (the projects' README pages): the
  environment, `KR_TREE_SPLIT=sah` and `KR_DATA` pointing at the scene data, and a card.
- **Pillow** for the JPEG frames, on `PYTHONPATH` for the server and `check.py` only; it
  is not among the repository's requirements. One way that leaves the environment as it
  is: `python -m pip install --target DIR pillow` with a Python of the environment's
  version, then `DIR` added to the server's `PYTHONPATH`.

## Starting it

```
python -m projects.viewer.server --host HOST --port PORT --token TOKEN \
    [--renderer raster|raster-textured|raster-lit|raytrace] [--scene tiny|sponza_overview|bistro_interior_overview|bistro_exterior_overview]
```

The server binds only the address `--host` gives. Open
`http://HOST:PORT/?token=TOKEN`: every request carries the token (the query `?token=`
on a GET, the header `X-Token` on a POST) and gets 403 without it; the page keeps the
token it was opened with. SIGINT or SIGTERM stops it: the render loop closes the
renderer (the card's session) before the process exits.

The first open of a scene builds what is missing (its tree, the lists, the kernels): a
public scene takes a while, and the page shows `loading` meanwhile.

## The page and its controls

- The frames as one image (a `multipart/x-mixed-replace` JPEG stream).
- A line of numbers from `/state`, twice a second: renderer, scene, the frame's ms with its label (`frame ms (host)` for the rasterizer,
  `frame ms (card)` for the ray tracer),
  frames a second over the last second, frame number, and the status: `loading`,
  `running`, `refused: N` (the card's result word; the last good image stays), or
  `error: ...` (the renderer is closed; choose a renderer or scene again).
- Two selects: renderer and scene. A new scene starts at its own camera.
- Click the image, then: **W A S D** move forward / left / back / right along the
  view, **Q E** down / up, **drag** with the mouse to turn (yaw, pitch), **shift**
  moves five times faster. The step is a tenth of the scene box's diagonal a second.
  The page posts the pose at most 30 times a second and only when it changed.

## What the server does

- One render thread owns the card's session: the process's main thread (tt-metal
  closes the device at the process's exit on the main thread, and that must be the
  thread that opened it); the web server runs on a second thread. When the renderer or
  the scene changes it closes the current one and opens the new one; it renders a
  frame for the latest pose and never the same pose twice in a row unless a second
  passes with nothing new (a keep-alive frame).
- The pose is a position, a yaw and a pitch (y is up). The server makes the 16 camera
  numbers from it as `reference/scenes.py` lays them out (right = y x forward, up =
  forward x right); the focal length, the centre and the near distance stay the scene's
  own. The first pose is the scene's own camera, and while the pose is that one the
  scene's camera is used as it is.
- `Raster`: the triangle lists are sized at four times `live.py`'s default slots; a
  frame whose result word is not 0 (lists too small for the moved camera) raises
  `FrameRefused` and the page shows `refused: N`. Its number, `frame ms (host)`, is
  the host's time from the camera's write to the pushed image in host memory (the
  card's own frame time is in the thread's region, read only after the program stops).
- `Raytrace`: the frame as one chain, `--treelets 128 --sort` on every scene but
  `tiny`; its number, `frame ms (card)`, is the chain command's time (`live.py`'s
  `card_frame_ms`).
- Bistro exterior (2.83 M triangles, 1 GB of texture pages) on card 1, 2026-11-23, poses
  posted as fast as the server took them: `raster` 34 frames a second, `raster-textured`
  and `raster-lit` 20, `raytrace` 1 (933 ms a frame); a switch to it took 2.7, 5.2, 5.7 and
  4.2 s. Its first `raytrace` open cuts its two treelet files (306 MB each) into `KR_DATA`.

Requests: `GET /` the page; `GET /stream` the frames; `POST /pose`
(`{"position": [x, y, z], "yaw": a, "pitch": b, "scene_serial": n}`); `POST /select`
(`{"renderer": r, "scene": s}`); `POST /sun` (`{"azimuth": a, "elevation": e,
"scene_serial": n}`, raster-lit); `GET /state` (the numbers, the status, the renderers
and scenes offered, the scene box's diagonal, the scene's first pose; raster-lit: `sun`,
`sun_own`, `sun_ms`).

The sun (raster-lit): two sliders, azimuth and elevation, start at the scene's own sun;
"reset sun" puts them back. A move posts the angles (at most ten times a second); the
render loop keeps the newest only and applies it between frames (`Loop.set_sun`: the
sun's numbers into LIGHT's table, the card's SUN pass renders the shadow map, none at or
below the horizon), and the header shows the sun and the last SUN pass's ms. Card 1,
2026-11-23 03:53, `check.py`: the sun moved to the own azimuth + 90 at elevation 40 changed
the frame and `/state` named it on all three scenes; SUN pass 8.6 ms (Sponza), 10.3
(Bistro interior), 21.5 (Bistro exterior), host side included.

## Checking it

```
python -m projects.viewer.check --url http://HOST:PORT --token TOKEN
```

For each renderer and scene: selects it, waits for `running`, posts 20 poses along a
short path, reads the frames from `/stream`, and reports the frames received, that they
decode to 1024 x 1024, that the first and the last differ, and the frame ms (with its
label) and frames a second `/state` showed; and that requests without the token get 403. One JSON line with
`ok`.
