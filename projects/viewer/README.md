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
| `renderers.py` | `Raster`, `RasterTextured` (`--shade`, offered on scenes with a materials file), `RasterLit` (`--light`, offered on scenes with a materials and a lighting file), `RasterGi` (`--light --indirect`, offered where the scene also has an indirect file) and `Raytrace`: `open(scene, size)`, `frame(camera_words) -> (image, card_ms)`, `set_size(size)`, `close()`, on the `Loop` class of each project's `live.py` |
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
    [--renderer raster|raster-textured|raster-lit|raster-gi|raytrace] [--scene tiny|sponza_overview|bistro_interior_overview|bistro_exterior_overview] \
    [--size WxH]
```

`--size` is the frame's size at the start, 1280x720 by default; a size the rasterizer's
card grid does not take stops the server with the reason.

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
- Width and height inputs and a **size** button (or Enter): the frame's size, applied
  between frames with no reload. A size the card's grid does not take (more than 1,024
  tiles of 32 x 32, more than 2,048 wide or 1,024 high) is refused and the reason shown
  beside the inputs. The numbers line shows the frames' size; the image is shown at its
  own size, scaled down (aspect kept) to fit the window. The ray tracer renders 1,024 x
  1,024 only (`(only)` after the size); a size posted to it is refused.
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
  scene's camera is used as it is. A rasterizer sizes the camera to the frame
  (`scenes.sized_camera`: centre at `(W / 2, H / 2)`, focal length times
  `max(W, H) / 1024`).
- The frame's size: a load takes the size asked last; `POST /size` (`{"width": w,
  "height": h}`) checks it against `card_thread.card_grid` (400 with `{"error":
  reason}` when refused, also when the loaded renderer is the ray tracer) and the
  render thread applies it before the next frame (`Loop.set_size`: the grid word into
  the camera's region, 0.2 to 0.5 ms on the host). `/state` has `size` (the frames'),
  `size_asked`, `size_ms` (the last change's host ms) and `sizes` (false: one size
  only).
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
`sun_own`, `sun_ms`; raster-lit and raster-gi: `tone`, `tone_own`; raster-gi:
`indirect`); `POST /tone` (`{"exposure": e, "saturation": s, "curve": 0 or 1,
"scene_serial": n}`, raster-lit and raster-gi: exposure clamped to 0.25 to 8,
saturation to 0 to 2); `POST /indirect` (`{"on": true or false, "scene_serial": n}`,
raster-gi).

The sun (raster-lit): two sliders, azimuth and elevation, start at the scene's own sun;
"reset sun" puts them back. A move posts the angles (at most ten times a second); the
render loop keeps the newest only and applies it between frames (`Loop.set_sun`: the
sun's numbers into LIGHT's table, the card's SUN pass renders the shadow map, none at or
below the horizon), and the header shows the sun and the last SUN pass's ms. Card 1,
2026-11-23 03:53, `check.py`: the sun moved to the own azimuth + 90 at elevation 40 changed
the frame and `/state` named it on all three scenes; SUN pass 8.6 ms (Sponza), 10.3
(Bistro interior), 21.5 (Bistro exterior), host side included.

raster-gi has the same sliders; its sun change also remakes the indirect field on the
host and writes its pages (`Loop.update_field`) before the next frame. Card 1,
2026-11-23 23:19 to 23:23, `check.py` (`ok`, every case): the sun change took 1,639 ms
(Sponza) and 3,402 ms (Bistro interior) in the render loop, the first frame with the new
sun came 1.7 and 3.5 s after the post (`follow_s`); frame ms (host) 14.9 and 23.6.
`check.py` waits up to 60 s for that frame, and does not ask raster-gi for Bistro
exterior (no indirect file).

The tone (raster-lit, raster-gi; hidden for the other renderers): an exposure slider
(0.25 to 8, logarithmic), a saturation slider (0 to 2), the curve (plain `x / (1 + x)`,
filmic) and "reset tone", back to the scene's numbers (its `lighting` entry). A change
posts the three (at most ten times a second); the render loop applies the newest between
frames (`Loop.set_tone`: LIGHT's table in the sun region, nothing reloaded), and the
header shows them. raster-gi also has the checkbox "indirect light" (on at a load): a
change posts it, and the loop switches the field off (lit as raster-lit, sky x
visibility and ambient) or on (`Loop.set_indirect`) before the next frame.
Card 1, 2026-11-24 00:31 to 00:37, `check.py` (`ok`, every case; the tone and the switch
changed the frame and `/state` named them): the tone shown with a frame after it in 0.06
to 0.11 s on raster-lit's three scenes and raster-gi's Bistro interior, 1.08 s on
raster-gi's Sponza; the indirect switch 1.07 s (Sponza) and 0.11 s (Bistro interior).

## Checking it

```
python -m projects.viewer.check --url http://HOST:PORT --token TOKEN
```

For each renderer and scene: selects it, waits for `running`, posts 20 poses along a
short path, reads the frames from `/stream`, and reports the frames received, that they
decode to the size `/state` reports, that the first and the last differ, and the frame ms (with its
label) and frames a second `/state` showed; and that requests without the token get 403. One JSON line with
`ok`. A lit renderer also moves the sun and posts the tone exposure 2, saturation 1.15,
filmic; raster-gi then switches the indirect light off: each must change the frame and
show in `/state`. Each case also posts 2048 x 1024 (64 x 32 = 2,048 tiles: 400 with the
reason), then on a rasterizer 1024 x 576 (`/state` and a frame of that size must follow)
and the start's size again; the ray tracer must refuse 1024 x 576 with 400.

Card 1, 2026-11-24 01:52 to 01:56, server `--size 1280x720` (default) on 127.0.0.1:8766,
`check.py` `ok`, every case: the rasterizers' frames 1,280 x 720 on every scene, the size
change shown with a 1,024 x 576 frame after it in 0.05 to 0.11 s (`size_ms` 0.20 to
0.47), back to 1,280 x 720 after; the refused size 400 "a frame of 2048 x 1024 pixels is
64 x 32 = 2048 tiles of 32 x 32: at most 1024"; the ray tracer 1,024 x 1,024, 1024 x 576
refused "raytrace renders 1024 x 1024 only". Host frame ms medians at 1,280 x 720
(`/state`): raster 3.6 (tiny), 6.1 (Sponza), 10.6 (interior), 20.2 (exterior);
raster-lit 12.2, 21.4, 32.8; raster-gi 14.2, 23.9.
