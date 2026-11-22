"""The interactive loops behind one interface: `open(scene)`, `frame(camera)` -> (image
(1024, 1024, 4) uint8, ms), `close()`. Raster: the rasterizer's `--push` loop;
RasterTextured: `--push --shade` (scenes with materials); Raytrace: the ray tracer's
`--show` loop. Each opens its own card session, owned by the thread calling them."""

import argparse
import contextlib

import numpy as np

import komorebi as km
from projects.raster import card as raster_card
from projects.raster import live as raster_live
from projects.raster import run as raster_run
from projects.raster import shade as raster_shade
from projects.raytrace import card as raytrace_card
from projects.raytrace import frame as raytrace_frame
from projects.raytrace import live as raytrace_live

SIDE = 1024
LIST_SLOTS_TIMES = 4  # the triangle lists: four times card_thread's default slots
TREELETS_KIB = 128  # the ray tracer's treelets from DRAM, on every scene but tiny


class FrameRefused(Exception):
    """A frame the card answered with a result word that is not 0: `word`."""

    def __init__(self, word):
        super().__init__(f"result word {word}")
        self.word = word


class Renderer:
    """What both share: the session's stack, the scene dict (`scene`), the image."""

    name = ""
    time_label = ""  # what frame's ms are, as the page labels them

    @staticmethod
    def offered(scene_name):
        """True when the renderer runs on the scene."""
        return True

    def __init__(self):
        self.stack = None
        self.scene = None
        self.image = np.zeros((SIDE, SIDE, 4), dtype=np.uint8)

    def close(self):
        """Stops the loop's program and closes the session (also after a failure)."""
        stack, self.stack = self.stack, None
        if stack is not None:
            stack.close()


class Raster(Renderer):
    """projects/raster/live.py's `--push` path."""

    name = "raster"
    time_label = "frame ms (host)"

    def materials(self, scene_name):
        """The materials the loop is loaded with (None: not shaded)."""
        return None

    def open(self, scene_name):
        self.close()
        scene, _ = raster_run.load(scene_name)
        materials = self.materials(scene_name)
        slots = LIST_SLOTS_TIMES * (2 * len(scene["indices"]) + 65536)
        self.stack = contextlib.ExitStack()
        try:
            session = self.stack.enter_context(km.Session(log=raster_card.log))
            self.live = raster_live.Loop(
                session, scene, True, slots, raster_live.show_table(), {}, materials
            )
            self.stack.callback(self.live.stop)
        except BaseException:
            self.close()
            raise
        self.scene = scene

    def frame(self, camera_words):
        """The FRAME command for the camera: (image, ms from the camera's write to the
        image in host memory)."""
        event, times = self.live.frame(camera_words, self.image)
        began, ended = times[0], times[4]
        if event.kind != "done":
            raise RuntimeError(f"FRAME: {event.kind}")
        if isinstance(event.result, str):
            raise RuntimeError(f"FRAME: {event.result}")
        if event.result != 0:
            raise FrameRefused(event.result)
        return self.image, (ended - began) * 1e3


class RasterTextured(Raster):
    """projects/raster/live.py's `--push --shade` path: textures by SHADE."""

    name = "raster-textured"

    @staticmethod
    def offered(scene_name):
        return raster_shade.has_materials(scene_name)

    def materials(self, scene_name):
        return raster_shade.load(scene_name)


class Raytrace(Renderer):
    """projects/raytrace/live.py's `--show` path."""

    name = "raytrace"
    time_label = "frame ms (card)"

    def open(self, scene_name):
        self.close()
        tiny = scene_name == "tiny"
        arguments = raytrace_live.frame_arguments(
            argparse.Namespace(
                scene=scene_name,
                treelets=0 if tiny else TREELETS_KIB,
                sort=not tiny,
            )
        )
        prepared = raytrace_frame.prepare(arguments)
        table, differs = raytrace_live.show_table_checked()
        if differs:
            raise ValueError(
                f"SHOW's table differs from png's rule on {differs} values"
            )
        self.stack = contextlib.ExitStack()
        try:
            session = self.stack.enter_context(raytrace_card.open_session())
            self.live = raytrace_live.Loop(session, arguments, prepared, table, {})
            self.stack.callback(self.live.stop)
        except BaseException:
            self.close()
            raise
        self.scene = prepared["scene"]

    def frame(self, camera_words):
        """The frame's chain command for the camera: (image, the command's card time
        in ms, as live.py's card_frame_ms)."""
        view = dict(self.scene, camera=np.asarray(camera_words, dtype=np.float32))
        event, _ = self.live.frame(view, self.image)
        if event.kind != "done":
            raise RuntimeError(f"frame chain: {event.kind}")
        return self.image, event.microseconds / 1e3


RENDERERS = {renderer.name: renderer for renderer in (Raster, RasterTextured, Raytrace)}


def offers(scenes):
    """{renderer name: the scenes of `scenes` it is offered on}."""
    return {
        name: [scene for scene in scenes if renderer.offered(scene)]
        for name, renderer in RENDERERS.items()
    }
