"""The loops behind one interface: `open(scene, size)`, `frame(camera)` -> (image (H, W,
4), ms), `set_size(size)`, `close()`; the camera of 1,024 x 1,024, sized here. Raster,
-Textured, -Lit (`set_sun`, `set_tone`), -Gi (`set_indirect`, `set_bounces`); Raytrace:
1,024 only. Each its own card session and thread (README.md)."""

import argparse
import contextlib

import numpy as np

import komorebi as km
from projects.raster import card as raster_card
from projects.raster import card_thread as raster_thread
from projects.raster import live as raster_live
from projects.raster import run as raster_run
from projects.raster import shade as raster_shade
from projects.raytrace import card as raytrace_card
from projects.raytrace import frame as raytrace_frame
from projects.raytrace import live as raytrace_live
from reference import indirect, native, public, raster, scenes
from reference import lights as light_data

SIDE = 1024  # the scenes' camera convention, and the ray tracer's frame
LIST_SLOTS_TIMES = 4  # the triangle lists: four times card_thread's default slots
TREELETS_KIB = 128  # the ray tracer's treelets from DRAM, on every scene but tiny


class FrameRefused(Exception):
    """A frame the card answered with a result word that is not 0: `word`."""

    def __init__(self, word):
        super().__init__(f"result word {word}")
        self.word = word


def check_size(size):
    """(W, H) as integers, any positive size (card_thread.card_grid); else ValueError
    with the reason. Memory alone bounds it, at the load (SizeRefused)."""
    width, height = (int(n) for n in size)
    raster_thread.card_grid((width, height))
    return width, height


class SizeRefused(ValueError):
    """A frame size a renderer does not render: the reason (the load's, for memory)."""


class Renderer:
    """What both share: the session's stack, the scene dict (`scene`), the image of
    the frame's size (`size`, (W, H)); `sizes`: False when only SIDE x SIDE."""

    name = ""
    time_label = ""  # what frame's ms are, as the page labels them
    sun_own = None  # a lit renderer: the scene's own sun (azimuth, elevation)
    tone_own = None  # a lit renderer: the scene's tone (exposure, saturation, curve)
    indirect = None  # raster-gi: the indirect switch (True: on)
    bounces = None  # raster-gi: the field's bounce passes in force (0 to 3)
    sizes = True  # takes a frame size (set_size)

    @staticmethod
    def offered(scene_name):
        """True when the renderer runs on the scene."""
        return True

    def __init__(self):
        self.stack = None
        self.scene = None
        self.sun = None  # a lit renderer: the sun's (azimuth, elevation) now
        self.tone = None  # a lit renderer: the tone now (exposure, saturation, curve)
        self.size = (SIDE, SIDE)
        self.image = np.zeros((SIDE, SIDE, 4), dtype=np.uint8)

    def sized(self, size):
        """The size this renderer renders for `size` asked: its own when it takes
        none."""
        return check_size(size) if self.sizes else (SIDE, SIDE)

    def field_s(self):
        """Seconds the last update of the indirect light's field took on the host (None
        before the first, or without the field)."""
        return None

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

    def lights(self, scene_name):
        """The lights the loop is loaded with (None: not lit)."""
        return None

    terms = raster.TERMS  # the lit loop's terms

    def open(self, scene_name, size=(SIDE, SIDE)):
        self.close()
        scene, _ = raster_run.load(scene_name)
        size = self.sized(size)
        materials, lights = self.materials(scene_name), self.lights(scene_name)
        slots = LIST_SLOTS_TIMES * (2 * len(scene["indices"]) + 65536)
        self.stack = contextlib.ExitStack()
        try:
            session = self.stack.enter_context(km.Session(log=raster_card.log))
            try:
                self.live = raster_live.Loop(
                    session,
                    scenes.with_size(scene, size),
                    True,
                    slots,
                    raster_live.show_table(),
                    {},
                    materials,
                    lights,
                    self.terms,
                )
            except raster_thread.L1Refused as error:
                raise SizeRefused(str(error)) from None
            self.stack.callback(self.live.stop)
        except BaseException:
            self.close()
            raise
        self.scene = scene
        self.size = size
        self.image = np.zeros((size[1], size[0], 4), dtype=np.uint8)

    def set_size(self, size):
        """The frame's size (W, H) from the next frame on (live.Loop.set_size): the grid
        word to the card, or past the loaded capacity a reload. Returns the host's ms;
        SizeRefused with the load's reason when memory does not hold it (the size as it
        was)."""
        size = check_size(size)
        try:
            ms = self.live.set_size(size)
        except raster_thread.L1Refused as error:
            raise SizeRefused(str(error)) from None
        self.size = size
        self.image = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        return ms

    def frame(self, camera_words):
        """The FRAME command for the camera (the scene's convention, sized here):
        (image, ms from the camera's write to the image in host memory)."""
        camera = scenes.sized_camera(camera_words, self.size)
        event, times = self.live.frame(camera, self.image)
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


class RasterLit(RasterTextured):
    """projects/raster/live.py's `--push --light` path: SHADE's surface, then LIGHT."""

    name = "raster-lit"

    @staticmethod
    def offered(scene_name):
        if not raster_shade.has_materials(scene_name):
            return False
        name = (
            public.views()[scene_name][0] if public.is_view(scene_name) else scene_name
        )
        return light_data.lighting_path(name, native.data_dir()).exists()

    def lights(self, scene_name):
        return light_data.load(scene_name, native.data_dir())

    def open(self, scene_name, size=(SIDE, SIDE)):
        self.sun_own = self.sun = self.tone_own = self.tone = None
        super().open(scene_name, size)
        own = light_data.sun_angles(self.live.lights["sun_direction"])
        self.sun_own = self.sun = (round(own[0], 4), round(own[1], 4))
        tone = self.live.tone
        self.tone_own = self.tone = (
            tone["exposure"],
            tone["saturation"],
            tone["curve"],
        )

    def set_tone(self, exposure, saturation, curve):
        """The tone step's numbers into LIGHT's table, applied to the next frame (no
        reload). Returns the host's ms."""
        ms = self.live.set_tone(exposure, saturation, curve)
        self.tone = (float(exposure), float(saturation), int(curve))
        return ms

    def set_sun(self, azimuth, elevation):
        """The sun moved to `azimuth`, `elevation` (degrees): its numbers and the card's
        SUN pass (none at or below the horizon), applied to the next frame. Returns the
        host's ms; a result word that is not 0 raises FrameRefused."""
        word, ms, _ = self.live.set_sun(azimuth, elevation)
        if isinstance(word, str):
            raise RuntimeError(f"SUN: {word}")
        if word not in (0, None):
            raise FrameRefused(word)
        self.sun = (azimuth, elevation)
        return ms


class RasterGi(RasterLit):
    """projects/raster/live.py's `--push --light --indirect` path: LIGHT with the
    field's indirect light; a sun change remakes the field on the host."""

    name = "raster-gi"
    terms = raster.TERMS + ("indirect",)

    def open(self, scene_name, size=(SIDE, SIDE)):
        self.indirect = self.bounces = None
        super().open(scene_name, size)
        self.indirect = True
        self.bounces = int(self.live.lights.get("bounces", light_data.BOUNCES))

    def set_bounces(self, bounces):
        """The field's bounce passes (0 to 3): the field recomputed on the host and
        written, applied to the next frame (no reload). Returns the host's ms."""
        ms = self.live.set_bounces(bounces)
        self.bounces = int(bounces)
        return ms

    def field_s(self):
        update = self.live.field_update
        return None if update is None else update["s"]

    def set_indirect(self, on):
        """The indirect switch (off: lit as without the field), applied to the next
        frame (no reload). Returns the host's ms."""
        ms = self.live.set_indirect(on)
        self.indirect = bool(on)
        return ms

    @staticmethod
    def offered(scene_name):
        return (
            RasterLit.offered(scene_name)
            and indirect.model_path(scene_name, native.data_dir()).exists()
        )


class Raytrace(Renderer):
    """projects/raytrace/live.py's `--show` path."""

    name = "raytrace"
    time_label = "frame ms (card)"
    sizes = False  # its frame is SIDE x SIDE

    @staticmethod
    def offered(scene_name):
        """True when the scene's box tree is saved (building one takes minutes)."""
        return (native.data_dir() / f"{scene_name}_tree.npz").exists()

    def open(self, scene_name, size=(SIDE, SIDE)):
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

    def set_size(self, size):
        raise SizeRefused(f"{self.name} renders {SIDE} x {SIDE} only")

    def frame(self, camera_words):
        """The frame's chain command for the camera: (image, the command's card time
        in ms, as live.py's card_frame_ms)."""
        view = dict(self.scene, camera=np.asarray(camera_words, dtype=np.float32))
        event, _ = self.live.frame(view, self.image)
        if event.kind != "done":
            raise RuntimeError(f"frame chain: {event.kind}")
        return self.image, event.microseconds / 1e3


RENDERERS = {
    renderer.name: renderer
    for renderer in (Raster, RasterTextured, RasterLit, RasterGi, Raytrace)
}


def offers(scenes):
    """{renderer name: the scenes of `scenes` it is offered on}."""
    return {
        name: [scene for scene in scenes if renderer.offered(scene)]
        for name, renderer in RENDERERS.items()
    }
