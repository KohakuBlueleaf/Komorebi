"""The viewer's camera from a pose (projects/viewer/server.py camera_of): a right-handed
frame with y up, and a scene's own pose gives back the scene's own camera."""

import numpy as np
import pytest

from reference import public, scenes

pytest.importorskip("PIL")  # the server encodes its frames with Pillow
server = pytest.importorskip("projects.viewer.server")


def own_cameras():
    """The tiny scene's camera and the camera of every view of the public table."""
    cameras = [scenes.make_camera(30.0 * np.sqrt(398 / 1e4))]
    for name, view in public.views().values():
        spec = public.table()[name]["views"][view]
        cameras.append(public.camera(spec["eye"], spec["target"], spec["fov"], 0.01))
    return cameras


def test_camera_of_is_right_handed_with_y_up():
    for yaw in np.linspace(-3.0, 3.0, 7):
        for pitch in (-1.2, 0.0, 0.7):
            pose = {"position": [1.0, 2.0, 3.0], "yaw": yaw, "pitch": pitch}
            cam = server.camera_of(np.zeros(16, np.float32), pose).astype(np.float64)
            right, up, forward = cam[3:6], cam[6:9], cam[9:12]
            for axis in (right, up, forward):
                assert abs(axis @ axis - 1) < 1e-6
            assert abs(right @ up) < 1e-6 and abs(right @ forward) < 1e-6
            assert abs(up @ forward) < 1e-6
            assert np.allclose(np.cross(right, up), -forward, atol=1e-6)
            assert abs(right[1]) < 1e-7 and up[1] > 0


def test_a_scenes_own_pose_gives_its_own_camera():
    for own in own_cameras():
        again = server.camera_of(own, server.pose_of(own))
        np.testing.assert_allclose(again, own, rtol=0, atol=1e-6)
