"""The viewer's camera: a pose (position, yaw, pitch) from a camera's 16 numbers, and
the 16 numbers of a pose; no card and no image library needed."""

import math

import numpy as np


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
    = forward x y, up = right x forward, as reference/scenes.py makes them); focal,
    centre and near the scene's own camera's."""
    yaw, pitch = pose["yaw"], pose["pitch"]
    forward = np.array(
        [
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
            math.cos(pitch) * math.cos(yaw),
        ]
    )
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    camera = np.array(own, dtype=np.float32)
    camera[0:3] = pose["position"]
    camera[3:6], camera[6:9], camera[9:12] = right, up, forward
    return camera
