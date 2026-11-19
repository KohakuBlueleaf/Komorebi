"""The data folder's files made by scripts that run side by side: whole or absent."""

import multiprocessing
import os

import numpy as np
import pytest

from reference import native, scenes


def _made(data):
    os.environ["KR_DATA"] = data
    native.ensure_scene("tiny")
    return scenes.checksum(scenes.load_scene(native.data_dir() / "tiny.npz"))


def test_four_makers_of_one_scene_at_once_leave_it_whole(tmp_path):
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        sums = pool.map(_made, [str(tmp_path)] * 4)
    assert len(set(sums)) == 1
    assert not [p.name for p in tmp_path.rglob(".*")]  # no maker's own name left


def test_a_save_that_fails_leaves_neither_file(tmp_path):
    def broken(name):
        np.save(name, np.zeros(3))
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        scenes.save_beside(tmp_path / "a.npy", broken)
    assert list(tmp_path.iterdir()) == []
