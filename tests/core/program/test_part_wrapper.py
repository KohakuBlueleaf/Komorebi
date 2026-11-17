"""Every builder of an overlay part writes its wrapper by overlay.wrapper_text: a
merged program's part as an overlay's (a handler's part with km_worker.hpp first)."""

import importlib
from types import SimpleNamespace

from komorebi.program import overlay

merge = importlib.import_module("komorebi.merge")  # km.merge is the function


def test_a_merged_part_s_wrapper_is_an_overlay_s(tmp_path, monkeypatch):
    source = tmp_path / "part.cpp"
    source.write_text("// a part\n")
    made = merge.MergedOverlay("merged", 3, {"compute": str(source)}, program=True)
    stub = SimpleNamespace(_recipes={}, merged_sources=[], code_overlays=set())
    object.__setattr__(made, "merged", stub)
    seen = []
    monkeypatch.setattr(
        merge,
        "build_units",
        lambda recipe, name, kernel, part, units, *rest: seen.extend(units),
    )
    made.build_part("recipe", "compute", "compute", tmp_path)
    assert seen[0][0] == overlay.wrapper_text(made, "compute", str(source))
    assert '#include "komorebi/device/km_worker.hpp"' not in seen[0][0]
    message = overlay.Overlay("handler", 4, {"compute": str(source)}, message=True)
    assert overlay.wrapper_text(message, "compute", source).count("km_worker.hpp") == 1
