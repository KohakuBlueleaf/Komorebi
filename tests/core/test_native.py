"""komorebi.native: where the dispatch kernel image and the native module come from."""

import pytest
from fake_tree import make_tree

from komorebi import _env, build, native


def test_cache_dir_and_source_key(tmp_path, monkeypatch):
    monkeypatch.setenv(native.CACHE_VARIABLE, str(tmp_path / "c"))
    assert native.cache_dir() == tmp_path / "c"
    monkeypatch.delenv(native.CACHE_VARIABLE)
    assert native.cache_dir().parts[-2:] == ("build", "komorebi")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.h").write_text("1")
    key = native.source_key([tmp_path / "a"], tmp_path)
    assert key == native.source_key([tmp_path / "a"], tmp_path)
    (tmp_path / "a" / "x.h").write_text("2")
    assert key != native.source_key([tmp_path / "a"], tmp_path)


def test_image_and_module_are_built_once_into_the_cache(tmp_path, monkeypatch):
    tree = make_tree(tmp_path / "tree")
    monkeypatch.setenv(native.CACHE_VARIABLE, str(tmp_path / "cache"))
    built = []

    def fake_image(tree_, out, sources, includes=(), defines=()):
        built.append(("image", sources, list(includes), list(defines)))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")

    def fake_cxx(kind, tree_, out, source, includes=(), defines=()):
        built.append((kind, source))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")

    monkeypatch.setattr(build, "build_image", fake_image)
    monkeypatch.setattr(build, "build_cxx", fake_cxx)
    image = native.image_path(tree)
    assert (
        image == native.image_path(tree) and image.parent.parent == tmp_path / "cache"
    )
    assert built[0][1] == [native.kernel_dir() / "main.c"]
    module = native.module_path(tree)
    assert module.name == native.MODULE_FILE and module == native.module_path(tree)
    assert [b[0] for b in built] == ["image", "pymodule"]
    variant = native.image_path(tree, defines=["X=1"])
    assert variant != image and built[-1][3] == ["X=1"]


def test_without_a_tree_the_build_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv(native.CACHE_VARIABLE, str(tmp_path / "cache"))
    monkeypatch.delenv("TT_METAL_HOME", raising=False)
    monkeypatch.setattr(_env, "package_dir", lambda: tmp_path / "pkg")
    with pytest.raises(build.BuildError, match="TT_METAL_HOME"):
        native.module_path()
