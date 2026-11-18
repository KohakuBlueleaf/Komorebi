"""N3 check (no card, about 10 s: four X280 builds): the thread image's cache key is
its build (sources' contents, the commands, the tree's linker script); the prebuilt
dispatcher image is taken only when no tree is given."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

S = Path(tempfile.mkdtemp(prefix="n3-"))
os.environ["KMR_CACHE"] = str(S / "cache")

from komorebi import _env  # noqa: E402
from komorebi.device import native  # noqa: E402

SOURCE = S / "clusters.c"
shutil.copy(Path(__file__).with_name("n3_clusters.c"), SOURCE)


class Stand:
    includes = ()

    def source_paths(self):
        return [SOURCE]

    def all_defines(self, programs=()):
        return {}


def build(label):
    out = native.thread_image(Stand())
    print(f"{label:30} {out.parent.name}")
    return out


tree = native.metal_tree()
first = build("first")
again = build("unchanged")
SOURCE.write_bytes(SOURCE.read_bytes() + b" ")
changed = build("one byte added to the source")
script = _env.x280_linker_script(tree)
copy = S / script.name
copy.write_bytes(script.read_bytes() + b"\n")
_env.x280_linker_script = lambda t: copy
moved = build("the tree's linker script changed")
fake = S / "package" / "komorebi"
fake.mkdir(parents=True)
(fake.parent / "komorebi_native").symlink_to(native.native_directory())
(fake / native.IMAGE).write_bytes(b"prebuilt")
_env.package_directory = lambda: fake
default = native.image_path() == fake / native.IMAGE
given = native.image_path(tree) != fake / native.IMAGE
print(f"image_path() prebuilt: {default}; image_path(tree) built: {given}")
ok = (
    again == first
    and changed.parent != first.parent
    and moved.parent != changed.parent
    and default
    and given
)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
