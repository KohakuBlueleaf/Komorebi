# Reference implementations

This folder holds the plain implementations that the programs under `examples/` and
`projects/` are checked and measured against: numpy, C++ and CUDA versions of the same
computations, the generator of the test scenes, and the downloader of public scenes. The
programs, their CPU tests and the comparison scripts import it; nothing here needs a
card or the framework.

Results from the card are compared against these as statistics (error, bias,
correlation), and exactly on inputs whose result is exactly representable.

| file | what |
|---|---|
| `scenes.py` | deterministic scenes for the rasterization and ray tracing benchmarks |
| `rng.py` | the per-index hash all implementations use for random numbers |
| `raster.py` | rasterization (and its shading stage), the definition the C++ and CUDA versions match |
| `bvh.py` | the binary box tree (8-word nodes), its cut into treelets, saving and loading |
| `raytrace.py` | ray tracing over that tree, the definition the C++ and CUDA versions match |
| `lights.py`, `indirect.py` | the lighting of `rasterize_lighted`; indirect light: elements, the transport matrix in the card's files, the field (raster reference README, "Indirect light") |
| `runner.py` | running the binaries and storing their results, shared by the project scripts |
| `results.py` | the result files an implementation hands to the comparison and picture scripts |
| `power.py` | GPU power from `nvidia-smi` and CPU package energy where readable |
| `png.py` | a PNG writer and colour map for the pictures (numpy and zlib only) |
| `native.py`, `native/` | building and running the C++ and CUDA references; shared headers |
| `meshio.py` | OBJ (with MTL) and PLY readers in numpy |
| `public.py`, `public_scenes.json` | public demo scenes: download, check, convert; their materials and textures ([SCENES.md](SCENES.md)) |

## Scenes

`python -m reference.scenes <folder>` writes four scenes as `.npz`; `make_scene(n)` makes one in
memory. A scene is spheres of mixed size (cube spheres, tessellation proportional to
area, a few large ones near the camera and many small ones far away) over a ground
grid, with per-vertex colour, one camera and one point light. Names and triangle counts:
`tiny` 398, `s10k` 11,994, `s100k` 102,308, `s1m` 997,806.

The arrays: `positions` (V, 3) float32, `colors` (V, 3) float32, `indices` (T, 3)
int32, `camera` (16,) float32 (eye, right, up, forward, focal length in pixels, principal
point, near distance), `light` (3,) float32, `eps` (1,) float32 (ray offset), `meta`
(5,) int32. Geometry uses only addition, subtraction, multiplication, division and square
root in float64, and every random number comes from one 32-bit hash of (index, stream,
seed), so the files are the same on every host. Nothing is shipped: the scenes are
generated.

Public demo scenes (Crytek Sponza, the conference room, the Amazon Lumberyard Bistro, San
Miguel) are downloaded and converted by `python -m reference.public fetch`; see
[SCENES.md](SCENES.md) for the list, the attribution each one asks for and how they are
converted.
