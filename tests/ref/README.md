# Reference renderer tests

These compare the reference renderers of `reference/` and `projects/` with each other: the
C++ and CUDA rasterizers and ray tracers against the numpy ones, bit for bit, and one
slow case of the numpy rasterizer. They test the references, not the framework;
the continuous integration runs them with the CPU tests, the CUDA ones (marked `gpu`)
deselected.

| File | What it checks |
|---|---|
| `test_ref_raster_native.py` | the C++ rasterizer on hand-made triangles, the tiny scene at 1, 3 and 16 threads and the 10^4-triangle scene; the CUDA one on three hand-made cases and the 10^4 scene |
| `test_ref_raytrace_native.py` | the C++ ray tracer on the tiny scene at 1, 3 and 16 threads, hand-made triangles and the 10^4 scene, on a lattice of pixels; the CUDA one on the last two |
| `test_ref_files.py` | four makers of one scene at once leave it whole and no file of their own; a failed save leaves nothing |
| `test_ref_raster_rectangle.py` | a rectangle cut into random triangles is covered exactly once by the numpy rasterizer (one whole-screen rendering per triangle) |
| `test_ref_indirect.py` | indirect light: in a closed box the light one element sends is what the others receive by the form factors; a pixel between two voxels blends them; an element behind a wall feeds only the directions it faces; the hierarchical apply equals the dense one |
| `test_ref_tungsten.py` | the Tungsten scene reader on a hand-written scene (an OBJ and a `.wo3` mesh, a texture, a quad emitter, a tinted glass, the camera): positions, winding, materials' kinds, `inside` |
| `test_ref_glass.py` | the glass layer on hand-made quads: a glass of transmission 1 leaves the colour behind unchanged, one of transmission 0 hides it, a nearer opaque surface hides the glass (`rasterize_lighted` and `rasterize_shaded`) |

Run them from the repository:

    python -m pytest -m "not gpu" tests/ref   # C++ and numpy
    KMR_GPU=1 python -m pytest tests/ref      # also CUDA, on a GPU with no process on it

The C++ tests need a C++ compiler and are skipped without one. The CUDA tests run only
with `KMR_GPU=1`, `nvcc`, and a GPU that has no process on it; without `KMR_GPU=1` they
never look for a GPU. Binaries and generated data go to a temporary directory.
