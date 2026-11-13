# Reference renderer tests

These compare the reference renderers of `ref/` and `projects/` with each other: the
C++ and CUDA rasterizers and ray tracers against the numpy ones, bit for bit, and one
slow case of the numpy rasterizer. They test the references, not the framework, so
`pytest tests/core` and the continuous integration do not collect them.

| File | What it checks |
|---|---|
| `test_ref_raster_native.py` | the C++ rasterizer on hand-made triangles, the tiny scene at 1, 3 and 16 threads and the 10^4-triangle scene; the CUDA one on three hand-made cases and the 10^4 scene |
| `test_ref_raytrace_native.py` | the C++ ray tracer on the tiny scene at 1, 3 and 16 threads, hand-made triangles and the 10^4 scene, on a lattice of pixels; the CUDA one on the last two |
| `test_ref_raster_rectangle.py` | a rectangle cut into random triangles is covered exactly once by the numpy rasterizer (one whole-screen rendering per triangle) |

Run them from the repository:

    python -m pytest tests/ref                # C++ and numpy; CUDA skipped
    KMR_GPU=1 python -m pytest tests/ref      # also CUDA, on a GPU with no process on it

The C++ tests need a C++ compiler and are skipped without one. The CUDA tests run only
with `KMR_GPU=1`, `nvcc`, and a GPU that has no process on it; without `KMR_GPU=1` they
never look for a GPU. Binaries and generated data go to a temporary directory.
