# Rasterization: reference implementations

The same rasterization algorithm three times, with no card and no framework: numpy
(`reference/raster.py`, the definition of correct), threaded C++ for the host CPU
(`cpu/`) and CUDA for one GPU (`gpu/`). The card versions are compared with these:
they use the same scenes ([`reference/scenes.py`](../../../reference/README.md)) and must produce the
same images.

![tiny scene](images/tiny_colour_256.png) ![10^4 scene](images/s10k_colour_256.png)

*Colour image of the tiny scene (398 triangles) and the 10^4 scene (11,994 triangles),
256 x 256 (the images are 1024 x 1024).*

## The algorithm

Opaque triangles, depth-tested, 1024 x 1024 pixels, tiles of 32 x 32 pixels (a 32 x 32
grid of tiles), float32 arithmetic in the order written in `reference/raster.py`, no fused
multiply-add.

1. **Setup**, one triangle at a time (no triangle depends on another). The three
   vertices are taken into camera space (`x`, `y`, `z` along right, up, forward). A
   triangle with no vertex beyond the near distance `n` (`z > n`) is dropped; one with
   all three beyond is kept whole; any other is **clipped** at the plane `z = n`: its
   edges are walked in order (0-1, 1-2, 2-0), each vertex beyond the plane is kept and
   each edge that crosses the plane adds the point where it crosses, computed from the
   vertex beyond (`a`) toward the other (`b`) as `s = (n - z_a) / (z_b - z_a)`,
   `x = x_a + s (x_b - x_a)`, `y` likewise, `z = n`, colour `c_a + s (c_b - c_a)`
   (float32, in this order). The 3 or 4 points give one triangle or two (0, 1, 2 and
   0, 2, 3), each with the original triangle's index; they go through the rest of the
   setup like whole triangles, so up to two records come from one triangle. Every
   point is then projected (`1/z`, screen `x` and `y`). Each triangle is turned to
   positive orientation by swapping two vertices, a zero-area triangle is dropped, and
   the record is built: three edge functions `a x + b y + c`, the planes of `1/z` and of the
   three colour channels (linear in screen space, no perspective correction), the box of
   pixel centres it can cover (empty box: dropped), and which edges are top or left.
2. **Binning.** A triangle goes to every tile its pixel box touches, in triangle order.
   The sizes of the tile lists are an output.
3. **Fill**, one tile at a time. A pixel centre `(x + 0.5, y + 0.5)` is inside when
   all three edge values are `> 0`, or `== 0` on a top or left edge (the top-left fill
   rule: the edge of a shared diagonal belongs to exactly one of its two triangles). The
   fragment with the larger `1/z` wins; an exact tie goes to the lower triangle index, so
   the result does not depend on the order of the list. Empty pixels keep the background
   colour and depth 0.

Outputs of every implementation (the arrays the comparison reads): `colour`
(1024, 1024, 3) float32, `depth` (the stored `1/z`; 0 where empty) (1024, 1024) float32,
`tri` (the index of the visible triangle, -1 where empty) (1024, 1024) int32 and
`tile_counts` (the list sizes) (32, 32) int32.

| | numpy | C++ | CUDA |
|---|---|---|---|
| source | `reference/raster.py` | `cpu/raster_cpu.cpp` | `gpu/raster_gpu.cu` |
| setup | all triangles as arrays | threads over chunks of 256 triangles | one thread per triangle |
| binning | sort of (tile, triangle) pairs | per-thread counts, prefix sum, scatter in order | atomic counts, one block scans, atomic scatter |
| fill | tile by tile, 128 triangles at a time as arrays | threads over tiles, largest list first | one block of 32 x 8 threads per tile, tiles sorted by list size, 64 records in shared memory at a time |

The C++ and CUDA versions share the setup and the pixel test (`raster_core.hpp`), so
they cannot drift apart; numpy is written separately.

## Running

```bash
python -m pip install -e ".[dev]"          # numpy, pytest, ruff, black
export KR_DATA=<folder for scenes and outputs>    # default: build/data
python -m reference.scenes $KR_DATA                      # the four scenes as .npz
python -m projects.raster.reference.agree          # numpy vs C++ vs CUDA (--fresh: rerun)
python -m projects.raster.reference.bench          # timings and power
python -m projects.raster.reference.pictures --out <folder>
```

## Result files and other implementations

Each implementation leaves, per scene, `<folder>/<scene>.npz` and `<folder>/<scene>.json`
(a folder per implementation; the three here write theirs under
`$KR_DATA/results/raster/`). `agree.py`, `pictures.py` and `bench.py` take
`--external DIR ...`: every sub-folder of DIR that holds files for the scene is added as
another implementation, so a card version only has to drop its files into a folder and
appears in the comparison, as a column of pictures and as a row of the table.

| array in `<scene>.npz` | shape, type | content |
|---|---|---|
| `colour` | (1024, 1024, 3) float32 | linear colour, background 0.05, 0.05, 0.08 |
| `depth` | (1024, 1024) float32 | the stored 1/z of the visible fragment, 0 where empty |
| `tile_counts` | (32, 32) int32 | triangles in each tile's list, row-major by tile row |
| `tri` (optional) | (1024, 1024) int32 | visible triangle index, -1 where empty |

`<scene>.json`: `{"name": ..., "scene": ..., "seconds_per_frame": ...,
"timings_ms": {"setup_ms": ..., "bin_ms": ..., "fill_ms": ..., "total_ms": ...},
"power": {"gpu_idle_w": ..., "gpu_mean_w": ..., "gpu_peak_w": ...,
"gpu_joules_per_frame": ..., "cpu_package_joules_per_frame": ...}}`. Only `name` and
`scene` are required; the stage times are what the picture page and table show. Scene
names are `tiny`, `s10k`, `s100k`, `s1m` (the scene files are generated by
`python -m reference.scenes`; the checksums are in `tests/core/test_ref_scenes.py`).
Differences are counted over `colour`, `depth` and, when both have it, `tri`.

Binaries are built on first use into `build/` (`KR_BUILD_DIR`) with `g++` (or `$CXX`) and
`nvcc` (`$NVCC`, `$CUDA_HOME` or `/usr/local/cuda`; `KR_CUDA_ARCH`, default `sm_70`). The
GPU runs use a GPU with no process on it, found with `nvidia-smi`; both the CUDA tests
and the CUDA rows are skipped when there is none.

## Agreement

All three implementations produce the same arrays bit for bit, on all four scenes
(numpy against C++ with 96 threads, and C++ against the CUDA version), including the
tile list sizes:

| scene | triangles | differing pixels (`tri`, `depth`, `colour`) | tile list sizes |
|---|---|---|---|
| tiny | 398 | 0 / 0 / 0 | equal |
| 10^4 | 11,994 | 0 / 0 / 0 | equal |
| 10^5 | 102,308 | 0 / 0 / 0 | equal |
| 10^6 | 997,806 | 0 / 0 / 0 | equal |

Why exact: every operation is a float32 add, subtract, multiply, divide or compare in a
fixed order (no fused multiply-add: `-ffp-contract=off`, `--fmad=false`), all
IEEE-exact, and the winner of a pixel does not depend on the order fragments arrive in.
`agree.py` reports, for any difference, how many pixels differ in each array, how many
of those lie on a triangle edge (a pixel whose 3 x 3 neighbourhood holds more than one
triangle) and the colour and depth errors. The tests (`tests/core/test_ref_raster*.py`,
`tests/ref/test_ref_raster_*.py`) check the fill rule on hand-made triangles (shared
diagonal, top, bottom, left and right edges, a rectangle cut into random triangles
covered exactly once), depth order, ties, culling, and run the C++ and CUDA versions on
hand-made and generated scenes against numpy; changing `>` to `>=` in the pixel test
makes six of them fail.

A card implementation will not be bit-exact if it uses another order of operations;
compare coverage and depth per pixel and count the differing pixels, which sit on
triangle edges.

## Timings and power

What is timed, per repetition: **setup** (all triangles), **binning** (counts, offsets,
lists, and for CUDA the read-back of the list size), **fill** (clearing the tile, the
triangle loop and writing the 1024 x 1024 colour, depth and index images). Not timed:
loading the scene, copying the mesh to the GPU, copying the images back. CPU: `std::thread`
pool, wall-clock time. GPU: CUDA events on the default stream, one Tesla V100 32 GB
(device name `Tesla PG503-216`, compute capability 7.0; one of four, idle when used).
Stage cells: median over 3 processes x 3 repetitions after one untimed warm-up
repetition per process, min and max in brackets, milliseconds.

"ms per frame" and the power columns come from one more run of at least 3 seconds per
row: wall time of the loop over frames divided by the number of frames. The GPU's draw is
`nvidia-smi` `power.draw` of the one GPU, sampled every 100 ms inside the loop (idle: the
second before the process started; mean and peak over the loop); joules per frame is the
mean draw times the seconds per frame. Only the GPU's own board power is included, not
the host CPU that drives it. The CPU package energy counter (RAPL) of this host is not
readable without root, so CPU joules are **not available** here (`bench.py` records them
when the host allows it). The host is a shared 96-thread AMD EPYC 7K62 (48 cores, SMT);
the first load average (`/proc/loadavg`) was between 6.6 and 12.4 during these runs, with
no other CUDA process on the GPUs; the per-process loads are in the JSON that
`bench.py --json` writes. The all-thread CPU numbers have the widest spread.

| scene | implementation | setup | binning | fill | total | ms per frame | GPU W idle / mean / peak | GPU J per frame |
|---|---|---|---|---|---|---|---|---|
| tiny | C++ 1 thread | 0.03 (0.03-0.03) | 0.02 (0.02-0.02) | 11.12 (11.04-11.22) | 11.17 (11.09-11.28) | 11.21 | n/a | n/a |
| tiny | C++ 96 threads | 0.25 (0.22-0.34) | 0.89 (0.85-1.21) | 1.01 (0.92-1.07) | 2.19 (2.01-2.43) | 2.66 | n/a | n/a |
| tiny | V100 | 0.07 (0.07-0.08) | 0.17 (0.16-0.17) | 0.06 (0.06-0.06) | 0.30 (0.29-0.31) | 0.29 | 55 / 89 / 91 | 0.026 |
| 10^4 | C++ 1 thread | 0.87 (0.86-1.37) | 0.17 (0.16-0.25) | 15.65 (15.59-16.03) | 16.68 (16.63-17.66) | 16.79 | n/a | n/a |
| 10^4 | C++ 96 threads | 0.33 (0.30-0.48) | 1.16 (0.90-1.57) | 1.42 (1.13-1.47) | 2.91 (2.48-3.36) | 3.32 | n/a | n/a |
| 10^4 | V100 | 0.03 (0.03-0.04) | 0.05 (0.05-0.06) | 0.16 (0.14-0.16) | 0.25 (0.22-0.26) | 0.25 | 57 / 112 / 114 | 0.028 |
| 10^5 | C++ 1 thread | 7.16 (7.15-7.21) | 1.23 (1.21-1.74) | 27.87 (27.80-31.14) | 36.26 (36.17-40.04) | 39.90 | n/a | n/a |
| 10^5 | C++ 96 threads | 0.92 (0.81-1.11) | 1.22 (1.11-1.32) | 1.90 (1.78-2.36) | 3.98 (3.76-4.77) | 4.08 | n/a | n/a |
| 10^5 | V100 | 0.12 (0.11-0.12) | 0.08 (0.07-0.08) | 0.57 (0.57-0.57) | 0.76 (0.75-0.78) | 0.79 | 65 / 110 / 114 | 0.087 |
| 10^6 | C++ 1 thread | 67.03 (66.89-68.86) | 15.10 (15.06-15.47) | 69.86 (69.70-70.63) | 152.32 (151.79-153.73) | 152.78 | n/a | n/a |
| 10^6 | C++ 96 threads | 4.86 (4.43-6.43) | 5.09 (4.69-7.07) | 3.06 (2.85-5.56) | 12.99 (11.99-18.25) | 17.87 | n/a | n/a |
| 10^6 | V100 | 0.77 (0.75-0.88) | 0.27 (0.27-0.32) | 3.34 (3.32-3.89) | 4.37 (4.35-5.09) | 4.42 | 55 / 110 / 115 | 0.484 |

Reading the table: at the three smaller sizes the 1-thread fill time is mostly
clearing and writing the output images (about 10 ms for 1024 x 1024 pixels on one core;
the tiny scene has almost no triangle work), so the CPU time grows little from tiny to
10^4. At 10^6 triangles setup (67 ms) is as large as the fill (70 ms) and binning
adds 15 ms; most of the 10^6 triangles are smaller than a pixel (5.5 % of them are
visible), so the tile lists are short but there are many entries (626,033, the
largest list has 8,780 triangles). No widely used software rasterizer was installed
for context: OpenGL needs a driver stack that the host's compute GPUs do not offer.

## Scenes and the work they make

| scene | triangles | after setup | list entries | largest list | non-empty tiles of 1024 | covered pixels |
|---|---|---|---|---|---|---|
| tiny | 398 | 289 | 4,047 | 28 | 664 | 62.8 % |
| 10^4 | 11,994 | 7,534 | 17,121 | 232 | 672 | 63.8 % |
| 10^5 | 102,308 | 67,219 | 92,778 | 1,302 | 709 | 66.9 % |
| 10^6 | 997,806 | 549,341 | 626,033 | 8,780 | 754 | 71.2 % |

About a third of the screen is empty sky (empty tiles), a few large near spheres give
tiles with thousands of small triangles, and the ground gives a gradient of
triangle size with distance.
