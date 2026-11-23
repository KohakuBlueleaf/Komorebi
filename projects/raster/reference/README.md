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
python -m projects.raster.reference.pictures --out ${KR_DATA:-build/data}/pictures
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
`python -m reference.scenes`; the checksums are in `tests/core/ref/test_ref_scenes.py`).
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
triangle) and the colour and depth errors. The tests (`tests/core/ref/test_ref_raster*.py`,
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

## Shading: materials and textures (SHADE)

On the scenes with a materials file (Sponza and the two Bistro scenes,
[SCENES.md](../../../reference/SCENES.md)) the frame is coloured from each pixel's
material instead of the vertex colours: a stage SHADE after FILL, defined by
`rasterize_shaded` in `reference/raster.py` and matched by the C++ version
(`raster_cpu --materials DIR`, the shared parts in `raster_core.hpp`). The CUDA version
has no SHADE.

1. **Two FILLs.** FILL runs once over the triangles whose material is not `masked` and
   once over those that are (the other set's triangles are collapsed to a point, so
   setup drops them and the rest keep their numbers), into two sets of planes. A pixel
   takes the masked layer where its fragment is nearer (the fill's own rule: larger
   1/z, a tie to the lower triangle number) and not a hole, otherwise the other
   layer. There is one masked layer: a hole shows the nearest surface that is not
   masked, never a second masked one behind it.
2. **Setup, per triangle.** Its first piece that setup keeps (the clip of the fill's
   setup, with the corners' u, v and light term cut along with the colour) gives the
   planes of u/z, v/z and light/z over the screen, built from the record's edge
   functions like the colour planes. For each of the material's textures one mip
   level: with the piece's doubled area `A` in pixels and `T` in level-0 texels,
   the number of `k = 1 .. levels - 1` with `T > A 2^(2k - 1)` (the level at which a
   pixel is nearest to one texel). For `lit`, one tangent frame: the normal of its
   positions turned toward the eye, the tangent along +u from its positions and
   coordinates (made orthogonal to the normal), the bitangent their cross product,
   turned toward +v.
3. **Per pixel**, from the winner's triangle number and the 1/z FILL wrote: u, v and
   the light term are the planes divided by that 1/z (perspective-correct); the texel
   is the nearest one, coordinates wrapping, row 0 at v = 1. An albedo byte becomes
   linear colour by the pictures' 2.2 gamma; the other textures' bytes are read as
   `b / 255` (normals `2 b / 255 - 1`).

| kind | colour of a pixel |
|---|---|
| `flat` | FILL's vertex colour |
| `textured` | albedo x base colour x light term |
| `masked` | as `textured`; a hole where the mask's alpha byte (or the albedo's) is under 128 |
| `lit` | albedo x base colour x (0.25 + 0.75 max(0, n . l)) + specular colour x (specular texel) x max(0, n . h)^exponent, n the normal map's normal in the frame, l toward the scene's point light, h the half vector |
| `emissive` | albedo x base colour, unlit |
| `transparent` | a third layer over the others ("Glass" below): `tau C + (1 - max tau)` x albedo x base colour x light term, `C` the colour behind, `tau` the transmission |

Everything is float32 in the order written (the power by squaring, low bit first),
so the two versions agree bit for bit. Not part of it: bilinear filtering, a level per
pixel and shadows. The C++ version has no `transparent` kind (below, "Glass"). With one level a triangle and a box-filtered mask, leaves
far from the camera fall below the threshold at the coarser levels and thin out.

What SHADE needs beyond the fill's 128-byte record, per triangle: the three planes
of u/z, v/z and light/z (nine floats), a level per texture, the material number and,
for `lit`, the frame (nine floats). The reference computes them from the scene and the
materials file; the record is unchanged.

```bash
python -m reference.public fetch sponza         # once: the public scene, into KR_DATA
python -m reference.public materials            # once: <scene>_materials.npz
python -m projects.raster.reference.shade agree          # numpy against C++, every view
python -m projects.raster.reference.shade pictures --out ${KR_DATA:-build/data}/pictures
python -m projects.raster.reference.shade stats --out ${KR_DATA:-build/data}/pictures
```

`agree` (results under `$KR_DATA/results/raster_shaded/`) on all six views (2026-11-14):
0 differing pixels in `tri`, `depth` and `colour`, tile list sizes equal. `pictures`
writes each view's vertex colours beside its shaded frame; `stats` the numbers a shading
pass on the cores is sized by: the kinds, materials and winners per 32 x 32 tile, the
kinds each of 120 cores meets with the fill's partition of the tiles, the texture pages
(32 x 32 texels of one level) a tile reads, and the hit rate of a page cache per core.

## Lighting

`rasterize_lighted(scene, materials, lights)` in `reference/raster.py` is SHADE with
direct light in place of the fixed per-vertex shade and the `lit` kind's hand-placed
light: a sun, a sky and point lights at the scene's lamps. It is defined in numpy
only: the C++ and CUDA versions do not have it. Every rule is per-pixel arithmetic, a
lookup in a table or a texel fetch, so a per-tile kernel can follow it; no ray is
traced per pixel. The lights and their precomputed data are in `reference/lights.py`,
the per-scene numbers in the `lighting` entry of `reference/public_scenes.json`
([SCENES.md](../../../reference/SCENES.md)).

```bash
python -m reference.public lighting sponza      # map, visibility bake, lamps, shadow map
```

writes `$KR_DATA/public/<scene>_lighting.npz`; the visibility bake is also kept in
`<scene>_visibility.npz` beside the materials file and reused unless `--rebake`.

1. **Environment.** One HDR environment map per scene (Poly Haven, CC0, the
   `environments` table), turned about the up axis (y) by the scene's `turn` (the
   sun's azimuth changes it: "Moving the sun" below). The
   sun is the brightest small region: the luminance-weighted centre of the excess
   over its surroundings inside 4 degrees of the brightest texel; its colour is that
   excess (luminance 1). The sky is the map with that cone replaced by the mean of
   the ring from 4 to 8 degrees, projected to 9 spherical-harmonic coefficients per
   colour; its irradiance on a normal is the quadratic form of Ramamoorthi and
   Hanrahan (2001), clamped at 0. The sun's irradiance facing it is `sun_to_sky`
   times the sky's irradiance on a horizontal surface (its luminance), in the sun's
   colour. `lights.environment` recomputes these from the file. When the entry has an
   authored `sun` (the direction it shines along, its colour), the sun's direction
   and colour are those; the map is turned about y so its own sun's azimuth equals
   the authored one (`turn` is computed) and the sky still comes from the map with its
   sun removed.
2. **Sky visibility**, baked per vertex: the share of 64 cosine-weighted rays (fixed
   directions per vertex from the hash of `vertex x 64 + ray`) from the vertex, moved
   the scene's `eps` along its area-weighted normal, that leave the scene without a
   hit, traced with the reference ray tracer (`raytrace.trace`, any hit). It takes
   the place of the light term at the corners, so the `light/z` plane interpolates it.
3. **Sun shadow.** The 1/z at `shadow_side` x `shadow_side` (the entry's, default
   2,048, saved with the map) from a camera 1000 scene radii away along the sun's
   direction, aimed at the centre of the scene's box, its frame covering the box
   (`sun_camera(..., side)`: focal and principal point scale with the side). A side of
   2,048 is four renders at FILL's 1,024 x 1,024: quadrant (i, j) uses that camera with
   its principal point moved by (-1024 i, -1024 j) (`quadrant_camera`), so its pixel
   (x, y) is texel (x + 1024 i, y + 1024 j). Each render has SHADE's two FILLs and holes: a texel takes the masked layer's 1/z where
   it is nearer and not a hole, else the other layer's, so plants and chains cast the
   shadow of their leaves, not of their quads. A pixel's point, moved along its
   lighting normal `n` (step 5: the normal map's where there is one, else the
   triangle's turned toward the eye) by one shadow texel's world size, is
   shadowed at a texel when its 1/z in that camera is below the texel's by more than
   the bias (`shadow_bias` texels at the camera's distance); texels outside the map
   are lit. The sun's share is the 2 x 2 percentage-closer filter: the comparisons at
   the four nearest texels, weighted bilinearly. One tap (the nearest texel,
   `pcf=False`) left sawtooth edges where the sun grazes a surface.
4. **Point lights.** Emissive triangles connected through shared vertices form
   groups; groups whose centroids are nearer than `merge_distance` are merged. A lamp
   is a point light at its area-weighted centroid, its colour the area-weighted
   emission colour (base colour x the emission texture's 1 x 1 level, scaled to a
   largest channel of 1), its intensity
   `lamp_scale` x its area. Irradiance at distance `d` on normal `n` toward `l`:
   `I max(0, n.l) / (d^2 + s^2) (1 - (d/r)^4)^2` for `d < r`, 0 from `r` on
   (`radius`, `size`). No shadows from point lights.
   When the scene's entry has a list `lights` (authored point lights: position,
   colour), the lamps are those instead, each of intensity `lamp_scale`; the emissive
   groups are used only when `lights` is absent (an empty list: no lamps).
5. **A pixel.** `a` = albedo x base colour as SHADE has it (`flat`: the base colour).
   The normal is the normal map's in the triangle's frame where the material has one
   and the frame is valid, else the triangle's normal turned toward the eye.
   `E = sky(n) x visibility + ambient + sun x max(0, n.s) x shadow + the lamps`;
   colour `= a / pi x E +` the `lit` kind's Blinn-Phong term once for the sun and
   once per lamp, each times that light's irradiance (with `n.l` and the shadow) over
   pi, only for `lit` materials with a specular colour. `emissive`: `a x
   emissive_scale`, unlit; `masked`: holes as SHADE. Background: the nearest texel of
   the octahedral map (below) in the pixel's view direction turned by `-turn` about y,
   `(x, z) -> (c x + s z, c z - s x)` with `c, s` the float32 cosine and sine of
   `-turn` (y kept). The frame is
   `x / (1 + x)` of `exposure x colour`, then the pictures' 8-bit conversion.
6. **Octahedral map.** `prepare` resamples the environment once into a side x side
   (the entry's `sky_side`, default 2,048; a reader takes it from the map's shape)
   float32 RGB map of directions in the environment's own frame (turn 0, y up), saved
   in the lighting file as `octahedral`, so the background needs no trigonometry per
   pixel and a new `turn` changes a number, never the map. A direction
   `d = (x, y, z)` (need not be unit) goes to `u = x / (|x| + |y| + |z|)`,
   `v = z / (|x| + |y| + |z|)`; when `y < 0` these are folded to
   `(u, v) = ((1 - |v|) sign u, (1 - |u|) sign v)` (both from the unfolded values,
   sign 0 counted as +). Its texel is column `floor((u + 1) / 2 x side)`, row
   `floor((v + 1) / 2 x side)`, each clamped to `side - 1`. The inverse, used to fill
   the map at texel centres `u, v = (i + 0.5) / side x 2 - 1`: `y = 1 - |u| - |v|`; when
   `y < 0`, `(u, v)` unfolded by the same rule; the direction is `(u, y, v)`
   normalized, whose nearest texel of the equirectangular map is taken. The sun and
   sky still come from the equirectangular file. The map is 2,048 and the files 4k
   (4096 x 2048) because a frame's pixel at 75 degrees over 1,024 pixels is 0.07
   degrees, a 512 texel about 0.4 and a 1k file's 0.35: the sky was blocky; a 2,048
   texel is about 0.1 and a 4k file's 0.09. (Before the movable sun the map
   was resampled turned and looked up unturned. On the Bistro exterior's own view the
   change moved 15,536 of its 24,108 background pixels, by 7.2 of 255 on average and
   95 at most, from nearest-texel resampling on a differently turned grid; against a
   direct lookup of the file the mean error went from 5.9 to 5.2. No surface pixel
   changed. Sponza and the interior, turn 0, are unchanged.)

**Moving the sun.** `lights.moved(lights, scene, materials, azimuth, elevation)` takes
the prepared lights (`lights.load`), the scene's mesh and materials, and two angles in
degrees, and returns the lights of a sun at that place without baking visibility again.
The angles are those of the direction toward the sun (`lights.sun_angles`): azimuth 0
at +x, 90 at -z (a right-handed turn about +y, the sense of `turned`), elevation
`atan2(y, sqrt(x^2 + z^2))` above the horizontal plane; the direction is
`(cos e cos a, sin e, -cos e sin a)`. The scenes' own suns: Sponza azimuth 144.08,
elevation 74.53; Bistro exterior azimuth -90, elevation 61.18 (the authored sun).

- **Azimuth** turns the environment with the sun: `turn` grows by the change of
  azimuth, so the map's own sun keeps the sun's azimuth. The sky's 9 coefficients per
  colour are rotated about y (`sh_turned`: the 9 x 9 matrix taking the basis to the
  basis at turned directions, fitted on 128 directions, exact since each band is closed
  under rotation); they agree with projecting the turned map again to float32 rounding
  (under 6e-8). The background follows through `turn` (step 5). The sky's irradiance on
  a horizontal surface does not change under a turn about y, so the sun's stays.
- **Elevation** moves the sun alone: the sky and the background do not change, so the
  map's own sun disc stays where the map has it (a known inconsistency: the
  background's sun and the light's sun part when the elevation moves).
- **Both** give a new direction and a new shadow map (`sun_camera`, `shadow_map`, the
  same rules); the bias stays `shadow_bias` texels at the new camera's distance. The
  sun's colour and irradiance do not change. At or below the horizon (elevation <= 0)
  the irradiance is 0 and the shadow map is a 1 x 1 map of 0 (all lit, unused).
- At the scene's own angles `moved` returns the prepared lights: the same direction,
  coefficients, turn and shadow map (Sponza bit for bit; the exterior's direction has
  x = 2.95e-17 for -0 and the same shadow map; Sponza at side 2,048 checked again bit
  for bit). Its time is the shadow map's: about 10 s for Sponza, 43 s for the Bistro
  exterior at side 2,048, 4 s and 17 s at 1,024 (numpy, 8 CPUs).

**Terms.** `rasterize_lighted(..., terms=...)` takes a subset of `sky`, `ambient`,
`sun`, `shadow`, `lamps`, `specular`, `background` (default: all of them, the frame
above). A term left out contributes nothing: without `shadow` the sun is unshadowed,
without `background` an empty pixel has the fill's constant background colour
(0.05, 0.05, 0.08); emissive materials are always drawn. The card's stages are
checked against these subsets.

**Returned arrays.** `colour` (H, W, 3, linear), `display`, `depth` (the merged 1/z)
and `tri` (the visible triangle, -1 empty), and `parts`, each per pixel of the taken
layer: `albedo` (`a`), `normal` (the normal used), `visibility`, `specular`
(specular colour x texel, 0 where the material has none), `material` (-1 empty),
`depth` (the merged 1/z), `layer` (0 the other layer, 1 the masked layer, -1
empty), and the diffuse parts `sky`, `sun` and `points` (`a / pi` x that light's
irradiance; 0 on emissive pixels).

Everything per pixel is float32. Not part of it: shadows from lamps, bounced light
(`ambient` stands in for it).
Visibility per vertex is coarse on large triangles: Sponza's courtyard floor is a few
triangles whose 23 vertices lie under the arcades, so it gets no sky light.
A shadow texel is 2.25 units on Sponza at side 2,048 (4.5 at 1,024); on a wall the sun
grazes (n.s 0.12 to 0.24) it stretches to about 9 to 19 units (19 to 37), so shadow
edges there are still steps. Turning Sponza's sun by 0.01 / 0.05 / 0.2 degrees changes
10.4 / 10.5 / 11.9 % of its default view's surface pixels by more than 8 of 255 at side
2,048, 11.3 / 11.0 / 13.1 % at 1,024; outside the pixels within 2 of a shadow edge in
both frames 2.2 / 1.6 / 2.1 % against 3.1 / 2.2 / 2.7 % (host, 2026-11-23 16:04). The
larger map sharpens the edges but does not settle them: most of the change is at the
edges and in the speckle of the bias's comparisons, which a turn re-rolls.
The quadrants agree with the 2,048 view: a quadrant's 1/z against a float64
evaluation of the triangle's plane on the 2,048 view's texel ray, 3,000 triangles,
median relative error 6.7e-7 (Sponza; 2.9e-6 for the 1,024 map against its own view);
across a quadrant border one triangle's 1/z steps by up to 6e-5 relative (inside a
quadrant one float32 unit), the two renders' own plane rounding, and no seam shows in a
picture of the map.

## Glass

A material whose transmission `tau` (per channel: `1 - Tf`, else `Tr`, else `1 - d`)
is above 0 has the kind `transparent` and keeps `tau` in the materials file
([SCENES.md](../../../reference/SCENES.md), "Materials"). It is drawn as a third layer
over the surface behind it, in numpy only (`rasterize_shaded` and `rasterize_lighted`
in `reference/raster.py`); no ray is traced, every rule is per-pixel arithmetic, a
table lookup or a texel fetch.

1. **Three FILLs.** FILL runs a third time over the transparent triangles only; the
   other two layers do not contain them. The surface behind is the merge of the other
   two (masked where nearer and not a hole, as SHADE). A pixel's glass is the nearest
   transparent triangle there, counted only where it is nearer than that surface (the
   fill's rule: larger 1/z, a tie to the lower triangle number); an empty pixel's
   surface is the background, so glass in front of the sky counts. One transparent
   layer: glass behind glass is not shown.
2. **A pixel with glass in front**, `C` the colour behind (the lit colour before the
   tone curve, or the background's radiance), `a` the glass's albedo x base colour,
   `n` its triangle's normal turned toward the eye (no normal map, also in the smooth
   variant), `v` toward the eye:
   `colour = tau C + (1 - max tau) a / pi E + S + F reflection R`. `E` is the glass's own
   irradiance by the rules of "Lighting" (sky x visibility, ambient, sun with shadow,
   lamps); `S` the `lit` kind's Blinn-Phong term for the sun and each lamp with the
   material's specular colour and exponent (always, when it has a specular colour);
   `R` the environment's radiance along `2 (n.v) n - v` (the octahedral map's nearest
   texel, turned like the background); `F = F0 + (1 - F0)(1 - n.v)^5`, `F0 = 0.04`,
   `n.v` clamped at 0; `reflection` the scene's number of its `lighting` entry. Then
   the tone curve as for every pixel. `rasterize_shaded` (no lighting) uses
   `tau C + (1 - max tau) a light` with its light term and no `S` or `R`; on scenes without
   transparent materials its results are unchanged (no third FILL runs).
3. **Shadows.** Transparent triangles cast none: the shadow map is `rasterize_shaded`'s
   1/z, which is the surface behind the glass. The baked sky visibility is not rebuilt:
   it still treats glass as opaque.
4. **Terms.** `glass` (default on): left out, transparent triangles are not drawn. `R`
   needs `background`; `S` needs `specular`. `parts` gains the glass layer's own planes
   where it is in front: `glass_depth` (1/z, 0 elsewhere), `glass_tri` and
   `glass_material` (-1 elsewhere), `glass_normal` (`n`). The other parts, `depth` and
   `tri` stay those of the surface behind. `rasterize_shaded` also returns the glass
   layer (`glass`) and where it is in front (`front`).

Chosen by the pictures (2026-11-23): `reflection` 0.5 in both Bistro scenes (in the
interior 0.25 left the clear bottles and glasses hard to see, rims only; in the exterior
the panes show the curtains behind them and a faint reflection); Sponza has no
transparent material. Transparent triangles: Bistro interior 220,840 (the bottles,
jars and glasses, the counter's display glass, the cash register's), exterior 3,778 (the
buildings' window panes `MASTER_Glass_Dirty` and `_Clean`, the street lights, a menu
sign). The bottles pass 0.95 to 0.97 (the `_02` bottles 0.8 to 1 by channel), the panes
0.8: the bottles read as clear glass with a faint reflection and highlights in front of
their opaque liquid (`Paris_LiquorBottle_01_Glass_Wine`), the windows as glass with the
rooms' curtains behind. No picture showed a material that should be clearly darker.

| view | pixels with glass in front | `rasterize_lighted` before, after (s) | `rasterize_shaded` before, after (s) |
|---|---:|---:|---:|
| Bistro interior overview | 5,910 | 9.7, 10.2 | 8.5, 8.7 |
| Bistro interior close | 15,416 | 9.2, 9.5 | 7.4, 8.4 |
| Bistro exterior overview | 30,498 | 14.8, 15.9 | 13.8, 15.7 |
| Sponza overview | 0 | 5.6, 5.7 | 4.8, 4.6 |

Numpy, one process pinned to 8 CPUs, 2026-11-23. Outside the glass the lit frames are
unchanged in the interior (the sun's new shadow map differs in 883 of 1,048,576 texels,
none on a visible surface) and in Sponza (bit for bit, both functions); in the exterior
7,942 pixels of the overview change because the sun now passes through the window
panes (1,803 texels of its shadow map). Not represented: refraction, glass behind
glass, coloured shadows (transparent triangles cast none). The C++ version
(`raster_core.hpp`) and the card's code have no kind 5 yet: on the Bistro scenes
`agree` and the card's checks need it first.
