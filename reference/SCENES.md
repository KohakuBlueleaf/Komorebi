# Public scenes

Besides the four generated scenes (`reference/scenes.py`), the rasterizer and ray tracer
references run on public demo scenes. Nothing of them is in this repository: `public.py`
downloads each one from its source, checks the SHA-256 recorded in `public_scenes.json`
and converts it to the scene layout of `reference/README.md`.

```sh
python -m reference.public list                  # scenes, views and attribution
python -m reference.public fetch sponza          # download, check, convert (all scenes if none named)
python -m projects.raster.reference.agree --scenes sponza_overview
```

Files go to `$KR_DATA/public/` (downloads in `downloads/`, the converted mesh as
`<scene>.npz`); each view is written as a scene of its own, `$KR_DATA/<scene>_<view>.npz`
and `$KR_DATA/<scene>_<view>/*.npy`. The reference scripts take a view name wherever
they take a scene name, and make the view from the converted mesh on first use
(`native.ensure_scene`). Only the fetch command downloads: a script given a scene that
has not been fetched stops with an error that names the command.

## Scenes and attribution

| name | scene | triangles | download | licence |
|---|---|---:|---:|---|
| `conference` | Conference room | 124,619 | 1.9 MB | credit required |
| `sponza` | Crytek Sponza | 262,266 | 80 MB | CC BY 3.0 |
| `bistro_interior` | Amazon Lumberyard Bistro, interior | 1,020,903 | 21 MB | CC BY 4.0 |
| `bistro_exterior` | Amazon Lumberyard Bistro, exterior | 2,833,330 | 64 MB | CC BY 4.0 |
| `san_miguel` | San Miguel 2.0 | 9,963,412 | 536 MB | CC BY 3.0 (see below) |

Triangle counts are after conversion. Attribution, as each source asks:

- `conference`: Conference Room by Anat Grynberg and Greg Ward, from Morgan McGuire's
  Computer Graphics Archive (https://casual-effects.com/data)
- `sponza`: Crytek Sponza by Frank Meinl, Crytek (2010), from Morgan McGuire's Computer
  Graphics Archive (https://casual-effects.com/data)
- `bistro_interior`, `bistro_exterior`: Amazon Lumberyard Bistro, Open Research Content
  Archive (ORCA), (c) 2017 Amazon Lumberyard; OBJ version from Morgan McGuire's Computer
  Graphics Archive (https://casual-effects.com/data)
- `san_miguel`: San Miguel by Guillermo M. Leal Llaguno (Evolucien Visual), 2017 version
  by Morgan McGuire, Guedis Cardenas, Michael Mara and Nicholas Hull, from Morgan
  McGuire's Computer Graphics Archive (https://casual-effects.com/data)

The archive page lists San Miguel as CC BY 3.0; the licence file inside the download says
"free for research and educational use with attribution". Use it under the narrower of
the two. Pictures of these scenes made with this repository carry the same attribution.

## Conversion

`meshio.py` reads OBJ (with its MTL) and PLY, from a file or from inside a zip, tar or gzip
archive, with numpy only:

- polygons are cut into fans from their first vertex;
- degenerate triangles (a repeated index, two equal positions, exactly collinear corners)
  are dropped and counted;
- a triangle's colour is its material's diffuse colour `Kd`; the scene file reads no
  texture (the materials file below does). For a material whose colour comes from a texture (`Kd` white with a `map_Kd`) the table gives
  a colour instead (`material_colours`): the mean linear colour of that texture over its
  opaque texels, times `Kd`, computed once from the downloaded textures and recorded as
  data;
- a ground plane (a 16 x 16 checker) is added under a single object when the table asks
  for it (`ground`, its half width as a multiple of the object's width).

Vertices are split by (vertex, colour), so every triangle keeps a flat colour. The
rasterizer has no lighting, so the colour is multiplied by a fixed shade per vertex,
`0.55 + 0.45 |n . d|` with `d = normalize(0.3, 1, 0.5)` and `n` the area-weighted
normals of the triangles around the vertex; without it a room is a few flat areas. The
ray tracer uses the same colours.

Per view the table gives the eye, the target and the field of view (square image,
1024 x 1024); the light is one point per scene. The ray offset `eps` is `2e-5` of the
scene's diagonal and the near distance `1e-4` of it unless the view gives `near`.

`$KR_SCENES_EXTRA` names a further JSON file in the layout of `public_scenes.json`; its
scenes are added to the table (for scenes used locally whose terms do not allow
publishing pictures).

## Materials

Sponza and the two Bistro scenes also have their materials and textures, in a file of
their own beside the scene file, which stays as above (everything that reads
`<scene>.npz` is unchanged):

```sh
pip install -e ".[scenes]"                      # Pillow, for decoding the textures
python -m reference.public materials sponza     # <scene>_materials.npz (all three if none named)
```

The command needs the scene converted (`fetch`) and the texture files of the table
(`texture_files`: the Bistro textures are three archives of their own, kept in
`downloads/textures/`, saved as the `file` their entry names: the archive serves them
without an extension); it downloads only what is missing, checked by SHA-256 like the
meshes. Pillow is the `scenes` extra of `pyproject.toml` and is imported by this
command only; without it the command stops with an error that names it.
`load_materials(name, data)` gives the arrays (a view's name works too).

| array | shape, type | content |
|---|---|---|
| `uv` | (T, 3, 2) float32 | texture coordinates of each triangle's corners, in the scene file's triangle order (0 where there are none) |
| `material` | (T,) int32 | the triangle's material |
| `light` | (T, 3) float32 | the light term of each corner, the shade the scene file's colours are multiplied by (`vertex_shade`), kept apart |
| `kind` | (M,) int32 | the material's kind of shader: 0 `flat`, 1 `textured`, 2 `masked`, 3 `lit`, 4 `emissive` |
| `base_colour` | (M, 3) float32 | `Kd`; for `emissive` without a texture, `Ke` clamped to 1 |
| `textures` | (M, 4) int32 | the material's albedo, mask, normal and specular texture, -1 for none |
| `exponent` | (M,) int32 | the specular exponent `Ns`, rounded, 1 to 1024 |
| `specular` | (M, 3) float32 | `Ks` |
| `texels` | (B,) uint8 | every level of every texture, RGBA8, rows from the top |
| `mip` | (N, 11, 3) int64 | per texture and level: the level's byte offset in `texels`, width, height |
| `levels` | (N,) int32 | the levels of each texture's chain |

Material 0 is `flat`: the generated ground, and every triangle without texture
coordinates or without a material. A material's kind comes from its MTL statements, in
this order: a glow (`map_Ke`, or `Ke` of 0.5 or more in a channel) makes it `emissive`,
its albedo the `map_Ke` texture; without `map_Kd` it is `flat` (the scene file's
colour); with `map_d` it is `masked` (the mask is `map_d`; when `map_d` names the albedo
image itself, the albedo's alpha); with a bump map (`map_bump` or `bump`) or a specular
map (`map_Ks`) it is `lit`; otherwise `textured`. Glass and other transparency are
opaque.

Textures are stored as decoded, RGBA8; a grey image is copied into the three colours. A
texture is reduced by halving (2 x 2 box filter, rounded) until neither side is above
1,024: most of the Bistro's textures are 2,048 or 4,096 texels wide and are kept at
1,024. Its mip chain halves again down to 1 x 1 (a side of 1 stays 1). A mask keeps its
alpha when it has one that is not all opaque, else its first channel, in all four
channels. Sponza's bump maps are heights; they are turned into tangent-space normal maps
(central differences, slope gain 2) before the chain is built; the Bistro's are normal
maps already. The same image used in two roles is stored once per role.

The converted sizes: Sponza 26 materials, 37 textures, 190 MB of texels; Bistro
interior 67 and 88, 373 MB; Bistro exterior 133 and 208, 1,007 MB.

## Limits

Not represented: transparency (glass is opaque), the scenes' own lights (one point light
per scene; emissive materials are drawn unlit and light nothing) and San Miguel's
textures (only their mean colour). Without the materials file, as in the ray tracer,
textures are represented by their mean colour and foliage cut out of quads is drawn as
whole opaque quads.
