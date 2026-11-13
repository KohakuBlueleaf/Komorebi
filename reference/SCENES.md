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
- a triangle's colour is its material's diffuse colour `Kd`. Textures are not read. For a
  material whose colour comes from a texture (`Kd` white with a `map_Kd`) the table gives
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

## Limits

Not represented: textures (only their mean colour), normal maps, alpha masks (foliage
cut out of quads is drawn as whole opaque quads), transparency (glass is opaque), the
scenes' own lights and emissive materials. Each of these would need a change to the
references (texture coordinates and filtering in the fill and shading, an alpha test in
the depth test and in the triangle test of the ray tracer, more than one light), which
these scenes are not meant to drive.
