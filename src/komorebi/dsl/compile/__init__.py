"""The class compiler in stages, driven by komorebi.dsl.compiler: each handler is a
`frame` that handler/'s `handler_compile` makes and `statements` lowers, a feature a
module of functions of the frame. Below them, here: `records`, `source`, `cxx`,
`layout`, `scope`, `expressions`, `handler_buffers`, `tile_lookup`; then dst/, pages/,
casts/ and tiles/, each importing only those before it. The families recurse into a
body through the frame's `statements` field."""
