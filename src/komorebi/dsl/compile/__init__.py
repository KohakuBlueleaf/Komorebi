"""The class compiler in stages, driven by komorebi.dsl.compiler: here the foundation
(records, frame, layout, source, cxx, scope, expressions, handler_buffers, tile_lookup),
then dst, pages, casts, tiles, handler: each imports only those before it, compile's
own modules, front, islands and the tile layer."""
