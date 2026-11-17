"""The class compiler in stages, driven by komorebi.dsl.compiler: a handler is a `frame`
that handler/ makes and lowers; here the stages' foundation (records, source, cxx,
layout, scope, expressions, handler_buffers, tile_lookup), then dst/, pages/, casts/,
tiles/, each importing only those before it; families recurse by frame.statements."""
