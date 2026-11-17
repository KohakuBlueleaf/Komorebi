"""The class compiler in stages, driven by komorebi.dsl.compiler: each handler is a
`frame` that `handler_compile` makes and `statements` lowers, a feature a module of
functions of the frame (below them `records`, `source`, `cxx`, `layout`, `dst_code`);
the families recurse into a body through the frame's `statements` field."""
