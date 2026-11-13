# manifest

Not implemented yet.

This package will hold the manifest of a program: its commands (the fixed-size values
that travel between threads), its homes (the state each core owns) and its handlers
(the code that runs between two breakpoints). From the manifest the package will
generate the glue between a program's compute code and the persistent kernel: command
layouts, handler tables and the group and placement tables the host uploads.

Until then, programs are written by hand against the headers in `komorebi_native`.
