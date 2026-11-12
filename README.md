# Komorebi

A programming model and runtime for the Tensix grid of Tenstorrent Blackhole cards.

A program is written as straight-line threads. A thread has a program counter and a
place: it stays on its core until it needs state held on another core, then it moves
there or sends a value there. A resident worker on every Tensix core runs the threads;
a small kernel on the card's X280 cores decides what to run.

The repository is being set up. Nothing here is released yet.
