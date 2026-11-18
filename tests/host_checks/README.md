# Host checks

The fast checks off the card, run together by `python -m komorebi.tools.host_check`
(from the repository's root; under a minute on a many-core machine, once the native
module and the mock's loads are built). Each folder's `checks.txt` lists its checks, one
a line: a limit in seconds, then the command. `{root}` is the repository, `{here}` the
folder, `{build}` the logs' folder; leading `NAME=value` words set the environment and
`cwd=P` the working folder. `kmr prebuild` runs a script on the mock card.

- `session/`: the session's open and close forced on the mock (the SIGTERM hook, the
  visible cards, a failed open closing what it built, a second open refused), the host
  transfers' flags forced by stand-ins, the reset's argument checks before the reset,
  the X280 thread image's rebuild rule and the clock line's form.
- `language/`: build-time refusals of the tile language (a reversed rectangle side,
  puts off the 16-byte phase, a broadcast without `.row()` or `.col()`, a local
  command whose groups cross an islands' edge), the tile cost model's table and the
  cost probe's kernels built on the mock.
- `tiles/`: the tile compiler's reader wrap, the Dst window check, wide-row costs, the
  16-bit Dst report, padding rules (on the mock, one card and two) and the refusals that
  name their way.
- `bounds/`: the commands of `experiments/language/bounds/README.md` that give a kernel
  buffer, built on the mock at that size: a kernel that grows past it fails here first.
- `islands/`: the island tensor table's growth and alignment on the mock, the dense
  addition's two-card kernels against linear.py's kernel buffer, and the stream and
  graph references of `experiments/checks/stream_checks` and
  `experiments/checks/graph_checks`.

A check passes on exit 0. Each run keeps its logs in `build/host_check/<time>/` (the last
10 runs); a failing check's line names its log.
