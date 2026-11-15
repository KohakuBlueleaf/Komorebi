# Stream checks

Tools for hunting slips in the micro-op stream simulator (`komorebi.engines.stream`).
They are not part of the test suite: the mutation run takes about nine seconds.

- `streams/`: 91 named reference streams, each a `.bin` upload with a `.json` beside it
  holding the simulation parameters and the simulator's answer. Another implementation,
  such as the kernel's interpreter, can replay them.
- `stream_reference.py` and `stream_reference_more.py`: build the reference streams.
  `python stream_reference.py` rewrites `streams/`; `--check` compares them with what
  the simulator gives and exits non-zero if any differs.
- `mutants.py`: about a hundred small text patches of the simulator and the check. Each
  is loaded as its own module and replayed on the references; every one must change some
  saved answer. It prints the survivors and exits non-zero if there is one.

Run from this directory with the package on the path:

    PYTHONPATH=../../../src python mutants.py
    PYTHONPATH=../../../src python stream_reference.py --check

Result: 104 of 104 mutants killed, 0 of 91 streams differing, about nine seconds
for the mutants. A change to the simulator's source that renames or reflows a line
makes the patches that name it fail to apply (`load` asserts each matches once): the
patches then follow the source.
