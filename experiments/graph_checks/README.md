# Graph checks

Tools for the graph executor (`komorebi.engines.graph` and
`src/komorebi_native/x280/engines/km_graph_engine.c`). They are run by hand, not by
the test suite.

- `graphs/`: 36 named reference graphs, each a `.bin` upload and a `.json` with the
  bindings, the simulation parameters and the answer. 20 run (chains, branches, the
  documented loop and the same upload kept and run with another binding, skips, CPU
  arithmetic, patched arguments and teams, results feeding records, queued and
  exclusive records, more ready records than slots, rounds, and each error at run
  time); 16 are uploads the check refuses, one per fault class. Another
  implementation, such as the kernel's executor, replays them: the issue set, the
  events and the final registers are binding, the order is not.
- `graph_reference.py`: builds them. `python graph_reference.py` rewrites `graphs/`;
  `--check` compares the Python executor and the C engine built for the host with them.
- `compare_engines.py`: random graphs through both executors, and mutated uploads
  through `check` and `kmg_check` (and, when accepted, through both executors). Any
  difference is printed and the exit code is 1.

Run from this directory with the package on the path (a C compiler is needed):

    PYTHONPATH=../../src python graph_reference.py --check
    PYTHONPATH=../../src python compare_engines.py --graphs 3000 --seed 7

Result on the commit that added this folder: 36 references, 0 differences; 3,000
random graphs and 15,000 mutated uploads, 0 differences (about 16 seconds).
