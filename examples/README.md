# Examples

Short, runnable programs that each show one thing: a thread program, its host script
and, where one is needed, the kernels it runs. An example runs on its own once
Komorebi is installed against a tt-metal tree (see `docs/install.md`), prints what it
checked, and does not depend on another example.

| Example | What it shows |
|---|---|
| [01_mapcollect](01_mapcollect/) | resident workers under the dispatch kernel, teams named per record, per-team completion, a timeout that names the core |
| [07_work_cluster](07_work_cluster/) | an X280 thread on a work cluster: regions, `kmx_par` on four cores, the result block, checked against its host fallback |
