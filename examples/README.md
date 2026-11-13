# Examples

Short, runnable programs that each show one thing: a thread program, its host script
and, where one is needed, the kernels it runs. An example runs on its own once
Komorebi is installed against a tt-metal tree (see `docs/install.md`), prints what it
checked, and does not depend on another example.

| Example | What it shows |
|---|---|
| [01_mapcollect](01_mapcollect/) | resident workers under the dispatch kernel, teams named per record, per-team completion, a timeout that names the core |
| [02_merge_tree](02_merge_tree/) | a value sent to one other core: each member's part merged at one owner or along a tree, bit-exact either way; a member that does not send ends in a timeout that names the port |
| [03_broadcast](03_broadcast/) | lane broadcast: blocks along the rows and down the columns of a rectangle team, a stalled receiver that times out with its lane and block named |
| [04_uop_stream](04_uop_stream/) | a micro-op stream decides the teams: random team sizes host-fed, issued one at a time by the X280, and queued by it, with the same tiles every way |
| [05_hop](05_hop/) | a dynamic segment: items hop between cores through per-core queues in DRAM until they retire, the run ends by itself, and every core's totals match a numpy replay |
| [06_graph](06_graph/) | a graph host-fed and on the X280: two branches in flight together, a loop that ends on a result word computed on the card, a kept graph run again with other bindings |
| [07_work_cluster](07_work_cluster/) | a control loop that issues records to changing teams until a result word from the card ends it; one C decision run by an X280 thread (or one on each work cluster at once), host-fed, and on the CPU against a fake ring |
| [08_parallel_thread](08_parallel_thread/) | parallel code in an X280 thread: a map, a reduction and a two-pass prefix sum on one hart and on four with `kmx_par`, exact against numpy and timed, and the rules the numbers give |
| [10_x280_ports](10_x280_ports/) | X280 cores and Tensix cores talk without the dispatch kernel: each Tensix core gets a value from its X280 core, asks it for more and tells it the hash, on 1, 4 and 12 X280 cores, exact against numpy, with the round trip timed; the serve ends at the record's event |
