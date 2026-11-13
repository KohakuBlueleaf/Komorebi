# Examples

Short, runnable programs that each show one thing: a thread program, its host script
and, where one is needed, the kernels it runs. An example runs on its own once
Komorebi is installed against a tt-metal tree (see `docs/install.md`), prints what it
checked, and does not depend on another example.

| Example | What it shows |
|---|---|
| [01_mapcollect](01_mapcollect/) | persistent kernels under the dispatcher, groups named per command, per-group completion, a timeout that names the core |
| [02_merge_tree](02_merge_tree/) | a value sent to one other core: each core's part merged at one owner or along a tree, bit-exact either way; a core that does not send ends in a timeout that names the port |
| [03_broadcast](03_broadcast/) | broadcast: blocks along the rows and down the columns of a rectangle group, a stalled receiver that times out with its broadcast and block named |
| [04_uop_stream](04_uop_stream/) | a micro-op stream decides the groups: random group sizes host-fed, submitted one at a time by the X280, and queued by it, with the same tiles every way |
| [05_forward](05_forward/) | a dynamic handler: messages are forwarded between cores through per-core queues in DRAM until they are consumed, the dispatcher finds the run's quiet end, and every core's totals match a numpy replay |
| [06_graph](06_graph/) | a graph host-fed and on the X280: two branches in flight together, a loop that ends on a result word computed on the card, a kept graph run again with other bindings |
| [07_work_cluster](07_work_cluster/) | a control loop that submits commands to changing groups until a result word from the card ends it; one C decision run by an X280 thread (or one on each work cluster at once), host-fed, and on the CPU against a fake ring |
| [08_parallel_thread](08_parallel_thread/) | parallel code in an X280 thread: a map, a reduction and a two-pass prefix sum on one core and on four with `kmx_par`, exact against numpy and timed, and the rules the numbers give |
| [09_bucket_sort](09_bucket_sort/) | the first pass of a bucket sort, driven by an X280 thread: COUNT on the Tensix group, PLAN on 1, 4 or 12 X280 cores, PLACE on the Tensix group, checked against numpy |
| [10_x280_ports](10_x280_ports/) | X280 cores and Tensix cores talk without the dispatcher: each Tensix core gets a value from its X280 core, asks it for more and tells it the hash, on 1, 4 and 12 X280 cores, exact against numpy, with the round trip timed; the serve ends at the command's event |
