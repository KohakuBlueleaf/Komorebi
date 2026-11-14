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
| [08_parallel_thread](08_parallel_thread/) | parallel code in an X280 thread: a map, a reduction and a two-pass prefix sum on one core and on four with `kmx_parallel`, exact against numpy and timed, and the rules the numbers give |
| [09_bucket_sort](09_bucket_sort/) | the first pass of a bucket sort, driven by an X280 thread: COUNT on the Tensix group, PLAN on 1, 4 or 12 X280 cores, PLACE on the Tensix group, checked against numpy |
| [10_x280_ports](10_x280_ports/) | X280 cores and Tensix cores talk without the dispatcher: each Tensix core gets a value from its X280 core, asks it for more and tells it the hash, on 1, 4 and 12 X280 cores, exact against numpy, with the round trip timed; the serve ends at the command's event |
| [11_movement](11_movement/) | both data-movement CPUs of each core on one stream: blocks dealt to the two movers by a pattern, every word checked through a copy, the read rate beside one mover alone |
| [12_overlays](12_overlays/) | overlays: code loaded into a slot of every core while the kernels run, one overlay with parts on all five cores of a tile; every result checked, each load and call timed |
| [13_thread_commands](13_thread_commands/) | commands to an X280 thread beside commands to core groups: the host and a graph (host-fed and on the X280) chain a grid command and a thread command whose word feeds the next round, every word against numpy, the thread's round beside a grid command's |
| [14_message_handlers](14_message_handlers/) | a message names its handler: threads look a key up at its owner, go on to the value's owner as a second handler and report at their origin as a third, forwarding to other cores and to their own; every count against a numpy replay, messages a second printed |
| [15_leader](15_leader/) | a leader picks what its group does next: it reads values, casts one of four choices as a word (alone, or with a tile), and every core of the broadcast takes that step, 10^6 casts with every word of every core checked |
| [16_two_programs](16_two_programs/) | two programs in one run, each in the runtime's form: on disjoint groups and combined on every core, commands to both interleaved, every result checked; the price of the runtime's form and of a change of program |
| [17_leader_overlay](17_leader_overlay/) | a leader casts which overlay its group runs: each round it reads a value and casts an overlay's number, and every core of the broadcast loads that overlay when it does not hold it and calls it; every core's data exact against numpy, timed on a row, a column and the grid beside the same operations compiled in |
| [18_thread_split](18_thread_split/) | a pull's answer names the code: each core pulls ranges from its X280 core, and the word with each range names one of three overlays, which the core loads if it does not hold it; the thread chooses from what the cores told it, against one overlay for everything and a split fixed by the host, every tile checked |
| [19_thread_programs](19_thread_programs/) | an X280 thread splits the grid between two programs every round: one command from the host, then the thread chooses each round how many cores each program gets from the last round's times, submits both and waits; against the best fixed split and the host feeding it, every unit checked |
| [20_message_overlays](20_message_overlays/) | a message's handler loads where the message arrives: example 14's three handlers as overlay parts under Komorebi's own kernels, every count against the numpy replay, with a slot for each handler and with fewer (loads counted), and a message for a handler the program lacks stopping the run |
| [21_thread_load](21_thread_load/) | an X280 thread gives cores another program: one command from the host, then each round the thread moves the boundary between two programs' groups with `kmx_load` and commands both; against the host issuing every load and command, every core's line of every round checked, and the dispatcher's refusals the thread sees |
| [23_message_body](23_message_body/) | a message carries a body: one thread's 4 KiB body is forwarded from rank to rank and changed at every hop, over 12 cores and the grid, every word against a numpy replay, a hop timed beside the 64-byte message alone |
