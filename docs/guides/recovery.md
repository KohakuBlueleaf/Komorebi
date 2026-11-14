---
title: Recover from a command that does not return
summary: What the card does when a command misses its deadline, how to read what it says, how one core is reset and loaded again without a card reset, and the cases in which a card reset is still the only way.
tags:
  - guide
  - debugging
  - recovery
---

# Recover from a command that does not return

Use this when a command has not returned and you want the run, or at least the card, to
survive it. [Debug a command that does not finish](debug-a-command-that-does-not-finish.md)
is the guide for finding the missing call in a kernel; this one is about what the card
does meanwhile and what is left for you to do. The pages with every rule are
[giving up one command](../reference/runtime.md#giving-up-one-command),
[inspecting a run](../reference/host.md#inspecting-a-run) and the
[watch on the dispatcher](../concepts/dispatch.md#the-watch).

## The ladder

A step is tried only when the one before it was not enough.

1. **The deadline.** Every command has one: `session.load(program, deadline=s)` is the
   seconds a command may take from submit to its group's completion, waiting behind
   earlier commands included (default 1.0). `wait=s` is the cores' own timeout on a wait
   for another core (default 2.0).
2. **The give-up.** At the deadline the dispatcher reads the group's state (an inspect),
   writes the command's sequence number into the slot's abort word on every core, and
   every runtime wait that polls the word returns. The cores leave the command, clear
   its pages and take the next one. The dispatcher frees the slot and sends the timeout
   once no kernel of the group runs the command. `run.wait` raises `km.CommandTimeout`
   with `given_up` True, and the run goes on.
3. **The reset of the cores that did not leave.** A core still in the command 20 ms after
   the abort, in a wait that polls nothing, is reset and loaded again by the dispatcher,
   if the dispatcher holds the write list for it (below). `CommandTimeout.recovered` names
   those cores. Without the list, that core ends the run with `KMD_ERROR_DEADLINE`.
4. **The card reset.** What none of these reaches, below.

## Steps

1. **Look while it hangs.** Submit the command, leave it time to hang (a quarter of the
   deadline, in the examples), then `run.inspect(group)` asks the dispatcher to read each
   core of the group:

    ```python
    handle = run.submit("hang", (0, 2))
    time.sleep(0.25)
    cores = run.inspect((0, 4))
    for core in cores:
        print(core["core"], core["live"]["reader"]["where"], core["given_up"])
    ```

    `live` has, per kernel (`reader`, `writer`, `unpack`, `math`, `pack`), the runtime wait
    it is in (a where code's name, or None) and the command it runs; `given_up` counts the
    commands the core has given up. A core the dispatcher could not reach is `{"core": i,
    "read": False, "reset": False}`. The inspect does not stop anything: the dispatcher
    reads one core a pass of its loop beside the commands in flight. Only one inspect runs
    at a time; one that finds its block overwritten by the deadline's raises
    `km.KomorebiError`, and can be made again. Over several cards it is
    `run.inspect(island, cores)`, in the island's own core numbers.

2. **Read the time-out.** `run.wait(handle)` raises:

    ```python
    try:
        run.wait(handle, timeout=10.0)
    except km.CommandTimeout as error:
        print(error.given_up, error.recovered)
        for core in error.cores:
            print(core["core"], core["live"])
    ```

    The text has this shape:

    ```text
    command <handle> was given up, the run goes on; at the deadline: slot <slot>,
    sequence <n>, group first <first> count <count>; cores: core <c> reader waits in
    <where>; recovered (reset and loaded again): cores [<c>]
    ```

    Each core of the group is named with the wait of each of its kernels (`reader waits in
    port`, `writer waits in data`, the where names of the
    [runtime reference](../reference/runtime.md#gave-up-and-where-codes)), or `no give-up
    recorded`. The `recovered` part is there only when the dispatcher reset cores. A time-out
    that was not given up says `timed out; not reported` and lists the cores that did not
    report: the run is stopped and read back, and cannot be used again. On islands, the text
    starts `island i:` and `error.island` is i; the cores are that island's own.

    A `km.WaitTimeout` is another thing: the host stopped waiting, the run is loaded, and
    the message ends with what the dispatcher was doing, including whether core 0 still runs
    its loop. The command may only be slow: wait again, or look at it with `run.inspect`.

3. **Keep going, or look at the core.** After a given-up command the next command on the
   same cores runs as usual. `run.state()` after the stop lists, per core, the first
   give-up of each kernel: `deadline` with the where code of the wait.

4. **Reset one core yourself.** The host can put one core of a loaded run in reset, write
   what the open writes for it (firmware, its tables, the boot jump), wait for its
   firmware and load its kernels again, while the other cores keep running theirs. It
   takes the core's grid position, and a limit in seconds for the firmware:

    ```python
    x, y = core % session.grid[0], core // session.grid[0]
    reset, firmware, load = run.backend.launch.reset_core(x, y, 2.0)
    back = firmware >= 0  # -1: the firmware never came back within the limit
    ```

    The three numbers are the seconds of the reset, of the wait for the firmware and of the
    reload. A core with no kernels of the program is refused (`ValueError`). This is the
    experiment's way, [`experiments/core_reset/ladder.py`](../../experiments/core_reset/ladder.py)
    (`--step 1`), and it is used on an idle core of a run; `run.backend` is not a
    public name yet.

5. **Let the dispatcher do it.** The dispatcher resets a core by its own command
   (`KMD_HANDLER_RESET`), with a list of the writes the host made for the core (kernels,
   tables) in its upload region. The host builds the list and uploads it:

    ```python
    writes = run.backend.launch.reset_list(x, y)
    run.backend.port.upload(0, writes)
    ```

    `reset_list(0, 0, every=True)` builds one list for every core, which the dispatcher
    keeps; at the end of the wait of any later give-up it then resets each core of the
    group still in the command and loads it again, and the run goes on. The list is refused
    for a core with semaphores, and the kept one when cores differ in kernels. The command
    that hands it over is sent through the run's own queue in the experiment (`keep_list`
    and `dispatcher_reset` in `ladder.py`, steps 2 and 5); the session does not do it for
    you yet, so a program that does not is in step 3 of the ladder without its list: a core
    that does not leave ends the run.

6. **The dispatcher's own loop.** Core 2 of the dispatcher's cluster watches the loop on
   core 0. A loop that makes no pass for 100 ms is reported to the host (`run.stalls`,
   stage "stalled"), and core 2 interrupts core 0, which re-enters its loop with the run's
   state kept ("back", about 1.2 microseconds later in the experiment); the commands after
   it run as before. If core 0 is not back 50 ms later, the host gets "lost" and stops the
   run with `km.KernelFailed`. Nothing has to be called for this to work.

## When a card reset is still the only way

- **The dispatcher's core 0 stopped executing**, for instance in an access that never
  completes. It takes no interrupt, so the watch cannot bring it back, and the runtime
  cannot restore the cluster. A `km.WaitTimeout` says so (core 0's pass marker did not
  move), and `km.KernelFailed` carries the runtime's account of each core.
- **An X280 thread that never returns and polls nothing.** The stop flag is seen only
  inside library calls (`kmx_wait`, `kmx_parallel`, `kmx_stop_requested`). A core stuck
  outside them cannot be stopped: `km.ThreadFailed`, and the next session needs the
  reset. A thread's own command is never given up, so its time-out ends the run too.
- **A kernel that leaves a compute unit running** when the process ends, and a packer that
  writes past its buffer. After a `km.GridTimeout` the runtime resets the cores at the
  process's exit; a stuck loop is recovered by the next open, a compute unit left running
  is not.
- **A process killed with SIGKILL.** Nothing can close the X280 runtime.
- **A dynamic run's waits.** The dispatcher gives no command of a dynamic run up; a
  core that gives up there ends the run with its reason (no card reset is needed, the
  next run starts on the loaded program).

The reset itself is `tt-smi -r N` ([install](../install.md)); Komorebi does not call it.
