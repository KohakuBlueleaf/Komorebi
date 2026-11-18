---
title: "Setup and the first run"
part: 1
summary: "checking the machine with kmr doctor and running the first program"
---

# Setup and the first run

This lesson gets your machine ready and runs the program of [[what-komorebi-is]]. At the
end you have run one program on the card yourself.

## What you need

- A computer with a Blackhole P150A card in it, running Linux on x86-64, with Python 3.13.
- The card set up once for Komorebi: tt-metal (Tenstorrent's own software for the card)
  built with the extra support Komorebi needs, and the card prepared. This is done once
  per machine, and it has [its own guide](../install.md#running-on-a-card). If you work on a shared machine
  somebody has done it already.
- Nobody else using the card at the same moment. One program at a time can open it.

## Get the code

The steps are the same everywhere. Replace the path of `TT_METAL_HOME` by
where tt-metal is on your machine.

```bash
git clone <this repository> komorebi
cd komorebi
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
export TT_METAL_HOME=~/tt-metal
```

`pip install -e .` installs the Python package `komorebi` from the folder you cloned, so
that `import komorebi` works. The `-e` means the package is read from the folder itself.
The last line tells Komorebi where tt-metal is.

### Or, the two wheels

There is a second way, with no tt-metal folder of your own: two
packages from PyPI, `komorebi-tt` (the framework) and `komorebi-runtime` (tt-metal,
already built). The packages do not hold the tutorial's programs, so you clone the
repository at the release's tag for them (0.3.0 here):

```bash
git clone --depth 1 --branch v0.3.0.post1 <this repository> komorebi
cd komorebi
python3.13 -m venv .venv
source .venv/bin/activate
pip install "komorebi-tt[runtime]"
kmr setup
```

The last line is done once: it fetches the compiler for the card's cores, then prepares
the card's X280 memory. On a card prepared before, it says so and changes nothing; on a
new card it asks you to reset the card once, with the command it prints
([the install guide](../install.md#setting-up-once) says more). Do not set
`TT_METAL_HOME` in this way of working: when it is set, Komorebi uses that folder
instead of the package. Everything else in this tutorial is the same.

## Check the machine

`kmr doctor` looks at what you have set up and prints one line for each thing. It does
not open the card.

```bash
kmr doctor
```

```{ .text .km-output }
--8<-- "tutorial/02_setup/doctor.txt"
```

The paths here are shortened; yours show where things really are. Read the lines in
pairs of name and answer:

- `komorebi` and `python`: the version of the package, and the Python it runs in. The
  path should be inside your `.venv`.
- `package` and `include`: where the package is, and where the C++ headers are that
  your own C++ files will include.
- `tt-metal` and `build`: the tt-metal folder it found and whether it comes from
  `TT_METAL_HOME`. With the two wheels the folder is inside your cache and the line
  under it says `from komorebi-runtime`.
- The two `x280` lines name a part of the card that [[the-card]] explains. The first must
  say `yes` and the second must show a path. Without them nothing in this tutorial runs.
- `card setup` names the command that prepares a card once, `kmr setup`. The doctor
  does not open the card, so it cannot tell whether that was done.
- A `card N` line for each card, read from the driver's files: its type, firmware and
  clock, whether its heartbeat moves, its device node, and the process that holds it.
  A card that cannot be opened gets a `NOT READY` line under it that says why and which
  reset to run. The clock reads its idle value (800 MHz on a P150A) until a program
  opens the card.

If `tt-metal` says `not found`, `TT_METAL_HOME` is not set in this shell, or it names a
folder that does not exist.

## Run the first program

The program of the first lesson is in the folder `tutorial/01_first_look` of the
repository. Run it from there:

```bash
cd tutorial/01_first_look
python add.py
```

```{ .text .km-output }
--8<-- "tutorial/01_first_look/output.txt"
```

If you see the same three lines, your machine can run everything in this tutorial.

Two things you will notice the first time:

- It takes a few seconds before the first line. Komorebi compiles the C++ files for the
  card, and keeps the result, so the next run starts faster.
- Besides these three lines your terminal fills with about 25 lines of log from
  tt-metal (lines that start with a date, and warnings from the compiler on the first
  run). They are not part of the program's output. The lessons show only what the
  program printed. To hide the log, run `python add.py 2>/dev/null`.

Every program in this tutorial lives in its own folder under `tutorial/`, named with the
number of its lesson, and every lesson shows the output that program printed when it ran
on a P150A card.

## Common mistakes

- `kmr: command not found`. The virtual environment is not active in this shell. Run
  `source .venv/bin/activate` again; it is needed in every new shell.
- `tt-metal   not found` in `kmr doctor`. `TT_METAL_HOME` is not exported in this shell.
- The program stops with `CardBusy` and the number of another process. Something else is
  using the card. Wait until it ends; a second program that opens the card would
  otherwise wait for it without end, so Komorebi refuses at once and tells you who holds
  it.
- The program stops with `CardNotReady`. The driver says the card cannot be opened (its
  heartbeat stands still, or the card or its device node is missing). `kmr doctor` shows
  the same; reset the card as the message says.

Next: [[the-card]] looks at the card itself, so that the next lessons have a picture to
work with.
