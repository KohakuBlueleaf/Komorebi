# Two reduces in one handler

`two_reduces.py`: a class whose handlers reduce twice, each sum against numpy, on one
card (`--cards 0`) or on two (`--cards 0,1`: the command's cores meet at the second
card's first rank, so its groups cross the edge between the cards).

| Case | What runs |
| --- | --- |
| grid | rank r's page r of s, then of z, summed over the command's 16 cores; the leader writes pages 0 and 1 of t |
| groups of 4 | the same in groups of 4 (`group=4` on both reduces); root g writes page g of t and of y |
| chain, groups of 2 | a chain's call: groups of 2 ranks sharing s's and z's 6 tiles, two reduces, root g writing page g of t and of y |
| two passes, a chain of three | a second pass that needs the first's sum at every rank: `mean` reduces s and writes m, `apart` streams s beside m's page and pushes (s_r - m / 16)^2 to d, `total` reduces d to u (a layer norm's two passes) |

The second reduce of a handler is a turn after the first: its senders wait for their
receiver's credit, as a reduce does in a range loop. A handler's reduces share one
form and group size; another is refused at its line (the turn restarts when the form
or the group changes: `reduce_turn_next`, `device/movement/km_port.hpp`).

A JSON line a case on stderr's log, then the summary (`bad_words`: words of the
integer sums not equal to numpy's; the two passes' `rms_rel` against float64).
