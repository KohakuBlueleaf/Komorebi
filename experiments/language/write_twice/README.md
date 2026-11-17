# A tile written twice, and on the branches of an if

`write_twice.py`: a class whose handlers write one tile, or a reduce's sum, to two
tensors, or to one of them by a command word, each case against numpy on one card
(`--cards 0`). On two cards (`--cards 0,1 --copy`) the command's cores meet at the
second card's first rank and y is a copy on each card.

| Case | What runs |
| --- | --- |
| a rank's tile to t and y | rank r sums page r of s (one page) and writes it to page r of t and of y |
| the sum to t and y | the 16 ranks' pages summed; the leader writes page 0 of t and of y |
| word 0: to t, word 1: to y | rank r's page to t when the command's word is 0, else to y |
| leader, word 0 or 1 | the sum, under the leader's if, to t or y by the word |
| only, word 1: nothing written | the sum written to t on word 0 alone: on word 1 it is freed, nothing written |
| only, word 0, after it | the same handler on word 0, after word 1's: the sum in page 0 of t |
| the sum to t and y, after both | `summed` again: its buffer's pages as at the start |

Compute packs the tile once and mover B writes its pages to t, then to y, then waits
for both writes and frees the pages. A page another card holds is written by a put
line that returns once the page is written. A copy's page is the writer's own card's
copy, so each card's copy holds the pages that card's ranks wrote.

t and y are zeroed before each case. A JSON line a case on stderr's log, then the
summary (`bad_words`: words of t and y not equal to numpy's).
