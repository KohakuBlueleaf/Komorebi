# A body's own buffers

Question: can a class give its compute bodies buffers that only they use, of the two
kinds the attention pass needs, with no line filling, waiting for or freeing them?
One is a float32 tile written in place. The other is a FIFO from the pack CPU to the
unpack CPU.

The line is a field whose value is `km.own()`
([Programs written as classes](../../../docs/reference/classes.md)):

```python
class OwnBuffers(km.Program):
    total: km.L1[km.fp32, 1, 32, 32] = km.own(push=None)  # written in place
    step: km.L1[km.bf16, 1, 32, 32] = km.own()  # pack's token: total is written
    delay: km.L1[km.bf16, 2, 32, 32] = km.own()  # pack to unpack, one step late
```

`own_class.py` has two handlers, each over a stream of `tiles` bfloat16 tiles split
over the cores:

- `running(source, sums, tiles)`. Each step copies the stream's tile to DST 0. After
  the first step it waits for the token `step`, copies `total` to DST 1 and adds it.
  The sum is packed back into `total`'s slot (`pack_tile<true>(0, total, 0)`), and
  pack pushes a token to `step`. After the stream, a body that gives a tile takes the
  last token and copies `total`. The line `sums.pages(self.rank, 1).write(whole)`
  writes it.
- `delayed(source, late, tiles)`. Each step's tile is copied, then pack pushes it to
  `delay`. A second body gives the tile pushed one step before (a zero tile at the
  first step), which the write stream writes. After the stream a body frees the last
  tile pushed.

The pack and unpack CPUs share no variable, so the token says when `total` holds the
new sum. The attention pass's state buffers use the same token.

```bash
python own_class.py --cores 8 --tiles 40
```

## What the own fields lower to

The build report, the program's three own buffers:

```
kFieldTotal (field, buffers 8): float32, 1 pages of 4096 B, 4096 B; field total: float32, 1 x 32 x 32; own: used by bodies, not filled by a line
kFieldStep (field, buffers 9): bfloat16, 1 pages of 2048 B, 2048 B; field step: bfloat16, 1 x 32 x 32; own: used by bodies, not filled by a line
kFieldDelay (field, buffers 10): bfloat16, 2 pages of 2048 B, 4096 B; field delay: bfloat16, 2 x 32 x 32; own: used by bodies, not filled by a line
```

They are counted in the build's L1 table with the other buffers. In each body that
takes one, the body gets the buffer's index by its name and nothing else:

```cpp
const uint32_t total = kFieldTotal;
const uint32_t step = kFieldStep;
```

There is no wait and no free. `total` is float32, so buffer 8 is unpacked to DST as
float32 (`unpack_to_dest_mode`), as a float32 stream's buffer is. The build looks for
`cb_push_back(<field>, ...)` in the bodies that take the field, and refuses:

- a push of a field declared `km.own(push=None)`;
- a push on a CPU other than the one named by `km.own(push="pack")`;
- pushes on two CPUs (`km.own()`, the default, takes the one CPU it finds).

The refusal names the other CPU's handler and line. A buffer with two producers was
the cause of the attention body's hang.

## Result

Card 0, a Blackhole p150a, 8 cores x 40 tiles (320 tiles of integers -4 to 4), each
output compared with numpy:

| Handler | Wrong values | Time a command |
|---|---|---|
| `running`: each core's float32 sum of its 40 tiles | 0 of 8,192 | 21.9 us |
| `delayed`: each core's tiles one step late, the first zero | 0 of 327,680 | 21.8 us |

Kernel text (mover A, mover B, unpack, math, pack): 14,836, 11,648, 9,984, 4,100,
5,376 B, 46,640 in all.

The first run gave `running` 8,063 values of 8,192 wrong while `delayed` was exact.
The cause was the example's body, not the line. `copy_init` does not set the
unpacker's data format, so `total` was read as bfloat16. Each copy now sets it first
(`reconfig_data_format_srca(tile_buffer)`, `reconfig_data_format_srca(total)`). A
class does not set it for a body, so a body that copies from buffers of two formats
must set it itself.
