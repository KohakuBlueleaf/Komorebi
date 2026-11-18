# A dynamic handler's step in lines

Question: does a dynamic handler whose step is written in lines, the scene's hit and
the quota left read from bodies' words (`hit = km.cxx(..., gives=int)`), give on the
card the totals the same rules give in one body and in Python, and what do the lines
cost in microseconds and kernel bytes?

`lines.py` makes two classes of one dynamic handler on 16 cores. Each core holds a
scene of 256 words in L1, the same on every core. 16 seed rays of depth 6 are sent, a
ray a core. A ray of depth 0, or one that meets a core with no quota left, is
consumed (its origin summed). Any other ray spawns a child at `origin + scene[origin &
255]` and is forwarded with its depth one less:

```python
for message in self.arrive():
    hit = km.cxx(HIT, takes=[scene, message.origin], gives=int)
    left = km.cxx("left = km_messages.quota();", gives=int)
    if message.depth == 0 or left == 0:
        ...
        self.consume()
    else:
        message.depth -= 1
        child = Ray(origin=message.origin + hit, depth=message.depth)
        self.send(child.origin % self.size, child)
        self.forward(message.origin % self.size, message)
```

`Body` writes the same step in one body, as before a body's word could be read in
lines: the walk, the quota test, `send`, `forward` and `consume` in C++. Each class is
loaded 5 times, a command a load; a core's totals go to its results words, and the
sums over the cores are compared with a Python walk of the same rules (1,024 rays
consumed, none for want of quota).

    python lines.py                  # one card (--cards N: that card)
    kmr prebuild lines.py            # the same on the mock
