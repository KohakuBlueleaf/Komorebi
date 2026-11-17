"""Who gave a dynamic run up, for MessagesGaveUp: the dispatcher's KMD_H_MESSAGES_GAVE
word (1 + the core whose status line said gave up, in its low 16 bits) and that core's
give-up words. The one place an X280 rank's or another island's part is added."""

from komorebi.session import state as device_state

# What a give-up's where asks of the program, by where.
WRITE = {
    "quota": (
        "a core's quota is max_live // the run's ranks: raise max_live, or write "
        "went = self.send(rank, m) and handle 0"
    ),
}


def named(result, blocks, h):
    """{core, where, position, text} of the core whose give-up ended a dynamic run,
    from a stopped run's result block and control blocks; None when none is named."""
    words = (result or {}).get("x280_cores")
    word = int(words[h["KMD_H_MESSAGES_GAVE"]]) & 0xFFFF if words else 0
    if word == 0:
        return None
    core = word - 1
    if blocks is None or core >= len(blocks):  # an X280 rank's line (not yet named)
        return {"core": core, "where": None, "position": None, "text": "an X280 rank"}
    gave = device_state.dynamic(blocks[core], h)["gave_up"] or {}
    where, position = gave.get("where"), gave.get("position")
    if where == "quota":
        what = f"quota (self.send to rank {position} with no quota left; "
        what += f"{WRITE['quota']})"
    else:
        what = f"{where} (position {position})" if where else "no give-up word"
    return {"core": core, "where": where, "position": position, "text": what}
