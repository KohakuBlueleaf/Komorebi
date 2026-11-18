"""Who gave a dynamic run up, for MessagesGaveUp: the dispatcher's KMD_H_MESSAGES_GAVE
word (1 + the core whose status line said gave up, in its low 16 bits) and that core's
give-up words (a link core's: its state word), or an X280 rank's index. The one place
another island's part is added."""

from komorebi.session import state as device_state

# What a give-up's where asks of the program, by where.
WRITE = {
    "quota": (
        "a core's quota is max_live // the run's ranks: raise max_live, or write "
        "went = self.send(rank, m) and handle 0"
    ),
    "refused": (
        "a link core refused a delivery line to another island's rank (its island, "
        "or its messages' or body's sizes past a delivery slot)"
    ),
}


def named(result, blocks, h, state=None):
    """{core, where, position, text} of the core whose give-up ended a dynamic run,
    from a stopped run's result block, control blocks and state words (`state`: a link
    core's give-up between commands, its where and code); None when none is named."""
    words = (result or {}).get("x280_cores")
    word = int(words[h["KMD_H_MESSAGES_GAVE"]]) & 0xFFFF if words else 0
    if word == 0:
        return None
    core = word - 1
    x280 = core - (h["KM_CORES_MAX"] - h["KM_X280_RANKS_MAX"])  # KM_X280_RANK_CORE
    if 0 <= x280 < h["KM_X280_RANKS_MAX"]:  # an X280 rank's status line said gave up
        return {
            "core": core,
            "x280": x280,
            "where": None,
            "position": None,
            "text": "its thread's give-up (KMX_GAVE_*)",
        }
    if blocks is None or core >= len(blocks):  # an X280 rank's line (not yet named)
        return {"core": core, "where": None, "position": None, "text": "an X280 rank"}
    gave = device_state.dynamic(blocks[core], h)["gave_up"] or {}
    where, position = gave.get("where"), gave.get("position")
    if where == "quota":
        what = f"quota (self.send to rank {position} with no quota left; "
        what += f"{WRITE['quota']})"
    elif where == "refused":
        what = f"refused ({WRITE['refused']})"
    elif where:
        what = f"{where} (position {position})"
    else:  # no run give-up word (a link core's between commands): its state word
        kept = device_state.core_state(state, core, h) if state is not None else {}
        word = kept.get("reader") or kept.get("writer")
        if word:  # kWherePages and the like: the runtime's own names
            where = device_state.where_names(h).get(word["where"], word["where"])
        what = f"{where}, {word['code']}" if word else "no give-up word"
    return {"core": core, "where": where, "position": position, "text": what}
