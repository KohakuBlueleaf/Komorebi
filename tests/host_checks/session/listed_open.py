"""A card listed in TT_VISIBLE_DEVICES is visible: with 0,1 set, Session(card=1) opens
(not only the first listed), and the variable reads 0,1 again after its close."""

import os

import komorebi as km

os.environ["TT_VISIBLE_DEVICES"] = "0,1"
seen = []
for card in (1,):
    session = km.Session(card=card)
    session.open()
    seen.append((card, os.environ.get("TT_VISIBLE_DEVICES")))
    session.close()
after = os.environ.get("TT_VISIBLE_DEVICES")
print("during", seen, "after", after)
assert seen == [(1, "1")] and after == "0,1", (seen, after)
print("all ok")
