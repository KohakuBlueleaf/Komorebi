"""islands= given to km.Session (card=) and to an IslandSession: -3, 5, True, 2."""

import komorebi as km

for islands in (-3, 5, True, 2):
    try:
        made = km.Session(card=0, islands=islands)
        print(f"islands={islands!r}: made, split {made.split} (not opened)")
    except Exception as error:  # noqa: BLE001 (printed)
        print(f"islands={islands!r}: raised {type(error).__name__}: {error}")
