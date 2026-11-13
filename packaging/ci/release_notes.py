"""The body of a GitHub release from the version's written notes.
Usage: release_notes.py NOTES TAG [REPOSITORY]
Prints the notes, a page of the documentation, without the front matter and the title,
and with their links to other files of the repository made absolute at TAG.
"""

import posixpath
import re
import sys

REPOSITORY = "KohakuBlueleaf/Komorebi"
LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)")


def body(text, notes_path, tag, repository=REPOSITORY):
    """The notes as a release body."""
    lines = text.splitlines()
    if lines and lines[0] == "---":
        lines = lines[lines.index("---", 1) + 1 :]
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].startswith("# "):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    folder = posixpath.dirname(notes_path)
    base = f"https://github.com/{repository}/blob/{tag}/"

    def absolute(match):
        target, _, anchor = match.group(1).partition("#")
        path = posixpath.normpath(posixpath.join(folder, target))
        return f"]({base}{path}{'#' + anchor if anchor else ''})"

    return LINK.sub(absolute, "\n".join(lines)) + "\n"


if __name__ == "__main__":
    notes, tag, *rest = sys.argv[1:]
    with open(notes, encoding="utf-8") as file:
        sys.stdout.write(body(file.read(), notes, tag, *rest))
