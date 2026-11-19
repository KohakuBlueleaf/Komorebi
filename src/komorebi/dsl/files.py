"""The files of a class's build folder: the folder named by its files' text, a file
written only when its text differs, the X280 thread's header of the class's commands."""

import hashlib
from pathlib import Path

from komorebi.program.overlay import write_into

TEXT_KEY_CHARS = 16  # hex digits of a build folder's name: the digest of its text


def camel(name):
    return "".join(part.capitalize() for part in name.split("_"))


def text_key(files):
    """A short hash of the files a build wrote: each name and text, in name order,
    and nothing else (not the report, which holds the folder's own path and the L1)."""
    digest = hashlib.sha256()
    for name in sorted(files):
        text = files[name].encode()
        digest.update(f"{name}\0{len(text)}\0".encode() + text)
    return digest.hexdigest()[:TEXT_KEY_CHARS]


def put(path, text):
    """Writes `text` to `path` unless the file already holds it: a folder named by its
    text is never trusted by its name, and a file that is equal is left as it is. A
    write is whole or absent (write_into): a build beside this one reads no half file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text() == text:
        return
    write_into(path, text)


def thread_header_name(compiled):
    return f"{compiled.name.lower()}_thread.h"


def thread_header(compiled):
    """The C header of an X280 thread that submits the class's commands: each
    handler's number, the place of each of its words, their count, and whether its
    command asks for the leader's result."""
    prefix = compiled.name.upper()
    lines = [
        f"/* {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        " * its handlers' numbers and each command word's place (a0 is 0), for a",
        " * thread's commands (kmx_submit). */",
        "#pragma once",
    ]
    for name, number in compiled.handlers.items():
        handler = f"{prefix}_{name.upper()}"
        words = compiled.words[name]
        lines += ["", f"/* {name.upper()}, line {compiled.lines[name]}. */"]
        lines.append(f"#define {prefix}_HANDLER_{name.upper()} {number}u")
        for place, (word, kind) in enumerate(words):
            suffix = {"pages": "_PAGES", "bank": "_BANK", "columns": "_COLUMNS"}.get(
                kind, ""
            )
            lines.append(f"#define {handler}_{word.upper()}{suffix} {place}u")
        lines.append(f"#define {handler}_WORDS {len(words)}u")
        if name in compiled.results:
            lines.append(f"#define {handler}_RESULT 1u /* KM_FLAG_RESULT */")
    return "\n".join(lines) + "\n"
