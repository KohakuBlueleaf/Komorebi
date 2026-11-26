"""The files of a class's build folder: where it is, the folder named by its files'
text, a file written only when its text differs, the X280 thread's header."""

import hashlib
from pathlib import Path

from komorebi.device import native
from komorebi.program.overlay import write_into

TEXT_KEY_CHARS = 16  # hex digits of a build folder's name: the digest of its text


def camel(name):
    return "".join(part.capitalize() for part in name.split("_"))


def builds(source):
    """The folder of a class's build folders in the cache directory (KMR_CACHE):
    classes/<folder>-<hash>, one a folder of class sources (`source`: a class file)."""
    folder = Path(source).resolve().parent
    named = hashlib.sha256(str(folder).encode()).hexdigest()[:8]
    return native.cache_directory() / "classes" / f"{folder.name}-{named}"


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
    lines = [
        f"/* {compiled.name}, made by komorebi.dsl from {Path(compiled.source).name}:",
        " * its handlers' numbers and each command word's place (a0 is 0), for a",
        " * thread's commands (kmx_submit). */",
        "#pragma once",
    ]
    for name in compiled.handlers:
        lines += ["", f"/* {name.upper()}, line {compiled.lines[name]}. */"]
        lines += [
            f"#define {define} {value}"
            for define, value in handler_defines(compiled, name)
        ]
    return "\n".join(lines) + "\n"


def handler_defines(compiled, name):
    """Handler `name`'s names in the thread header and their values: its number,
    each word's place, the words' count, whether its command asks for the result."""
    prefix = compiled.name.upper()
    handler = f"{prefix}_{name.upper()}"
    words = compiled.words[name]
    defines = [(f"{prefix}_HANDLER_{name.upper()}", f"{compiled.handlers[name]}u")]
    for place, (word, kind) in enumerate(words):
        suffix = {"pages": "_PAGES", "bank": "_BANK", "columns": "_COLUMNS"}.get(
            kind, ""
        )
        defines.append((f"{handler}_{word.upper()}{suffix}", f"{place}u"))
    defines.append((f"{handler}_WORDS", f"{len(words)}u"))
    if name in compiled.results:
        defines.append((f"{handler}_RESULT", "1u /* KM_FLAG_RESULT */"))
    return defines
