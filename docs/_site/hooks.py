"""MkDocs hooks of the documentation site.

Three jobs, all at build time:

- the READMEs of the examples (outside docs/) become pages of the site;
- a link from a page to another file of the repository goes to that page when the site
  has it, and to the file on GitHub otherwise;
- a code listing with `callouts="text|text"` is read from the file it names (the same
  `--8<-- "path:section"` line the snippets extension takes), and the first line that
  holds each text gets a numbered mark, Material's code annotation. The explanation is
  the numbered list that follows the listing;
- the lessons: `docs/tutorial/order.txt` lists them in order, in parts; the navigation
  of every page is built from it, a lesson page gets its number, its previous and next
  links, and `[[slug]]` becomes a link to that lesson by its number and title.
"""

import posixpath
import re
from pathlib import Path

from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "main"
FENCE = re.compile(r"^(?P<indent>[ ]*)(?P<ticks>`{3,})")
LINK = re.compile(r"(\]\()(?P<target>[^)\s]+)(?P<title>\s+\"[^\"]*\")?(\))")
CALLOUT_BLOCK = re.compile(
    r"^(?P<indent>[ ]*)```[ ]*(?:\{[ ]*\.)?(?P<lang>[\w+-]+)(?P<attrs>[^\n]*)\n"
    r"(?P=indent)--8<-- \"(?P<path>[^\"]+)\"[ ]*\n"
    r"(?P=indent)```[ ]*$",
    re.M,
)
SECTION_BLOCK = re.compile(
    r"^(?P<indent>[ ]*)```(?P<head>[^\n]*)\n"
    r"(?P=indent)--8<-- \"(?P<path>[^\":]+):(?P<name>[\w-]+)\"[ ]*\n"
    r"(?P=indent)```[ ]*$",
    re.M,
)
SECTION_MARK = re.compile(r"--8<--\s*\[(start|end):([\w-]+)\]")
COMMENT = {"cpp": "//", "c": "//", "python": "#", "py": "#"}


LESSONS = ROOT / "docs" / "tutorial"
WIKI = re.compile(r"\[\[([a-z0-9-]+)(?:\|([^\]]+))?\]\]")


def front_matter(path):
    """The `key: value` lines between the `---` marks at the top of a page."""
    text = path.read_text()
    if not text.startswith("---\n"):
        return {}
    head = text.split("\n---\n", 1)[0].split("\n")[1:]
    pairs = (line.split(": ", 1) for line in head if ": " in line)
    return {key: value.strip().strip('"') for key, value in pairs}


def lesson_list():
    """[(part title, goal, [slug, ...])] from order.txt; the lessons number 1, 2, ...
    through all parts."""
    parts = []
    for line in (LESSONS / "order.txt").read_text().split("\n"):
        if line.startswith("#"):
            parts.append([line.lstrip("# ").strip(), "", []])
        elif line.startswith(">"):
            parts[-1][1] = line.lstrip("> ").strip()
        elif line.strip():
            parts[-1][2].append(line.strip())
    return [tuple(part) for part in parts]


def lesson_table():
    """{slug: (number, title, planned)} in lesson order."""
    table, number = {}, 0
    for _, _, slugs in lesson_list():
        for slug in slugs:
            number += 1
            meta = front_matter(LESSONS / f"{slug}.md")
            table[slug] = (number, meta["title"], meta.get("status") == "planned")
    return table


def on_config(config):
    """The tutorial's part of the navigation is the lesson list, written or not."""
    table = lesson_table()
    parts = []
    for title, _, slugs in lesson_list():
        entries = []
        for slug in slugs:
            number, name, planned = table[slug]
            label = f"{number}. {name}" + (" (planned)" if planned else "")
            entries.append({label: f"tutorial/{slug}.md"})
        parts.append({title: entries})
    for position, item in enumerate(config["nav"]):
        if isinstance(item, dict) and "Tutorial" in item:
            config["nav"][position] = {"Tutorial": ["tutorial/index.md", *parts]}
    return config


def repo_path(file):
    """The path of a site file from the repository root."""
    if Path(file.src_dir).resolve() == ROOT / "docs":
        return "docs/" + file.src_uri
    return file.src_uri


def on_files(files, config):
    """The examples' READMEs join the site at the same paths as in the repository."""
    for readme in sorted((ROOT / "examples").glob("*/README.md")) + [
        ROOT / "examples" / "README.md"
    ]:
        rel = readme.relative_to(ROOT).as_posix()
        files.append(
            File(
                rel,
                str(ROOT),
                config["site_dir"],
                config["use_directory_urls"],
            )
        )
    return files


def github(config, path, directory):
    kind = "tree" if directory else "blob"
    return f"{config['repo_url']}/{kind}/{BRANCH}/{path}"


def rewrite_links(markdown, page, config, files):
    by_path = {repo_path(f): f for f in files if f.src_uri.endswith(".md")}
    here = repo_path(page.file)
    page_dir = posixpath.dirname(page.file.src_uri)

    def fix(match):
        target = match["target"]
        if re.match(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", target, re.I):
            return match.group(0)
        path, _, anchor = target.partition("#")
        anchor = "#" + anchor if anchor else ""
        full = posixpath.normpath(posixpath.join(posixpath.dirname(here), path))
        if full.startswith(".."):
            return match.group(0)
        site = by_path.get(full) or by_path.get(full.rstrip("/") + "/README.md")
        if site is not None:
            new = posixpath.relpath(site.src_uri, page_dir or ".") + anchor
        elif (ROOT / full).exists():
            new = github(config, full.rstrip("/"), (ROOT / full).is_dir()) + anchor
        else:
            return match.group(0)
        return f"{match[1]}{new}{match['title'] or ''}{match[4]}"

    out, fence = [], None
    for line in markdown.split("\n"):
        m = FENCE.match(line)
        if m and fence is None:
            fence = m["ticks"]
        elif m and fence and m["ticks"].startswith(fence) and not line.strip(" `"):
            fence = None
        out.append(line if fence or m else LINK.sub(fix, line))
    return "\n".join(out)


def section(path, name):
    """(first line number in the file, the section's lines) of `path[:name]`: the
    lines between `--8<-- [start:name]` and `--8<-- [end:name]`, dedented, or all
    lines; section marks never appear in the result."""
    lines = (ROOT / path).read_text().split("\n")
    first, keep = 1, list(enumerate(lines, 1))
    if name:
        start = end = None
        for number, line in enumerate(lines, 1):
            mark = SECTION_MARK.search(line)
            if mark and mark[2] == name:
                if mark[1] == "start":
                    start = number
                else:
                    end = number
        if start is None or end is None or end < start:
            raise PluginError(f"{path}: no section {name!r}")
        keep, first = keep[start : end - 1], start + 1
    keep = [(n, line) for n, line in keep if "--8<--" not in line]
    body = [line for _, line in keep]
    indent = min((len(x) - len(x.lstrip()) for x in body if x.strip()), default=0)
    return first, [x[indent:] for x in body]


def with_section(match):
    """A listing of one section of a file, written out dedented (the snippets
    extension keeps the section's own indentation)."""
    indent = match["indent"]
    _, code = section(match["path"], match["name"])
    body = [indent + line if line else line for line in code]
    return "\n".join([f"{indent}```{match['head']}", *body, f"{indent}```"])


def with_callouts(match):
    indent, lang, attrs = match["indent"], match["lang"], match["attrs"]
    path, _, name = match["path"].partition(":")
    marks = re.search(r"callouts=\"([^\"]*)\"", attrs)
    if marks is None:
        return match.group(0)
    attrs = attrs.replace(marks.group(0), "").strip().removesuffix("}").strip()
    first, code = section(path, name)
    comment = COMMENT.get(lang, "//")
    used = set()
    for number, text in enumerate(marks[1].split("|"), 1):
        for i, line in enumerate(code):
            if text in line and i not in used:
                break
        else:
            raise PluginError(f"{path}:{name}: no line holds {text!r}")
        used.add(i)
        line = code[i]
        has_comment = re.search(rf"(?:^|\s){re.escape(comment)}\s", line)
        if has_comment or line.strip().startswith(comment):
            code[i] = f"{line} ({number})"
        else:
            code[i] = f"{line} {comment} ({number})!"
    head = f'{indent}```{{ .{lang} .annotate linenums="{first}" {attrs} }}'
    return "\n".join([head, *[indent + x if x else x for x in code], f"{indent}```"])


def lesson_bar(slug, table):
    """The line of links above and below a lesson: previous, the list, next."""
    slugs = list(table)
    here = slugs.index(slug)

    def link(other, arrow_first):
        number, title, _ = table[other]
        text = f"{number}. {title}"
        if arrow_first:
            return f"[&larr; {text}]({other}.md)"
        return f"[{text} &rarr;]({other}.md)"

    before = link(slugs[here - 1], True) if here else ""
    after = link(slugs[here + 1], False) if here + 1 < len(slugs) else ""
    return (
        '<nav class="km-lesson-bar" markdown="span">'
        f'<span class="km-prev">{before}</span>'
        '<span class="km-up">[All lessons](index.md)</span>'
        f'<span class="km-next">{after}</span></nav>'
    )


def lesson_index(table):
    """The parts of the tutorial, each with its goal and a table of its lessons: the
    hub the lesson pages hang from."""
    out = []
    for title, goal, slugs in lesson_list():
        out += [f"## {title}", "", goal, ""]
        out += ["| Lesson | What it teaches | |", "|---|---|---|"]
        for slug in slugs:
            number, name, planned = table[slug]
            idea = front_matter(LESSONS / f"{slug}.md").get("summary", "")
            state = '<span class="chip">planned</span>' if planned else ""
            out.append(f"| [{number}. {name}]({slug}.md) | {idea} | {state} |")
        out.append("")
    return "\n".join(out)


def lessons_page(markdown, page):
    """Number the lesson, add the bars, and turn [[slug]] into a link."""
    table = lesson_table()
    slug = page.file.src_uri.removeprefix("tutorial/").removesuffix(".md")
    markdown = markdown.replace("<!-- lessons -->", lesson_index(table))
    if slug in table:
        number, title, _ = table[slug]
        part = front_matter(LESSONS / f"{slug}.md").get("part", "")
        names = [name for name, _, _ in lesson_list()]
        bar = lesson_bar(slug, table)
        markdown = markdown.replace(
            f"# {title}\n",
            f'<p class="km-eyebrow">{names[int(part) - 1]} &middot; lesson {number} of '
            f"{len(table)}</p>\n\n# {number}. {title}\n\n{bar}\n",
            1,
        )
        markdown = markdown.rstrip("\n") + f"\n\n{bar}\n"

    def wiki(match):
        other, text = match[1], match[2]
        if other not in table:
            raise PluginError(f"{page.file.src_uri}: no lesson {other!r}")
        number, title, _ = table[other]
        return f"[{text or f'lesson {number}, {title}'}]({other}.md)"

    return WIKI.sub(wiki, markdown)


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri.startswith("tutorial/"):
        markdown = lessons_page(markdown, page)
    markdown = rewrite_links(markdown, page, config, files)
    markdown = CALLOUT_BLOCK.sub(with_callouts, markdown)
    return SECTION_BLOCK.sub(with_section, markdown)
