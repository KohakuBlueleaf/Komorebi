"""The tutorial site's colours: every hex lives in docs/assets/gem.css, and each is
one of the palette's seven gems (three shades each) or its warm stone surfaces."""

import re
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[3] / "docs" / "assets"
HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")

GEMS = {
    "iolite": ("#DDD0F0", "#5A4FCF", "#312A7A"),
    "sapphire": ("#D6E3F8", "#0F52BA", "#082567"),
    "aquamarine": ("#D4EDE8", "#4C9989", "#1B6B5A"),
    "taaffeite": ("#E8D5ED", "#A57EAE", "#6B4670"),
    "amber": ("#F5E6C8", "#D4920A", "#8B5E00"),
    "coral": ("#F5D5D5", "#D46B6B", "#8B3A3A"),
    "sage": ("#D5E8DA", "#5A9E6F", "#3A6B48"),
}
STONE = (
    # light: page, card, alternate, border, text, muted, header, side navigation
    (
        "#f2eee8",
        "#ffffff",
        "#efebe5",
        "#e5dfd7",
        "#2e2a26",
        "#6c655d",
        "#e3ddd5",
        "#dcd5cb",
    ),
    # dark: page, card, alternate, border, text, muted
    ("#1a1816", "#252220", "#1e1c1a", "#3d3835", "#e8e0d8", "#8a8078"),
)
ALLOWED = {h.upper() for shades in GEMS.values() for h in shades} | {
    h.upper() for scheme in STONE for h in scheme
}


def test_no_colour_outside_the_gem_file():
    for path in sorted(ASSETS.rglob("*")):
        if (
            path.is_file()
            and path.name != "gem.css"
            and path.suffix in {".css", ".js", ".svg"}
        ):
            assert not HEX.findall(path.read_text()), f"hex colour in {path.name}"


def test_every_colour_of_the_gem_file_is_in_the_palette():
    used = {h.upper() for h in HEX.findall((ASSETS / "gem.css").read_text())}
    assert used
    assert used <= ALLOWED, sorted(used - ALLOWED)
