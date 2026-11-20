"""Builds the VSIX of the extension in `komorebi-ir/` (standard library only).

python3 build_vsix.py [OUT]: OUT is a file, or a folder for `komorebi-ir-<version>.vsix`
(default: this script's folder). The same files give the same bytes.
"""

import json
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "komorebi-ir"
# Files of the folder that are not part of the extension.
SKIP = {".gitignore"}
STAMP = (2020, 1, 1, 0, 0, 0)

CONTENT_TYPES = """<?xml version="1.0" encoding="utf-8"?>
<Types>
  <Default Extension=".json" ContentType="application/json"/>
  <Default Extension=".md" ContentType="text/markdown"/>
  <Default Extension=".vsixmanifest" ContentType="text/xml"/>
</Types>
"""

MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0"
    xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">
  <Metadata>
    <Identity Language="en-US" Id={name} Version={version} Publisher={publisher}/>
    <DisplayName>{display}</DisplayName>
    <Description xml:space="preserve">{description}</Description>
    <Categories>{categories}</Categories>
    <GalleryFlags>Public</GalleryFlags>
    <Properties>
      <Property Id="Microsoft.VisualStudio.Code.Engine" Value={engine}/>
    </Properties>
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code"/>
  </Installation>
  <Dependencies/>
  <Assets>
    <Asset Type="Microsoft.VisualStudio.Code.Manifest"
           Path="extension/package.json" Addressable="true"/>
    <Asset Type="Microsoft.VisualStudio.Services.Content.Details"
           Path="extension/README.md" Addressable="true"/>
  </Assets>
</PackageManifest>
"""


def manifest_of(package: dict) -> str:
    """The `extension.vsixmanifest` text for the fields of `package.json`."""
    return MANIFEST.format(
        name=quoteattr(package["name"]),
        version=quoteattr(package["version"]),
        publisher=quoteattr(package["publisher"]),
        display=escape(package.get("displayName", package["name"])),
        description=escape(package.get("description", "")),
        categories=escape(",".join(package.get("categories", []))),
        engine=quoteattr(package["engines"]["vscode"]),
    )


def build(out: Path) -> Path:
    """Writes the VSIX to `out` (a file, or a folder to put it in); returns its path."""
    package = json.loads((SOURCE / "package.json").read_text(encoding="utf-8"))
    if out.is_dir():
        out = out / f"{package['name']}-{package['version']}.vsix"
    files = sorted(p for p in SOURCE.rglob("*") if p.is_file() and p.name not in SKIP)
    entries = [
        ("[Content_Types].xml", CONTENT_TYPES.encode()),
        ("extension.vsixmanifest", manifest_of(package).encode()),
    ]
    entries += [
        ("extension/" + p.relative_to(SOURCE).as_posix(), p.read_bytes()) for p in files
    ]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            info = zipfile.ZipInfo(name, STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    return out


def main(argv: list[str]) -> int:
    """Command line: an optional output path."""
    if len(argv) > 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[1]) if len(argv) == 2 else HERE
    print(build(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
