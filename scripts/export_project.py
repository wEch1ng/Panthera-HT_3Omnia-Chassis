"""Export a source distribution without cache, virtualenv or historical duplicates."""

import argparse
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, default=root / "dist/panthera-lab-source.tar.gz")
args = p.parse_args()
if args.output.exists():
    p.error("Output exists; choose another --output")
args.output.parent.mkdir(parents=True, exist_ok=True)
entries = [
    "src",
    "assets/robots",
    "assets/tools",
    "checkpoints",
    "configs",
    "licenses",
    "docs",
    "examples",
    "scripts",
    "tests",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitignore",
    ".gitattributes",
]


def keep(info):
    if "__pycache__" in Path(info.name).parts or info.name.endswith((".pyc", ".pyo")):
        return None
    return info


with tarfile.open(args.output, "w:gz") as tar:
    for entry in entries:
        tar.add(root / entry, arcname="panthera-lab/" + entry, filter=keep)
print(args.output)
