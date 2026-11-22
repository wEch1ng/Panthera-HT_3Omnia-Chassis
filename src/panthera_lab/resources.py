"""Read-only resources in a checkout or installed wheel; never tied to cwd."""

import os
from pathlib import Path


def resource_root() -> Path:
    override = os.getenv("PANTHERA_RESOURCE_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
    else:
        bundled = Path(__file__).parent / "_resources"
        root = bundled if bundled.is_dir() else Path(__file__).resolve().parents[2]
    if not (root / "assets/robots/panthera/mjcf/chassis.xml").is_file():
        raise FileNotFoundError(
            f"Robot assets missing under {root}; install the complete project/wheel."
        )
    return root


def resource(*parts: str) -> Path:
    path = resource_root().joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"Required project resource not found: {path}")
    return path
