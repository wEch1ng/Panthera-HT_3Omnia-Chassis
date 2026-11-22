"""Verify portable local mesh references in canonical URDF and MJCF assets."""

from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
count = 0
for folder in ("assets/robots/panthera/urdf", "assets/robots/panthera/mjcf", "assets/tools/paddle"):
    for xml in (root / folder).glob("*"):
        if xml.suffix not in (".urdf", ".xml"):
            continue
        for mesh in ET.parse(xml).getroot().iter("mesh"):
            ref = mesh.get("filename") or mesh.get("file")
            if not ref:
                continue
            assert not Path(ref).is_absolute() and not ref.startswith("package:"), (xml, ref)
            target = (xml.parent / ref).resolve()
            assert target.is_file(), (xml, target)
            assert target.is_relative_to(root), (xml, target)
            count += 1
print(f"PASS {count} mesh references stay inside the project")
