"""Ball/paddle task assets; the base robot knows nothing about ping-pong."""

import xml.etree.ElementTree as ET
from panthera_lab.resources import resource
from panthera_lab.sim.model import robot_xml, compile_model


def build_model(force_controls=False):
    root = robot_xml()
    world = root.find("worldbody")
    assets = root.find("asset")
    directory = resource("assets", "tools", "paddle")
    paddle = ET.parse(directory / "paddle.xml").getroot()
    for item in paddle.find("asset"):
        if item.tag == "mesh":
            item.set("file", str(directory / item.get("file")))
        assets.append(item)
    grip = world.find(".//body[@name='gripper_center_link']")
    for name, y in [("L", 0.014789), ("R", -0.015211)]:
        finger = grip.find(f"body[@name='gripper_{name}_link']")
        pos = finger.get("pos").split()
        pos[1] = str(y)
        finger.set("pos", " ".join(pos))
    tool = ET.SubElement(
        grip, "body", name="paddle_mount", pos="-0.098 0 0", quat="-0.5 -0.5 0.5 0.5"
    )
    blade = paddle.find(".//body[@name='paddle']")
    tool.append(blade)
    for geom in blade.findall("geom"):
        geom.attrib.pop("class", None)
        coll = geom.get("name") == "paddle_blade"
        geom.set("contype", "4" if coll else "0")
        geom.set("conaffinity", "8" if coll else "0")
    ball = paddle.find(".//body[@name='ball']")
    world.append(ball)
    ball.find("geom").attrib.update(contype="8", conaffinity="5")
    world.find("geom[@name='floor']").set("conaffinity", "10")
    if force_controls:
        from panthera_lab.sim.forces import add_force_controls

        add_force_controls(root)
    return compile_model(root)
