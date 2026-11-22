"""Compose a wheeled arm once, then let a task add its own tools and objects."""

import xml.etree.ElementTree as ET
import mujoco as mj
from panthera_lab.resources import resource
from panthera_lab.robot.config import PHYSICS_HZ, WHEEL_RADIUS, WHEEL_BODIES, CAMERA_FOVY
from panthera_lab.robot.motor_limits import NAMES, STALL_TORQUE


def robot_xml():
    directory = resource("assets", "robots", "panthera", "mjcf")
    root = ET.parse(directory / "chassis.xml").getroot()
    world, assets = root.find("worldbody"), root.find("asset")
    for mesh in assets.findall("mesh"):
        mesh.set("file", str(directory / mesh.get("file")))
    reference = ET.parse(directory / "arm.xml").getroot()
    arm = reference.find("worldbody/body[@name='base_link']")
    arm.set("name", "arm_base_link")
    world.find("body[@name='base_link']").append(arm)
    for mesh in reference.findall("asset/mesh"):
        mesh.set("name", "ref_" + mesh.get("name"))
        mesh.set("file", str(directory / mesh.get("file")))
        assets.append(mesh)
    for geom in arm.iter("geom"):
        if geom.get("mesh"):
            geom.set("mesh", "ref_" + geom.get("mesh"))
    for motor in reference.find("actuator"):
        root.find("actuator").append(motor)
    camera_link = arm.find(".//body[@name='camera_link']")
    ET.SubElement(
        camera_link, "camera", name="wrist", pos="0 0 0", quat="0 1 0 0", fovy=str(CAMERA_FOVY)
    )
    for name in WHEEL_BODIES:
        body = world.find(f".//body[@name='{name}']")
        body.find("joint").set("damping", ".01")
        geom = body.find("geom[@group='3']")
        geom.attrib.clear()
        geom.attrib.update(
            name=name + "_contact",
            type="sphere",
            size=str(WHEEL_RADIUS),
            group="3",
            contype="2",
            conaffinity="1",
            condim="1",
            priority="2",
        )
    for name, stall in zip(NAMES, STALL_TORQUE):
        world.find(f".//joint[@name='{name}']").set("actuatorfrcrange", f"{-stall} {stall}")
        root.find(f"actuator/motor[@joint='{name}']").set("ctrlrange", f"{-stall} {stall}")
    root.find("option").set("timestep", str(1 / PHYSICS_HZ))
    return root


def compile_model(root):
    return mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
