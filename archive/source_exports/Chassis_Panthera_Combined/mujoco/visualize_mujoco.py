"""URDF to MuJoCo: torque-controlled, gravity-compensated mouse dragging."""
from pathlib import Path
import argparse
import copy
import time
import queue
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "robot_color.urdf"
OUTPUT = Path(__file__).resolve().parent
OBJ_DIR = OUTPUT / "meshes" / "obj"
HOME = np.r_[0., 0., 0., 1., 0., 0., 0., np.zeros(9)]


def build():
    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    robot = ET.parse(SOURCE).getroot()
    extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(extension, "compiler", discardvisual="false", fusestatic="false",
                  strippath="false")
    for link in robot.findall("link"):
        for visual in list(link.findall("visual")):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            path = ROOT / mesh.get("filename")
            scene = trimesh.load(path, force="scene")
            for index, node in enumerate(scene.graph.nodes_geometry):
                transform, geometry = scene.graph[node]
                part = scene.geometry[geometry].copy()
                part.apply_transform(transform)
                filename = OBJ_DIR / f"{link.get('name')}_{index}.obj"
                part.export(filename, include_texture=False)
                replacement = copy.deepcopy(visual)
                replacement.find("geometry/mesh").set("filename", filename.relative_to(OUTPUT).as_posix())
                material = replacement.find("material")
                if material is not None:
                    replacement.remove(material)
                material = ET.SubElement(replacement, "material", name=f"{link.get('name')}_{index}")
                color = (part.visual.material.main_color if part.visual.kind == "texture"
                         else part.visual.main_color)
                rgba = np.asarray(color, dtype=float) / 255
                ET.SubElement(material, "color", rgba=" ".join(map(str, rgba)))
                link.append(replacement)
            link.remove(visual)
        for mesh in link.findall("collision/geometry/mesh"):
            source = ROOT / mesh.get("filename")
            target = OBJ_DIR / f"{link.get('name')}_collision.obj"
            trimesh.load(source, force="mesh").export(target)
            mesh.set("filename", target.relative_to(OUTPUT).as_posix())
    converted = OUTPUT / "robot.urdf"
    ET.ElementTree(robot).write(converted, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(converted))
    # Position the floor below the lowest visual vertex at the initial pose.
    initial = mujoco.MjData(model)
    mujoco.mj_forward(model, initial)
    lowest = 0.0
    for geom_id in range(model.ngeom):
        mesh_id = model.geom_dataid[geom_id]
        if mesh_id < 0 or model.geom_group[geom_id] == 3:
            continue
        start = model.mesh_vertadr[mesh_id]
        count = model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start:start + count]
        world_vertices = vertices @ initial.geom_xmat[geom_id].reshape(3, 3).T + initial.geom_xpos[geom_id]
        lowest = min(lowest, float(world_vertices[:, 2].min()))
    xml_path = OUTPUT / "scene.xml"
    mujoco.mj_saveLastXML(str(xml_path), model)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ET.SubElement(root, "option", timestep="0.001", gravity="0 0 -9.81",
                  integrator="implicitfast")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "quality", shadowsize="4096")
    ET.SubElement(visual, "map", znear="0.03", zfar="50", shadowscale="1")
    asset = root.find("asset")
    for mesh in asset.findall("mesh"):
        mesh.set("file", "meshes/obj/" + Path(mesh.get("file")).name)
    # Same sky, checker texture, scale and reflection as Kinova_System.xml.
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
    ET.SubElement(asset, "texture", type="2d", name="groundplane", builtin="checker",
                  mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3",
                  markrgb="0.8 0.8 0.8", width="300", height="300")
    ET.SubElement(asset, "material", name="groundplane", texture="groundplane",
                  texuniform="true", texrepeat="5 5", reflectance="0.2")
    world = root.find("worldbody")
    ET.SubElement(world.find("body[@name='base_link']"), "freejoint", name="base_free_joint")
    for geom in world.iter("geom"):
        if geom.get("mesh", "").endswith("_collision"):
            geom.set("group", "3")
            # STL convex hulls overfill the assembly: enable floor contact only.
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
    actuator = ET.SubElement(root, "actuator")
    for joint in world.iter("joint"):
        joint.set("damping", "1.0")
        joint.set("armature", "0.01")
        ET.SubElement(actuator, "motor", name=f"{joint.get('name')}_torque",
                      joint=joint.get("name"), gear="1", ctrllimited="true",
                      ctrlrange=joint.get("actuatorfrcrange", "-10 10"))
    # An oblique spotlight avoids the vertical light's grazing-angle shadow
    # artifacts on the gripper and motor housing while retaining floor shadows.
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 1 -2", directional="false",
                  cutoff="40", castshadow="true",
                  ambient="0.15 0.15 0.15", diffuse="0.9 0.9 0.9", specular="0.3 0.3 0.3")
    # Uniform fill also reaches the distant floor; no extra shadow map means
    # the directional light cannot reintroduce the previous shadow stripes.
    ET.SubElement(world, "light", name="fill", pos="0 0 2", dir="0 0 -1",
                  directional="true", castshadow="false", ambient="0.2 0.2 0.2",
                  diffuse="0.35 0.35 0.35", specular="0 0 0")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05",
                  pos=f"0 0 {lowest - 0.002}", material="groundplane", contype="1", conaffinity="2")
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="home", qpos=" ".join(map(str, HOME)))
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(xml_path))


def step(model, data, compensate=True):
    # Unit motor gear: ctrl is joint torque in Nm. Bias compensation includes
    # gravity and Coriolis; passive joint damping dissipates motion after release.
    mujoco.mj_forward(model, data)
    # Only actuated hinge DOFs receive compensation; the free base stays unactuated.
    motor_dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    data.ctrl[:] = np.clip(data.qfrc_bias[motor_dofs] if compensate else np.zeros(model.nu),
                           model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    mujoco.mj_step(model, data)


def check_dynamics(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(2000):
        step(model, data)
    assert model.nq == 16 and model.nv == 15 and model.nu == 9
    assert model.joint("base_free_joint").type == mujoco.mjtJoint.mjJNT_FREE
    assert data.ncon > 0, "Robot has no ground contact"
    assert abs(data.qpos[2]) < 0.05, "Base did not stay supported by the floor"
    assert np.linalg.norm(data.qvel) < 0.1, "Robot did not settle"
    # Lift the robot clear of the floor: its unactuated base must fall under gravity.
    falling = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, falling, 0)
    falling.qpos[2] += 0.5
    for _ in range(100):
        step(model, falling)
    assert falling.qpos[2] < 0.48 and falling.qvel[2] < -0.5
    before = data.qpos[:3].copy()
    body = model.body("base_link").id
    data.xfrc_applied[body, 0] = 100.0
    for _ in range(500):
        step(model, data)
    displacement = np.linalg.norm(data.qpos[:3] - before)
    assert displacement > 0.01, "External force did not move the free base"
    data.xfrc_applied[:] = 0
    for _ in range(2000):
        step(model, data)
    assert np.isfinite(data.qpos).all()
    assert not np.any(data.warning.number), "MuJoCo reported dynamics warnings"
    print(f"Dynamics OK: floor support, free fall, base force moved robot {displacement:.3f} m.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Build and validate without opening a window")
    parser.add_argument("--screenshot", type=Path, help="Render a PNG without opening the viewer")
    parser.add_argument("--no-gravity-compensation", action="store_true")
    args = parser.parse_args()
    model = build()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    print(f"Loaded {model.nbody - 1} bodies, {model.njnt} joints, {model.ngeom} geoms", flush=True)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.25]
    camera.distance = 1.65
    camera.azimuth = 135
    camera.elevation = -20
    if args.screenshot:
        from PIL import Image
        model.vis.global_.offwidth = 1200
        model.vis.global_.offheight = 900
        render_options = mujoco.MjvOption()
        render_options.geomgroup[3] = 0
        with mujoco.Renderer(model, height=900, width=1200) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=render_options)
            Image.fromarray(renderer.render()).save(args.screenshot)
        return
    if args.check:
        check_dynamics(model)
        return
    from mujoco import viewer as mjviewer
    events = queue.SimpleQueue()
    paused = False
    compensate = not args.no_gravity_compensation
    with mjviewer.launch_passive(model, data, key_callback=events.put) as viewer:
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance = camera.distance
        viewer.cam.azimuth = camera.azimuth
        viewer.cam.elevation = camera.elevation
        viewer.opt.geomgroup[3] = 0
        print("双击选中连杆，Ctrl+右键拖动施力，Ctrl+左键拖动施加力矩。\n"
              "Space: 暂停/继续；R: 恢复初始姿态；G: 切换重力补偿。", flush=True)
        while viewer.is_running():
            start = time.monotonic()
            while not events.empty():
                key = events.get()
                if key == 32:
                    paused = not paused
                elif key == ord("R"):
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                elif key == ord("G"):
                    compensate = not compensate
                    print(f"Gravity compensation: {compensate}", flush=True)
            if not paused:
                for _ in range(10):
                    step(model, data, compensate)
            else:
                mujoco.mj_forward(model, data)
            # sync transfers native mouse perturbations into xfrc_applied.
            data.xfrc_applied[:] = 0
            viewer.sync()
            time.sleep(max(0, 10 * model.opt.timestep - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
