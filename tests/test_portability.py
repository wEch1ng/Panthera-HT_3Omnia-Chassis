import hashlib, json
import numpy as np
import mujoco as mj
from panthera_lab.resources import resource
from panthera_lab.sim.model import robot_xml, compile_model
from panthera_lab.robot.layout import RobotLayout
from panthera_lab.robot.config import HOME
from panthera_lab.sim.plant import RobotPlant
from panthera_lab.control.omni_drive import OmniDrive


def test_assets_and_checkpoints_from_arbitrary_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for entry in json.loads(resource("checkpoints", "manifest.json").read_text()).values():
        assert (
            hashlib.sha256(resource("checkpoints", entry["file"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )
    model = compile_model(robot_xml())
    layout = RobotLayout(model)
    assert model.nu == 9 and model.body("camera_link").id > 0
    assert len(set(layout.qpos)) == 9
    assert model.camera("wrist").id == 0


def test_generic_plant_raw_images_without_ball_task():
    model = compile_model(robot_xml())
    plant = RobotPlant(model)
    try:
        plant.d.qpos[plant.layout.arm_qpos] = HOME
        mj.mj_forward(model, plant.d)
        packet = plant.read_sensors()
        assert packet.camera.rgb.shape == (96, 128, 3)
        assert packet.camera.depth.shape == (96, 128)
        assert packet.motors.position.shape == (9,)
        plant.apply_command(HOME, np.zeros(3))
        assert np.isfinite(plant.d.qpos).all()
    finally:
        plant.close()


def test_omni_forward_inverse_and_saturation():
    drive = OmniDrive()
    for twist in ([0.08, 0, 0], [0, 0.08, 0], [0, 0, 0.3], [0.06, -0.02, 0.1]):
        np.testing.assert_allclose(drive.body_velocity(drive.wheels(twist)), twist, atol=1e-10)
    assert np.max(np.abs(drive.wheels([10, 10, 10]))) <= 0.9 * 82 * np.pi / 30 + 1e-9
