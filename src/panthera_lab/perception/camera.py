"""URDF-mounted RGB-D renderer. Simulation state stays inside this sensor backend."""

import numpy as np
import mujoco as mj
from panthera_lab.interfaces import CameraFrame
from panthera_lab.robot.config import CAMERA_WIDTH, CAMERA_HEIGHT


class WristCamera:
    def __init__(self, model):
        self.model = model
        self.camera_id = model.camera("wrist").id
        self.renderer = mj.Renderer(model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH)
        self.options = mj.MjvOption()
        self.options.geomgroup[3] = 0
        self.geoms = np.array(
            [
                g
                for g in range(model.ngeom)
                if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH and model.geom_group[g] != 3
            ]
        )
        corners = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
        self.box = (
            model.geom_aabb[self.geoms, None, :3] + corners * model.geom_aabb[self.geoms, None, 3:]
        )

    def capture(self, data):
        r = self.renderer
        cam = self.camera_id
        idx = self.geoms
        r.update_scene(data, camera="wrist", scene_option=self.options)
        r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
        r.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
        corners = (
            self.box @ data.geom_xmat[idx].reshape(-1, 3, 3).transpose(0, 2, 1)
            + data.geom_xpos[idx, None, :]
            - data.cam_xpos[cam]
        ) @ data.cam_xmat[cam].reshape(3, 3)
        x, y, z = corners.transpose(2, 0, 1)
        hy = -z * 0.56 + 0.002
        hx = -z * 0.75 + 0.002
        out = set(
            idx[
                (x > hx).all(1)
                | (x < -hx).all(1)
                | (y > hy).all(1)
                | (y < -hy).all(1)
                | (z > 0).all(1)
            ]
        )
        for geom in r.scene.geoms[: r.scene.ngeom]:
            if geom.objtype == mj.mjtObj.mjOBJ_GEOM and geom.objid in out:
                geom.type = mj.mjtGeom.mjGEOM_SPHERE
                geom.size[:] = 0
        rgb = r.render().copy()
        r.enable_depth_rendering()
        try:
            depth = r.render().copy()
        finally:
            r.disable_depth_rendering()
        return CameraFrame(rgb, depth, float(self.model.cam_fovy[cam]))

    def close(self):
        self.renderer.close()
