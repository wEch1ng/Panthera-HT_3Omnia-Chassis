"""Fixed conventions of the bundled robot, independent of any particular task."""

import numpy as np

HOME = np.array([0.0, 1.0, 1.7, -0.7, 0.0, 0.0])
WHEEL_BODIES = ("left_wheel", "right_wheel", "back_wheel")
WHEEL_JOINTS = tuple(name + "_joint" for name in WHEEL_BODIES)
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
MOTOR_JOINTS = WHEEL_JOINTS + ARM_JOINTS
WHEEL_RADIUS = 0.0654
PHYSICS_HZ = 480
CONTROL_HZ = 30
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOVY = 128, 96, 58.0
