"""Body-frame kiwi drive. x forward, y left, yaw counterclockwise; wheel output rad/s."""

import numpy as np


class OmniDrive:
    def __init__(self, radius=0.0654):
        self.radius = radius
        # Wheel centers and joint axes from chassis URDF, ordered left/right/back.
        centers = np.array([[0.105242, 0.182294], [0.105242, -0.182294], [-0.2105, 0.0]])
        axes = np.array([[-0.5, -0.866025403784], [-0.5, 0.866025403784], [1.0, 0.0]])
        tangent = np.column_stack((axes[:, 1], -axes[:, 0]))
        self.matrix = (
            np.column_stack(
                (tangent, -tangent[:, 0] * centers[:, 1] + tangent[:, 1] * centers[:, 0])
            )
            / radius
        )
        self.inverse = np.linalg.inv(self.matrix)

    def wheels(self, twist, limit=0.9 * 82 * np.pi / 30):
        wheel = self.matrix @ np.asarray(twist, dtype=float)
        return wheel / max(1.0, np.max(np.abs(wheel)) / limit)

    def body_velocity(self, wheel_velocity):
        return self.inverse @ np.asarray(wheel_velocity, dtype=float)
