"""Panthera joint-module limits transcribed from the supplied High Torque catalog.

The 6-DOF policy controls joints 1..6. Joint 7 is the HTDW-3532 gripper and is
recorded separately because the paddle simulation keeps the gripper fixed.
"""

import numpy as np

MOTOR_MODEL = (
    "HTDW-5036-02-DNE",
    "HTDW-6036-02-DNE",
    "HTDW-6036-02-DNE",
    "HTDW-5036-02-DNE",
    "HTDW-4438-30-NE",
    "HTDW-4438-30-NE",
)

# Catalog module-output values, ordered by robot joint 1..6.
RATED_TORQUE_NM = np.array([6.0, 10.0, 10.0, 6.0, 2.0, 2.0])
STALL_TORQUE_NM = np.array([21.0, 36.0, 36.0, 21.0, 10.0, 10.0])
RATED_SPEED_RPM = np.array([50.0, 50.0, 50.0, 50.0, 40.0, 40.0])
NO_LOAD_SPEED_RPM = np.array([75.0, 60.0, 60.0, 75.0, 160.0, 160.0])
RATED_CURRENT_A = np.array([2.0, 3.0, 3.0, 2.0, 0.7, 0.7])
PEAK_CURRENT_A = np.array([10.0, 15.0, 15.0, 10.0, 10.0, 10.0])
TORQUE_CONSTANT_NM_A = np.array([0.54, 0.531, 0.531, 0.54, 0.272, 0.272])

GRIPPER_MOTOR = dict(
    model="HTDW-3532-02-DNE",
    rated_torque_nm=0.6,
    stall_torque_nm=3.7,
    rated_speed_rpm=60.0,
    no_load_speed_rpm=300.0,
    rated_current_a=0.35,
    peak_current_a=10.0,
    torque_constant_nm_a=0.125,
    rated_voltage_v=24.0,
)

RPM_TO_RAD_S = 2 * np.pi / 60
RATED_SPEED_RAD_S = RATED_SPEED_RPM * RPM_TO_RAD_S
NO_LOAD_SPEED_RAD_S = NO_LOAD_SPEED_RPM * RPM_TO_RAD_S
