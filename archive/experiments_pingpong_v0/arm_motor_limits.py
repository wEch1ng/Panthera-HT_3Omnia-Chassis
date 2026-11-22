"""Panthera joint-module limits transcribed from the supplied High Torque catalog.

The 6-DOF policy controls joints 1..6. Joint 7 is the HTDW-3532 gripper and is
recorded separately because the paddle simulation keeps the gripper fixed.
"""
import numpy as np

MOTOR_MODEL = ('HTDW-5036-02-DNE', 'HTDW-6036-02-DNE',
               'HTDW-6036-02-DNE', 'HTDW-5036-02-DNE',
               'HTDW-4438-30-NE', 'HTDW-4438-30-NE')

# Catalog module-output values, ordered by robot joint 1..6.
RATED_TORQUE_NM = np.array([6., 10., 10., 6., 2., 2.])
STALL_TORQUE_NM = np.array([21., 36., 36., 21., 10., 10.])
RATED_SPEED_RPM = np.array([50., 50., 50., 50., 40., 40.])
NO_LOAD_SPEED_RPM = np.array([75., 60., 60., 75., 160., 160.])
RATED_CURRENT_A = np.array([2., 3., 3., 2., .7, .7])
PEAK_CURRENT_A = np.array([10., 15., 15., 10., 10., 10.])
TORQUE_CONSTANT_NM_A = np.array([.54, .531, .531, .54, .272, .272])

GRIPPER_MOTOR = dict(model='HTDW-3532-02-DNE', rated_torque_nm=.6,
                     stall_torque_nm=3.7, rated_speed_rpm=60.,
                     no_load_speed_rpm=300., rated_current_a=.35,
                     peak_current_a=10., torque_constant_nm_a=.125,
                     rated_voltage_v=24.)

RPM_TO_RAD_S = 2*np.pi/60
RATED_SPEED_RAD_S = RATED_SPEED_RPM*RPM_TO_RAD_S
NO_LOAD_SPEED_RAD_S = NO_LOAD_SPEED_RPM*RPM_TO_RAD_S
