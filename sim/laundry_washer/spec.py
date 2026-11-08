"""Shared geometry for the laundry scene.

CPU stepping, the MJCF generator, and the software rasterizer all read this
module so the task gates and the Newton model describe the same machine.
Lengths are metres. Angles are radians. Gravity is the laundry baseline,
not the lift-task ADR difficulty-0 vector (0, 0, 0).
"""

from __future__ import annotations

from .runtime_config import GRAVITY, LABEL_CLOTH, LABEL_FLUID, LABEL_NO_POLICY, LABEL_PHYSICAL, LABEL_SCRIPTED

DT = 0.01
CONTROL_SUBSTEPS = 4

# Washer cabinet, front face toward +X.
CABINET_MIN = (-0.30, -0.30, 0.0)
CABINET_MAX = (0.30, 0.30, 0.85)
FRONT_X = 0.30
OPENING_Y = (-0.20, 0.20)
OPENING_Z = (0.22, 0.62)

DRUM_CENTER = (0.02, 0.0, 0.42)
DRUM_RADIUS = 0.18
DRUM_X = (-0.10, 0.16)
DRUM_SPEED = 3.0  # rad/s while washing
WASH_TURNS = 2.0  # full revolutions required

# Door hinge on the left stile. Axis is -Z so a positive angle swings outward (+X).
DOOR_HINGE = (0.30, -0.24, 0.42)
DOOR_AXIS_Z = -1.0
DOOR_RANGE = (0.0, 1.70)
DOOR_OPEN = 1.20
DOOR_CLOSED = 0.10
DOOR_HANDLE_LOCAL = (0.03, 0.42, 0.0)

DRAWER_ORIGIN = (0.10, 0.0, 0.74)
DRAWER_RANGE = (0.0, 0.20)
DRAWER_OPEN = 0.15
DRAWER_CLOSED = 0.02
DRAWER_HANDLE_LOCAL = (0.12, 0.0, 0.0)

# Cups ride on the drawer. The slot is the fixed mouth of the dispenser
# housing on the cabinet front; pouring there while the drawer is shut
# does not fill a cup.
DISPENSERS = {
    "main": (-0.02, 0.09, 0.035),
    "softener": (-0.02, 0.00, 0.035),
    "bleach": (-0.02, -0.09, 0.035),
}
DISPENSER_SLOTS = {
    "main": (0.34, 0.09, 0.78),
    "softener": (0.34, 0.00, 0.78),
    "bleach": (0.34, -0.09, 0.78),
}
POUR_RADIUS = 0.06
CORRECT_DISPENSER = "main"

CAP_RANGE = (0.0, 2.60)
CAP_OPEN = 1.20
CAP_CLOSED = 0.20
MOUTH_LOCAL_Z = 0.09
POUR_TILT = 2.10
POUR_TILT_MIN = 1.80
UPRIGHT_TILT = 0.15

BOTTLE_HOME = (0.55, 0.42, 0.74)
BOTTLE_RADIUS = 0.04
LIQUID_ML = 150.0
DOSE_ML = 40.0
CAPACITY_ML = 60.0
MIN_DOSE_ML = 30.0

# Knob on the front panel. Nearest mark within PROGRAM_TOL is the selection.
PROGRAMS = (
    ("off", 0.00, False),
    ("cotton", 0.80, True),
    ("synthetic", 1.60, True),
    ("delicate", 2.40, True),
    ("spin_only", 3.20, False),
)
PROGRAM_TOL = 0.28
KNOB_POS = (0.32, 0.20, 0.66)
BUTTON_POS = (0.32, 0.20, 0.56)
BUTTON_PRESSED = -0.006
LOCK_ENGAGED = 0.02
LOCK_POS = (0.30, 0.22, 0.42)

BASKET_CENTER = (0.62, -0.42, 0.0)
BASKET_HALF = (0.16, 0.14)
BASKET_FLOOR_Z = 0.06
COUNTER_CENTER = (0.55, 0.42)
COUNTER_HALF = (0.14, 0.12)
COUNTER_TOP_Z = 0.70

# Distinct rigid proxies. Half-extents are the box half-sizes.
GARMENTS = (
    {"name": "shirt", "pos": (0.56, -0.48, 0.072), "half": (0.045, 0.030, 0.012), "rgba": (0.75, 0.18, 0.16, 1.0), "mass": 0.18},
    {"name": "pants", "pos": (0.68, -0.38, 0.074), "half": (0.050, 0.028, 0.014), "rgba": (0.15, 0.28, 0.72, 1.0), "mass": 0.22},
    {"name": "towel", "pos": (0.56, -0.34, 0.070), "half": (0.040, 0.036, 0.010), "rgba": (0.20, 0.62, 0.28, 1.0), "mass": 0.16},
    {"name": "sock", "pos": (0.70, -0.50, 0.068), "half": (0.022, 0.016, 0.008), "rgba": (0.90, 0.78, 0.15, 1.0), "mass": 0.06},
)

ARM_BASE = (0.78, 0.05, 0.0)
ARM_SHOULDER_Z = 0.48
ARM_L1 = 0.34
ARM_L2 = 0.32
ARM_L3 = 0.12
ARM_HOME = (0.0, 0.55, -1.15, 0.55)  # yaw, shoulder, elbow, wrist
ARM_YAW_RANGE = (-3.14159, 3.14159)
ARM_SHOULDER_RANGE = (-1.6, 2.2)
ARM_ELBOW_RANGE = (-2.7, 2.6)
ARM_WRIST_RANGE = (-2.6, 2.6)
GRIP_SPEED = 0.75  # m/s cartesian limit while tracking IK
JOINT_VEL = {
    "door_hinge": 1.6,
    "drawer_slide": 0.12,
    "cap_hinge": 1.8,
    "program_knob": 1.5,
    "start_button": 0.05,
    "lock_pin": 0.08,
    "bottle_tilt": 1.6,
    "drum_spin": 4.0,
    "arm_yaw": 1.8,
    "arm_shoulder": 1.6,
    "arm_elbow": 2.0,
    "arm_wrist": 2.2,
}
JOINT_EFFORT = {
    "door_hinge": 8.0,
    "drawer_slide": 12.0,
    "cap_hinge": 1.5,
    "program_knob": 1.0,
    "start_button": 2.0,
    "lock_pin": 6.0,
    "bottle_tilt": 1.2,
    "drum_spin": 4.0,
    "arm_yaw": 20.0,
    "arm_shoulder": 20.0,
    "arm_elbow": 12.0,
    "arm_wrist": 6.0,
}

NEAR_HANDLE = 0.10
MAX_STEPS = 12000
LOAD_POINT = (0.08, 0.0, 0.42)
FLOOR_POUR_MOUTH = (0.95, 0.45, 0.35)

ASSUMPTIONS = (
    LABEL_CLOTH,
    LABEL_FLUID,
    LABEL_NO_POLICY,
    "door, drawer, cap, knob, button and lock are actuated joints with range and speed limits",
    "CPU backend is a deterministic rigid integrator plus an aperture constraint; it is not MuJoCo contact",
    "Newton/MuJoCo MJCF is the contact scene for the one-env GPU path",
    "normal gravity (0, 0, -9.81); the lift ADR zero-gravity schedule is not loaded",
)

OBS_NAMES = (
    "door_q",
    "drawer_q",
    "cap_q",
    "knob_q",
    "button_q",
    "lock_q",
    "drum_q",
    "drum_qd",
    "bottle_x",
    "bottle_y",
    "bottle_z",
    "bottle_tilt",
    "liquid_ml",
    "main_ml",
    "softener_ml",
    "bleach_ml",
    "spill_ml",
    "shirt_x",
    "shirt_y",
    "shirt_z",
    "shirt_in_drum",
    "pants_x",
    "pants_y",
    "pants_z",
    "pants_in_drum",
    "towel_x",
    "towel_y",
    "towel_z",
    "towel_in_drum",
    "sock_x",
    "sock_y",
    "sock_z",
    "sock_in_drum",
    "gripper_x",
    "gripper_y",
    "gripper_z",
    "grasped",
    "running",
    "wash_turns",
    "program_valid",
    "overfill",
    "gravity_z",
)

# Continuous action, one control interval. Later policies can use this vector.
# 0 door target rad, 1 drawer target m, 2 cap target rad, 3 knob target rad,
# 4 button command 0..1, 5-7 gripper target xyz, 8 grasp (-1 release, +1 hold),
# 9 pour trigger 0..1, 10 bottle tilt target rad.
ACTION_NAMES = (
    "door_target",
    "drawer_target",
    "cap_target",
    "knob_target",
    "button",
    "grip_x",
    "grip_y",
    "grip_z",
    "grasp",
    "pour",
    "tilt_target",
)
ACTION_LOW = (-0.2, -0.05, -0.2, -0.3, 0.0, -0.2, -1.2, 0.0, -1.0, 0.0, -0.2)
ACTION_HIGH = (1.8, 0.25, 2.8, 3.5, 1.0, 1.4, 1.0, 1.4, 1.0, 1.0, 2.6)

__all__ = [name for name in globals() if name.isupper()]
