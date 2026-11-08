"""Planar 3-link arm inverse kinematics.

The arm is a yaw joint plus two pitch links and a straight wrist (wrist
command is held at 0 during position tracking). It is a reach layout for
this scene, not the Franka lift policy.
"""

from __future__ import annotations

import math

from .spec import ARM_BASE, ARM_ELBOW_RANGE, ARM_L1, ARM_L2, ARM_L3, ARM_SHOULDER_RANGE, ARM_SHOULDER_Z, ARM_WRIST_RANGE, ARM_YAW_RANGE


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def forward(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    yaw, shoulder, elbow, wrist = q
    pitch2 = shoulder + elbow
    pitch3 = pitch2 + wrist
    radial = (
        ARM_L1 * math.cos(shoulder)
        + ARM_L2 * math.cos(pitch2)
        + ARM_L3 * math.cos(pitch3)
    )
    z = (
        ARM_SHOULDER_Z
        + ARM_L1 * math.sin(shoulder)
        + ARM_L2 * math.sin(pitch2)
        + ARM_L3 * math.sin(pitch3)
    )
    x = ARM_BASE[0] + radial * math.cos(yaw)
    y = ARM_BASE[1] + radial * math.sin(yaw)
    return (x, y, z)


def inverse(target: tuple[float, float, float]) -> tuple[float, float, float, float] | None:
    """Elbow-bent IK with the wrist held straight. None if out of reach or joint limits."""
    dx = target[0] - ARM_BASE[0]
    dy = target[1] - ARM_BASE[1]
    dz = target[2] - ARM_SHOULDER_Z
    horizontal = math.hypot(dx, dy)
    reach = math.hypot(horizontal, dz)
    length = ARM_L2 + ARM_L3
    max_r = ARM_L1 + length - 1e-4
    min_r = abs(ARM_L1 - length) + 1e-4
    if reach > max_r or reach < min_r or horizontal < 1e-6 and reach < 1e-6:
        return None
    cos_q2 = (reach * reach - ARM_L1 * ARM_L1 - length * length) / (2.0 * ARM_L1 * length)
    cos_q2 = _clip(cos_q2, -1.0, 1.0)
    q2 = -math.acos(cos_q2)
    q1 = math.atan2(dz, horizontal) - math.atan2(length * math.sin(q2), ARM_L1 + length * math.cos(q2))
    yaw = math.atan2(dy, dx)
    # Split the straight forearm across elbow and a zero wrist.
    elbow = q2
    wrist = 0.0
    if not (ARM_YAW_RANGE[0] <= yaw <= ARM_YAW_RANGE[1]):
        return None
    if not (ARM_SHOULDER_RANGE[0] <= q1 <= ARM_SHOULDER_RANGE[1]):
        return None
    if not (ARM_ELBOW_RANGE[0] <= elbow <= ARM_ELBOW_RANGE[1]):
        return None
    if not (ARM_WRIST_RANGE[0] <= wrist <= ARM_WRIST_RANGE[1]):
        return None
    return (yaw, q1, elbow, wrist)


def chain(q: tuple[float, float, float, float]) -> list[tuple[float, float, float]]:
    """Shoulder, elbow, wrist, and gripper positions for drawing the arm."""
    yaw, shoulder, elbow, wrist = q
    pitch2 = shoulder + elbow
    pitch3 = pitch2 + wrist
    origin = (ARM_BASE[0], ARM_BASE[1], ARM_SHOULDER_Z)

    def _tip(length: float, pitch: float, base: tuple[float, float, float]):
        radial = length * math.cos(pitch)
        return (
            base[0] + radial * math.cos(yaw),
            base[1] + radial * math.sin(yaw),
            base[2] + length * math.sin(pitch),
        )

    elbow_p = _tip(ARM_L1, shoulder, origin)
    wrist_p = _tip(ARM_L2, pitch2, elbow_p)
    tip = _tip(ARM_L3, pitch3, wrist_p)
    return [origin, elbow_p, wrist_p, tip]


def reachable(target: tuple[float, float, float], tol: float = 0.03) -> bool:
    q = inverse(target)
    if q is None:
        return False
    got = forward(q)
    return math.dist(got, target) <= tol
