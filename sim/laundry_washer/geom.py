"""Pose helpers shared by the task gates and the CPU scene."""

from __future__ import annotations

import math

from .spec import (
    DISPENSER_SLOTS,
    DISPENSERS,
    DOOR_HANDLE_LOCAL,
    DOOR_HINGE,
    DRAWER_HANDLE_LOCAL,
    DRAWER_ORIGIN,
    DRUM_CENTER,
    DRUM_RADIUS,
    DRUM_X,
    FRONT_X,
    MOUTH_LOCAL_Z,
    OPENING_Y,
    OPENING_Z,
    PROGRAM_TOL,
    PROGRAMS,
)


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a, s: float):
    return (a[0] * s, a[1] * s, a[2] * s)


def dist(a, b) -> float:
    return math.dist(a, b)


def door_handle(angle: float) -> tuple[float, float, float]:
    """Handle position. Positive angle swings the door outward about -Z."""
    lx, ly, lz = DOOR_HANDLE_LOCAL
    c = math.cos(angle)
    s = math.sin(angle)
    x = c * lx + s * ly
    y = -s * lx + c * ly
    return (DOOR_HINGE[0] + x, DOOR_HINGE[1] + y, DOOR_HINGE[2] + lz)


def drawer_handle(slide: float) -> tuple[float, float, float]:
    return (
        DRAWER_ORIGIN[0] + slide + DRAWER_HANDLE_LOCAL[0],
        DRAWER_ORIGIN[1] + DRAWER_HANDLE_LOCAL[1],
        DRAWER_ORIGIN[2] + DRAWER_HANDLE_LOCAL[2],
    )


def cup_world(name: str, slide: float) -> tuple[float, float, float]:
    lx, ly, lz = DISPENSERS[name]
    return (DRAWER_ORIGIN[0] + slide + lx, DRAWER_ORIGIN[1] + ly, DRAWER_ORIGIN[2] + lz)


def mouth_offset(tilt: float) -> tuple[float, float, float]:
    """Mouth relative to the bottle center. tilt=0 is upright; large tilt points down and toward -X."""
    return (
        -MOUTH_LOCAL_Z * math.sin(tilt),
        0.0,
        MOUTH_LOCAL_Z * math.cos(tilt),
    )


def bottle_center_for_mouth(mouth, tilt: float):
    return sub(mouth, mouth_offset(tilt))


def in_drum(pos) -> bool:
    if not (DRUM_X[0] <= pos[0] <= DRUM_X[1]):
        return False
    dy = pos[1] - DRUM_CENTER[1]
    dz = pos[2] - DRUM_CENTER[2]
    return dy * dy + dz * dz <= DRUM_RADIUS * DRUM_RADIUS


def in_opening(pos, radius: float = 0.0) -> bool:
    return (
        OPENING_Y[0] - radius <= pos[1] <= OPENING_Y[1] + radius
        and OPENING_Z[0] - radius <= pos[2] <= OPENING_Z[1] + radius
    )


def crosses_closed_door(old, new, door_q: float, door_open: float, radius: float) -> bool:
    """True when a body would pass inward through the shut aperture."""
    if door_q >= door_open:
        return False
    plane = FRONT_X - 0.005
    if old[0] >= plane and new[0] < plane and (in_opening(old, radius) or in_opening(new, radius)):
        return True
    return False


def selected_program(knob_q: float) -> tuple[str, bool]:
    best_name = "between"
    best_valid = False
    best_d = 1e9
    for name, angle, valid in PROGRAMS:
        gap = abs(knob_q - angle)
        if gap < best_d:
            best_d = gap
            best_name = name
            best_valid = valid
    if best_d > PROGRAM_TOL:
        return "between", False
    return best_name, best_valid


def slot_world(name: str):
    return DISPENSER_SLOTS[name]
