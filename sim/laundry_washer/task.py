"""Task gates for the washer episode.

Success is a conjunction of physical predicates (garments inside the drum
volume, door shut, drawer shut, dose in the main cup, valid program, latch
engaged, drum revolutions). It is not a sequence counter. Ordering can vary.

This module does not step MuJoCo, Newton, or a GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geom import cup_world, dist, in_drum, mouth_offset, selected_program, slot_world
from .spec import (
    CAP_CLOSED,
    CAP_OPEN,
    CAPACITY_ML,
    CORRECT_DISPENSER,
    DOOR_CLOSED,
    DOOR_OPEN,
    DOSE_ML,
    DRAWER_CLOSED,
    DRAWER_OPEN,
    LABEL_CLOTH,
    LABEL_FLUID,
    LABEL_NO_POLICY,
    LABEL_PHYSICAL,
    LABEL_SCRIPTED,
    LOCK_ENGAGED,
    MIN_DOSE_ML,
    OBS_NAMES,
    POUR_RADIUS,
    POUR_TILT_MIN,
    WASH_TURNS,
)


@dataclass
class Snapshot:
    n_in_drum: int
    main_ml: float
    spill_ml: float
    overfill: bool
    running: bool
    wash_complete: bool
    rejected_locked: bool


def validation_label(action_mode: str) -> str:
    if action_mode == "scripted_kinematic":
        return LABEL_SCRIPTED
    return LABEL_PHYSICAL


def program_of(world) -> tuple[str, bool]:
    return selected_program(world.q["program_knob"])


def garments_in_drum(world) -> dict[str, bool]:
    return {name: in_drum(body.pos) for name, body in world.garments.items()}


def all_in_drum(world) -> bool:
    flags = garments_in_drum(world)
    return bool(flags) and all(flags.values())


def bottle_mouth(world):
    return (
        world.bottle.pos[0] + mouth_offset(world.q["bottle_tilt"])[0],
        world.bottle.pos[1] + mouth_offset(world.q["bottle_tilt"])[1],
        world.bottle.pos[2] + mouth_offset(world.q["bottle_tilt"])[2],
    )


def _dose_ok(world) -> bool:
    main = world.doses["main"]
    return (not world.overfill) and MIN_DOSE_ML <= main <= CAPACITY_ML + 1e-6


def start_blockers(world) -> list[str]:
    """Why a start press must be refused. Empty means the machine may run."""
    reasons: list[str] = []
    if not all_in_drum(world):
        reasons.append("clothes_not_in_drum")
    if world.q["door_hinge"] > DOOR_CLOSED:
        reasons.append("door_open")
    if world.q["drawer_slide"] > DRAWER_CLOSED:
        reasons.append("drawer_open")
    if world.overfill:
        reasons.append("overfill")
    if not _dose_ok(world):
        reasons.append("detergent_not_in_main")
    name, valid = program_of(world)
    if not valid:
        reasons.append("invalid_program" if name != "between" else "invalid_program")
    if world.q["cap_hinge"] > CAP_CLOSED:
        reasons.append("cap_open")
    if in_drum(world.bottle.pos):
        reasons.append("bottle_in_drum")
    if world.doses["main"] < MIN_DOSE_ML and (
        world.doses["softener"] > 0.0 or world.doses["bleach"] > 0.0
    ):
        reasons.append("wrong_dispenser")
    return reasons


def holds_success(world) -> bool:
    if not world.wash_complete or world.failed_after_start:
        return False
    if not all_in_drum(world):
        return False
    if world.q["door_hinge"] > DOOR_CLOSED:
        return False
    if world.q["drawer_slide"] > DRAWER_CLOSED:
        return False
    if world.q["lock_pin"] < LOCK_ENGAGED * 0.5:
        return False
    if not _dose_ok(world):
        return False
    _name, valid = program_of(world)
    if not valid:
        return False
    if world.q["cap_hinge"] > CAP_CLOSED:
        return False
    if world.wash_angle < 2.0 * math.pi * WASH_TURNS - 1e-6:
        return False
    return True


def stage_name(world) -> str:
    if holds_success(world):
        return "wash_cycle_complete"
    if world.running:
        return "washing"
    if not start_blockers(world):
        return "ready_to_start"
    if all_in_drum(world) and world.q["door_hinge"] <= DOOR_CLOSED and world.q["drawer_slide"] <= DRAWER_CLOSED:
        return "loaded_needs_dose_or_program"
    if all_in_drum(world):
        return "loaded_still_open"
    if world.q["door_hinge"] >= DOOR_OPEN:
        return "door_open_loading"
    return "need_load"


def attempt_pour(world) -> str:
    """Pose+dose pour. Not a fluid simulation. Returns a result token."""
    if world.q["cap_hinge"] < CAP_OPEN:
        world.last_pour = "closed_cap"
        return world.last_pour
    if world.liquid_ml <= 1e-6:
        world.last_pour = "empty"
        return world.last_pour
    if world.q["bottle_tilt"] < POUR_TILT_MIN:
        world.last_pour = "not_tilted"
        return world.last_pour

    mouth = bottle_mouth(world)
    drawer_open = world.q["drawer_slide"] >= DRAWER_OPEN

    if not drawer_open:
        for name in ("main", "softener", "bleach"):
            if dist(mouth, slot_world(name)) <= POUR_RADIUS:
                _consume(world, world.doses, name, into_cup=False, result="closed_drawer")
                return world.last_pour

    best_name = None
    best_d = 1e9
    for name in ("main", "softener", "bleach"):
        gap = dist(mouth, cup_world(name, world.q["drawer_slide"]))
        if gap < best_d:
            best_d = gap
            best_name = name

    if best_name is not None and best_d <= POUR_RADIUS:
        if not drawer_open:
            _consume(world, world.doses, best_name, into_cup=False, result="closed_drawer")
            return world.last_pour
        _consume(world, world.doses, best_name, into_cup=True, result=best_name)
        return world.last_pour

    _consume(world, world.doses, CORRECT_DISPENSER, into_cup=False, result="floor")
    return world.last_pour


def _consume(world, doses, name: str, into_cup: bool, result: str) -> None:
    take = min(DOSE_ML, world.liquid_ml)
    if take <= 0.0:
        world.last_pour = "empty"
        return
    world.liquid_ml -= take
    if not into_cup:
        world.spill_ml += take
        world.last_pour = result
        return
    space = CAPACITY_ML - doses[name]
    if take > space + 1e-9:
        accepted = max(0.0, space)
        doses[name] += accepted
        world.spill_ml += take - accepted
        world.overfill = True
        world.last_pour = "overfill"
        return
    doses[name] += take
    world.last_pour = result


def compute_reward(prev: Snapshot, cur: Snapshot) -> float:
    reward = -0.002  # per physics step (10 ms); about -0.2 per second
    reward += 1.0 * (cur.n_in_drum - prev.n_in_drum)
    if cur.main_ml > prev.main_ml + 1e-9 and not cur.overfill:
        reward += 2.0
    if cur.spill_ml > prev.spill_ml + 1e-9:
        reward -= 1.0
    if cur.overfill and not prev.overfill:
        reward -= 2.0
    if cur.running and not prev.running:
        reward += 5.0
    if cur.wash_complete and not prev.wash_complete:
        reward += 10.0
    if cur.rejected_locked and not prev.rejected_locked:
        reward -= 0.5
    return reward


def observation_vector(world) -> list[float]:
    flags = garments_in_drum(world)
    _name, valid = program_of(world)
    mouth_tilt = world.q["bottle_tilt"]
    values = {
        "door_q": world.q["door_hinge"],
        "drawer_q": world.q["drawer_slide"],
        "cap_q": world.q["cap_hinge"],
        "knob_q": world.q["program_knob"],
        "button_q": world.q["start_button"],
        "lock_q": world.q["lock_pin"],
        "drum_q": world.q["drum_spin"],
        "drum_qd": world.qd["drum_spin"],
        "bottle_x": world.bottle.pos[0],
        "bottle_y": world.bottle.pos[1],
        "bottle_z": world.bottle.pos[2],
        "bottle_tilt": mouth_tilt,
        "liquid_ml": world.liquid_ml,
        "main_ml": world.doses["main"],
        "softener_ml": world.doses["softener"],
        "bleach_ml": world.doses["bleach"],
        "spill_ml": world.spill_ml,
        "gripper_x": world.fk[0],
        "gripper_y": world.fk[1],
        "gripper_z": world.fk[2],
        "grasped": 0.0 if world.grasp is None else 1.0,
        "running": 1.0 if world.running else 0.0,
        "wash_turns": world.wash_angle / (2.0 * math.pi),
        "program_valid": 1.0 if valid else 0.0,
        "overfill": 1.0 if world.overfill else 0.0,
        "gravity_z": world.gravity[2],
    }
    for name in ("shirt", "pants", "towel", "sock"):
        pos = world.garments[name].pos
        values[f"{name}_x"] = pos[0]
        values[f"{name}_y"] = pos[1]
        values[f"{name}_z"] = pos[2]
        values[f"{name}_in_drum"] = 1.0 if flags[name] else 0.0
    return [float(values[name]) for name in OBS_NAMES]


def assumption_lines() -> tuple[str, ...]:
    return (
        LABEL_CLOTH,
        LABEL_FLUID,
        LABEL_NO_POLICY,
        LABEL_SCRIPTED,
        LABEL_PHYSICAL,
    )
