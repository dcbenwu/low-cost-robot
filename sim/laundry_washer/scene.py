"""Deterministic rigid scene used by the CPU task tests.

Joints are velocity-limited position servos (they do not jump to the
target). Free bodies in physical mode integrate ``gravity`` with
semi-implicit Euler and stop on the floor, the basket, the counter, or
the drum shell. A shut door is an aperture constraint: a body cannot be
carried inward through the front opening. The MJCF actuators carry the
effort limits used by Newton; this CPU backend does not pretend to be
that contact solver.

This is not a MuJoCo or Newton contact solve. The MJCF in ``mjcf.py`` is
the scene those solvers load. Tests that use this class are task-logic
tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import arm
from .geom import add, crosses_closed_door, sub
from .spec import (
    ARM_ELBOW_RANGE,
    ARM_HOME,
    ARM_SHOULDER_RANGE,
    ARM_WRIST_RANGE,
    ARM_YAW_RANGE,
    BASKET_CENTER,
    BASKET_FLOOR_Z,
    BASKET_HALF,
    BOTTLE_HOME,
    CAP_RANGE,
    COUNTER_CENTER,
    COUNTER_HALF,
    COUNTER_TOP_Z,
    DOOR_CLOSED,
    DOOR_OPEN,
    DOOR_RANGE,
    DRAWER_ORIGIN,
    DRAWER_RANGE,
    DRUM_CENTER,
    DRUM_RADIUS,
    DRUM_SPEED,
    DRUM_X,
    DT,
    GARMENTS,
    GRAVITY,
    JOINT_VEL,
    LIQUID_ML,
    LOCK_ENGAGED,
    WASH_TURNS,
)
from .task import Snapshot, all_in_drum, attempt_pour, holds_success, start_blockers

ARM_JOINTS = ("arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist")
JOINT_LIMITS = {
    "door_hinge": DOOR_RANGE,
    "drawer_slide": DRAWER_RANGE,
    "cap_hinge": CAP_RANGE,
    "program_knob": (-0.2, 3.4),
    "start_button": (-0.01, 0.0),
    "lock_pin": (0.0, 0.025),
    "bottle_tilt": (0.0, 2.4),
    "drum_spin": (-1e6, 1e6),
    "arm_yaw": ARM_YAW_RANGE,
    "arm_shoulder": ARM_SHOULDER_RANGE,
    "arm_elbow": ARM_ELBOW_RANGE,
    "arm_wrist": ARM_WRIST_RANGE,
}


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class Body:
    name: str
    pos: tuple[float, float, float]
    vel: tuple[float, float, float]
    half: tuple[float, float, float]
    mass: float

    @property
    def radius(self) -> float:
        return max(self.half)


class LaundryScene:
    def __init__(self, kinematic: bool = False):
        self.kinematic = kinematic
        self.gravity = (float(GRAVITY[0]), float(GRAVITY[1]), float(GRAVITY[2]))
        self.q: dict[str, float] = {}
        self.qd: dict[str, float] = {}
        self.target: dict[str, float] = {}
        self.garments: dict[str, Body] = {}
        self.bottle = Body("bottle", BOTTLE_HOME, (0.0, 0.0, 0.0), (0.04, 0.04, 0.04), 0.35)
        self.grasp: str | None = None
        self.offset = (0.0, 0.0, 0.0)
        self.arm_setpoint: tuple[float, float, float] | None = None
        self.blocked = False
        self.step_count = 0
        self.liquid_ml = LIQUID_ML
        self.doses = {"main": 0.0, "softener": 0.0, "bleach": 0.0}
        self.spill_ml = 0.0
        self.overfill = False
        self.running = False
        self.wash_angle = 0.0
        self.wash_complete = False
        self.failed_after_start = False
        self.rejected_locked = False
        self.last_pour = ""
        self.last_reject = ""
        self.last_start_reasons: list[str] = []
        self.fk = (0.0, 0.0, 0.0)
        self._integrated_gravity_z = 0.0
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        for name, limits in JOINT_LIMITS.items():
            home = 0.0
            if name == "arm_yaw":
                home = ARM_HOME[0]
            elif name == "arm_shoulder":
                home = ARM_HOME[1]
            elif name == "arm_elbow":
                home = ARM_HOME[2]
            elif name == "arm_wrist":
                home = ARM_HOME[3]
            self.q[name] = home
            self.qd[name] = 0.0
            self.target[name] = home
        self.garments = {}
        for item in GARMENTS:
            self.garments[item["name"]] = Body(
                item["name"],
                tuple(item["pos"]),
                (0.0, 0.0, 0.0),
                tuple(item["half"]),
                float(item["mass"]),
            )
        self.bottle = Body("bottle", BOTTLE_HOME, (0.0, 0.0, 0.0), (0.04, 0.04, 0.04), 0.35)
        self.grasp = None
        self.offset = (0.0, 0.0, 0.0)
        self.arm_setpoint = None
        self.blocked = False
        self.step_count = 0
        self.liquid_ml = LIQUID_ML
        self.doses = {"main": 0.0, "softener": 0.0, "bleach": 0.0}
        self.spill_ml = 0.0
        self.overfill = False
        self.running = False
        self.wash_angle = 0.0
        self.wash_complete = False
        self.failed_after_start = False
        self.rejected_locked = False
        self.last_pour = ""
        self.last_reject = ""
        self.last_start_reasons = []
        self._integrated_gravity_z = 0.0
        self.fk = arm.forward(self._arm_q())

    def place(self, name: str, pos, vel=(0.0, 0.0, 0.0)) -> None:
        body = self.garments[name] if name in self.garments else self.bottle
        body.pos = (float(pos[0]), float(pos[1]), float(pos[2]))
        body.vel = (float(vel[0]), float(vel[1]), float(vel[2]))
        if self.grasp == name:
            self.grasp = None

    def body(self, name: str) -> Body:
        if name == "bottle":
            return self.bottle
        return self.garments[name]

    def _arm_q(self):
        return tuple(self.q[name] for name in ARM_JOINTS)

    def gripper_position(self):
        return arm.forward(self._arm_q())

    # ----------------------------------------------------------------- inputs
    def set_arm_setpoint(self, point) -> None:
        self.arm_setpoint = None if point is None else (float(point[0]), float(point[1]), float(point[2]))

    def grasp_body(self, name: str) -> bool:
        body = self.body(name)
        gap = math.dist(self.gripper_position(), body.pos)
        if gap > 0.05:
            self.last_reject = "grasp_too_far"
            return False
        self.grasp = name
        self.offset = sub(body.pos, self.gripper_position())
        body.vel = (0.0, 0.0, 0.0)
        return True

    def grasp_nearest(self) -> str | None:
        best = None
        best_d = 1e9
        for name, body in list(self.garments.items()) + [("bottle", self.bottle)]:
            gap = math.dist(self.gripper_position(), body.pos)
            if gap < best_d:
                best_d = gap
                best = name
        if best is None or best_d > 0.05:
            return None
        if self.grasp_body(best):
            return best
        return None

    def release(self) -> None:
        self.grasp = None
        self.offset = (0.0, 0.0, 0.0)

    def pour(self) -> str:
        return attempt_pour(self)

    def try_start(self) -> bool:
        reasons = start_blockers(self)
        self.last_start_reasons = list(reasons)
        if reasons:
            self.last_reject = "start_rejected:" + ",".join(reasons)
            return False
        self.running = True
        self.target["lock_pin"] = 0.025
        self.target["door_hinge"] = min(self.q["door_hinge"], DOOR_CLOSED * 0.25)
        self.last_reject = ""
        return True

    # ------------------------------------------------------------------- step
    def step(self) -> None:
        self.blocked = False
        self._apply_arm_setpoint()
        before = {name: self.q[name] for name in ARM_JOINTS}
        before_qd = {name: self.qd[name] for name in ARM_JOINTS}
        self._enforce_lock_target()
        for name, limits in JOINT_LIMITS.items():
            if name == "drum_spin" and self.running:
                continue
            self._servo(name, limits[0], limits[1])
        self._enforce_lock_position()
        self.fk = self.gripper_position()
        if self.grasp is not None:
            body = self.body(self.grasp)
            proposed = add(self.fk, self.offset)
            if crosses_closed_door(body.pos, proposed, self.q["door_hinge"], DOOR_OPEN, body.radius):
                for name in ARM_JOINTS:
                    self.q[name] = before[name]
                    self.qd[name] = before_qd[name]
                self.fk = self.gripper_position()
                self.blocked = True
                self.last_reject = "door_blocks_entry"
            else:
                body.pos = proposed
                body.vel = (0.0, 0.0, 0.0)
        if not self.kinematic:
            self._integrate_free_bodies()
        self._spin_drum()
        self.step_count += 1
        self.fk = self.gripper_position()

    def _apply_arm_setpoint(self) -> None:
        if self.arm_setpoint is None:
            return
        solved = arm.inverse(self.arm_setpoint)
        if solved is None:
            return
        for name, value in zip(ARM_JOINTS, solved):
            self.target[name] = value

    def _servo(self, name: str, lo: float, hi: float) -> None:
        target = _clip(self.target[name], lo, hi)
        vel_lim = JOINT_VEL[name]
        q = self.q[name]
        err = target - q
        if name == "arm_yaw":
            err = (err + math.pi) % (2.0 * math.pi) - math.pi
        max_step = vel_lim * DT
        dq = _clip(err, -max_step, max_step)
        q = q + dq
        if name == "arm_yaw":
            q = (q + math.pi) % (2.0 * math.pi) - math.pi
        else:
            q = _clip(q, lo, hi)
        self.q[name] = q
        self.qd[name] = dq / DT

    def _enforce_lock_target(self) -> None:
        if not self.running:
            return
        self.target["lock_pin"] = 0.025
        if self.target["door_hinge"] > DOOR_CLOSED:
            self.rejected_locked = True
            self.last_reject = "door_locked_while_running"
            self.target["door_hinge"] = min(self.q["door_hinge"], DOOR_CLOSED * 0.25)

    def _enforce_lock_position(self) -> None:
        if self.running and self.q["lock_pin"] >= LOCK_ENGAGED * 0.5 and self.q["door_hinge"] > DOOR_CLOSED:
            self.q["door_hinge"] = DOOR_CLOSED * 0.25
            self.qd["door_hinge"] = 0.0
            self.failed_after_start = True
            self.last_reject = "door_locked_while_running"

    def _spin_drum(self) -> None:
        if not self.running:
            return
        self.qd["drum_spin"] = DRUM_SPEED
        self.q["drum_spin"] += DRUM_SPEED * DT
        if not self.failed_after_start:
            self.wash_angle += DRUM_SPEED * DT
        if (
            not self.failed_after_start
            and self.wash_angle >= 2.0 * math.pi * WASH_TURNS
            and all_in_drum(self)
            and self.q["door_hinge"] <= DOOR_CLOSED
        ):
            self.wash_complete = True

    def _integrate_free_bodies(self) -> None:
        gz = self.gravity[2]
        self._integrated_gravity_z = gz
        for body in list(self.garments.values()) + [self.bottle]:
            if self.grasp == body.name:
                continue
            vx, vy, vz = body.vel
            vz = vz + gz * DT
            x = body.pos[0] + vx * DT
            y = body.pos[1] + vy * DT
            z = body.pos[2] + vz * DT
            x, y, z, vz = self._support(body, x, y, z, vz)
            body.pos = (x, y, z)
            body.vel = (vx, vy, vz)

    def _support(self, body: Body, x: float, y: float, z: float, vz: float):
        r = body.half[2]
        # Basket interior.
        if abs(x - BASKET_CENTER[0]) <= BASKET_HALF[0] and abs(y - BASKET_CENTER[1]) <= BASKET_HALF[1]:
            floor = BASKET_FLOOR_Z + r
            if z < floor:
                z = floor
                vz = 0.0 if vz < 0.0 else vz
        elif abs(x - COUNTER_CENTER[0]) <= COUNTER_HALF[0] and abs(y - COUNTER_CENTER[1]) <= COUNTER_HALF[1]:
            floor = COUNTER_TOP_Z + r
            if z < floor:
                z = floor
                vz = 0.0 if vz < 0.0 else vz
        elif DRUM_X[0] <= x <= DRUM_X[1]:
            dy = y - DRUM_CENTER[1]
            remain = DRUM_RADIUS * DRUM_RADIUS - dy * dy
            if remain > r * r:
                floor = DRUM_CENTER[2] - math.sqrt(remain) + r
                if z < floor:
                    z = floor
                    vz = 0.0 if vz < 0.0 else vz
        if z < r:
            z = r
            vz = 0.0 if vz < 0.0 else vz
        return x, y, z, vz

    def snapshot(self) -> Snapshot:
        return Snapshot(
            n_in_drum=sum(1 for body in self.garments.values() if _in(body.pos)),
            main_ml=self.doses["main"],
            spill_ml=self.spill_ml,
            overfill=self.overfill,
            running=self.running,
            wash_complete=self.wash_complete and holds_success(self),
            rejected_locked=self.rejected_locked,
        )

    def diagnostics(self) -> dict:
        return {
            "backend": "numpy_rigid",
            "gpu_physics_validation": False,
            "isaac_validation": False,
            "newton_validation": False,
            "kinematic": self.kinematic,
            "gravity": list(self.gravity),
            "gravity_used_by_integrator": self._integrated_gravity_z if self.step_count else self.gravity[2],
            "step": self.step_count,
            "sim_time_s": self.step_count * DT,
            "dt": DT,
            "joints": {
                name: {"q": self.q[name], "qd": self.qd[name], "target": self.target[name]}
                for name in JOINT_LIMITS
            },
            "running": self.running,
            "wash_turns": self.wash_angle / (2.0 * math.pi),
            "liquid_ml": self.liquid_ml,
            "doses_ml": dict(self.doses),
            "spill_ml": self.spill_ml,
            "overfill": self.overfill,
            "last_pour": self.last_pour,
            "last_reject": self.last_reject,
            "note": (
                "CPU rigid integrator and task gates. Not a MuJoCo, Newton, or GPU result. "
                "Gravity is the value added to vertical velocity each step."
            ),
        }


def _in(pos) -> bool:
    from .geom import in_drum

    return in_drum(pos)
