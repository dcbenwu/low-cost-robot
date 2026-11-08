"""Map the task scene onto the MJCF generalized coordinates.

Newton and MuJoCo both store this MJCF as one qpos vector: hinge and slide
joints take one value, each free joint takes position plus a wxyz quaternion.
Bottle tilt is not its own MJCF joint. It is the bottle free-joint rotation
about Y, using the same convention as ``geom.mouth_offset``.

The CPU tests pack and unpack this vector. They do not start Newton.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .spec import GARMENTS


@dataclass(frozen=True)
class Dof:
    name: str
    kind: str  # "hinge", "slide", or "free"
    adr: int
    nq: int
    default: tuple[float, ...]


FREE_BODIES = ("shirt", "pants", "towel", "sock", "bottle")
SCALAR_FROM_SCENE = (
    "door_hinge",
    "drawer_slide",
    "drum_spin",
    "program_knob",
    "start_button",
    "lock_pin",
    "cap_hinge",
    "arm_yaw",
    "arm_shoulder",
    "arm_elbow",
    "arm_wrist",
)


def quat_from_tilt(tilt: float) -> tuple[float, float, float, float]:
    """wxyz quaternion for the bottle tilt used by the pour check."""
    half = -0.5 * float(tilt)
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def tilt_from_quat(quat) -> float:
    qw = float(quat[0])
    qy = float(quat[2])
    return -2.0 * math.atan2(qy, qw)


def build_index(xml_text: str) -> list[Dof]:
    """Depth-first joint order, which is MuJoCo's qpos order for this file."""
    root = ET.fromstring(xml_text)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("MJCF has no worldbody")
    dofs: list[Dof] = []
    adr = 0

    def add(body, element) -> None:
        nonlocal adr
        if element.tag == "freejoint":
            name = element.get("name") or f"{body.get('name')}_free"
            pos = _vec(body.get("pos"), 3)
            default = (pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0)
            dofs.append(Dof(name, "free", adr, 7, default))
            adr += 7
            return
        kind = element.get("type") or "hinge"
        if kind not in ("hinge", "slide"):
            raise ValueError(f"unsupported joint {element.get('name')} type {kind}")
        name = element.get("name")
        if not name:
            raise ValueError("scalar joint is missing a name")
        dofs.append(Dof(name, kind, adr, 1, (float(element.get("ref") or 0.0),)))
        adr += 1

    def walk(body) -> None:
        for child in list(body):
            if child.tag in ("joint", "freejoint"):
                add(body, child)
            elif child.tag == "body":
                walk(child)

    for child in list(world):
        if child.tag == "body":
            walk(child)
        elif child.tag in ("joint", "freejoint"):
            add(world, child)
    return dofs


def nq(index: list[Dof]) -> int:
    if not index:
        return 0
    last = index[-1]
    return last.adr + last.nq


def pack_scene(scene, index: list[Dof], *, free_bodies: bool = True) -> list[float]:
    """Write task joints and, unless disabled, free-body poses into qpos."""
    vector = [0.0] * nq(index)
    for dof in index:
        vector[dof.adr : dof.adr + dof.nq] = dof.default
    for dof in index:
        if dof.kind == "free":
            if not free_bodies:
                continue
            body_name = dof.name[: -len("_free")] if dof.name.endswith("_free") else dof.name
            if body_name == "bottle":
                pos = scene.bottle.pos
                quat = quat_from_tilt(scene.q["bottle_tilt"])
            elif body_name in scene.garments:
                pos = scene.garments[body_name].pos
                quat = (1.0, 0.0, 0.0, 0.0)
            else:
                continue
            vector[dof.adr : dof.adr + 7] = (
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                quat[0],
                quat[1],
                quat[2],
                quat[3],
            )
        elif dof.name in scene.q:
            vector[dof.adr] = float(scene.q[dof.name])
    return vector


def ledger_from_scene(scene) -> dict:
    return {
        "liquid_ml": float(scene.liquid_ml),
        "doses": {key: float(value) for key, value in scene.doses.items()},
        "spill_ml": float(scene.spill_ml),
        "overfill": bool(scene.overfill),
        "running": bool(scene.running),
        "wash_angle": float(scene.wash_angle),
        "wash_complete": bool(scene.wash_complete),
        "failed_after_start": bool(scene.failed_after_start),
    }


def apply_ledger(scene, ledger: dict) -> None:
    scene.liquid_ml = float(ledger["liquid_ml"])
    scene.doses = {key: float(value) for key, value in ledger["doses"].items()}
    scene.spill_ml = float(ledger["spill_ml"])
    scene.overfill = bool(ledger["overfill"])
    scene.running = bool(ledger["running"])
    scene.wash_angle = float(ledger["wash_angle"])
    scene.wash_complete = bool(ledger["wash_complete"])
    scene.failed_after_start = bool(ledger["failed_after_start"])


def unpack_scene(scene, vector, index: list[Dof], ledger: dict | None = None) -> None:
    """Load qpos back into the task scene. Dose state comes from the ledger."""
    for dof in index:
        if dof.kind == "free":
            pos = (
                float(vector[dof.adr]),
                float(vector[dof.adr + 1]),
                float(vector[dof.adr + 2]),
            )
            quat = vector[dof.adr + 3 : dof.adr + 7]
            body_name = dof.name[: -len("_free")] if dof.name.endswith("_free") else dof.name
            if body_name == "bottle":
                scene.bottle.pos = pos
                scene.q["bottle_tilt"] = tilt_from_quat(quat)
            elif body_name in scene.garments:
                scene.garments[body_name].pos = pos
        elif dof.name in scene.q:
            scene.q[dof.name] = float(vector[dof.adr])
    if ledger is not None:
        apply_ledger(scene, ledger)
    scene.fk = scene.gripper_position()


def _vec(text: str | None, count: int) -> tuple[float, ...]:
    if not text:
        return tuple(0.0 for _ in range(count))
    parts = [float(item) for item in text.split()]
    if len(parts) != count:
        raise ValueError(f"expected {count} values, got {text!r}")
    return tuple(parts)


def garment_names() -> tuple[str, ...]:
    return tuple(item["name"] for item in GARMENTS)
