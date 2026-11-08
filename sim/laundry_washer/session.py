"""One workflow over a physics backend.

The command sequence is the laundry task. After every command the backend
stores the named MJCF pose. Gates are evaluated on the pose read back from
that backend, together with the dose ledger. A memory backend is the CPU
stand-in for the same qpos contract Newton uses. It is not a Newton run.
"""

from __future__ import annotations

from .env import LaundryEnv, full_wash_commands
from .geom import in_drum, selected_program
from .mjcf import build_mjcf
from .qpos_map import build_index, ledger_from_scene, nq, pack_scene, unpack_scene
from .runtime_config import LABEL_PHYSICAL, LABEL_SCRIPTED
from .scene import LaundryScene
from .spec import GARMENTS, LIQUID_ML
from .task import holds_success, stage_name

KEY_TAGS = ("loaded", "pour", "start", "washing")


class MemoryBackend:
    """Holds one qpos vector. Scripted mode writes free bodies; physical mode does not."""

    def __init__(self):
        self.index = build_index(build_mjcf())
        self.qpos = [0.0] * nq(self.index)
        self.ledger = ledger_from_scene(LaundryScene(kinematic=True))
        self.captures: list[str] = []
        self.steps = 0
        self.writes = 0
        self.free_body_writes = 0
        self.newton_executed = False

    def write_scene(self, scene, *, free_bodies: bool) -> None:
        self.qpos = pack_scene(scene, self.index, free_bodies=free_bodies)
        self.ledger = ledger_from_scene(scene)
        self.writes += 1
        if free_bodies:
            self.free_body_writes += 1

    def settle(self, count: int = 1) -> None:
        # Kinematic hold: the stored qpos is the state the next read returns.
        self.steps += int(count)

    def read_scene(self) -> LaundryScene:
        scene = LaundryScene(kinematic=True)
        unpack_scene(scene, self.qpos, self.index, self.ledger)
        return scene

    def capture(self, tag: str) -> None:
        self.captures.append(tag)

    def close(self) -> None:
        return None


def run_session(
    backend=None,
    commands: list[str] | None = None,
    action_mode: str = "scripted_kinematic",
    hold_steps: int = 4,
    on_capture=None,
) -> dict:
    if action_mode not in ("scripted_kinematic", "physical"):
        raise ValueError("action_mode must be scripted_kinematic or physical")
    if backend is None:
        backend = MemoryBackend()
    commands = list(commands if commands is not None else full_wash_commands())
    free_bodies = action_mode == "scripted_kinematic"
    env = LaundryEnv(action_mode=action_mode)
    records = []
    backend.write_scene(env.scene, free_bodies=free_bodies)
    backend.settle(hold_steps)
    backend.capture("reset")
    if on_capture is not None:
        on_capture("reset", backend.read_scene())

    for command in commands:
        info = env.run(command)
        backend.write_scene(env.scene, free_bodies=free_bodies)
        backend.settle(hold_steps)
        read = backend.read_scene()
        tag = _tag(command, read)
        if tag is not None:
            backend.capture(tag)
            if on_capture is not None:
                on_capture(tag, read)
        records.append(_record(command, info, read))

    final = backend.read_scene()
    success = holds_success(final)
    env.reset()
    backend.write_scene(env.scene, free_bodies=True)
    backend.settle(1)
    restored = backend.read_scene()
    summary = {
        "learned_policy": False,
        "action_mode": action_mode,
        "validation_label": LABEL_SCRIPTED if action_mode == "scripted_kinematic" else LABEL_PHYSICAL,
        "pose_source": (
            "scripted_qpos_written_into_model"
            if free_bodies
            else "actuator_targets_only_free_bodies_stay_dynamic"
        ),
        "physics_backend": "memory_qpos" if isinstance(backend, MemoryBackend) else "newton",
        "newton_executed": bool(getattr(backend, "newton_executed", False)),
        "gpu_execution_claimed": False,
        "workflow_commands": commands,
        "command_records": records,
        "success_from_readback": success,
        "stage": stage_name(final),
        "keyframes": list(getattr(backend, "captures", [])),
        "hold_steps": hold_steps,
        "reset_after": _reset_report(restored),
        "free_body_writes": int(getattr(backend, "free_body_writes", 0)),
        "cloth": "rigid garment proxies — not deformable cloth",
        "detergent": "detergent pose+dose abstraction — not fluid",
    }
    return summary


def _tag(command: str, scene) -> str | None:
    if command == "pour":
        return "pour"
    if command == "press_start":
        return "start"
    if command == "wash":
        return "washing"
    if command == "release" and all(in_drum(body.pos) for body in scene.garments.values()):
        return "loaded"
    return None


def _record(command: str, info: dict, scene) -> dict:
    program, valid = selected_program(scene.q["program_knob"])
    return {
        "command": command,
        "command_done": bool(info.get("command_done")),
        "rejected": bool(info.get("rejected")),
        "stage": stage_name(scene),
        "door_q": scene.q["door_hinge"],
        "drawer_q": scene.q["drawer_slide"],
        "cap_q": scene.q["cap_hinge"],
        "knob_q": scene.q["program_knob"],
        "program": program,
        "program_valid": valid,
        "lock_q": scene.q["lock_pin"],
        "bottle_tilt": scene.q["bottle_tilt"],
        "n_in_drum": sum(1 for body in scene.garments.values() if in_drum(body.pos)),
        "main_ml": scene.doses["main"],
        "liquid_ml": scene.liquid_ml,
        "running": scene.running,
        "success": holds_success(scene),
    }


def _reset_report(scene) -> dict:
    home = {item["name"]: item["pos"] for item in GARMENTS}
    garment_error = 0.0
    for name, pos in home.items():
        got = scene.garments[name].pos
        garment_error = max(garment_error, max(abs(got[i] - pos[i]) for i in range(3)))
    program, valid = selected_program(scene.q["program_knob"])
    ok = (
        abs(scene.q["door_hinge"]) < 1e-6
        and abs(scene.q["cap_hinge"]) < 1e-6
        and abs(scene.q["drawer_slide"]) < 1e-6
        and abs(scene.liquid_ml - LIQUID_ML) < 1e-6
        and scene.doses["main"] == 0.0
        and not scene.running
        and not scene.wash_complete
        and program == "off"
        and not valid
        and garment_error < 1e-6
    )
    return {
        "ok": ok,
        "door_q": scene.q["door_hinge"],
        "cap_q": scene.q["cap_hinge"],
        "liquid_ml": scene.liquid_ml,
        "main_ml": scene.doses["main"],
        "running": scene.running,
        "program": program,
        "garment_error_m": garment_error,
    }


# Re-export so callers can build a view without importing qpos internals.
__all__ = ["MemoryBackend", "run_session"]
