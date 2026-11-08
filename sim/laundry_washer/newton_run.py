"""Direct Newton entry for the laundry workflow.

This is the command to run on uDoge. It loads the same MJCF the task names,
writes that task's joint and body coordinates into the Newton state, steps
the one-world solver, and scores success from the coordinates read back.

  export PYGLET_HEADLESS=1
  export PYGLET_HEADLESS_DEVICE=0
  python -m laundry_washer.newton_run --mode scripted_kinematic --record --out /tmp/laundry_newton
  python -m laundry_washer.newton_run --mode manual --commands open_door,grasp:shirt,carry:drum,release --record --out /tmp/laundry_manual
  python -m laundry_washer.newton_run --mode physical --commands open_door --out /tmp/laundry_phys

``scripted_kinematic`` writes free-body poses as well as hinges. That is
scripted scene validation, not learned manipulation and not cloth or fluid.
``physical`` writes actuator targets only; free bodies stay dynamic, so a
finished script is not reported as success unless the read-back actually
puts the clothes in the drum.

One world, 640x360, no CUDA graph, no HDR, no Kit viewport. This process
exits 2 when Newton is not installed. A missing frame is a GL error, not a
drawn substitute. The summary file is written before the viewer is closed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .qpos_map import pack_scene
from .runtime_config import (
    EYE,
    GRAVITY,
    ITERATIONS,
    LOOKAT,
    LS_ITERATIONS,
    NCONMAX,
    NJMAX,
    NUM_WORLDS,
    SIM_DT,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    eye_lookat_to_pitch_yaw,
)
from .session import KEY_TAGS, run_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Newton laundry workflow. Not a trained policy.")
    parser.add_argument("--mode", default="scripted_kinematic", choices=("scripted_kinematic", "physical", "manual"))
    parser.add_argument("--commands", default="", help="Comma-separated task commands. Empty runs the full wash in scripted/physical mode.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--hold-steps", type=int, default=4, dest="hold_steps")
    parser.add_argument("--steps", type=int, default=0, help="If set, overrides --hold-steps.")
    parser.add_argument("--width", type=int, default=WINDOW_WIDTH)
    parser.add_argument("--height", type=int, default=WINDOW_HEIGHT)
    args = parser.parse_args(argv)
    if NUM_WORLDS != 1:
        print("refusing to run: NUM_WORLDS must stay 1", file=sys.stderr)
        return 2
    mode = "scripted_kinematic" if args.mode == "manual" else args.mode
    commands = [item.strip() for item in args.commands.split(",") if item.strip()] or None
    if args.mode == "manual" and not commands:
        print("manual mode needs --commands, for example open_door,grasp:shirt,carry:drum,release", file=sys.stderr)
        return 2
    hold = args.steps if args.steps else args.hold_steps

    os.environ.setdefault("PYGLET_HEADLESS", "1")
    os.environ.setdefault("PYGLET_HEADLESS_DEVICE", "0")
    try:
        import pyglet
        pyglet.options["headless"] = True
        import newton
        import warp as wp
    except Exception as exc:  # noqa: BLE001
        print(f"Newton/pyglet/warp is not importable here: {exc}", file=sys.stderr)
        print("No Newton execution is claimed. CPU mapping tests do not use this process.", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    backend = NewtonBackend(newton, wp, record=args.record, width=args.width, height=args.height, out_dir=args.out)
    try:
        summary = run_session(backend, commands=commands, action_mode=mode, hold_steps=hold)
    except Exception as exc:  # noqa: BLE001
        summary = {
            "learned_policy": False,
            "newton_executed": False,
            "gpu_execution_claimed": False,
            "error": repr(exc),
        }
        _write_summary(args.out, summary)
        backend.close()
        print(repr(exc), file=sys.stderr)
        return 1
    summary["frames"] = list(backend.frames)
    summary["render_error"] = backend.render_error
    summary["gravity"] = backend.gravity
    summary["gravity_source"] = backend.gravity_source
    summary["joint_layout"] = backend.layout_note
    summary["qpos_length"] = getattr(backend, "qpos_length", None)
    summary["expected_nq"] = getattr(backend, "expected_nq", None)
    summary["camera"] = {"eye": list(EYE), "lookat": list(LOOKAT), "width": args.width, "height": args.height}
    summary["summary_written_before_viewer_close"] = True
    summary["gpu_execution_claimed"] = False
    path = _write_summary(args.out, summary)
    backend.close()
    print(json.dumps({
        "summary": path,
        "mode": summary.get("action_mode"),
        "validation_label": summary.get("validation_label"),
        "success_from_readback": summary.get("success_from_readback"),
        "reset_ok": (summary.get("reset_after") or {}).get("ok"),
        "frames": summary.get("frames"),
        "render_error": backend.render_error,
        "newton_executed": summary.get("newton_executed"),
    }, indent=2))
    if backend.gravity is None or abs(backend.gravity[2] + 9.81) > 1e-2:
        print("gravity readback is not (0, 0, -9.81)", file=sys.stderr)
        return 1
    if not summary.get("newton_executed"):
        return 1
    if args.record and backend.render_error is not None:
        return 3
    if mode == "scripted_kinematic" and commands is None and not summary.get("success_from_readback"):
        return 1
    return 0


class NewtonBackend:
    def __init__(self, newton, wp, *, record: bool, width: int, height: int, out_dir: str):
        from .mjcf import build_mjcf

        self.newton = newton
        self.wp = wp
        self.out_dir = out_dir
        self.record = record
        self.frames: list[dict] = []
        self.render_error = None
        self.viewer = None
        self.captures: list[str] = []
        self.writes = 0
        self.free_body_writes = 0
        self.steps = 0
        self.newton_executed = False
        self._held = None
        self.layout_note = ""
        xml_path = os.path.join(out_dir, "laundry_washer.xml")
        with open(xml_path, "w", encoding="utf-8") as handle:
            handle.write(build_mjcf())
        builder = newton.ModelBuilder()
        if hasattr(builder, "gravity"):
            try:
                builder.gravity = wp.vec3(*GRAVITY)
            except Exception:
                builder.gravity = GRAVITY
        builder.add_mjcf(xml_path, parse_mujoco_options=True, parse_meshes=False)
        self.model = builder.finalize()
        self.gravity, self.gravity_source = _read_gravity(self.model, builder)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            njmax=NJMAX,
            nconmax=NCONMAX,
            iterations=ITERATIONS,
            ls_iterations=LS_ITERATIONS,
            use_mujoco_cpu=False,
            use_mujoco_contacts=True,
        )
        self.state = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        self.index = _newton_index(self.model)
        self.layout_note = self.index.get("note", "")
        self._time = 0.0
        if record:
            self._open_viewer(width, height)

    def write_scene(self, scene, *, free_bodies: bool) -> None:
        vector = pack_scene(scene, self.index["dofs"], free_bodies=free_bodies)
        self._assign_qpos(vector)
        self._assign_targets(scene, free_bodies=free_bodies)
        self._held = list(vector) if free_bodies else None
        self.ledger = {
            "liquid_ml": float(scene.liquid_ml),
            "doses": {key: float(value) for key, value in scene.doses.items()},
            "spill_ml": float(scene.spill_ml),
            "overfill": bool(scene.overfill),
            "running": bool(scene.running),
            "wash_angle": float(scene.wash_angle),
            "wash_complete": bool(scene.wash_complete),
            "failed_after_start": bool(scene.failed_after_start),
        }
        self.writes += 1
        if free_bodies:
            self.free_body_writes += 1

    def settle(self, count: int = 1) -> None:
        for _ in range(max(int(count), 1)):
            contacts = self.model.collide(self.state) if hasattr(self.model, "collide") else None
            self.solver.step(self.state, self.state_out, self.control, contacts, SIM_DT)
            self.state, self.state_out = self.state_out, self.state
            self.steps += 1
            self._time += SIM_DT
            if self._held is not None:
                # Scripted validation keeps the commanded coordinates after the
                # solver step. Free bodies are not claimed to be contact-stable.
                self._assign_qpos(self._held)
        self.newton_executed = self.steps > 0
        flat = self.state.joint_q.numpy().reshape(-1)
        self.qpos_length = int(flat.shape[0])
        self.expected_nq = len(self._held) if self._held is not None else int(flat.shape[0])

    def read_scene(self):
        from .qpos_map import unpack_scene
        from .scene import LaundryScene

        vector = self._read_qpos()
        scene = LaundryScene(kinematic=True)
        unpack_scene(scene, vector, self.index["dofs"], getattr(self, "ledger", None))
        return scene

    def capture(self, tag: str) -> None:
        self.captures.append(tag)
        if not self.record or tag not in ("reset",) + KEY_TAGS:
            return
        if self.viewer is None:
            return
        try:
            self.viewer.begin_frame(self._time)
            self.viewer.log_state(self.state)
            self.viewer.end_frame()
            frame = self.viewer.get_frame().numpy()
            path = os.path.join(self.out_dir, f"newton_{tag}.png")
            _write_png(path, frame)
            self.frames.append({"tag": tag, "path": path})
        except Exception as exc:  # noqa: BLE001
            self.render_error = repr(exc)

    def close(self) -> None:
        viewer = self.viewer
        self.viewer = None
        if viewer is not None:
            viewer.close()

    def _open_viewer(self, width: int, height: int) -> None:
        try:
            pitch, yaw = eye_lookat_to_pitch_yaw(EYE, LOOKAT)
            self.viewer = self.newton.viewer.ViewerGL(width=width, height=height, headless=True, vsync=False)
            self.viewer.set_model(self.model)
            self.viewer.set_camera(pos=self.wp.vec3(*EYE), pitch=pitch, yaw=yaw)
        except Exception as exc:  # noqa: BLE001
            self.render_error = repr(exc)
            self.viewer = None

    def _assign_qpos(self, vector) -> None:
        array = self.state.joint_q
        current = array.numpy().copy()
        flat = current.reshape(-1)
        count = min(len(flat), len(vector))
        flat[:count] = vector[:count]
        array.assign(current)

    def _read_qpos(self):
        return [float(value) for value in self.state.joint_q.numpy().reshape(-1)]

    def _assign_targets(self, scene, *, free_bodies: bool) -> None:
        # Position targets keep hinges near the written coordinates when the
        # solver steps. Free bodies are not actuator-driven.
        for name in ("joint_target_pos", "joint_target", "joint_target_q"):
            if not hasattr(self.control, name):
                continue
            try:
                values = getattr(self.control, name).numpy().copy()
                flat = values.reshape(-1)
                for dof in self.index["dofs"]:
                    if dof.kind == "free" or dof.name not in scene.q:
                        continue
                    if dof.adr < len(flat):
                        flat[dof.adr] = float(scene.q[dof.name])
                getattr(self.control, name).assign(values)
                return
            except Exception:
                continue
        _ = free_bodies


def _newton_index(model) -> dict:
    """Prefer Newton's own joint names and qpos addresses when it publishes them."""
    from .mjcf import build_mjcf
    from .qpos_map import build_index

    xml_index = build_index(build_mjcf())
    names = []
    for attr in ("joint_label", "joint_key", "joint_names"):
        if hasattr(model, attr):
            try:
                names = [str(item) for item in list(getattr(model, attr))]
            except Exception:
                names = []
            if names:
                break
    starts = None
    for attr in ("joint_q_start", "joint_qposadr", "jnt_qposadr"):
        if hasattr(model, attr):
            try:
                starts = [int(value) for value in list(getattr(model, attr).numpy().reshape(-1))]
            except Exception:
                try:
                    starts = [int(value) for value in list(getattr(model, attr))]
                except Exception:
                    starts = None
            if starts:
                break
    note = "mjcf_depth_first_qpos"
    dofs = xml_index
    if names and starts and len(names) == len(starts) and len(names) == len(xml_index):
        by_xml = {dof.name: dof for dof in xml_index}
        if all(name in by_xml for name in names):
            rebuilt = []
            for name, adr in zip(names, starts):
                src = by_xml[name]
                rebuilt.append(type(src)(src.name, src.kind, adr, src.nq, src.default))
            dofs = rebuilt
            note = "newton_joint_q_start"
    return {"dofs": dofs, "note": note, "newton_names": names[:32]}


def _read_gravity(model, builder):
    for obj, label in ((model, "model"), (builder, "builder")):
        for name in ("gravity", "gravity_vector"):
            if not hasattr(obj, name):
                continue
            raw = getattr(obj, name)
            try:
                return [float(raw[0]), float(raw[1]), float(raw[2])], f"{label}.{name}"
            except Exception:
                continue
    return None, None


def _write_summary(out_dir: str, summary: dict) -> str:
    path = os.path.join(out_dir, "newton_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return path


def _write_png(path: str, frame) -> None:
    import numpy as np
    from .render_png import write_png

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    write_png(path, array)


if __name__ == "__main__":
    raise SystemExit(main())
