"""One-env Newton 1.6 smoke for the laundry MJCF.

Supported on uDoge, not in this cloud sandbox:

  export PYGLET_HEADLESS=1
  export PYGLET_HEADLESS_DEVICE=0
  python -m laundry_washer.newton_run --steps 180 --out /tmp/laundry_newton
  python -m laundry_washer.newton_run --steps 180 --record --out /tmp/laundry_newton

Backend: newton.solvers.SolverMuJoCo, one world, mujoco_warp, no CUDA graph.
Camera: newton.viewer.ViewerGL headless 640x360, eye/lookat from runtime_config.
No Kit RTX viewport, no HDR, no Franka checkpoint.

This module does not import newton until main() runs. Exit code 2 means
Newton is not installed. A viewer failure is reported in diagnostics and
does not invent frames.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _as_floats(value):
    if value is None:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return None


def _read_gravity(model, builder):
    for obj in (model, builder):
        for name in ("gravity", "gravity_vector"):
            if hasattr(obj, name):
                got = _as_floats(getattr(obj, name))
                if got is not None:
                    return got, f"{type(obj).__name__}.{name}"
    return None, None


def _names(model):
    for attr in ("joint_label", "joint_key", "joint_names"):
        if hasattr(model, attr):
            raw = getattr(model, attr)
            try:
                return [str(item) for item in list(raw)], attr
            except Exception:
                continue
    return [], None


def main(argv: list[str] | None = None) -> int:
    from .mjcf import build_mjcf
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
        supported_summary,
    )

    parser = argparse.ArgumentParser(description="One-env Newton laundry smoke. Not a trained policy.")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--out", required=True)
    parser.add_argument("--record", action="store_true", help="NewtonGL headless frames. Requires a GL context.")
    parser.add_argument("--width", type=int, default=WINDOW_WIDTH)
    parser.add_argument("--height", type=int, default=WINDOW_HEIGHT)
    args = parser.parse_args(argv)
    if NUM_WORLDS != 1:
        print("refusing to run: NUM_WORLDS must stay 1", file=sys.stderr)
        return 2

    os.environ.setdefault("PYGLET_HEADLESS", "1")
    os.environ.setdefault("PYGLET_HEADLESS_DEVICE", "0")
    try:
        import pyglet
        pyglet.options["headless"] = True
        import newton
        import warp as wp
    except Exception as exc:  # noqa: BLE001 — report the missing stack, do not fake a run
        print(f"Newton/pyglet/warp is not importable here: {exc}", file=sys.stderr)
        print("This cloud sandbox does not run the GPU path. Use the CPU tests instead.", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    xml_path = os.path.join(args.out, "laundry_washer.xml")
    with open(xml_path, "w", encoding="utf-8") as handle:
        handle.write(build_mjcf())

    builder = newton.ModelBuilder()
    if hasattr(builder, "gravity"):
        try:
            builder.gravity = wp.vec3(*GRAVITY)
        except Exception:
            builder.gravity = GRAVITY
    builder.add_mjcf(xml_path, parse_mujoco_options=True, parse_meshes=False)
    model = builder.finalize()
    gravity, gravity_source = _read_gravity(model, builder)
    solver = newton.solvers.SolverMuJoCo(
        model,
        njmax=NJMAX,
        nconmax=NCONMAX,
        iterations=ITERATIONS,
        ls_iterations=LS_ITERATIONS,
        use_mujoco_cpu=False,
        use_mujoco_contacts=True,
    )
    state = model.state()
    state_out = model.state()
    control = model.control()
    joint_names, joint_attr = _names(model)

    def collide(current):
        if hasattr(model, "collide"):
            return model.collide(current)
        return None

    def joint_q(current):
        array = getattr(current, "joint_q", None)
        if array is None:
            return None
        try:
            return [float(v) for v in array.numpy().reshape(-1)[:32]]
        except Exception:
            return None

    q_before = joint_q(state)
    contacts = collide(state)
    solver.step(state, state_out, control, contacts, SIM_DT)
    state, state_out = state_out, state
    q_after_one = joint_q(state)

    # Drive the door joint toward open if Newton exposes a position target.
    door_index = joint_names.index("door_hinge") if "door_hinge" in joint_names else None
    target_attr = None
    for candidate in ("joint_target_pos", "joint_target", "joint_target_q"):
        if hasattr(control, candidate):
            target_attr = candidate
            break
    target_applied = False
    if door_index is not None and target_attr is not None:
        try:
            values = getattr(control, target_attr).numpy()
            values = values.copy()
            flat = values.reshape(-1)
            flat[door_index] = 1.2
            getattr(control, target_attr).assign(values)
            target_applied = True
        except Exception as exc:  # noqa: BLE001
            target_applied = False
            target_attr = f"{target_attr} failed: {exc}"

    for _ in range(max(args.steps - 1, 0)):
        contacts = collide(state)
        solver.step(state, state_out, control, contacts, SIM_DT)
        state, state_out = state_out, state
    q_final = joint_q(state)

    frames = []
    render_error = None
    if args.record:
        try:
            pitch, yaw = eye_lookat_to_pitch_yaw(EYE, LOOKAT)
            viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True, vsync=False)
            viewer.set_model(model)
            viewer.set_camera(pos=wp.vec3(*EYE), pitch=pitch, yaw=yaw)
            viewer.begin_frame(args.steps * SIM_DT)
            viewer.log_state(state)
            viewer.end_frame()
            frame = viewer.get_frame().numpy()
            frame_path = os.path.join(args.out, "newton_frame.png")
            _write_frame(frame_path, frame)
            frames.append(frame_path)
            viewer.close()
        except Exception as exc:  # noqa: BLE001
            render_error = repr(exc)

    door_delta = None
    if q_before is not None and q_final is not None and door_index is not None and door_index < len(q_final):
        door_delta = q_final[door_index] - q_before[door_index]
    motion = "motion_observed" if door_delta is not None and abs(door_delta) > 1e-3 else "no_joint_delta_recorded"
    summary = {
        "learned_policy": False,
        "validation_label": "physical actuator mode / NOT a learned policy",
        "num_worlds": 1,
        "solver": "newton.solvers.SolverMuJoCo",
        "use_cuda_graph": False,
        "njmax": NJMAX,
        "nconmax": NCONMAX,
        "iterations": ITERATIONS,
        "ls_iterations": LS_ITERATIONS,
        "sim_dt": SIM_DT,
        "steps": args.steps,
        "gravity": gravity,
        "gravity_source": gravity_source,
        "gravity_expected": list(GRAVITY),
        "joint_name_attr": joint_attr,
        "joint_names_head": joint_names[:24],
        "door_index": door_index,
        "target_attr": target_attr,
        "target_applied": target_applied,
        "joint_q_before_head": None if q_before is None else q_before[:8],
        "joint_q_after_one_head": None if q_after_one is None else q_after_one[:8],
        "joint_q_final_head": None if q_final is None else q_final[:8],
        "door_delta": door_delta,
        "motion_label": motion,
        "frames": frames,
        "render_error": render_error,
        "camera": supported_summary(),
        "record_requested": args.record,
        "note": (
            "One Newton world. No Kit viewport, no HDR, no Franka checkpoint. "
            "A missing frame means the GL capture failed; it is not filled in."
        ),
    }
    summary_path = os.path.join(args.out, "newton_diagnostics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"summary": summary_path, "gravity": gravity, "motion_label": motion, "render_error": render_error}, indent=2))
    if gravity is None or abs(gravity[2] + 9.81) > 1e-3:
        print("gravity readback is not (0, 0, -9.81)", file=sys.stderr)
        return 1
    if args.record and render_error is not None:
        return 3
    return 0


def _write_frame(path: str, frame) -> None:
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
