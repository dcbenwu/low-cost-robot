"""Optional Isaac Lab entry for the same one-env laundry scene.

Do not use this as the default on uDoge while ComfyUI is resident. It starts
Isaac Lab's app. The supported small path is ``python -m laundry_washer.newton_run``.

This file does not modify the Isaac Lab checkout. It only imports the
public classes at develop 93d7bef:

  NewtonCfg + MJWarpSolverCfg (njmax 80, nconmax 48, iterations 20, no CUDA graph)
  NewtonGLVisualizerCfg(headless=True, 640x360, streaming_view=False, no sky, no shadows)
  gravity (0, 0, -9.81) set on SimulationCfg — the lift ADR zero-gravity
  schedule is not loaded and the Franka checkpoint is not loaded.

  cd /home/hock/IsaacLab
  # do not git pull, do not uv sync, do not edit this tree
  source env_isaaclab/bin/activate
  export PYGLET_HEADLESS=1
  export PYGLET_HEADLESS_DEVICE=0
  python /path/to/laundry_washer_experiment/laundry_washer/isaac_run.py --headless --num-envs 1 --steps 120 --out /tmp/laundry_isaac

Exit 2 if Isaac Lab is not importable. This cloud sandbox does not run it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> int:
    from .mjcf import build_mjcf
    from .runtime_config import GRAVITY, ITERATIONS, LS_ITERATIONS, NCONMAX, NJMAX, SIM_DT, newton_gl_cfg_kwargs

    parser = argparse.ArgumentParser(description="Optional one-env Isaac/Newton laundry launch. Not a trained policy.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--out", default="/tmp/laundry_isaac")
    parser.add_argument("--device", default="cuda:0")
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:  # noqa: BLE001
        print(f"Isaac Lab is not importable: {exc}", file=sys.stderr)
        print("Do not install or upgrade Isaac Lab from this script.", file=sys.stderr)
        return 2
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs != 1:
        print("refusing: this baseline is one environment so it fits beside ComfyUI", file=sys.stderr)
        return 2
    args.headless = True
    os.environ.setdefault("PYGLET_HEADLESS", "1")
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    try:
        from isaaclab.sim import SimulationCfg, SimulationContext
    except ImportError:
        from isaaclab.sim.simulation_cfg import SimulationCfg
        from isaaclab.sim.simulation_context import SimulationContext
    try:
        from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    except ImportError:
        from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
        from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    os.makedirs(args.out, exist_ok=True)
    xml_path = os.path.join(args.out, "laundry_washer.xml")
    with open(xml_path, "w", encoding="utf-8") as handle:
        handle.write(build_mjcf())

    visualizer = NewtonGLVisualizerCfg(**newton_gl_cfg_kwargs())
    sim_cfg = SimulationCfg(
        dt=SIM_DT,
        gravity=GRAVITY,
        device=args.device,
        render_interval=4,
        use_newton_actuators=True,
        physics=NewtonCfg(
            num_substeps=1,
            use_cuda_graph=False,
            solver_cfg=MJWarpSolverCfg(
                njmax=NJMAX,
                nconmax=NCONMAX,
                iterations=ITERATIONS,
                ls_iterations=LS_ITERATIONS,
                solver="newton",
                integrator="implicitfast",
                use_mujoco_cpu=False,
                use_mujoco_contacts=True,
            ),
        ),
        visualizer_cfgs=[visualizer],
    )
    sim = SimulationContext(sim_cfg)
    spawn_error = None
    try:
        asset = sim_utils.MjcfFileCfg(asset_path=xml_path, fix_base=True)
        asset.func("/World/Laundry", asset)
    except Exception as exc:  # noqa: BLE001
        spawn_error = repr(exc)

    gravity_read = None
    for source in (getattr(sim, "cfg", None), getattr(sim, "physics_manager", None)):
        if source is not None and hasattr(source, "gravity"):
            raw = source.gravity
            try:
                gravity_read = [float(raw[0]), float(raw[1]), float(raw[2])]
            except Exception:
                gravity_read = None
    if gravity_read is None:
        gravity_read = [float(v) for v in sim_cfg.gravity]

    sim.reset()
    for _ in range(args.steps):
        sim.step()

    summary = {
        "learned_policy": False,
        "num_envs": 1,
        "device": args.device,
        "gravity_cfg": [float(v) for v in sim_cfg.gravity],
        "gravity_read": gravity_read,
        "steps": args.steps,
        "use_cuda_graph": False,
        "njmax": NJMAX,
        "nconmax": NCONMAX,
        "visualizer": "NewtonGLVisualizerCfg",
        "visualizer_type": getattr(visualizer, "visualizer_type", "newton_gl"),
        "video_source_alias": "visualizer:newton",
        "window": [visualizer.window_width, visualizer.window_height],
        "eye": list(visualizer.eye),
        "lookat": list(visualizer.lookat),
        "streaming_view": visualizer.streaming_view,
        "spawn_error": spawn_error,
        "franka_checkpoint_loaded": False,
        "adr_loaded": False,
        "note": "One env. No lift-task ADR. No HDR sky. Kit RTX viewport is not requested.",
    }
    path = os.path.join(args.out, "isaac_diagnostics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({"summary": path, "gravity_read": gravity_read, "spawn_error": spawn_error}, indent=2))
    simulation_app.close()
    if spawn_error is not None:
        return 1
    if abs(gravity_read[2] + 9.81) > 1e-3:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
