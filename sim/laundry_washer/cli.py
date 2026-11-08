"""Command line for the laundry environment.

CPU commands (no GPU, no Isaac, no Newton):

  python -m laundry_washer.cli config
  python -m laundry_washer.cli demo --mode scripted_kinematic --out /tmp/laundry_debug_raster
  python -m laundry_washer.cli reset --state /tmp/laundry_state.json
  python -m laundry_washer.cli step --state /tmp/laundry_state.json --action open_door
  python -m laundry_washer.cli diagnostics --state /tmp/laundry_state.json

GPU commands are separate modules and are not invoked by the unit tests:

  python -m laundry_washer.newton_run --help
  python -m laundry_washer.isaac_run --help
"""

from __future__ import annotations

import argparse
import json
import sys

from .demo import run_scripted_validation
from .env import LaundryEnv
from .runtime_config import supported_summary
from .spec import GARMENTS


def _dump_state(env: LaundryEnv, path: str) -> None:
    scene = env.scene
    payload = {
        "action_mode": env.action_mode,
        "q": scene.q,
        "qd": scene.qd,
        "target": scene.target,
        "garments": {name: {"pos": body.pos, "vel": body.vel} for name, body in scene.garments.items()},
        "bottle": {"pos": scene.bottle.pos, "vel": scene.bottle.vel},
        "liquid_ml": scene.liquid_ml,
        "doses": scene.doses,
        "spill_ml": scene.spill_ml,
        "overfill": scene.overfill,
        "running": scene.running,
        "wash_angle": scene.wash_angle,
        "wash_complete": scene.wash_complete,
        "failed_after_start": scene.failed_after_start,
        "step_count": scene.step_count,
        "grasp": scene.grasp,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_state(path: str) -> LaundryEnv:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    env = LaundryEnv(action_mode=payload["action_mode"])
    scene = env.scene
    scene.q.update(payload["q"])
    scene.qd.update(payload["qd"])
    scene.target.update(payload["target"])
    for name, item in payload["garments"].items():
        scene.garments[name].pos = tuple(item["pos"])
        scene.garments[name].vel = tuple(item["vel"])
    scene.bottle.pos = tuple(payload["bottle"]["pos"])
    scene.bottle.vel = tuple(payload["bottle"]["vel"])
    scene.liquid_ml = payload["liquid_ml"]
    scene.doses = payload["doses"]
    scene.spill_ml = payload["spill_ml"]
    scene.overfill = payload["overfill"]
    scene.running = payload["running"]
    scene.wash_angle = payload["wash_angle"]
    scene.wash_complete = payload["wash_complete"]
    scene.failed_after_start = payload["failed_after_start"]
    scene.step_count = payload["step_count"]
    scene.grasp = payload["grasp"]
    scene.fk = scene.gripper_position()
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Laundry washer environment (no trained policy)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("config", help="print the supported one-env Newton/camera contract")

    demo = sub.add_parser("demo", help="scripted full-process validation and CPU frames")
    demo.add_argument("--mode", default="scripted_kinematic", choices=("scripted_kinematic", "physical"))
    demo.add_argument("--out", required=True)
    demo.add_argument("--seed", type=int, default=0)

    reset_p = sub.add_parser("reset", help="write a fresh episode state file")
    reset_p.add_argument("--state", required=True)
    reset_p.add_argument("--mode", default="physical", choices=("physical", "scripted_kinematic"))

    step_p = sub.add_parser("step", help="run one primitive against a state file")
    step_p.add_argument("--state", required=True)
    step_p.add_argument("--action", required=True, help="for example open_door, grasp:shirt, pour, press_start")

    diag = sub.add_parser("diagnostics", help="print gravity, joints, and step count")
    diag.add_argument("--state", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "config":
        print(json.dumps(supported_summary(), indent=2))
        return 0
    if args.cmd == "demo":
        summary = run_scripted_validation(args.out, action_mode=args.mode, seed=args.seed)
        print(json.dumps({
            "validation_label": summary["validation_label"],
            "success": summary["success"],
            "steps": summary["steps"],
            "frames": len(summary["frames"]),
            "summary": summary["summary_path"],
            "renderer": summary["renderer"],
            "learned_policy": False,
        }, indent=2))
        return 0 if summary["success"] else 1
    if args.cmd == "reset":
        env = LaundryEnv(action_mode=args.mode)
        _dump_state(env, args.state)
        info = env.diagnostics()
        info["garments"] = [item["name"] for item in GARMENTS]
        print(json.dumps({"wrote": args.state, "step": info["step"], "gravity": info["gravity"]}, indent=2))
        return 0
    if args.cmd == "step":
        env = _load_state(args.state)
        info = env.run(args.action)
        _dump_state(env, args.state)
        print(json.dumps({
            "command": args.action,
            "command_done": info["command_done"],
            "rejected": info["rejected"],
            "timed_out": info["timed_out"],
            "step": info["step"],
            "stage": info["stage"],
            "validation_label": info["validation_label"],
            "learned_policy": False,
        }, indent=2))
        return 0
    if args.cmd == "diagnostics":
        env = _load_state(args.state)
        print(json.dumps(env.diagnostics(), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
