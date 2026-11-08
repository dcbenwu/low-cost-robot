"""Scripted full-process validation.

The kinematic mode is explicitly not a learned controller. Frames are
rasters of the scene state after each recorded step. The summary is written
before the function returns; there is no viewer object left open.
"""

from __future__ import annotations

import json
import os

from .env import LaundryEnv, full_wash_commands
from .render_png import render_state, write_png
from .runtime_config import LABEL_SCRIPTED
from .task import validation_label


def run_scripted_validation(out_dir: str, action_mode: str = "scripted_kinematic", seed: int = 0) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    frame_dir = os.path.join(out_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    env = LaundryEnv(action_mode=action_mode, seed=seed)
    records = []
    frame_paths = []

    def grab(tag: str) -> None:
        image = render_state(env.scene)
        path = os.path.join(frame_dir, f"frame_{len(frame_paths):03d}_{tag}.png")
        write_png(path, image)
        frame_paths.append(path)

    grab("reset")
    for command in full_wash_commands():
        info = env.run(command)
        records.append(
            {
                "command": command,
                "command_done": info["command_done"],
                "rejected": info["rejected"],
                "timed_out": info["timed_out"],
                "step": info["step"],
                "stage": info["stage"],
                "last_pour": info["last_pour"],
                "last_reject": info["last_reject"],
                "blockers": info["blockers"],
            }
        )
        grab(command.replace(":", "_"))
        if action_mode == "scripted_kinematic" and info["validation_label"] != LABEL_SCRIPTED:
            raise RuntimeError("scripted mode lost its validation label")

    # A few extra frames while the drum stripe is moving, if the cycle ran.
    if env.scene.running or env.success:
        for _ in range(3):
            env.scene.step()
            grab("drum")

    summary = {
        "action_mode": action_mode,
        "validation_label": validation_label(action_mode),
        "learned_policy": False,
        "renderer": "software_raster_of_simulated_state",
        "gpu_physics_validation": False,
        "isaac_validation": False,
        "newton_validation": False,
        "success": env.success,
        "steps": env.scene.step_count,
        "episode_reward": env.episode_reward,
        "stage_transitions": env.transitions,
        "commands": records,
        "frames": frame_paths,
        "diagnostics": env.diagnostics(),
        "cloud_note": (
            "These frames are a CPU raster of the numpy/kinematic scene. "
            "They are not NewtonGL and not an Isaac viewport. "
            "GPU recording is a separate command and was not run here."
        ),
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    # Summary is on disk before the function returns. There is no GL context to close.
    summary["summary_path"] = summary_path
    return summary
