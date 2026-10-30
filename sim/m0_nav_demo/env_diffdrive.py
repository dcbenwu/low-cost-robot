"""M0 demo env: a low-cost diff-drive mobile base must reach a random goal.

Task: goal-reaching within an episode budget.
Obs (8): goal in body frame (2), body heading (sin/cos, 2), body linear vel (2), wheel rates (2).
Action (2): normalized motor torques for left / right wheel, each in [-1, 1].
Reward: (prev_dist - dist) potential shaping + success bonus (20) - 0.01/step; ends on success or timeout.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

XML_PATH = Path(__file__).parent / "robot.xml"
GOAL_RADIUS = 0.12          # success radius [m]
GOAL_RANGE = (0.6, 1.5)     # goal distance band [m]
MAX_EPISODE_STEPS = 1000    # 1000 * 0.01s = 10s budget (0.24 m/s => can reach full 1.5m)
SUCCESS_BONUS = 20.0        # dominant, clean success signal (see step() shaping)
STEP_COST = 0.01            # small per-step cost: prefer reaching on time


def _quat_heading(q: np.ndarray) -> float:
    """Heading (yaw) of the body that the world x-axis maps to under the
    body quaternion, i.e. the robot's facing direction in the world plane."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))


class DiffDriveGoalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        self.goal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        # goal body carries a free joint; its qpos start = qpos address of that joint
        import numpy as _np

        self.goal_qpos_adr = int(self.model.jnt_qposadr[int(_np.where(self.model.jnt_bodyid == self.goal_id)[0][0])])
        # robot free-joint qvel start (linear world velocities live at qvel[adr..adr+2])
        self.body_vel_adr = int(self.model.jnt_dofadr[int(_np.where(self.model.jnt_bodyid == self.body_id)[0][0])])
        self._reset = getattr(mujoco, "mj_resetData", None) or mujoco.mj_reset
        self.wheel_ids = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_l_joint"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_r_joint"),
        )
        self.render_mode = render_mode
        self.viewer = None
        self.goal = np.zeros(2)
        self.success = False
        self.steps = 0
        self._prev_dist = float("inf")

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.full(2, -1.0), high=np.full(2, 1.0), shape=(2,), dtype=np.float32)
        self._reset_externally = True  # SB3 contract: reset() is called before step()

    # ------------------------------------------------------------------ obs
    def _obs(self) -> np.ndarray:
        pos = self.data.xpos[self.body_id][:2].copy()
        vel = self.data.qvel[self.body_vel_adr : self.body_vel_adr + 2].copy()
        heading = _quat_heading(self.data.xquat[self.body_id])
        # goal relative in the body frame
        dx, dy = self.goal - pos
        c, s = np.cos(-heading), np.sin(-heading)
        gx = c * dx - s * dy
        gy = s * dx + c * dy
        wheel_rates = np.array(
            [self.data.qvel[self.model.jnt_dofadr[j]] for j in self.wheel_ids],
            dtype=np.float64,
        )
        obs = np.concatenate(
            [[gx, gy], [np.sin(heading), np.cos(heading)], vel, wheel_rates]
        ).astype(np.float32)
        return obs

    # ---------------------------------------------------------------- reset
    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._reset(self.model, self.data)
        r = np.random.uniform(*GOAL_RANGE)
        a = np.random.uniform(0.0, 2.0 * np.pi)
        self.goal = np.array([r * np.cos(a), r * np.sin(a)])
        self.data.qpos[self.goal_qpos_adr : self.goal_qpos_adr + 2] = self.goal
        self.success = False
        self.steps = 0
        for _ in range(30):  # let the free body settle onto the floor before the first obs
            self.data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, self.data)
        p0 = self.data.xpos[self.body_id][:2]
        self._prev_dist = float(np.linalg.norm(self.goal - p0))
        obs = self._obs()
        info = {}
        return obs, info

    # ----------------------------------------------------------------- step
    def _physics_step(self, action: np.ndarray) -> None:
        self.data.ctrl[0] = float(np.clip(action[0], -1.0, 1.0))
        self.data.ctrl[1] = float(np.clip(action[1], -1.0, 1.0))
        mujoco.mj_step(self.model, self.data)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        self._physics_step(action)
        pos = self.data.xpos[self.body_id][:2]
        dist = float(np.linalg.norm(self.goal - pos))
        # Potential-based shaping: reward = (prev_dist - dist). The per-episode
        # shaping telescopes to (d0 - d_T) and is *bounded*, so SUCCESS_BONUS
        # dominates the return and the policy sees a clean, learnable signal.
        reward = float(self._prev_dist - dist) - STEP_COST
        self._prev_dist = dist
        done = self.steps >= MAX_EPISODE_STEPS
        info: dict = {"distance": dist}
        if dist < GOAL_RADIUS:
            self.success = True
            reward += SUCCESS_BONUS
            done = True
            info["success"] = True
        return self._obs(), float(reward), bool(done), False, info

    # ---------------------------------------------------------------- render
    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self.viewer is None:
            from mujoco import renderer

            self.viewer = renderer.Renderer(self.model)
        self.viewer.update_scene(self.data)
        img = self.viewer.render()
        import cv2  # only imported when actually rendering

        cv2.imshow("lcr m0", img[:, :, ::-1])
        cv2.waitKey(1)

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
