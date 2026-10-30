"""M0 demo env: a low-cost diff-drive mobile base must reach a random goal.

Task: goal-reaching within an episode budget.
Obs (8): goal in body frame (2), body heading (sin/cos, 2), body linear vel (2), wheel rates (2).
Action (2): normalized motor torques for left / right wheel, each in [-1, 1].
Reward: -dist + success bonus (10); episode ends on success or timeout.
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
MAX_EPISODE_STEPS = 400     # 400 * 0.01s = 4s budget
SUCCESS_BONUS = 10.0


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
        self.goal_qpos_adr = self.model.body_qposadr[self.goal_id]
        self.wheel_ids = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_l_joint"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wheel_r_joint"),
        )
        self.render_mode = render_mode
        self.viewer = None
        self.goal = np.zeros(2)
        self.success = False
        self.steps = 0

        low = np.zeros(8, dtype=np.float32)
        high = np.inf * low
        self.observation_space = spaces.Box(low=low, high=high, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.full(2, -1.0), high=np.full(2, 1.0), shape=(2,), dtype=np.float32)
        self._reset_externally = True  # SB3 contract: reset() is called before step()

    # ------------------------------------------------------------------ obs
    def _obs(self) -> np.ndarray:
        pos = self.data.xpos[self.body_id][:2].copy()
        vel = self.data.xvel[self.body_id][:2].copy()
        heading = _quat_heading(self.data.xquat[self.body_id])
        # goal relative in the body frame
        dx, dy = self.goal - pos
        c, s = np.cos(-heading), np.sin(-heading)
        gx = c * dx - s * dy
        gy = s * dx + c * dy
        wheel_rates = np.array(
            [self.data.qvel[self.model.jnt_qposadr[j]] for j in self.wheel_ids],
            dtype=np.float64,
        )
        obs = np.concatenate(
            [[gx, gy], [np.sin(heading), np.cos(heading)], [vel], wheel_rates]
        ).astype(np.float32)
        return obs

    # ---------------------------------------------------------------- reset
    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_reset(self.model, self.data)
        r = np.random.uniform(*GOAL_RANGE)
        a = np.random.uniform(0.0, 2.0 * np.pi)
        self.goal = np.array([r * np.cos(a), r * np.sin(a)])
        self.data.qpos[self.goal_qpos_adr : self.goal_qpos_adr + 2] = self.goal
        self.success = False
        self.steps = 0
        self._physics_step(np.zeros(2))  # let contacts settle before the first obs
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
        done = self.steps >= MAX_EPISODE_STEPS
        reward = -dist
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
