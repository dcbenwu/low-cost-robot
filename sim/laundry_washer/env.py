"""Episode API for the laundry task.

``run`` executes one physically limited primitive (approach, then actuate).
``step`` applies the 11-D action vector for one control interval. Neither
path loads a policy. ``scripted_kinematic`` is labeled and is not a learned
controller; ``physical`` uses the torque/speed-limited integrator.
"""

from __future__ import annotations

from . import arm
from .geom import add, bottle_center_for_mouth, cup_world, dist, door_handle, drawer_handle, slot_world, sub
from .runtime_config import LABEL_NO_POLICY
from .scene import LaundryScene
from .spec import (
    ACTION_NAMES,
    BOTTLE_HOME,
    BUTTON_POS,
    CAP_CLOSED,
    CAP_OPEN,
    CONTROL_SUBSTEPS,
    DOOR_CLOSED,
    DOOR_OPEN,
    DOOR_RANGE,
    DRAWER_CLOSED,
    DRAWER_OPEN,
    DRAWER_ORIGIN,
    FLOOR_POUR_MOUTH,
    KNOB_POS,
    LOAD_POINT,
    LOCK_ENGAGED,
    MAX_STEPS,
    OBS_NAMES,
    POUR_TILT,
    PROGRAMS,
)
from .task import compute_reward, holds_success, observation_vector, stage_name, start_blockers, validation_label

_BUTTON_PRESSED = -0.006
_WAYPOINT = 0.04


def full_wash_commands(
    program: str = "cotton",
    load: bool = True,
    close_door: bool = True,
    pour: str = "pour_main",
    open_cap: bool = True,
    extra_pour: bool = False,
    wash: bool = True,
) -> list[str]:
    commands = ["open_door"]
    if load:
        for name in ("shirt", "pants", "towel", "sock"):
            commands += [f"grasp:{name}", "carry:drum", "release"]
    commands.append("open_drawer")
    commands.append("grasp:bottle")
    if open_cap:
        commands.append("open_cap")
    commands.append(f"carry:{pour}")
    commands.append("pour")
    if extra_pour:
        commands.append("pour")
    commands += ["carry:counter", "close_cap", "release", "close_drawer"]
    if close_door:
        commands.append("close_door")
    commands += [f"set_program:{program}", "press_start"]
    if wash:
        commands.append("wash")
    return commands


class LaundryEnv:
    metadata = {"render_modes": ["rgb_array"], "learned_policy": False}

    def __init__(self, action_mode: str = "physical", seed: int = 0):
        if action_mode not in ("physical", "scripted_kinematic"):
            raise ValueError("action_mode must be 'physical' or 'scripted_kinematic'")
        self.action_mode = action_mode
        self.seed = int(seed)
        self.scene = LaundryScene(kinematic=action_mode == "scripted_kinematic")
        self.episode_reward = 0.0
        self.transitions: list[dict] = []
        self._cmd = ""
        self._phase = ""
        self._waypoints: list[tuple[float, float, float]] = []
        self._tilt = None
        self._cmd_done = True
        self._rejected = False
        self._timed_out = False
        self._started_try = False
        self._note_stage()

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
        self.scene.reset()
        self.episode_reward = 0.0
        self.transitions = []
        self._cmd_done = True
        self._rejected = False
        self._timed_out = False
        self._note_stage()
        return self._obs(), self._info()

    def step(self, action):
        values = [float(v) for v in action]
        if len(values) != len(ACTION_NAMES):
            raise ValueError(f"expected {len(ACTION_NAMES)} actions, got {len(values)}")
        scene = self.scene
        scene.target["door_hinge"] = values[0]
        scene.target["drawer_slide"] = values[1]
        scene.target["cap_hinge"] = values[2]
        scene.target["program_knob"] = values[3]
        scene.target["bottle_tilt"] = values[10]
        if values[4] >= 0.5:
            scene.target["start_button"] = -0.01
        else:
            scene.target["start_button"] = 0.0
        scene.set_arm_setpoint((values[5], values[6], values[7]))
        prev = scene.snapshot()
        for _ in range(CONTROL_SUBSTEPS):
            if values[8] > 0.5 and scene.grasp is None:
                scene.grasp_nearest()
            elif values[8] < -0.5:
                scene.release()
            scene.step()
            if values[4] >= 0.5 and scene.q["start_button"] <= _BUTTON_PRESSED and not scene.running:
                scene.try_start()
            if values[9] > 0.5 and _ == 0:
                scene.pour()
        reward = compute_reward(prev, scene.snapshot())
        self.episode_reward += reward
        self._note_stage()
        terminated = holds_success(scene)
        truncated = scene.step_count >= MAX_STEPS
        return self._obs(), reward, terminated, truncated, self._info()

    def run(self, command: str, budget: int = 1600) -> dict:
        self._begin(command)
        for _ in range(budget):
            if self._cmd_done:
                break
            prev = self.scene.snapshot()
            self._tick()
            self.scene.step()
            self.episode_reward += compute_reward(prev, self.scene.snapshot())
            self._after_step()
            self._note_stage()
        else:
            if not self._cmd_done:
                self._timed_out = True
                self._cmd_done = True
        return self._command_info()

    def run_sequence(self, commands: list[str]) -> list[dict]:
        return [self.run(command) for command in commands]

    @property
    def success(self) -> bool:
        return holds_success(self.scene)

    def diagnostics(self) -> dict:
        data = self.scene.diagnostics()
        data.update(
            {
                "action_mode": self.action_mode,
                "validation_label": validation_label(self.action_mode),
                "learned_policy": False,
                "stage": stage_name(self.scene),
                "stage_transitions": list(self.transitions),
                "blockers": start_blockers(self.scene),
                "success": self.success,
                "episode_reward": self.episode_reward,
                "assumptions_label_cloth": "rigid garment proxies — not deformable cloth",
                "assumptions_label_fluid": "detergent pose+dose abstraction — not fluid",
                "policy_label": LABEL_NO_POLICY,
            }
        )
        return data

    # --------------------------------------------------------------- commands
    def _begin(self, command: str) -> None:
        self._cmd = command
        self._phase = "approach"
        self._waypoints = []
        self._tilt = None
        self._cmd_done = False
        self._rejected = False
        self._timed_out = False
        self._started_try = False
        scene = self.scene
        scene.set_arm_setpoint(None)
        if command == "open_door":
            scene.target["door_hinge"] = scene.q["door_hinge"]
        elif command == "close_door":
            scene.target["door_hinge"] = scene.q["door_hinge"]
        elif command == "open_drawer":
            scene.target["drawer_slide"] = scene.q["drawer_slide"]
        elif command == "close_drawer":
            if _bottle_in_tray(scene):
                self._rejected = True
                self._cmd_done = True
                scene.last_reject = "bottle_blocks_drawer"
        elif command.startswith("grasp:"):
            self._phase = "goto"
        elif command in ("open_cap", "close_cap"):
            self._phase = "approach"
        elif command.startswith("carry:"):
            where = command.split(":", 1)[1]
            if scene.grasp is None:
                self._rejected = True
                self._cmd_done = True
                scene.last_reject = "carry_without_grasp"
                return
            self._waypoints, self._tilt = _carry_waypoints(scene, where)
            if not self._waypoints:
                self._rejected = True
                self._cmd_done = True
                scene.last_reject = "unreachable"
                return
            self._phase = "follow"
        elif command == "release":
            scene.release()
            self._cmd_done = True
        elif command == "pour":
            scene.pour()
            self._cmd_done = True
        elif command.startswith("set_program:"):
            program = command.split(":", 1)[1]
            angle = _program_angle(program)
            if angle is None:
                self._rejected = True
                self._cmd_done = True
                scene.last_reject = "unknown_program"
            else:
                self._tilt = angle  # reuse as knob goal
        elif command == "press_start":
            self._phase = "approach"
        elif command == "wash":
            self._phase = "spin"
            scene.set_arm_setpoint(None)
        elif command.startswith("dwell:"):
            self._phase = "dwell"
            self._tilt = float(command.split(":", 1)[1])
        else:
            self._rejected = True
            self._cmd_done = True
            scene.last_reject = "unknown_command"

    def _tick(self) -> None:
        if self._cmd_done:
            return
        scene = self.scene
        cmd = self._cmd
        if cmd in ("open_door", "close_door"):
            opening = cmd == "open_door"
            handle = door_handle(scene.q["door_hinge"])
            scene.set_arm_setpoint(handle)
            goal_reached = scene.q["door_hinge"] >= DOOR_OPEN if opening else scene.q["door_hinge"] <= DOOR_CLOSED
            if self._phase == "approach" and dist(scene.fk, handle) <= 0.08:
                self._phase = "actuate"
                scene.target["door_hinge"] = DOOR_RANGE[1] if opening else 0.0
            elif self._phase == "actuate" and goal_reached:
                self._cmd_done = True
        elif cmd in ("open_drawer", "close_drawer"):
            opening = cmd == "open_drawer"
            handle = drawer_handle(scene.q["drawer_slide"])
            scene.set_arm_setpoint(handle)
            goal_reached = scene.q["drawer_slide"] >= DRAWER_OPEN if opening else scene.q["drawer_slide"] <= DRAWER_CLOSED
            if self._phase == "approach" and dist(scene.fk, handle) <= 0.08:
                self._phase = "actuate"
                scene.target["drawer_slide"] = 0.20 if opening else 0.0
            elif self._phase == "actuate" and goal_reached:
                self._cmd_done = True
        elif cmd.startswith("grasp:"):
            name = cmd.split(":", 1)[1]
            target = scene.body(name).pos
            scene.set_arm_setpoint(target)
            if dist(scene.fk, target) <= 0.045:
                if scene.grasp_body(name):
                    self._cmd_done = True
                else:
                    self._rejected = True
                    self._cmd_done = True
        elif cmd.startswith("carry:"):
            if self._tilt is not None and scene.grasp == "bottle":
                scene.target["bottle_tilt"] = float(self._tilt)
            if not self._waypoints:
                self._cmd_done = True
                return
            scene.set_arm_setpoint(self._waypoints[0])
            tilt_ok = True
            if self._tilt is not None and scene.grasp == "bottle":
                tilt_ok = abs(scene.q["bottle_tilt"] - float(self._tilt)) < 0.15
            if dist(scene.fk, self._waypoints[0]) <= _WAYPOINT and (len(self._waypoints) > 1 or tilt_ok):
                self._waypoints.pop(0)
                if not self._waypoints and tilt_ok:
                    self._cmd_done = True
        elif cmd in ("open_cap", "close_cap"):
            opening = cmd == "open_cap"
            scene.set_arm_setpoint(scene.bottle.pos)
            goal = CAP_OPEN if opening else 0.0
            reached = scene.q["cap_hinge"] >= CAP_OPEN if opening else scene.q["cap_hinge"] <= CAP_CLOSED
            if self._phase == "approach" and dist(scene.fk, scene.bottle.pos) <= 0.08:
                self._phase = "actuate"
                scene.target["cap_hinge"] = 2.2 if opening else 0.0
            elif self._phase == "actuate" and reached:
                self._cmd_done = True
            _ = goal
        elif cmd.startswith("set_program:"):
            scene.set_arm_setpoint(KNOB_POS)
            if self._phase == "approach" and dist(scene.fk, KNOB_POS) <= 0.08:
                self._phase = "actuate"
                scene.target["program_knob"] = float(self._tilt)
            elif self._phase == "actuate" and abs(scene.q["program_knob"] - float(self._tilt)) < 0.08:
                self._cmd_done = True
        elif cmd == "press_start":
            scene.set_arm_setpoint(BUTTON_POS)
            if self._phase == "approach" and dist(scene.fk, BUTTON_POS) <= 0.08:
                self._phase = "press"
                scene.target["start_button"] = -0.01
            elif self._phase == "press" and scene.q["start_button"] <= _BUTTON_PRESSED and not self._started_try:
                self._started_try = True
                if not scene.try_start():
                    self._rejected = True
                    self._cmd_done = True
                else:
                    self._phase = "lock"
            elif self._phase == "lock" and scene.q["lock_pin"] >= LOCK_ENGAGED:
                scene.target["start_button"] = 0.0
                self._cmd_done = True
        elif cmd == "wash":
            if holds_success(scene) or scene.failed_after_start:
                self._cmd_done = True
        elif cmd.startswith("dwell:"):
            self._tilt -= 1.0
            if self._tilt <= 0:
                self._cmd_done = True

    def _after_step(self) -> None:
        scene = self.scene
        if scene.blocked and self._cmd.startswith("carry:"):
            self._rejected = True
            self._cmd_done = True
        if scene.last_reject == "door_locked_while_running" and self._cmd == "open_door":
            self._rejected = True
            self._cmd_done = True
        if self._cmd == "wash" and holds_success(scene):
            self._cmd_done = True

    def _note_stage(self) -> None:
        name = stage_name(self.scene)
        if not self.transitions or self.transitions[-1]["stage"] != name:
            self.transitions.append({"step": self.scene.step_count, "stage": name})

    def _obs(self):
        return observation_vector(self.scene)

    def _info(self) -> dict:
        scene = self.scene
        return {
            "action_mode": self.action_mode,
            "validation_label": validation_label(self.action_mode),
            "learned_policy": False,
            "success": holds_success(scene),
            "stage": stage_name(scene),
            "stage_transitions": list(self.transitions),
            "blockers": start_blockers(scene),
            "last_pour": scene.last_pour,
            "last_reject": scene.last_reject,
            "last_start_reasons": list(scene.last_start_reasons),
            "step": scene.step_count,
            "gravity": list(scene.gravity),
            "assumptions": [
                "rigid garment proxies — not deformable cloth",
                "detergent pose+dose abstraction — not fluid",
                LABEL_NO_POLICY,
            ],
        }

    def _command_info(self) -> dict:
        info = self._info()
        info.update(
            {
                "command": self._cmd,
                "command_done": self._cmd_done and not self._timed_out and not self._rejected,
                "timed_out": self._timed_out,
                "rejected": self._rejected,
                "door_q": self.scene.q["door_hinge"],
                "drawer_q": self.scene.q["drawer_slide"],
                "fk": self.scene.fk,
            }
        )
        return info


def _program_angle(name: str) -> float | None:
    for program, angle, _valid in PROGRAMS:
        if program == name:
            return angle
    return None


def _bottle_in_tray(scene: LaundryScene) -> bool:
    pos = scene.bottle.pos
    x = DRAWER_ORIGIN[0] + scene.q["drawer_slide"]
    return abs(pos[0] - x) < 0.12 and abs(pos[1] - DRAWER_ORIGIN[1]) < 0.17 and abs(pos[2] - DRAWER_ORIGIN[2]) < 0.08


def _grip_points(scene: LaundryScene, object_points: list[tuple[float, float, float]]):
    offset = scene.offset
    points = []
    for point in object_points:
        gripper = sub(point, offset)
        if arm.reachable(gripper, tol=0.045):
            points.append(gripper)
    if not points:
        return []
    final = sub(object_points[-1], offset)
    if dist(points[-1], final) > 0.02:
        if not arm.reachable(final, tol=0.045):
            return []
        points.append(final)
    return points


def _carry_waypoints(scene: LaundryScene, where: str):
    body = scene.body(scene.grasp)
    tilt = None
    if where == "drum":
        lift_z = max(body.pos[2] + 0.16, 0.30)
        objects = [
            (body.pos[0], body.pos[1], lift_z),
            (0.50, 0.0, 0.62),
            LOAD_POINT,
        ]
        return _grip_points(scene, objects), tilt
    if where == "counter":
        tilt = 0.0
        return _grip_points(scene, [body.pos, BOTTLE_HOME]), tilt
    if where == "pour_floor":
        tilt = POUR_TILT
        center = bottle_center_for_mouth(FLOOR_POUR_MOUTH, tilt)
        return _grip_points(scene, [body.pos, center]), tilt
    if where == "slot_main":
        tilt = POUR_TILT
        mouth = add(slot_world("main"), (0.0, 0.0, 0.02))
        center = bottle_center_for_mouth(mouth, tilt)
        return _grip_points(scene, [body.pos, center]), tilt
    pour_name = {"pour_main": "main", "pour_softener": "softener", "pour_bleach": "bleach"}.get(where)
    if pour_name is None:
        scene.last_reject = "unknown_carry"
        return [], None
    tilt = POUR_TILT
    mouth = add(cup_world(pour_name, max(scene.q["drawer_slide"], DRAWER_OPEN)), (0.0, 0.0, 0.03))
    center = bottle_center_for_mouth(mouth, tilt)
    return _grip_points(scene, [body.pos, center]), tilt

