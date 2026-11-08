"""CPU task-logic tests for the laundry environment.

These tests step the numpy rigid scene and the task predicates.
They do not import Isaac Sim, Newton, Warp, or a GPU, and they are not
MuJoCo contact validation or render-fidelity validation.

A shut door, a closed cap, the wrong cup, an empty bottle, a bad program,
an overfilled cup, and a locked door are rejected because the predicates
read joint positions and body positions. Nothing sets a success bit directly.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from laundry_washer.demo import run_scripted_validation
from laundry_washer.env import LaundryEnv, full_wash_commands
from laundry_washer.geom import in_drum
from laundry_washer.mjcf import build_mjcf
from laundry_washer.runtime_config import (
    GRAVITY,
    LABEL_SCRIPTED,
    NUM_WORLDS,
    USE_CUDA_GRAPH,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from laundry_washer.spec import (
    ACTION_NAMES,
    CAPACITY_ML,
    DOOR_CLOSED,
    DT,
    GARMENTS,
    LIQUID_ML,
    LOCK_ENGAGED,
    OBS_NAMES,
)
from laundry_washer.task import observation_vector


def _play(commands, mode="physical", liquid=None) -> LaundryEnv:
    env = LaundryEnv(action_mode=mode)
    if liquid is not None:
        env.scene.liquid_ml = liquid
    env.run_sequence(commands)
    return env


class TaskGateTests(unittest.TestCase):
    def test_positive_full_workflow_physical(self):
        env = _play(full_wash_commands())
        self.assertTrue(env.success)
        self.assertGreater(env.scene.step_count, 100)
        self.assertGreater(env.episode_reward, 0.0)
        self.assertEqual(env.scene.doses["main"], 40.0)
        self.assertFalse(env.scene.overfill)
        self.assertLessEqual(env.scene.q["door_hinge"], DOOR_CLOSED)
        self.assertGreaterEqual(env.scene.q["lock_pin"], LOCK_ENGAGED)
        self.assertGreaterEqual(env.scene.wash_angle, 4.0 * 3.14159265)
        for body in env.scene.garments.values():
            self.assertTrue(in_drum(body.pos))
        self.assertEqual(env.diagnostics()["learned_policy"], False)
        self.assertIn("wash_cycle_complete", [item["stage"] for item in env.transitions])
        self.assertFalse(env.diagnostics()["gpu_physics_validation"])

    def test_scripted_kinematic_is_labeled_and_still_gated(self):
        env = LaundryEnv(action_mode="scripted_kinematic")
        self.assertEqual(env._info()["validation_label"], LABEL_SCRIPTED)
        info = env.run("grasp:shirt")
        self.assertTrue(info["command_done"])
        info = env.run("carry:drum")
        self.assertTrue(info["rejected"])
        self.assertFalse(in_drum(env.scene.garments["shirt"].pos))
        self.assertEqual(info["validation_label"], LABEL_SCRIPTED)
        done = _play(full_wash_commands(), mode="scripted_kinematic")
        self.assertTrue(done.success)
        self.assertEqual(done.diagnostics()["validation_label"], LABEL_SCRIPTED)

    def test_clothes_not_loaded(self):
        env = _play(full_wash_commands(load=False, wash=False))
        self.assertFalse(env.success)
        self.assertFalse(env.scene.running)
        self.assertIn("clothes_not_in_drum", env.scene.last_start_reasons)

    def test_open_door_at_start(self):
        env = _play(full_wash_commands(close_door=False, wash=False))
        self.assertFalse(env.success)
        self.assertFalse(env.scene.running)
        self.assertGreater(env.scene.q["door_hinge"], DOOR_CLOSED)
        self.assertIn("door_open", env.scene.last_start_reasons)

    def test_pour_closed_cap(self):
        env = _play(["open_drawer", "grasp:bottle", "carry:pour_main", "pour"])
        self.assertEqual(env.scene.last_pour, "closed_cap")
        self.assertEqual(env.scene.doses["main"], 0.0)
        self.assertEqual(env.scene.liquid_ml, LIQUID_ML)
        self.assertEqual(env.scene.spill_ml, 0.0)

    def test_pour_wrong_dispenser(self):
        env = _play(["open_drawer", "grasp:bottle", "open_cap", "carry:pour_softener", "pour"])
        self.assertEqual(env.scene.last_pour, "softener")
        self.assertEqual(env.scene.doses["softener"], 40.0)
        self.assertEqual(env.scene.doses["main"], 0.0)
        self.assertIn("detergent_not_in_main", env.diagnostics()["blockers"])

    def test_pour_closed_drawer(self):
        env = _play(["grasp:bottle", "open_cap", "carry:slot_main", "pour"])
        self.assertEqual(env.scene.last_pour, "closed_drawer")
        self.assertEqual(env.scene.doses["main"], 0.0)
        self.assertEqual(env.scene.spill_ml, 40.0)
        self.assertLess(env.scene.q["drawer_slide"], 0.05)

    def test_pour_on_floor(self):
        env = _play(["grasp:bottle", "open_cap", "carry:pour_floor", "pour"])
        self.assertEqual(env.scene.last_pour, "floor")
        self.assertEqual(env.scene.doses["main"], 0.0)
        self.assertEqual(env.scene.spill_ml, 40.0)

    def test_empty_detergent(self):
        env = _play(
            ["open_drawer", "grasp:bottle", "open_cap", "carry:pour_main", "pour"],
            liquid=0.0,
        )
        self.assertEqual(env.scene.last_pour, "empty")
        self.assertEqual(env.scene.doses["main"], 0.0)
        self.assertEqual(env.scene.liquid_ml, 0.0)

    def test_wrong_program(self):
        env = _play(full_wash_commands(program="spin_only", wash=False))
        self.assertFalse(env.success)
        self.assertFalse(env.scene.running)
        self.assertIn("invalid_program", env.scene.last_start_reasons)

    def test_overfill(self):
        env = _play(full_wash_commands(extra_pour=True, wash=False))
        self.assertTrue(env.scene.overfill)
        self.assertEqual(env.scene.doses["main"], CAPACITY_ML)
        self.assertGreater(env.scene.spill_ml, 0.0)
        self.assertFalse(env.success)
        self.assertFalse(env.scene.running)
        self.assertIn("overfill", env.scene.last_start_reasons)

    def test_door_locked_while_running(self):
        env = _play(full_wash_commands(wash=False))
        self.assertTrue(env.scene.running)
        self.assertGreaterEqual(env.scene.q["lock_pin"], LOCK_ENGAGED)
        door_before = env.scene.q["door_hinge"]
        info = env.run("open_door")
        self.assertTrue(info["rejected"])
        self.assertEqual(info["last_reject"], "door_locked_while_running")
        self.assertLessEqual(env.scene.q["door_hinge"], DOOR_CLOSED)
        self.assertAlmostEqual(env.scene.q["door_hinge"], door_before, places=3)

    def test_reset_restores_episode(self):
        env = _play(full_wash_commands(wash=False))
        self.assertTrue(env.scene.running)
        self.assertNotEqual(env.scene.liquid_ml, LIQUID_ML)
        env.reset()
        self.assertEqual(env.scene.step_count, 0)
        self.assertFalse(env.scene.running)
        self.assertFalse(env.scene.wash_complete)
        self.assertEqual(env.scene.liquid_ml, LIQUID_ML)
        self.assertEqual(env.scene.doses, {"main": 0.0, "softener": 0.0, "bleach": 0.0})
        self.assertEqual(env.scene.spill_ml, 0.0)
        self.assertFalse(env.scene.overfill)
        self.assertEqual(env.scene.q["door_hinge"], 0.0)
        self.assertEqual(env.scene.q["cap_hinge"], 0.0)
        self.assertEqual(env.scene.q["program_knob"], 0.0)
        self.assertEqual(env.scene.q["lock_pin"], 0.0)
        for item in GARMENTS:
            self.assertEqual(env.scene.garments[item["name"]].pos, tuple(item["pos"]))
            self.assertFalse(in_drum(env.scene.garments[item["name"]].pos))

    def test_alternate_order_still_succeeds(self):
        commands = ["open_drawer", "set_program:cotton", "open_door"]
        for name in ("towel", "sock", "shirt", "pants"):
            commands += [f"grasp:{name}", "carry:drum", "release"]
        commands += [
            "grasp:bottle",
            "open_cap",
            "carry:pour_main",
            "pour",
            "close_cap",
            "carry:counter",
            "release",
            "close_door",
            "close_drawer",
            "press_start",
            "wash",
        ]
        env = _play(commands)
        self.assertTrue(env.success)
        self.assertEqual(env.scene.doses["main"], 40.0)

    def test_closed_door_blocks_entry(self):
        env = LaundryEnv()
        self.assertTrue(env.run("grasp:shirt")["command_done"])
        info = env.run("carry:drum")
        self.assertTrue(info["rejected"])
        self.assertEqual(info["last_reject"], "door_blocks_entry")
        self.assertFalse(in_drum(env.scene.garments["shirt"].pos))

    def test_gravity_joint_and_step_diagnostics(self):
        env = LaundryEnv()
        env.scene.place("shirt", (1.2, -1.1, 1.0))
        samples = []
        for _ in range(20):
            env.scene.step()
            samples.append(env.scene.garments["shirt"].pos[2])
        g = env.scene.gravity[2]
        expected = 1.0 + g * DT * DT * 20 * 21 / 2.0
        self.assertEqual(env.scene.gravity, GRAVITY)
        self.assertAlmostEqual(g, -9.81)
        self.assertAlmostEqual(samples[-1], expected, places=5)
        self.assertLess(samples[-1], samples[0])
        diag = env.diagnostics()
        self.assertEqual(diag["gravity"], [0.0, 0.0, -9.81])
        self.assertEqual(diag["step"], 20)
        self.assertGreater(diag["sim_time_s"], 0.0)
        self.assertFalse(diag["gpu_physics_validation"])
        env.scene.target["door_hinge"] = 1.2
        q0 = env.scene.q["door_hinge"]
        env.scene.step()
        q1 = env.scene.q["door_hinge"]
        env.scene.step()
        q2 = env.scene.q["door_hinge"]
        self.assertGreater(q1, q0)
        self.assertGreater(q2, q1)
        self.assertLess(q2, 1.2)
        self.assertEqual(env.scene.step_count, 22)

    def test_observation_and_action_api(self):
        env = LaundryEnv()
        obs = observation_vector(env.scene)
        self.assertEqual(len(obs), len(OBS_NAMES))
        self.assertEqual(obs[OBS_NAMES.index("gravity_z")], -9.81)
        hold = list(env.scene.fk)
        action = [0.0] * len(ACTION_NAMES)
        action[0] = 1.0  # door target
        action[5:8] = hold
        _obs, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(len(_obs), len(OBS_NAMES))
        self.assertGreater(env.scene.q["door_hinge"], 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["learned_policy"])
        self.assertIsInstance(reward, float)

    def test_mjcf_is_texture_free_and_has_earth_gravity(self):
        xml = build_mjcf()
        root = ET.fromstring(xml)
        option = root.find("option")
        self.assertIsNotNone(option)
        self.assertEqual(option.get("gravity"), "0.00000 0.00000 -9.81000")
        self.assertNotIn("http://", xml)
        self.assertNotIn("https://", xml)
        self.assertNotIn(".hdr", xml)
        self.assertNotIn(".png", xml)
        names = {joint.get("name") for joint in root.iter("joint")}
        for required in (
            "door_hinge",
            "drawer_slide",
            "cap_hinge",
            "program_knob",
            "start_button",
            "lock_pin",
            "drum_spin",
            "arm_yaw",
        ):
            self.assertIn(required, names)
        for garment in ("shirt", "pants", "towel", "sock"):
            self.assertIn(garment, {body.get("name") for body in root.iter("body")})

    def test_supported_runtime_stays_one_small_env(self):
        self.assertEqual(NUM_WORLDS, 1)
        self.assertEqual((WINDOW_WIDTH, WINDOW_HEIGHT), (640, 360))
        self.assertFalse(USE_CUDA_GRAPH)
        self.assertEqual(GRAVITY, (0.0, 0.0, -9.81))

    def test_scripted_demo_writes_real_frames_and_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = run_scripted_validation(folder, action_mode="scripted_kinematic")
            self.assertTrue(summary["success"])
            self.assertEqual(summary["validation_label"], LABEL_SCRIPTED)
            self.assertFalse(summary["learned_policy"])
            self.assertEqual(summary["renderer"], "software_raster_of_simulated_state")
            self.assertFalse(summary["gpu_physics_validation"])
            self.assertGreaterEqual(len(summary["frames"]), 4)
            self.assertTrue(os.path.isfile(summary["summary_path"]))
            with open(summary["summary_path"], encoding="utf-8") as handle:
                on_disk = json.loads(handle.read())
            self.assertGreaterEqual(len(on_disk["stage_transitions"]), 2)
            stages = [item["stage"] for item in on_disk["stage_transitions"]]
            self.assertIn("wash_cycle_complete", stages)
            with open(summary["frames"][0], "rb") as handle:
                early = handle.read()
            with open(summary["frames"][-1], "rb") as handle:
                later = handle.read()
            self.assertEqual(early[:8], b"\x89PNG\r\n\x1a\n")
            self.assertNotEqual(early, later)


if __name__ == "__main__":
    unittest.main()
