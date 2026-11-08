"""Cross-layer tests: task state packed into the MJCF qpos vector and read back.

No Newton, Warp, or GPU is imported. A passing test means the named door,
drawer, cap, clothes, bottle tilt, and program share one coordinate vector
with the scene file. It does not mean Newton executed that vector.
"""

from __future__ import annotations

import unittest

from laundry_washer.env import LaundryEnv, full_wash_commands
from laundry_washer.geom import in_drum
from laundry_washer.mjcf import build_mjcf
from laundry_washer.qpos_map import (
    build_index,
    nq,
    pack_scene,
    quat_from_tilt,
    tilt_from_quat,
    unpack_scene,
)
from laundry_washer.runtime_config import LABEL_PHYSICAL, LABEL_SCRIPTED
from laundry_washer.scene import LaundryScene
from laundry_washer.session import MemoryBackend, run_session
from laundry_washer.spec import ARM_HOME, LIQUID_ML
from laundry_washer.task import holds_success


class StateMapTests(unittest.TestCase):
    def test_mjcf_index_covers_task_joints_and_free_bodies(self):
        index = build_index(build_mjcf())
        kinds = {dof.name: dof.kind for dof in index}
        for name in ("door_hinge", "cap_hinge", "program_knob", "drum_spin", "arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist"):
            self.assertEqual(kinds[name], "hinge")
        for name in ("drawer_slide", "start_button", "lock_pin", "gripper_slide"):
            self.assertEqual(kinds[name], "slide")
        for name in ("shirt_free", "pants_free", "towel_free", "sock_free", "bottle_free"):
            self.assertEqual(kinds[name], "free")
        self.assertNotIn("bottle_tilt", kinds)
        self.assertEqual(nq(index), sum(dof.nq for dof in index))
        addresses = [dof.adr for dof in index]
        self.assertEqual(addresses, sorted(addresses))

    def test_tilt_roundtrip_matches_pour_convention(self):
        for tilt in (0.0, 0.4, 1.2, 2.1):
            self.assertAlmostEqual(tilt_from_quat(quat_from_tilt(tilt)), tilt, places=6)

    def test_full_workflow_roundtrip_preserves_gates(self):
        env = LaundryEnv(action_mode="scripted_kinematic")
        env.run_sequence(full_wash_commands())
        self.assertTrue(env.success)
        index = build_index(build_mjcf())
        vector = pack_scene(env.scene, index, free_bodies=True)
        restored = LaundryScene(kinematic=True)
        from laundry_washer.qpos_map import ledger_from_scene

        unpack_scene(restored, vector, index, ledger_from_scene(env.scene))
        self.assertTrue(holds_success(restored))
        self.assertEqual(restored.doses["main"], 40.0)
        self.assertLessEqual(restored.q["door_hinge"], 0.10)
        self.assertLessEqual(restored.q["cap_hinge"], 0.20)
        for name, body in env.scene.garments.items():
            self.assertTrue(in_drum(restored.garments[name].pos))
            for axis in range(3):
                self.assertAlmostEqual(restored.garments[name].pos[axis], body.pos[axis], places=5)
        self.assertAlmostEqual(restored.q["bottle_tilt"], env.scene.q["bottle_tilt"], places=5)
        self.assertAlmostEqual(restored.q["program_knob"], 0.80, places=2)
        self.assertAlmostEqual(restored.q["arm_shoulder"], env.scene.q["arm_shoulder"], places=5)

    def test_home_free_bodies_match_mjcf_defaults(self):
        index = build_index(build_mjcf())
        scene = LaundryScene()
        vector = pack_scene(scene, index, free_bodies=True)
        by_name = {dof.name: dof for dof in index}
        for dof in index:
            if dof.kind != "free":
                continue
            for axis in range(3):
                self.assertAlmostEqual(vector[dof.adr + axis], dof.default[axis], places=5)
        self.assertAlmostEqual(vector[by_name["arm_shoulder"].adr], ARM_HOME[1], places=5)
        self.assertNotAlmostEqual(ARM_HOME[1], 0.0)

    def test_memory_session_scripted_workflow_and_reset(self):
        summary = run_session(MemoryBackend(), action_mode="scripted_kinematic", hold_steps=1)
        self.assertEqual(summary["validation_label"], LABEL_SCRIPTED)
        self.assertEqual(summary["physics_backend"], "memory_qpos")
        self.assertFalse(summary["newton_executed"])
        self.assertFalse(summary["gpu_execution_claimed"])
        self.assertTrue(summary["success_from_readback"])
        self.assertEqual(summary["pose_source"], "scripted_qpos_written_into_model")
        for tag in ("loaded", "pour", "start", "washing"):
            self.assertIn(tag, summary["keyframes"])
        pour = next(row for row in summary["command_records"] if row["command"] == "pour")
        self.assertGreaterEqual(pour["n_in_drum"], 4)
        self.assertGreater(pour["cap_q"], 1.0)
        self.assertGreater(pour["drawer_q"], 0.1)
        self.assertEqual(pour["main_ml"], 40.0)
        start = next(row for row in summary["command_records"] if row["command"] == "press_start")
        self.assertTrue(start["program_valid"])
        self.assertLessEqual(start["door_q"], 0.10)
        self.assertTrue(summary["reset_after"]["ok"])
        self.assertEqual(summary["reset_after"]["liquid_ml"], LIQUID_ML)
        self.assertFalse(summary["reset_after"]["running"])

    def test_physical_mode_does_not_write_free_bodies(self):
        backend = MemoryBackend()
        summary = run_session(
            backend,
            commands=["open_door"],
            action_mode="physical",
            hold_steps=1,
        )
        self.assertEqual(summary["validation_label"], LABEL_PHYSICAL)
        self.assertEqual(summary["pose_source"], "actuator_targets_only_free_bodies_stay_dynamic")
        self.assertFalse(summary["success_from_readback"])
        opened = summary["command_records"][0]
        self.assertEqual(opened["command"], "open_door")
        self.assertGreater(opened["door_q"], 1.0)
        self.assertEqual(opened["n_in_drum"], 0)
        # Only the closing reset writes free-body coordinates. The command itself does not.
        self.assertEqual(backend.free_body_writes, 1)
        self.assertTrue(summary["reset_after"]["ok"])


if __name__ == "__main__":
    unittest.main()
