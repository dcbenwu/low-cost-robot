"""MuJoCo compiler check for the laundry MJCF.

This is not a Newton run, not a GPU run, and not the numpy task-gate suite.
It loads the same XML Newton compiles and rejects any jointed body whose
mass or inertia is at or below 1e-8 (MuJoCo's mjMINVAL is 1e-15).
"""

from __future__ import annotations

import unittest

from laundry_washer.mjcf import build_mjcf

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None


@unittest.skipIf(mujoco is None, "mujoco is not installed; this skip is not a physics pass")
class MjcfCompileTests(unittest.TestCase):
    def test_every_jointed_body_has_mass_and_inertia(self):
        model = mujoco.MjModel.from_xml_string(build_mjcf())
        self.assertAlmostEqual(float(model.opt.gravity[2]), -9.81, places=5)
        checked = []
        for i in range(model.nbody):
            if int(model.body_jntnum[i]) == 0:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            mass = float(model.body_mass[i])
            inertia = [float(v) for v in model.body_inertia[i]]
            self.assertGreater(mass, 1e-8, name)
            self.assertGreater(min(inertia), 1e-8, name)
            checked.append(name)
        self.assertIn("arm_yaw_link", checked)
        self.assertGreaterEqual(len(checked), 17)


if __name__ == "__main__":
    unittest.main()
