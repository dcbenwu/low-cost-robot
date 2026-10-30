"""Direction probe: for each of 4 command patterns, measure actual body displacement.

Tells us:
  a) does [a,a] (a>0) move the robot along +body-x (true forward) or -body-x?
  b) does [a,-a] rotate the correct way for a +angle error?
"""
import numpy as np

from env_diffdrive import DiffDriveGoalEnv, _quat_heading


def run(env, action, n=80):
    obs, _ = env.reset()
    p0 = env.data.xpos[env.body_id][:2].copy()
    h0 = _quat_heading(env.data.xquat[env.body_id])
    for _ in range(n):
        obs, r, term, trunc, _ = env.step(np.asarray(action, np.float64))
        if term or trunc:  # shouldn't happen in 80 steps
            break
    p1 = env.data.xpos[env.body_id][:2]
    h1 = _quat_heading(env.data.xquat[env.body_id])
    disp = p1 - p0
    # express displacement in the *initial* body frame
    c, s = np.cos(-h0), np.sin(-h0)
    body = [c * disp[0] - s * disp[1], s * disp[0] + c * disp[1]]
    return body, h1 - h0


env = DiffDriveGoalEnv()
for name, a in (
    ("forward  [ 1, 1]", [1.0, 1.0]),
    ("forward  [-1,-1]", [-1.0, -1.0]),
    ("turn+    [ 1,-1]", [1.0, -1.0]),
    ("turn-    [-1, 1]", [-1.0, 1.0]),
):
    body, dh = run(env, a)
    print(f"{name}: body-frame disp=({body[0]:+.3f}, {body[1]:+.3f}) m   dheading={dh:+.3f} rad")
