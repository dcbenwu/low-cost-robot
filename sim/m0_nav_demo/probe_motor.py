"""Fine motor probe: isolate each wheel, measure joint qvel + body motion."""
import mujoco
import numpy as np

from env_diffdrive import DiffDriveGoalEnv, _quat_heading


def n2id(m, typ, i):
    return mujoco.mj_id2name(m, typ, i)


def run(env, action, n=40, label=""):
    env.reset()
    h0 = _quat_heading(env.data.xquat[env.body_id])
    p0 = env.data.xpos[env.body_id][:2].copy()
    for _ in range(n):
        env.step(np.asarray(action, np.float64))
    p1 = env.data.xpos[env.body_id][:2]
    h1 = _quat_heading(env.data.xquat[env.body_id])
    jw = [env.data.qvel[env.model.jnt_dofadr[j]] for j in env.wheel_ids]
    d = p1 - p0
    mag, dh = float(np.hypot(*d)), h1 - h0
    print(f"{label:14s} | wheel qvel L/R = {jw[0]:+7.3f} {jw[1]:+7.3f} "
          f"| disp=({d[0]:+.3f},{d[1]:+.3f}) {mag:+.3f}m | dheading={dh:+.3f}")


env = DiffDriveGoalEnv()
m = env.model
# check actuator mapping first
for i in range(m.nu):
    jid = int(m.actuator_trnid[i][0])
    print(f"actuator[{i}] = {n2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} "
          f"-> joint {n2id(m, mujoco.mjtObj.mjOBJ_JOINT, jid)}")
print(f"wheel ids: L={n2id(m, mujoco.mjtObj.mjOBJ_JOINT, env.wheel_ids[0])} "
      f"R={n2id(m, mujoco.mjtObj.mjOBJ_JOINT, env.wheel_ids[1])}")
print()
run(env, [1.0, 0.0], label="right-off [1,0]")
run(env, [0.0, 1.0], label="left-off  [0,1]")
run(env, [1.0, 1.0], label="both+ [1,1]")
run(env, [-1.0, -1.0], label="both- [-1,-1]")
run(env, [1.0, -1.0], label="turn+ [1,-1]")
run(env, [-1.0, 1.0], label="turn- [-1,1]")
run(env, [-1.0, 0.0], label="mirror [-1,0]")
run(env, [0.0, -1.0], label="mirror [0,-1]")
