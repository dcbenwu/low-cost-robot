import numpy as np, sys
sys.path.insert(0, "/home/hock/lcr/m0_nav_demo")
import env_diffdrive as E
def load(p):
    d=np.load(p,allow_pickle=True).item()
    for k in ["mu","sd","W0","b0","W1","b1","W2","b2"]: d[k]=d[k].astype(np.float64)
    return d
def infer(d,obs):
    x=(obs-d["mu"])/d["sd"]; h0=np.tanh(d["W0"]@x+d["b0"]); h1=np.tanh(d["W1"]@h0+d["b1"]); return np.tanh(d["W2"]@h1+d["b2"])
def evalone(d,N=200):
    env=E.DiffDriveGoalEnv(); ok=0
    for s in range(N):
        o,_=env.reset(seed=10000+s)
        while True:
            o,_,t,tr,info=env.step(np.array(infer(d,o),dtype=np.float32))
            if t or tr:
                ok+=bool(info.get("success")); break
    env.close(); return ok,N
P="/home/hock/lcr/m0_nav_demo/"
for name in ["bc_policy_vanillaBC_backup.npz.npy","bc_policy.npz.npy","bc_policy_dagger_r1.npy","bc_policy_dagger_r2.npy","bc_policy_dagger_r3.npy","bc_policy_dagger_r4.npy","bc_policy_dagger_r5.npy"]:
    try:
        d=load(P+name); ok,N=evalone(d); print(f"{name:42s} {ok}/{N} ({100.0*ok/N:.0f}%)  [training domain, N=200]")
    except Exception as e: print(f"{name:42s} ERR {e}")
