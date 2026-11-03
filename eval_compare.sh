#!/usr/bin/env bash
# Not the seed=0 protocol. Cited M0 rates are eval_all.py / gen_test.py in
# sim/m0_nav_demo/RESULTS.md. This script overwrites bc_policy.npz.npy.
cd /home/hock/lcr/m0_nav_demo
PY=/home/hock/miniconda3/envs/env_isaaclab/bin/python
LOG=/home/hock/lcr/eval_compare.log
{
  echo "=== A. DAGGER r2 (trusted eval_bc.py) ==="
  cp bc_policy_dagger_r2.npy bc_policy.npz.npy
  for k in 1 2 3; do
    echo "[run$k]"
    $PY eval_bc.py 100 2>/dev/null | grep -viE "warning|warn|box|dtype|logger"
  done
  echo "=== B. VANILLA BC (trusted eval_bc.py) ==="
  cp bc_policy_vanillaBC_backup.npz.npy bc_policy.npz.npy
  for k in 1 2 3; do
    echo "[run$k]"
    $PY eval_bc.py 100 2>/dev/null | grep -viE "warning|warn|box|dtype|logger"
  done
  echo "=== C. copy dagger r2 onto bc_policy.npz.npy (not a ranking) ==="
  cp bc_policy_dagger_r2.npy bc_policy.npz.npy
  echo "DONE"
} > "$LOG" 2>&1
