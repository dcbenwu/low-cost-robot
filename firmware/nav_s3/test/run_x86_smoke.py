#!/usr/bin/env python3
"""Compile the nav_s3 obs pipeline and policy.c on x86 and check numpy parity.

policy_forward is the embedded graph: (obs - mu) / sd, tanh, tanh, linear head.
This does not apply the extra output tanh used by the sim rollout scripts.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT / "firmware" / "nav_s3" / "main"
TEST = ROOT / "firmware" / "nav_s3" / "test"
sys.path.insert(0, str(ROOT / "sim" / "m0_nav_demo"))

import sim_obs_bridge as bridge  # noqa: E402


def _u32_array(text: str, name: str) -> np.ndarray:
    match = re.search(rf"static const uint32_t {name}\[(\d+)\] = \{{(.*?)\}};", text, re.S)
    if not match:
        raise SystemExit(f"missing {name} in policy_weights.h")
    words = [int(w, 16) for w in re.findall(r"0x[0-9A-Fa-f]+", match.group(2))]
    if len(words) != int(match.group(1)):
        raise SystemExit(f"{name}: parsed {len(words)} words, header says {match.group(1)}")
    return np.array(words, dtype=np.uint32).view(np.float32).copy()


def load_policy():
    text = (MAIN / "policy_weights.h").read_text()
    mu = _u32_array(text, "obs_mu_bits")
    sd = _u32_array(text, "obs_sd_bits")
    w0 = _u32_array(text, "W0_bits").reshape(64, 8)
    b0 = _u32_array(text, "b0_bits")
    w1 = _u32_array(text, "W1_bits").reshape(64, 64)
    b1 = _u32_array(text, "b1_bits")
    w2 = _u32_array(text, "W2_bits").reshape(2, 64)
    b2 = _u32_array(text, "b2_bits")
    return mu, sd, w0, b0, w1, b1, w2, b2


def policy_linear(obs, weights) -> np.ndarray:
    mu, sd, w0, b0, w1, b1, w2, b2 = weights
    x = (np.asarray(obs, dtype=np.float32) - mu) / sd
    h0 = np.tanh(w0 @ x + b0)
    h1 = np.tanh(w1 @ h0 + b1)
    return (w2 @ h1 + b2).astype(np.float32)


def policy_tanh_head(obs, weights) -> np.ndarray:
    return np.tanh(policy_linear(obs, weights)).astype(np.float32)


def compile_smoke(bin_path: Path) -> None:
    cmd = [
        "gcc",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-O2",
        "-DNAV_POSIX_UDP",
        "-I",
        str(MAIN),
        "-o",
        str(bin_path),
        str(TEST / "smoke_main.c"),
        str(MAIN / "obs_builder.c"),
        str(MAIN / "sensors.c"),
        str(MAIN / "host_link.c"),
        str(MAIN / "policy.c"),
        "-lm",
    ]
    subprocess.check_call(cmd)
    subprocess.check_call(
        [
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            "-I",
            str(MAIN),
            "-I",
            str(TEST / "stubs"),
            str(MAIN / "main.c"),
        ]
    )


def _floats(line: str) -> np.ndarray:
    nums = []
    for tok in line.split():
        try:
            nums.append(float(tok))
        except ValueError:
            continue
    return np.array(nums, dtype=np.float64)


def check_program(bin_path: Path, weights) -> None:
    constants = bridge.load_link_constants()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        pkt = bridge.pack_lcr1(
            [0.2, -0.4],
            pose=(0.0, 0.0, 0.0, 0.0, 0.0),
            wheels=(1.5, -1.5),
            constants=constants,
        )
        # Goal-from-ToF bit must not ride along; this packet is a normal inject.
        if (constants["flag_tof"] & 0x80) != 0x80:
            raise SystemExit("NAV_FLAG_GOAL_FROM_TOF is no longer the rejected bit")
        tmp.write(pkt)
        packet_path = tmp.name

    out = subprocess.check_output([str(bin_path), "--packet", packet_path], text=True)
    print(out, end="" if out.endswith("\n") else "\n")
    fails = [line for line in out.splitlines() if line.startswith("CHECK ") and "FAIL" in line]
    if fails or "CHECKS 0" not in out:
        raise SystemExit("c smoke checks failed")

    yaw90 = None
    packet_obs = None
    policies = []
    for line in out.splitlines():
        if line.startswith("OBS yaw90 "):
            yaw90 = _floats(line)[:8]
        elif line.startswith("OBS packet "):
            packet_obs = _floats(line)[:8]
        elif line.startswith("POLICY "):
            vals = _floats(line)
            policies.append((vals[:8], vals[8:]))
    if yaw90 is None or packet_obs is None or len(policies) != 3:
        raise SystemExit("smoke output missing OBS/POLICY lines")

    yaw = math.pi / 2
    expect_yaw = bridge.build_obs([0, 0], yaw, [0, 0], [1, 0], [0.25, -0.5])
    if not np.allclose(yaw90, expect_yaw, atol=1e-5):
        raise SystemExit(f"C yaw90 obs {yaw90} != python {expect_yaw}")

    expect_pkt = bridge.build_obs([0, 0], 0.0, [0, 0], [0.2, -0.4], [1.5, -1.5])
    if not np.allclose(packet_obs, expect_pkt, atol=1e-5):
        raise SystemExit(f"C packet obs {packet_obs} != python {expect_pkt}")

    saw_ref = False
    for obs, act in policies:
        got = policy_linear(obs, weights)
        if not np.allclose(act, got, atol=1e-5, rtol=1e-5):
            raise SystemExit(f"policy mismatch obs={obs} c={act} py={got}")
        if np.allclose(obs[:2], bridge.REF_RESET[:2], atol=1e-6):
            saw_ref = True
            tanh_act = policy_tanh_head(obs, weights)
            if np.allclose(got, tanh_act, atol=1e-3):
                raise SystemExit("linear head matched tanh(head); the lock would not see a squash")
    if not saw_ref:
        raise SystemExit("policy cases did not include the published reset sample")
    print("POLICY_PARITY PASS")


def check_udp(bin_path: Path) -> None:
    proc = subprocess.Popen(
        [str(bin_path), "--listen"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    banner: list[str] = []
    try:
        deadline = time.time() + 3
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                raise SystemExit("listener exited before LISTEN:\n" + "".join(banner))
            banner.append(line)
            if line.startswith("LISTEN"):
                break
        else:
            raise SystemExit("listener did not bind:\n" + "".join(banner))
        env_proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "sim" / "m0_nav_demo" / "sim_obs_bridge.py"),
                "--steps",
                "1",
                "--seed",
                "0",
                "--udp",
                "127.0.0.1:9000",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            rest, _ = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise SystemExit("listener did not consume the LCR1 packet")
        text = "".join(banner) + (rest or "")
        print(env_proc.stdout, end="" if env_proc.stdout.endswith("\n") else "\n")
        print(text, end="" if text.endswith("\n") else "\n")
        env_lines = [ln for ln in env_proc.stdout.splitlines() if ln.startswith("ENV_OBS ")]
        udp_lines = [ln for ln in text.splitlines() if ln.startswith("OBS udp ")]
        if not env_lines or not udp_lines:
            raise SystemExit("missing ENV_OBS or OBS udp")
        env_obs = _floats(env_lines[0])
        udp_obs = _floats(udp_lines[0])
        if not np.allclose(env_obs, udp_obs, atol=1e-5):
            raise SystemExit(f"UDP rebuild {udp_obs} != env {env_obs}")
        if proc.returncode != 0:
            raise SystemExit(f"listener exit {proc.returncode}")
        print("UDP_PARITY PASS")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def main() -> None:
    weights = load_policy()
    with tempfile.TemporaryDirectory() as td:
        bin_path = Path(td) / "nav_s3_smoke"
        compile_smoke(bin_path)
        check_program(bin_path, weights)
        check_udp(bin_path)
    print("X86_SMOKE PASS")


if __name__ == "__main__":
    main()
