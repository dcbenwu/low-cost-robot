"""Procedural MJCF for the front-loading washer scene.

Primitive geoms only: no meshes, no HDR, no remote assets. Newton 1.6
`ModelBuilder.add_mjcf` and MuJoCo can both load the string. The CPU task
tests do not execute this file as a physics engine.
"""

from __future__ import annotations

from .spec import (
    ARM_BASE,
    ARM_ELBOW_RANGE,
    ARM_L1,
    ARM_L2,
    ARM_L3,
    ARM_SHOULDER_RANGE,
    ARM_SHOULDER_Z,
    ARM_WRIST_RANGE,
    ARM_YAW_RANGE,
    BOTTLE_HOME,
    BUTTON_POS,
    DISPENSER_SLOTS,
    DISPENSERS,
    DOOR_HINGE,
    DOOR_RANGE,
    DRAWER_ORIGIN,
    DRAWER_RANGE,
    DRUM_CENTER,
    DRUM_RADIUS,
    GARMENTS,
    GRAVITY,
    KNOB_POS,
    LOCK_POS,
    MOUTH_LOCAL_Z,
)


def _rgba(rgba: tuple[float, float, float, float]) -> str:
    return " ".join(f"{c:.3f}" for c in rgba)


def build_mjcf() -> str:
    g = " ".join(f"{v:.5f}" for v in GRAVITY)
    garments = []
    for item in GARMENTS:
        px, py, pz = item["pos"]
        hx, hy, hz = item["half"]
        garments.append(
            f"""
    <body name="{item["name"]}" pos="{px:.4f} {py:.4f} {pz:.4f}">
      <freejoint name="{item["name"]}_free"/>
      <inertial pos="0 0 0" mass="{item["mass"]:.3f}" diaginertia="0.0004 0.0004 0.0002"/>
      <geom name="{item["name"]}_geom" type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}" mass="{item["mass"]:.3f}" rgba="{_rgba(item["rgba"])}" friction="1.0 0.05 0.01"/>
    </body>"""
        )
    cups = []
    colors = {"main": "0.20 0.55 0.85 1", "softener": "0.85 0.45 0.75 1", "bleach": "0.85 0.75 0.25 1"}
    for name, (lx, ly, lz) in DISPENSERS.items():
        cups.append(
            f'      <geom name="cup_{name}" type="cylinder" pos="{lx:.3f} {ly:.3f} {lz:.3f}" size="0.025 0.018" rgba="{colors[name]}"/>'
        )
        cups.append(
            f'      <site name="cup_{name}" pos="{lx:.3f} {ly:.3f} {lz + 0.02:.3f}" size="0.008"/>'
        )
    slots = []
    for name, (sx, sy, sz) in DISPENSER_SLOTS.items():
        slots.append(f'    <site name="slot_{name}" pos="{sx:.3f} {sy:.3f} {sz:.3f}" size="0.008" rgba="0.9 0.9 0.9 1"/>')

    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="lcr_laundry_washer">
  <!-- rigid garment proxies, not deformable cloth; detergent is not a fluid -->
  <!-- no textures, no external assets; gravity is the laundry baseline -->
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.01" gravity="{g}" integrator="implicitfast" iterations="20" ls_iterations="8"/>
  <visual>
    <headlight diffuse="0.85 0.85 0.85" ambient="0.35 0.35 0.35"/>
    <rgba haze="0.16 0.18 0.22 1"/>
  </visual>
  <asset>
    <material name="cabinet" rgba="0.90 0.91 0.93 1"/>
    <material name="drum" rgba="0.45 0.48 0.52 1"/>
    <material name="rubber" rgba="0.12 0.12 0.13 1"/>
  </asset>
  <worldbody>
    <light directional="true" diffuse="0.9 0.9 0.9" pos="1.2 -0.8 2.2" dir="-0.4 0.3 -1"/>
    <geom name="floor" type="plane" size="1.6 1.6 0.02" rgba="0.55 0.56 0.58 1"/>

    <geom name="cab_back" type="box" pos="-0.28 0 0.425" size="0.02 0.30 0.425" material="cabinet"/>
    <geom name="cab_left" type="box" pos="0 0.28 0.425" size="0.30 0.02 0.425" material="cabinet"/>
    <geom name="cab_right" type="box" pos="0 -0.28 0.425" size="0.30 0.02 0.425" material="cabinet"/>
    <geom name="cab_top" type="box" pos="0 0 0.83" size="0.30 0.30 0.02" material="cabinet"/>
    <geom name="cab_bottom" type="box" pos="0 0 0.04" size="0.30 0.30 0.04" material="cabinet"/>
    <geom name="cab_front_top" type="box" pos="0.29 0 0.74" size="0.015 0.30 0.11" material="cabinet"/>
    <geom name="cab_front_bot" type="box" pos="0.29 0 0.12" size="0.015 0.30 0.12" material="cabinet"/>
    <geom name="cab_front_left" type="box" pos="0.29 0.25 0.42" size="0.015 0.05 0.20" material="cabinet"/>
    <geom name="cab_front_right" type="box" pos="0.29 -0.25 0.42" size="0.015 0.05 0.20" material="cabinet"/>
    <geom name="panel" type="box" pos="0.312 0.20 0.64" size="0.008 0.09 0.12" rgba="0.22 0.24 0.28 1"/>
    <geom name="mark_off" type="box" pos="0.322 0.26 0.72" size="0.003 0.008 0.006" rgba="0.45 0.45 0.48 1" contype="0" conaffinity="0"/>
    <geom name="mark_cotton" type="box" pos="0.322 0.24 0.70" size="0.003 0.008 0.006" rgba="0.20 0.45 0.90 1" contype="0" conaffinity="0"/>
    <geom name="mark_synthetic" type="box" pos="0.322 0.22 0.68" size="0.003 0.008 0.006" rgba="0.20 0.70 0.45 1" contype="0" conaffinity="0"/>
    <geom name="mark_delicate" type="box" pos="0.322 0.20 0.66" size="0.003 0.008 0.006" rgba="0.85 0.55 0.75 1" contype="0" conaffinity="0"/>
    <geom name="mark_spin" type="box" pos="0.322 0.18 0.64" size="0.003 0.008 0.006" rgba="0.85 0.25 0.20 1" contype="0" conaffinity="0"/>

    <body name="door" pos="{DOOR_HINGE[0]:.3f} {DOOR_HINGE[1]:.3f} {DOOR_HINGE[2]:.3f}">
      <joint name="door_hinge" type="hinge" axis="0 0 -1" range="{DOOR_RANGE[0]} {DOOR_RANGE[1]}" damping="1.2" armature="0.02"/>
      <geom name="door_panel" type="box" pos="0.012 0.24 0" size="0.008 0.24 0.24" rgba="0.86 0.88 0.90 1"/>
      <geom name="door_rim" type="cylinder" pos="0.018 0.24 0.02" size="0.175 0.012" euler="0 1.5707963 0" rgba="0.72 0.76 0.80 1" contype="0" conaffinity="0"/>
      <geom name="door_glass" type="cylinder" pos="0.028 0.24 0.02" size="0.145 0.006" euler="0 1.5707963 0" rgba="0.12 0.20 0.28 0.55" contype="0" conaffinity="0"/>
      <geom name="door_handle" type="capsule" fromto="0.03 0.40 -0.05 0.03 0.40 0.07" size="0.012" material="rubber"/>
      <site name="door_handle" pos="0.03 0.42 0" size="0.01"/>
    </body>

    <body name="drawer" pos="{DRAWER_ORIGIN[0]:.3f} {DRAWER_ORIGIN[1]:.3f} {DRAWER_ORIGIN[2]:.3f}">
      <joint name="drawer_slide" type="slide" axis="1 0 0" range="{DRAWER_RANGE[0]} {DRAWER_RANGE[1]}" damping="4"/>
      <geom name="drawer_box" type="box" pos="0 0 0" size="0.09 0.15 0.028" rgba="0.70 0.78 0.86 1"/>
      <geom name="drawer_front" type="box" pos="0.10 0 0" size="0.012 0.16 0.04" rgba="0.88 0.89 0.92 1"/>
{chr(10).join(cups)}
      <site name="drawer_handle" pos="0.12 0 0" size="0.008"/>
    </body>

    <body name="drum" pos="{DRUM_CENTER[0]:.3f} {DRUM_CENTER[1]:.3f} {DRUM_CENTER[2]:.3f}">
      <joint name="drum_spin" type="hinge" axis="1 0 0" damping="0.05" armature="0.01"/>
      <geom name="drum_shell" type="cylinder" size="{DRUM_RADIUS:.3f} 0.12" euler="0 1.5707963 0" rgba="0.40 0.43 0.48 0.35" contype="0" conaffinity="0"/>
      <geom name="drum_stripe" type="box" pos="0 0.12 0" size="0.10 0.015 0.02" rgba="0.85 0.25 0.18 1" contype="0" conaffinity="0"/>
      <geom name="drum_floor" type="box" pos="0 0 -0.12" size="0.10 0.12 0.015" rgba="0.35 0.37 0.40 1" friction="1.2 0.05 0.01"/>
    </body>

    <body name="knob" pos="{KNOB_POS[0]:.3f} {KNOB_POS[1]:.3f} {KNOB_POS[2]:.3f}">
      <joint name="program_knob" type="hinge" axis="1 0 0" range="-0.2 3.4" damping="0.4"/>
      <geom name="knob_body" type="cylinder" size="0.028 0.012" euler="0 1.5707963 0" rgba="0.15 0.15 0.18 1"/>
      <geom name="knob_pointer" type="box" pos="0 0.02 0.02" size="0.008 0.006 0.016" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
      <site name="program_knob" pos="0.02 0 0" size="0.008"/>
    </body>
    <body name="start_button" pos="{BUTTON_POS[0]:.3f} {BUTTON_POS[1]:.3f} {BUTTON_POS[2]:.3f}">
      <joint name="start_button" type="slide" axis="1 0 0" range="-0.01 0" damping="0.2"/>
      <geom name="start_button_cap" type="cylinder" size="0.016 0.008" euler="0 1.5707963 0" rgba="0.15 0.65 0.25 1"/>
      <site name="start_button" pos="0.015 0 0" size="0.008"/>
    </body>
    <body name="door_lock" pos="{LOCK_POS[0]:.3f} {LOCK_POS[1]:.3f} {LOCK_POS[2]:.3f}">
      <joint name="lock_pin" type="slide" axis="1 0 0" range="0 0.025" damping="1"/>
      <geom name="lock_pin_geom" type="box" pos="0.01 0 0" size="0.012 0.01 0.01" rgba="0.95 0.72 0.15 1"/>
    </body>

    <geom name="basket_floor" type="box" pos="0.62 -0.42 0.03" size="0.16 0.14 0.03" rgba="0.45 0.32 0.18 1"/>
    <geom name="basket_px" type="box" pos="0.78 -0.42 0.10" size="0.012 0.14 0.08" rgba="0.45 0.32 0.18 1"/>
    <geom name="basket_nx" type="box" pos="0.46 -0.42 0.10" size="0.012 0.14 0.08" rgba="0.45 0.32 0.18 1"/>
    <geom name="basket_py" type="box" pos="0.62 -0.28 0.10" size="0.16 0.012 0.08" rgba="0.45 0.32 0.18 1"/>
    <geom name="basket_ny" type="box" pos="0.62 -0.56 0.10" size="0.16 0.012 0.08" rgba="0.45 0.32 0.18 1"/>
    <geom name="counter" type="box" pos="0.55 0.42 0.35" size="0.14 0.12 0.35" rgba="0.62 0.64 0.66 1"/>
{chr(10).join(slots)}
{chr(10).join(garments)}

    <body name="bottle" pos="{BOTTLE_HOME[0]:.3f} {BOTTLE_HOME[1]:.3f} {BOTTLE_HOME[2]:.3f}">
      <freejoint name="bottle_free"/>
      <inertial pos="0 0 0" mass="0.35" diaginertia="0.001 0.001 0.0004"/>
      <geom name="bottle_body" type="capsule" size="0.035 0.055" rgba="0.20 0.72 0.82 1" friction="1.0 0.05 0.01"/>
      <geom name="bottle_level" type="cylinder" pos="0 0 -0.01" size="0.028 0.04" rgba="0.25 0.45 0.95 1" contype="0" conaffinity="0"/>
      <site name="pour_mouth" pos="0 0 {MOUTH_LOCAL_Z:.3f}" size="0.008"/>
      <body name="cap" pos="0 0 0.10">
        <joint name="cap_hinge" type="hinge" axis="0 1 0" range="0 2.6" damping="0.05"/>
        <geom name="cap_geom" type="cylinder" size="0.038 0.012" rgba="0.12 0.14 0.16 1"/>
      </body>
    </body>

    <body name="arm_base" pos="{ARM_BASE[0]:.3f} {ARM_BASE[1]:.3f} {ARM_BASE[2]:.3f}">
      <geom name="arm_pedestal" type="cylinder" pos="0 0 0.16" size="0.06 0.16" rgba="0.25 0.27 0.30 1"/>
      <body name="arm_yaw_link" pos="0 0 {ARM_SHOULDER_Z:.3f}">
          <joint name="arm_yaw" type="hinge" axis="0 0 1" range="{ARM_YAW_RANGE[0]} {ARM_YAW_RANGE[1]}" damping="1.5" armature="0.05"/>
        <body name="arm_shoulder_link">
          <joint name="arm_shoulder" type="hinge" axis="0 1 0" range="{ARM_SHOULDER_RANGE[0]} {ARM_SHOULDER_RANGE[1]}" damping="1.5" armature="0.05"/>
          <geom name="arm_upper" type="capsule" fromto="0 0 0 {ARM_L1:.3f} 0 0" size="0.028" rgba="0.95 0.55 0.15 1"/>
          <body name="arm_elbow_link" pos="{ARM_L1:.3f} 0 0">
            <joint name="arm_elbow" type="hinge" axis="0 1 0" range="{ARM_ELBOW_RANGE[0]} {ARM_ELBOW_RANGE[1]}" damping="1.0" armature="0.03"/>
            <geom name="arm_fore" type="capsule" fromto="0 0 0 {ARM_L2:.3f} 0 0" size="0.024" rgba="0.95 0.55 0.15 1"/>
            <body name="arm_wrist_link" pos="{ARM_L2:.3f} 0 0">
              <joint name="arm_wrist" type="hinge" axis="0 1 0" range="{ARM_WRIST_RANGE[0]} {ARM_WRIST_RANGE[1]}" damping="0.6" armature="0.02"/>
              <geom name="arm_tool" type="capsule" fromto="0 0 0 {ARM_L3:.3f} 0 0" size="0.018" rgba="0.85 0.85 0.88 1"/>
              <body name="gripper" pos="{ARM_L3:.3f} 0 0">
                <joint name="gripper_slide" type="slide" axis="0 0 1" range="0 0.03" damping="0.5"/>
                <geom name="finger_a" type="box" pos="0 0.018 0" size="0.018 0.006 0.012" rgba="0.15 0.15 0.16 1"/>
                <geom name="finger_b" type="box" pos="0 -0.018 0" size="0.018 0.006 0.012" rgba="0.15 0.15 0.16 1"/>
                <site name="gripper_site" pos="0 0 0" size="0.01"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_door" joint="door_hinge" kp="40" ctrlrange="{DOOR_RANGE[0]} {DOOR_RANGE[1]}" forcerange="-8 8"/>
    <position name="act_drawer" joint="drawer_slide" kp="80" ctrlrange="{DRAWER_RANGE[0]} {DRAWER_RANGE[1]}" forcerange="-12 12"/>
    <position name="act_cap" joint="cap_hinge" kp="8" ctrlrange="0 2.6" forcerange="-1.5 1.5"/>
    <position name="act_knob" joint="program_knob" kp="6" ctrlrange="-0.2 3.4" forcerange="-1 1"/>
    <position name="act_button" joint="start_button" kp="20" ctrlrange="-0.01 0" forcerange="-2 2"/>
    <position name="act_lock" joint="lock_pin" kp="40" ctrlrange="0 0.025" forcerange="-6 6"/>
    <velocity name="act_drum" joint="drum_spin" kv="1.5" ctrlrange="-4 4" forcerange="-4 4"/>
    <position name="act_yaw" joint="arm_yaw" kp="60" ctrlrange="{ARM_YAW_RANGE[0]} {ARM_YAW_RANGE[1]}" forcerange="-20 20"/>
    <position name="act_shoulder" joint="arm_shoulder" kp="60" ctrlrange="{ARM_SHOULDER_RANGE[0]} {ARM_SHOULDER_RANGE[1]}" forcerange="-20 20"/>
    <position name="act_elbow" joint="arm_elbow" kp="40" ctrlrange="{ARM_ELBOW_RANGE[0]} {ARM_ELBOW_RANGE[1]}" forcerange="-12 12"/>
    <position name="act_wrist" joint="arm_wrist" kp="20" ctrlrange="{ARM_WRIST_RANGE[0]} {ARM_WRIST_RANGE[1]}" forcerange="-6 6"/>
    <position name="act_grip" joint="gripper_slide" kp="20" ctrlrange="0 0.03" forcerange="-4 4"/>
  </actuator>
</mujoco>
"""
    return xml


def write_mjcf(path: str) -> str:
    text = build_mjcf()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text
