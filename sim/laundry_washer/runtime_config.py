"""Supported runtime for the one-env laundry baseline.

These numbers are the contract for uDoge. The GPU path is Newton 1.6
SolverMuJoCo with one world and a 640x360 NewtonGL framebuffer. It does not
start Kit RTX, does not load HDR/dome textures, and does not use the Franka
lift checkpoint. ComfyUI is left resident; this scene is sized for the
remaining ~10.6 GB, not for a full Isaac Sim viewport.

CPU task tests do not import this module's GPU stack. They only check that
the published numbers stay small and that gravity is Earth-normal.
"""

from __future__ import annotations

# Newton 1.6 / mujoco_warp, one world. Not the 64-env lift task.
NUM_WORLDS = 1
SOLVER = "newton.solvers.SolverMuJoCo"
SOLVER_BACKEND = "mujoco_warp"
USE_MUJOCO_CPU = False
USE_CUDA_GRAPH = False
NJMAX = 80
NCONMAX = 48
ITERATIONS = 20
LS_ITERATIONS = 8
SIM_DT = 1.0 / 120.0

# NewtonGL rasterizer. Same eye/lookat fields NewtonGLVisualizerCfg inherits
# from VisualizerCfg at Isaac Lab develop 93d7bef. Pitch/yaw match
# isaaclab_visualizers.newton._eye_lookat_to_pitch_yaw (degrees, Z-up).
VISUALIZER = "newton.viewer.ViewerGL"
ISAAC_VISUALIZER_CFG = "isaaclab_visualizers.newton.NewtonGLVisualizerCfg"
# Recorder source that produced real frames on this machine (alias of newton_gl).
VIDEO_SOURCE = "visualizer:newton"
VISUALIZER_TYPE = "newton_gl"
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 360
# Front three-quarter view: door and drum opening, basket on -Y, bottle on +Y,
# drawer, knob and start button. Not a close crop of one part.
EYE = (1.85, -0.95, 1.05)
LOOKAT = (0.20, -0.02, 0.48)
BACKGROUND_COLOR = (0.16, 0.18, 0.22)
STREAMING_VIEW = False
ENABLE_SHADOWS = False
ENABLE_SKY = False
SHOW_PARTICLES = False
ENABLE_MARKERS = False
ENABLE_LIVE_PLOTS = False
TEXTURES = "none — primitive rgba geoms only; no HDR, no dome latlong, no USD sky"

# Explicit baseline. The lift-task ADR schedule at difficulty 0 writes
# gravity (0, 0, 0). This environment does not load that schedule.
GRAVITY = (0.0, 0.0, -9.81)

# pyglet binds X11 on the first viewer import. DISPLAY is empty over SSH.
PYGLET_HEADLESS = True
PYGLET_HEADLESS_DEVICE = 0

LABEL_SCRIPTED = "scripted environment validation / NOT learned robotic manipulation"
LABEL_PHYSICAL = "physical actuator mode / NOT a learned policy"
LABEL_CLOTH = "rigid garment proxies — not deformable cloth"
LABEL_FLUID = "detergent pose+dose abstraction — not fluid"
LABEL_NO_POLICY = "no trained policy loaded"


def eye_lookat_to_pitch_yaw(
    eye: tuple[float, float, float] = EYE,
    lookat: tuple[float, float, float] = LOOKAT,
) -> tuple[float, float]:
    """Pitch and yaw in degrees, Z-up, yaw from +X. Matches Isaac Lab 93d7bef."""
    import math

    dx = lookat[0] - eye[0]
    dy = lookat[1] - eye[1]
    dz = lookat[2] - eye[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-8:
        return 0.0, 0.0
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, dz / length))))
    yaw = math.degrees(math.atan2(dy, dx))
    return pitch, yaw


def newton_gl_cfg_kwargs() -> dict:
    """Keyword arguments for NewtonGLVisualizerCfg at Isaac Lab 93d7bef."""
    return {
        "headless": True,
        "window_width": WINDOW_WIDTH,
        "window_height": WINDOW_HEIGHT,
        "eye": EYE,
        "lookat": LOOKAT,
        "background_color": BACKGROUND_COLOR,
        "streaming_view": STREAMING_VIEW,
        "enable_shadows": ENABLE_SHADOWS,
        "enable_sky": ENABLE_SKY,
        "show_particles": SHOW_PARTICLES,
        "enable_markers": ENABLE_MARKERS,
        "enable_live_plots": ENABLE_LIVE_PLOTS,
        "enable_picking": False,
        "max_visible_envs": NUM_WORLDS,
        "update_frequency": 1,
    }


def supported_summary() -> dict:
    pitch, yaw = eye_lookat_to_pitch_yaw()
    return {
        "num_worlds": NUM_WORLDS,
        "solver": SOLVER,
        "solver_backend": SOLVER_BACKEND,
        "use_mujoco_cpu": USE_MUJOCO_CPU,
        "use_cuda_graph": USE_CUDA_GRAPH,
        "njmax": NJMAX,
        "nconmax": NCONMAX,
        "iterations": ITERATIONS,
        "ls_iterations": LS_ITERATIONS,
        "sim_dt": SIM_DT,
        "gravity": list(GRAVITY),
        "visualizer": VISUALIZER,
        "isaac_visualizer_cfg": ISAAC_VISUALIZER_CFG,
        "visualizer_type": VISUALIZER_TYPE,
        "video_source": VIDEO_SOURCE,
        "window": [WINDOW_WIDTH, WINDOW_HEIGHT],
        "eye": list(EYE),
        "lookat": list(LOOKAT),
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "background_color": list(BACKGROUND_COLOR),
        "streaming_view": STREAMING_VIEW,
        "enable_shadows": ENABLE_SHADOWS,
        "enable_sky": ENABLE_SKY,
        "textures": TEXTURES,
        "pyglet_headless": PYGLET_HEADLESS,
        "pyglet_headless_device": PYGLET_HEADLESS_DEVICE,
        "vram_note": (
            "One world, 640x360, no HDR. Leave ComfyUI resident. "
            "Do not launch a Kit RTX viewport and do not clone the 64-env lift scene."
        ),
    }
