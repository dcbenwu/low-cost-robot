"""Software rasterizer of the simulated rigid state.

Frames are produced from the body positions the CPU scene actually has.
They are not NewtonGL frames and they are not a learned-policy video.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

from .arm import chain
from .geom import door_handle
from .runtime_config import EYE, LOOKAT
from .spec import DRUM_CENTER, FRONT_X, GARMENTS

# 5x7 glyphs, rows of 5 bits. Enough for the on-frame labels.
_GLYPHS = {
    " ": [0, 0, 0, 0, 0, 0, 0],
    "-": [0, 0, 0, 31, 0, 0, 0],
    "/": [1, 2, 4, 8, 16, 0, 0],
    ":": [0, 4, 4, 0, 4, 4, 0],
    ".": [0, 0, 0, 0, 0, 4, 4],
    "+": [0, 4, 4, 31, 4, 4, 0],
    "0": [14, 17, 19, 21, 25, 17, 14],
    "1": [4, 12, 4, 4, 4, 4, 14],
    "2": [14, 17, 1, 6, 8, 16, 31],
    "3": [14, 17, 1, 6, 1, 17, 14],
    "4": [2, 6, 10, 18, 31, 2, 2],
    "5": [31, 16, 30, 1, 1, 17, 14],
    "6": [6, 8, 16, 30, 17, 17, 14],
    "7": [31, 1, 2, 4, 8, 8, 8],
    "8": [14, 17, 17, 14, 17, 17, 14],
    "9": [14, 17, 17, 15, 1, 2, 12],
    "A": [14, 17, 17, 31, 17, 17, 17],
    "B": [30, 17, 17, 30, 17, 17, 30],
    "C": [14, 17, 16, 16, 16, 17, 14],
    "D": [30, 17, 17, 17, 17, 17, 30],
    "E": [31, 16, 16, 30, 16, 16, 31],
    "F": [31, 16, 16, 30, 16, 16, 16],
    "G": [14, 17, 16, 23, 17, 17, 14],
    "H": [17, 17, 17, 31, 17, 17, 17],
    "I": [14, 4, 4, 4, 4, 4, 14],
    "K": [17, 18, 20, 24, 20, 18, 17],
    "L": [16, 16, 16, 16, 16, 16, 31],
    "M": [17, 27, 21, 21, 17, 17, 17],
    "N": [17, 17, 25, 21, 19, 17, 17],
    "O": [14, 17, 17, 17, 17, 17, 14],
    "P": [30, 17, 17, 30, 16, 16, 16],
    "R": [30, 17, 17, 30, 20, 18, 17],
    "S": [14, 17, 16, 14, 1, 17, 14],
    "T": [31, 4, 4, 4, 4, 4, 4],
    "U": [17, 17, 17, 17, 17, 17, 14],
    "V": [17, 17, 17, 17, 10, 10, 4],
    "W": [17, 17, 17, 21, 21, 21, 10],
    "X": [17, 17, 10, 4, 10, 17, 17],
    "Y": [17, 17, 10, 4, 4, 4, 4],
}


def write_png(path: str, rgb: np.ndarray) -> None:
    image = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("rgb image required")
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw, 6))
    payload += chunk(b"IEND", b"")
    with open(path, "wb") as handle:
        handle.write(payload)


def _basis(eye, lookat):
    forward = np.array(lookat, dtype=float) - np.array(eye, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    if up[2] < 0.0:
        up = -up
        right = -right
    return right, up, forward


def _project(points, eye, right, up, forward, width, height):
    rel = np.asarray(points, dtype=float) - np.asarray(eye, dtype=float)
    depth = rel @ forward
    x = rel @ right
    y = rel @ up
    focal = 0.85 * width
    sx = width * 0.5 + focal * x / np.maximum(depth, 1e-3)
    sy = height * 0.42 - focal * y / np.maximum(depth, 1e-3)
    return sx, sy, depth


def _disc(img, x, y, radius, color, zbuf, depth):
    h, w, _ = img.shape
    r = max(int(radius), 1)
    x0 = max(int(x) - r, 0)
    x1 = min(int(x) + r + 1, w)
    y0 = max(int(y) - r, 0)
    y1 = min(int(y) + r + 1, h)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    view = zbuf[y0:y1, x0:x1]
    paint = mask & (depth < view)
    view[paint] = depth
    img[y0:y1, x0:x1][paint] = color


def _text(img, x, y, text, color=(255, 255, 255)):
    cursor = x
    for ch in text.upper():
        rows = _GLYPHS.get(ch)
        if rows is None:
            cursor += 4
            continue
        for row, bits in enumerate(rows):
            for col in range(5):
                if bits & (1 << (4 - col)):
                    px = cursor + col
                    py = y + row
                    if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                        img[py, px] = color
        cursor += 6


def render_state(scene, width: int = 320, height: int = 180) -> np.ndarray:
    """Rasterize the current CPU scene. Returns uint8 HxWx3."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (32, 36, 44)
    # Floor band so an empty projection is still a scene, not a blank card.
    img[int(height * 0.55) :, :] = (70, 74, 78)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    eye = EYE
    right, up, forward = _basis(eye, LOOKAT)

    def draw(pos, radius, color):
        sx, sy, depth = _project([pos], eye, right, up, forward, width, height)
        if depth[0] <= 0.05:
            return
        _disc(img, sx[0], sy[0], radius * 220.0 / depth[0], color, zbuf, float(depth[0]))

    draw((0.0, 0.0, 0.02), 0.9, (90, 94, 98))
    draw((0.0, 0.0, 0.42), 0.34, (210, 214, 218))
    draw(DRUM_CENTER, 0.16, (70, 76, 84))
    # Drum stripe rotates with the spin joint so a running machine is visible.
    stripe_a = scene.q["drum_spin"]
    draw(
        (
            DRUM_CENTER[0],
            DRUM_CENTER[1] + 0.10 * np.cos(stripe_a),
            DRUM_CENTER[2] + 0.10 * np.sin(stripe_a),
        ),
        0.03,
        (210, 70, 55),
    )
    hinge_color = (40, 140, 70) if scene.running else (180, 186, 192)
    draw(door_handle(scene.q["door_hinge"]), 0.035, hinge_color)
    draw((FRONT_X + scene.q["drawer_slide"], 0.0, 0.74), 0.06, (120, 150, 175))
    draw((0.62, -0.42, 0.08), 0.12, (120, 84, 48))
    colors = {item["name"]: tuple(int(c * 255) for c in item["rgba"][:3]) for item in GARMENTS}
    for name, body in scene.garments.items():
        draw(body.pos, 0.045, colors[name])
    tilt = scene.q["bottle_tilt"]
    draw(scene.bottle.pos, 0.04, (40, 170, 190))
    mouth = (
        scene.bottle.pos[0] - 0.09 * np.sin(tilt),
        scene.bottle.pos[1],
        scene.bottle.pos[2] + 0.09 * np.cos(tilt),
    )
    draw(mouth, 0.02, (20, 40, 50) if scene.q["cap_hinge"] < 0.5 else (230, 200, 40))
    pts = chain(tuple(scene.q[name] for name in ("arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist")))
    for point in pts:
        draw(point, 0.03, (230, 140, 40))

    mode = "SCRIPTED KINEMATIC / NOT LEARNED" if scene.kinematic else "PHYSICAL ACTUATORS / NOT A LEARNED POLICY"
    _text(img, 4, 4, mode, (255, 230, 120))
    _text(img, 4, 14, "RIGID GARMENT PROXIES - NOT DEFORMABLE CLOTH", (220, 220, 220))
    _text(img, 4, 24, "DETERGENT POSE+DOSE - NOT FLUID", (180, 220, 255))
    _text(img, 4, 34, "NO TRAINED POLICY", (255, 160, 160))
    state = "WASHING" if scene.running else "IDLE"
    _text(
        img,
        4,
        height - 22,
        f"STEP {scene.step_count} DOOR {scene.q['door_hinge']:.2f} {state} G {scene.gravity[2]:.2f}",
        (255, 255, 255),
    )
    return img
