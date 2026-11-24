"""Build an original prompt-driven character using the pinned YSM AI Engine."""
import json
import math
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from ysm_engine.builder import YSMModel, make_vgradient_patch, parse_color, shade_color
from ysm_engine.package import package_all
from ysm_engine.renderer import BedrockScene, render_view

ROOT = Path(__file__).resolve().parent
HAIR = {"bob", "long", "braids", "ponytail", "hood"}
OUTFITS = {"ranger", "armor", "coat", "dress", "streetwear"}
PALETTE = {"skin", "hair", "eyes", "primary", "secondary", "accent", "boots"}
FACES = {"Head", "HairBack", "Body", "RightArm", "LeftArm", "RightLeg", "LeftLeg", "Skirt"}


def read_design(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1][:-3].strip()
    spec = json.loads(text)
    if not isinstance(spec, dict) or set(spec) != {"name", "hair", "outfit", "palette", "details"}:
        raise ValueError("expected name, hair, outfit, palette and details")
    if not isinstance(spec["name"], str) or not 1 <= len(spec["name"].strip()) <= 80:
        raise ValueError("name must have 1 to 80 characters")
    if spec["hair"] not in HAIR or spec["outfit"] not in OUTFITS:
        raise ValueError("unknown hair or outfit silhouette")
    palette = spec["palette"]
    if not isinstance(palette, dict) or set(palette) != PALETTE:
        raise ValueError("expected seven palette colors")
    if any(not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color) for color in palette.values()):
        raise ValueError("palette colors must be #RRGGBB")
    details = spec["details"]
    if not isinstance(details, list) or len(details) > 20:
        raise ValueError("expected at most 20 accessory parts")
    for detail in details:
        if not isinstance(detail, dict) or set(detail) != {"bone", "origin", "size", "rotation", "color"}:
            raise ValueError("bad accessory fields")
        if detail["bone"] not in FACES or detail["color"] not in PALETTE:
            raise ValueError("unknown accessory color or bone")
        for field, low, high in (("origin", -12, 40), ("size", 0.12, 14), ("rotation", -65, 65)):
            values = detail[field]
            if not isinstance(values, list) or len(values) != 3 or any(
                type(x) not in (int, float) or not math.isfinite(x) or not low <= x <= high for x in values
            ):
                raise ValueError(f"accessory {field} is out of range")
    return spec


def rig(model):
    bones = [
        ("All", None, (0, 0, 0)), ("Waist", "All", (0, 14, 0)),
        ("Body", "Waist", (0, 14, 0)), ("AllHead", "Body", (0, 24.3, 0)),
        ("Head", "AllHead", (0, 24.3, 0)), ("ViewLocator", "Head", (0, 29, 0)),
        ("EyeRight", "Head", (-1.28, 28.4, -3.12)), ("EyeLeft", "Head", (1.28, 28.4, -3.12)),
        ("HairBack", "Head", (0, 31.5, 2.65)),
        ("HairBackLower", "HairBack", (0, 25, 2.9)),
        ("HairSideRight", "Head", (-3.2, 30, 0)), ("HairSideLeft", "Head", (3.2, 30, 0)),
        ("Skirt", "Waist", (0, 15, 0)),
        ("SkirtFront", "Skirt", (0, 15, -1.7)), ("SkirtBack", "Skirt", (0, 15, 1.7)),
        ("SkirtTailRight", "Skirt", (-2, 15, 0)), ("SkirtTailLeft", "Skirt", (2, 15, 0)),
        ("ElytraLocator", "Body", (0, 23, 1.8)), ("BackpackLocator", "Body", (0, 20, 1.8)),
    ]
    for side, sign in (("Right", -1), ("Left", 1)):
        x = 3.65 * sign
        bones.extend([
            (side + "Arm", "Body", (x, 23.2, 0)),
            (side + "Hand", side + "Arm", (x, 16.7, 0)),
            (side + "HandLocator", side + "Hand", (x, 13.35, -0.4)),
            (side + "Leg", "Waist", (1.25 * sign, 14, 0)),
            (side + "ShoulderLocator", "Body", (x, 24, 0)),
        ])
    for bone, parent, pivot in bones:
        model.add_bone(bone, parent, pivot)


def sculpt(model, palette):
    def block(bone, origin, size, color, rotation=None, painter=None, scale=None):
        pigment = palette[color] if color in palette else color
        model.add_cube(
            bone, origin, size, rotation=rotation, face_painter=painter, uv_scale=scale,
            vgradient3=(shade_color(pigment, 1.14), pigment, shade_color(pigment, 0.62)),
        )
    return block


def face_patch(color):
    def paint(face, w, h):
        lip = shade_color(color, 0.72)
        pixels = make_vgradient_patch(w, h, shade_color(color, 0.79), shade_color(color, 0.9), mid_color=color)
        if face == "north":
            y = int(h * 0.78)
            pixels[y:y + max(1, h // 36), int(w * 0.44):int(w * 0.56)] = lip
            cheek = (np.array(parse_color(color)) * 0.78 + np.array((208, 122, 122, 255)) * 0.22).astype(np.uint8)
            for x in (0.21, 0.74):
                pixels[int(h * 0.59):int(h * 0.66), int(w*x):int(w*(x + 0.12))] = cheek
        return pixels
    return paint


def eye_patch(iris):
    def paint(_face, w, h):
        y, x = np.mgrid[0:h, 0:w]
        x = (x + 0.5) / w
        y = (y + 0.5) / h
        image = make_vgradient_patch(w, h, "#B4A2AB", "#FFFBF6")
        inside = ((x - 0.52)/0.29)**2 + ((y - 0.52)/0.54)**2 < 1
        glow = make_vgradient_patch(w, h, shade_color(iris, 0.38), shade_color(iris, 1.4))
        image[inside] = glow[inside]
        image[((x - 0.53)/0.117)**2 + ((y - 0.49)/0.31)**2 < 1] = (31, 24, 34, 255)
        image[((x - 0.43)/0.10)**2 + ((y - 0.38)/0.14)**2 < 1] = (255, 250, 255, 255)
        image[y < 0.22 + (x - 0.50)**2 * 0.6] = (42, 29, 43, 255)
        return image
    return paint


def build_head(block, palette, hair):
    block("Head", (-2.9, 24.7, -3.1), (5.8, 6.0, 5.9), "skin", painter=face_patch(palette["skin"]), scale=8)
    block("Head", (-2.55, 24.4, -2.82), (5.1, 1.1, 5.35), "skin")
    for side, sign in (("Right", -1), ("Left", 1)):
        eye_origin = -2.36 if sign < 0 else 0.31
        block("Eye" + side, (eye_origin, 27.7, -3.168), (2.06, 1.38, 0.07),
              "eyes", painter=eye_patch(palette["eyes"]), scale=12)
        block("Head", (eye_origin, 29.22, -3.155), (1.92, 0.13, 0.07), "hair", (0, 0, sign * 6))
        block("Head", (-3.09 if sign < 0 else 2.73, 26.4, -0.2), (0.36, 1.42, 1.2), "skin")

    hood = hair == "hood"
    crown = "primary" if hood else "hair"
    for i, (width, depth) in enumerate(((6.94, 6.73), (6.62, 6.42), (5.98, 5.83), (4.8, 4.92))):
        block("HairBack", (-width/2, 30.28 + i*0.51, -depth/2 + 0.23),
              (width, 0.64, depth), crown)
    block("HairBack", (-3.32, 27.1, 1.92), (6.64, 3.85, 1.58), crown)
    for i in range(8):
        x = (i-3.5)*0.76
        length = 2.35 - abs(x)*0.18
        block("HairBackLower", (x-0.37, 27.31-length, 2.35), (0.79, length, 0.78),
              crown, (6, 0, x*3))
    for i, x in enumerate((-2.82, -2.09, -1.35, -0.57, 0.19, 0.94, 1.69, 2.39)):
        length = (2.05, 1.56, 1.32, 1.12, 1.04, 1.24, 1.51, 2.21)[i]
        block("Head", (x, 31.12-length, -3.48), (0.83, length, 0.65), "hair", (3, 0, (x-0.35)*5))
    hair_length = {"bob": 3.25, "long": 8.35, "braids": 7.15, "ponytail": 3.55, "hood": 2.75}[hair]
    for side, sign in (("Right", -1), ("Left", 1)):
        bone = "HairSide" + side
        for i in range(5):
            x = sign * (2.86 + i*0.11)
            height = hair_length - i*0.24
            block(bone, (x - 0.38, 30.18-height, -2.01+i*0.68), (0.77, height, 0.78),
                  crown, (i * 2, 0, sign * (4+i*1.7)))
        if hair == "braids":
            for i in range(7):
                width = 0.9-i*0.075
                x = sign*(2.93+i*0.07)
                block(bone, (x-width/2, 27.4-i*0.92, -2.33), (width, 1.05, 0.77),
                      "hair", (0, 0, sign*((-1)**i)*14))
            block(bone, (sign*3.34-0.4, 21.28, -2.38), (0.82, 0.31, 0.83), "accent")
    if hair in {"long", "ponytail"}:
        for i in range(8):
            x = (i-3.5)*0.71
            length = (9.3 if hair == "long" else 6.7) - abs(x)*1.1
            block("HairBackLower", (x, 29.3-length, 2.62+abs(x)*0.17),
                  (0.76, length, 0.86), "hair", (7, 0, x*2.7))
        if hair == "ponytail":
            block("HairBack", (-0.9, 28.1, 2.8), (1.8, 1.04, 1.28), "accent")
    if hood:
        for i, x in enumerate((-2.16, 1.37)):
            block("HairBack", (x, 33.0, -0.8), (0.9, 1.75, 1.18), "primary", (-5, 0, -10 if i else 10))
            block("HairBack", (x+0.19, 33.28, -0.87), (0.49, 1.14, 0.14), "secondary")


def build_outfit(block, outfit):
    block("Body", (-2.46, 14.2, -1.56), (4.92, 6.34, 3.12), "primary")
    block("Body", (-2.87, 20.35, -1.7), (5.74, 3.42, 3.4), "primary")
    block("Body", (-0.83, 23.2, -0.87), (1.66, 1.68, 1.74), "skin")
    block("Body", (-1.17, 22.92, -1.1), (2.34, 0.58, 2.2), "secondary")
    block("Body", (-2.55, 15.35, -1.81), (5.1, 0.78, 3.62), "boots")
    block("Body", (-0.68, 15.27, -2.02), (1.36, 0.93, 0.34), "accent")
    block("Body", (-0.42, 15.46, -2.07), (0.84, 0.53, 0.37), "boots")
    for i in range(4):
        block("Body", (-0.24, 17.05+i*1.16, -1.76), (0.48, 0.36, 0.23), "accent")
    for side, sign in (("Right", -1), ("Left", 1)):
        x = 3.65 * sign
        arm = side + "Arm"
        block(arm, (x-0.94, 17.1, -0.96), (1.88, 5.97, 1.92), "primary", (0, 0, sign*2))
        block(arm, (x-1.04, 16.9, -1.08), (2.08, 0.68, 2.16), "secondary")
        block(side + "Hand", (x-0.73, 13.55, -0.71), (1.46, 3.45, 1.42), "skin")
        block(side + "Hand", (x-0.34, 13.29, -0.83), (0.68, 1.76, 0.52), "boots")
        leg = side + "Leg"
        lx = sign*1.25
        block(leg, (lx-0.99, 4.0, -1.06), (1.98, 10.15, 2.12), "secondary")
        block(leg, (lx-1.03, 1.04, -1.12), (2.06, 4.72, 2.24), "boots")
        block(leg, (lx-1.13, 0.26, -1.68), (2.26, 1.42, 2.93), "boots")
        block(leg, (lx-1.16, 0, -1.75), (2.32, 0.36, 3.07), "secondary")
        block(leg, (lx-0.91, 4.27, -1.24), (1.82, 0.37, 0.24), "accent")
        if outfit in {"armor", "ranger"}:
            for j in range(3):
                block(arm, (x-1.16+j*0.09, 20.98+j*0.83, -1.18+j*0.03),
                      (2.32-j*0.18, 1.12, 2.36-j*0.06), "secondary", (0, 0, sign * (5+j*3)))
        if outfit in {"coat", "streetwear", "dress"}:
            for j in range(3):
                block("Body", (sign*(0.61+j*0.35)-0.35, 20.45+j*0.75, -1.9),
                      (0.7, 1.08, 0.29), "secondary", (0, 0, sign*22))
    if outfit == "ranger":
        for i in range(9):
            x = (i-4)*0.54
            block("Body", (x-0.29, 15.4, 1.65+abs(x)*0.06), (0.61, 7.5-abs(x)*0.32, 0.57),
                  "secondary", (0, 0, x*2.8))
        block("Body", (-0.48, 22.2, -1.94), (0.96, 0.52, 0.29), "accent", (0, 0, 12))
    if outfit in {"ranger", "coat", "dress"}:
        depth = {"ranger": 5.5, "coat": 7.2, "dress": 6.4}[outfit]
        for i in range(10):
            x = (i-4.5)*0.53
            bone = "SkirtBack" if outfit == "ranger" else ("SkirtTailRight" if x < 0 else "SkirtTailLeft")
            block(bone, (x, 15.15-depth-abs(x)*0.11, 1.27), (0.58, depth, 0.66),
                  "secondary" if outfit == "ranger" else "primary", (8, 0, x*3))
        if outfit == "dress":
            for i in range(8):
                x = (i-3.5)*0.64
                block("SkirtFront", (x, 10.1, -1.89), (0.73, 5.7, 0.61),
                      "primary", (-10, 0, x*4))
    if outfit == "armor":
        for i in range(5):
            width = 4.74-i*0.31
            block("Body", (-width/2, 18.9+i*0.82, -1.97-i*0.024),
                  (width, 0.95, 0.48), "secondary" if i%2 else "primary")
        block("Body", (-0.59, 21.4, -2.57), (1.18, 1.18, 0.28), "accent", (0, 0, 45))
    if outfit == "streetwear":
        block("Body", (-2.12, 17.0, -1.94), (4.24, 2.5, 0.4), "secondary")
        for x in (-0.55, 0.34):
            block("Body", (x, 19.72, -1.94), (0.17, 3.37, 0.24), "accent", (0, 0, x*8))


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def build(spec, pack):
    model = YSMModel("geometry.generated", texture_width=512, texture_height=512, texels_per_unit=3)
    rig(model)
    block = sculpt(model, spec["palette"])
    build_head(block, spec["palette"], spec["hair"])
    build_outfit(block, spec["outfit"])
    for detail in spec["details"]:
        block(detail["bone"], detail["origin"], detail["size"], detail["color"], detail["rotation"])
    model.export_main_json(pack / "models/main.json")
    model.export_arm_json(pack / "models/arm.json")
    model.export_texture(pack / "textures/default.png")
    for kind in ("main", "arm"):
        animations = json.loads((ROOT / f"animations/{kind}.animation.json").read_text(encoding="utf8"))
        write_json(pack / f"animations/{kind}.animation.json", animations)
    write_json(pack / "ysm.json", {
        "spec": 2,
        "metadata": {"name": spec["name"], "license": {"type": "Unspecified"}, "authors": [{"name": "Lilyco Studio", "role": "Procedural modeling"}]},
        "properties": {"height_scale": 0.94, "width_scale": 0.92, "preview_animation": "gui"},
        "files": {"player": {"model": {"main": "models/main.json", "arm": "models/arm.json"},
                             "animation": {"main": "animations/main.animation.json", "arm": "animations/arm.animation.json"},
                             "texture": [{"uv": "textures/default.png"}]}},
    })


def preview(pack):
    scene = BedrockScene(pack / "models/main.json", pack / "textures/default.png")
    preview = Image.new("RGB", (630, 340))
    for i, yaw in enumerate((0, 85, 180)):
        view = render_view(scene, width=210, height=340, yaw_deg=yaw, pitch_deg=4,
                           target=(0, 16.4, 0), ortho_scale=37, bg_color=(228, 232, 239))
        preview.paste(view, (i*210, 0))
    path = pack / "renders"
    path.mkdir()
    preview.save(path / "turnaround.png")


def main():
    spec = read_design(sys.stdin.read())
    with TemporaryDirectory(prefix="lilyco-ysm-") as directory:
        pack = Path(directory) / "character"
        build(spec, pack)
        preview(pack)
        # Validation errors go to stderr. Stdout is reserved for the binary artifact.
        archive, _ = package_all(pack, export_bb=False, quiet=True)
        sys.stdout.buffer.write(archive.read_bytes())


if __name__ == "__main__":
    main()
