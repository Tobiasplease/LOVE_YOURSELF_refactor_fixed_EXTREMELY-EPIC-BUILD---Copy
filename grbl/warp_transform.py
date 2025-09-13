# import matplotlib.pyplot as plt
import math
import re

# Load robot arm dimensions from config file
import json
import os

def load_warp_config():
    """Load warp transform configuration from JSON file"""
    config_path = os.path.join(os.path.dirname(__file__), "warp_config.json")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"[WARN] Warp config not found at {config_path}, using default values")
        return None

# Load configuration or use defaults
warp_config = load_warp_config()
if warp_config and "robot_arm_dimensions" in warp_config:
    arm_dims = warp_config["robot_arm_dimensions"]
    biceps = arm_dims.get("biceps", 100)
    underarm = arm_dims.get("underarm", 100)
    tendon_biceps = arm_dims.get("tendon_biceps", 50)
    tendon_underarm = arm_dims.get("tendon_underarm", 4)
    mirror = arm_dims.get("mirror", -1)

    # Apply coordinate system adjustments if available
    if "coordinate_system" in warp_config:
        coord_sys = warp_config["coordinate_system"]
        origin_offset_x = coord_sys.get("origin_offset_x", 0)
        origin_offset_y = coord_sys.get("origin_offset_y", 0)
        scale_factor = coord_sys.get("scale_factor", 1.0)
    else:
        origin_offset_x = origin_offset_y = 0
        scale_factor = 1.0

    if warp_config.get("debug_settings", {}).get("enable_debug_output", False):
        print(f"[WARP] Loaded config: biceps={biceps}, underarm={underarm}, tendons={tendon_biceps}/{tendon_underarm}, mirror={mirror}")
        print(f"[WARP] Coordinate system: offset=({origin_offset_x},{origin_offset_y}), scale={scale_factor}")
else:
    # Default values (JBE's original Swedish mathematician values)
    biceps = 100
    underarm = 100
    tendon_biceps = 50
    tendon_underarm = 4
    mirror = -1
    origin_offset_x = origin_offset_y = 0
    scale_factor = 1.0

# det här är bara för utplottningen
# min_l = abs(tendon_underarm - tendon_biceps) + 1
# max_l = tendon_biceps + tendon_underarm - 1


# hjälpfunktion
def add_vectors(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1])


# hjälpfunktion
def rotation(cos_v, sin_v):
    def rotation_v(x, y):
        return (x * cos_v - y * mirror * sin_v, x * mirror * sin_v + y * cos_v)

    return rotation_v


# det här simulerar felet som maskinen skapar
def trans(theta, l):
    cos_phi = (l**2 + tendon_underarm**2 - tendon_biceps**2) / (2 * l * tendon_underarm)
    joint_tendon = (tendon_biceps + tendon_underarm + l) / 2
    sin_phi = 2 * math.sqrt(l * (joint_tendon - tendon_underarm) * (joint_tendon - tendon_biceps) * (joint_tendon - l)) / (tendon_underarm * l)
    rotation_v = rotation(cos_phi, sin_phi)
    p = add_vectors(rotation_v(0, -underarm), (0, underarm))
    rotation_v2 = rotation(math.cos(theta), math.sin(theta))
    return rotation_v2(p[0], p[1])


# DET HÄR är en hjälpfunktion för inversen
# den behöver vara med i filen
def theta_calc(x, y):
    if x == 0:
        return 0
    cos_alpha = (biceps**2 + (x**2 + y**2) - underarm**2) / (2 * biceps * math.sqrt(x**2 + y**2))
    alpha = math.acos(cos_alpha)
    x_y_angle = math.atan(y / x)
    if x > 0:
        return alpha + x_y_angle - math.pi / 2
    if x < 0:
        return alpha + x_y_angle + math.pi / 2


# DET HÄR är funktionen som ska konvertera tillbaka skjuvningen
# koppla in den där koordinater för g-code skapas
def inverse(x, y):
    x = mirror * x
    cos_phi = (biceps**2 + underarm**2 - (x**2 + y**2)) / (2 * biceps * underarm)
    l = tendon_underarm * cos_phi + math.sqrt((tendon_underarm * cos_phi) ** 2 - (tendon_underarm**2 - tendon_biceps**2))
    theta = theta_calc(x, y)
    return mirror * theta, l  # type: ignore


# inställningar för utplottningen
# plt.scatter(0, 0, color="black", marker="x")
# plt.gca().set_aspect("equal", adjustable="box")
# plt.xlim(-300, 200)
# plt.ylim(-80, 300)

# for x in range(0, 15, 1):
#     for y in range(min_l, max_l, 1):
#         # p = (x* 0.1, y)
#         p = trans(x * 0.1, y)
#         # p = inverse(p[0], p[1]) # den här raden är bara för att testa inversen
#         plt.scatter(p[0], p[1], color="blue", marker=".")

# plt.show()


def warp_transform_line(gcode_line):
    x_match = re.search(r"X([-+]?\d*\.?\d+)", gcode_line, re.IGNORECASE)
    y_match = re.search(r"Y([-+]?\d*\.?\d+)", gcode_line, re.IGNORECASE)

    if x_match and y_match:
        # Extract original coordinates
        original_x = float(x_match.group(1))
        original_y = float(y_match.group(1))

        # Apply scale factor and origin offset before warp transform
        scaled_x = (original_x * scale_factor) + origin_offset_x
        scaled_y = (original_y * scale_factor) + origin_offset_y

        # Apply JBE's inverse warp transform
        transformed_x, transformed_y = inverse(scaled_x, scaled_y)

        # Debug output if enabled
        debug_enabled = warp_config and warp_config.get("debug_settings", {}).get("log_transformations", False)
        if debug_enabled:
            print(f"[WARP] {original_x:.2f},{original_y:.2f} -> {scaled_x:.2f},{scaled_y:.2f} -> {transformed_x:.4f},{transformed_y:.4f}")

        # Update G-code line with transformed coordinates
        gcode_line = re.sub(r"X[-+]?\d*\.?\d+", f"X{transformed_x:.4f}", gcode_line, flags=re.IGNORECASE)
        gcode_line = re.sub(r"Y[-+]?\d*\.?\d+", f"Y{transformed_y:.4f}", gcode_line, flags=re.IGNORECASE)
    else:
        if warp_config and warp_config.get("debug_settings", {}).get("enable_debug_output", False):
            print("[WARN] No X or Y found in line, skipping transformation.")

    return gcode_line


if __name__ == "__main__":
    gcode_file = "/Users/jbe/repos/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy/grbl/test_files/impostor-20250815_201047_00001__center_lined_servo_adjusted.gcode"
    try:
        with open(gcode_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"G-code file not found: {gcode_file}")
    lines = lines[3:]  # vpype junk

    for line in lines:
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith(";") or line.startswith("%"):
                continue

            try:
                if line.startswith(("G0", "G1", "G00", "G01")):
                    print(f"[ORG] {line}")
                    line = warp_transform_line(line)
                    print(f"[ADJ] {line}")
                    print("=====")

            except Exception as e:
                print(f"[ERROR] Failed to process line {line_num}: {line}. Error: {e}")
                continue
