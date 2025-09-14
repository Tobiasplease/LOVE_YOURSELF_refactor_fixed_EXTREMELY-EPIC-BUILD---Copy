#!/usr/bin/env python3
"""
Test Coordinate System Transformations for Warp Transform
========================================================

Based on Mikael's email, the math is correct but needs coordinate system adjustments:
1. May need transpose/mirror/rotate the inverse
2. Origin (0,0) position assumptions may be wrong
3. Need to calibrate against Andreas's working A4 coordinates

The mathematician said: "Man kan t ex behöva transponera/spegla/rotera inversen"
"""

import math


# === ORIGINAL FUNCTIONS (assume these are mathematically correct) ===
biceps = 295  # Upper arm length (mm)
underarm = 325  # Lower arm length (mm)
tendon_biceps = 160  # Tendon length from biceps to elbow (mm)
tendon_underarm = 50  # Tendon length from elbow to attachment (mm)
mirror = -1  # Mirror factor (-1 or 1 depending on orientation)


def add_vectors(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1])


def rotation(cos_v, sin_v):
    def rotation_v(x, y):
        return (x * cos_v - y * mirror * sin_v, x * mirror * sin_v + y * cos_v)
    return rotation_v


def trans(theta, l):
    """Forward transform: robot parameters → actual position (simulates robot distortion)"""
    cos_phi = (l**2 + tendon_underarm**2 - tendon_biceps**2) / (2 * l * tendon_underarm)
    joint_tendon = (tendon_biceps + tendon_underarm + l) / 2
    sin_phi = 2 * math.sqrt(l * (joint_tendon - tendon_underarm) * (joint_tendon - tendon_biceps) * (joint_tendon - l)) / (tendon_underarm * l)
    rotation_v = rotation(cos_phi, sin_phi)
    p = add_vectors(rotation_v(0, -underarm), (0, underarm))
    rotation_v2 = rotation(math.cos(theta), math.sin(theta))
    return rotation_v2(p[0], p[1])


def theta_calc(x, y):
    """Helper function for inverse kinematics"""
    if x == 0:
        return 0
    cos_alpha = (biceps**2 + (x**2 + y**2) - underarm**2) / (2 * biceps * math.sqrt(x**2 + y**2))
    alpha = math.acos(cos_alpha)
    x_y_angle = math.atan(y / x)
    if x > 0:
        return alpha + x_y_angle - math.pi / 2
    if x < 0:
        return alpha + x_y_angle + math.pi / 2


def translate(x, y, x_delta, y_delta):
    """Translate coordinate system origin"""
    return (x + x_delta, y + y_delta)


def inverse_original(x, y):
    """Original inverse transform: desired position → robot parameters"""
    (x, y) = translate(x, y, 60, 60)
    cos_phi = (biceps**2 + underarm**2 - (x**2 + y**2)) / (2 * biceps * underarm)
    l = tendon_underarm * cos_phi + math.sqrt((tendon_underarm * cos_phi) ** 2 - (tendon_underarm**2 - tendon_biceps**2))
    theta = theta_calc(x, y)
    return theta, l


# === COORDINATE SYSTEM VARIATIONS ===

def inverse_transposed(x, y):
    """Try swapping X and Y coordinates"""
    return inverse_original(y, x)


def inverse_mirrored_x(x, y):
    """Try mirroring X coordinate"""
    return inverse_original(-x, y)


def inverse_mirrored_y(x, y):
    """Try mirroring Y coordinate"""
    return inverse_original(x, -y)


def inverse_mirrored_both(x, y):
    """Try mirroring both coordinates"""
    return inverse_original(-x, -y)


def inverse_rotated_90(x, y):
    """Try 90-degree rotation: (x,y) → (-y,x)"""
    return inverse_original(-y, x)


def inverse_rotated_180(x, y):
    """Try 180-degree rotation: (x,y) → (-x,-y)"""
    return inverse_original(-x, -y)


def inverse_rotated_270(x, y):
    """Try 270-degree rotation: (x,y) → (y,-x)"""
    return inverse_original(y, -x)


def inverse_different_origin(x, y, offset_x=0, offset_y=0):
    """Try different origin offsets"""
    (x, y) = translate(x, y, offset_x, offset_y)
    cos_phi = (biceps**2 + underarm**2 - (x**2 + y**2)) / (2 * biceps * underarm)
    l = tendon_underarm * cos_phi + math.sqrt((tendon_underarm * cos_phi) ** 2 - (tendon_underarm**2 - tendon_biceps**2))
    theta = theta_calc(x, y)
    return theta, l


# === ANDREAS'S WORKING A4 COORDINATES ===
ANDREAS_A4_CORNERS = {
    "bottom_left": (66, -2),    # Should map to A4 (0, 0)
    "bottom_right": (111, -1),  # Should map to A4 (210, 0)
    "top_left": (-2, 67),       # Should map to A4 (0, 297)
    "top_right": (24, 67)       # Should map to A4 (210, 297)
}

A4_CORNERS = {
    "bottom_left": (0, 0),
    "bottom_right": (210, 0),
    "top_left": (0, 297),
    "top_right": (210, 297)
}


def test_coordinate_transformations():
    """Test different coordinate system transformations"""
    print("=== COORDINATE SYSTEM TRANSFORMATION TEST ===")
    print("Testing different ways to apply the inverse transform")
    print()

    # Test with a simple square
    test_square = [(0, 0), (100, 0), (100, 100), (0, 100)]

    transformations = [
        ("Original", inverse_original),
        ("Transposed", inverse_transposed),
        ("Mirrored X", inverse_mirrored_x),
        ("Mirrored Y", inverse_mirrored_y),
        ("Mirrored Both", inverse_mirrored_both),
        ("Rotated 90°", inverse_rotated_90),
        ("Rotated 180°", inverse_rotated_180),
        ("Rotated 270°", inverse_rotated_270)
    ]

    print(f"{'Transform':<15} {'Point':<10} {'→ Robot Params':<20} {'Status'}")
    print("-" * 60)

    for transform_name, transform_func in transformations:
        valid_count = 0

        for x, y in test_square:
            try:
                theta, l = transform_func(x, y)

                # Check if parameters are reasonable
                theta_deg = math.degrees(theta)
                is_valid = (-180 <= theta_deg <= 180) and (100 <= l <= 300)

                if is_valid:
                    valid_count += 1
                    status = "✓"
                else:
                    status = "✗"

                print(f"{transform_name:<15} ({x:3},{y:3})    θ={theta:6.3f} l={l:6.1f}   {status}")

            except Exception as e:
                print(f"{transform_name:<15} ({x:3},{y:3})    ERROR: {str(e)[:20]:<20}   ✗")

        print(f"  → {valid_count}/4 points valid")
        print()


def test_origin_variations():
    """Test different origin offset values"""
    print("=== ORIGIN OFFSET VARIATIONS ===")
    print("Testing different translation offsets")
    print()

    test_point = (50, 50)
    offsets = [
        (0, 0),      # No offset
        (30, 30),    # Half of current
        (60, 60),    # Current offset
        (120, 120),  # Double current
        (-60, -60),  # Negative offset
        (100, 0),    # X only
        (0, 100)     # Y only
    ]

    print(f"{'Offset':<12} {'Point':<10} {'→ Robot Params':<20} {'Distance':<10} {'Status'}")
    print("-" * 65)

    for offset_x, offset_y in offsets:
        try:
            theta, l = inverse_different_origin(test_point[0], test_point[1], offset_x, offset_y)

            # Calculate translated point distance from origin
            translated_x, translated_y = translate(test_point[0], test_point[1], offset_x, offset_y)
            distance = math.sqrt(translated_x**2 + translated_y**2)

            # Check if reasonable
            is_valid = (100 <= l <= 300) and (distance < 500)
            status = "✓" if is_valid else "✗"

            print(f"({offset_x:3},{offset_y:3})     ({test_point[0]:2},{test_point[1]:2})    θ={theta:6.3f} l={l:6.1f}   {distance:6.1f}mm  {status}")

        except Exception as e:
            print(f"({offset_x:3},{offset_y:3})     ({test_point[0]:2},{test_point[1]:2})    ERROR: {str(e)[:15]:<15}        ✗")

    print()


def analyze_andreas_coordinates():
    """Analyze Andreas's working coordinates to understand the coordinate system"""
    print("=== ANDREAS'S A4 COORDINATE ANALYSIS ===")
    print("Analyzing the working coordinates to understand coordinate system")
    print()

    print("Andreas's working coordinates:")
    for corner_name, (x, y) in ANDREAS_A4_CORNERS.items():
        a4_x, a4_y = A4_CORNERS[corner_name]
        print(f"  {corner_name:<12}: ({x:4},{y:4}) should draw at A4 ({a4_x:3},{a4_y:3})")

    print("\nCoordinate system analysis:")

    # Calculate coordinate system transformation from Andreas's data
    bottom_left_grbl = ANDREAS_A4_CORNERS["bottom_left"]
    bottom_right_grbl = ANDREAS_A4_CORNERS["bottom_right"]
    top_left_grbl = ANDREAS_A4_CORNERS["top_left"]

    # Calculate vectors
    width_vector = (bottom_right_grbl[0] - bottom_left_grbl[0],
                   bottom_right_grbl[1] - bottom_left_grbl[1])
    height_vector = (top_left_grbl[0] - bottom_left_grbl[0],
                    top_left_grbl[1] - bottom_left_grbl[1])

    print(f"  Width vector (210mm A4):  {width_vector}")
    print(f"  Height vector (297mm A4): {height_vector}")

    # Calculate scaling
    width_scale = math.sqrt(width_vector[0]**2 + width_vector[1]**2) / 210
    height_scale = math.sqrt(height_vector[0]**2 + height_vector[1]**2) / 297

    print(f"  Width scaling: {width_scale:.3f}")
    print(f"  Height scaling: {height_scale:.3f}")

    # Calculate rotation
    width_angle = math.degrees(math.atan2(width_vector[1], width_vector[0]))
    height_angle = math.degrees(math.atan2(height_vector[1], height_vector[0]))

    print(f"  Width vector angle: {width_angle:.1f}°")
    print(f"  Height vector angle: {height_angle:.1f}°")
    print()


if __name__ == "__main__":
    print("Coordinate System Transformation Test")
    print("=" * 60)

    # Analyze Andreas's working coordinates first
    analyze_andreas_coordinates()

    # Test different coordinate transformations
    test_coordinate_transformations()

    # Test different origin offsets
    test_origin_variations()

    print("=" * 60)
    print("NEXT STEPS:")
    print("1. Find which transformation produces reasonable robot parameters")
    print("2. Calibrate coordinate system to match Andreas's working coordinates")
    print("3. Test with square drawing to validate approach")