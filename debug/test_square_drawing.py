#!/usr/bin/env python3
"""
Test Complete Square Drawing with Corrected Warp Transform
=========================================================

Now that we understand the coordinate system, let's create a proper coordinate mapping
that converts from theoretical coordinates to Andreas's GRBL coordinate system.

Goal: Draw an accurate square by applying the warp correction properly.
"""

import math
import numpy as np


# === ORIGINAL FUNCTIONS ===
biceps = 295  # Upper arm length (mm)
underarm = 325  # Lower arm length (mm)
tendon_biceps = 160  # Tendon length from biceps to elbow (mm)
tendon_underarm = 50  # Tendon length from elbow to attachment (mm)
mirror = -1  # Mirror factor


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
    return (x + x_delta, y + y_delta)


def inverse_original(x, y):
    """Original inverse transform: desired position → robot parameters"""
    (x, y) = translate(x, y, 60, 60)
    cos_phi = (biceps**2 + underarm**2 - (x**2 + y**2)) / (2 * biceps * underarm)
    l = tendon_underarm * cos_phi + math.sqrt((tendon_underarm * cos_phi) ** 2 - (tendon_underarm**2 - tendon_biceps**2))
    theta = theta_calc(x, y)
    return theta, l


# === COORDINATE MAPPING BASED ON ANDREAS'S DATA ===

# Andreas's working A4 coordinates
ANDREAS_A4_CORNERS = [
    (66, -2),    # bottom_left  → A4 (0, 0)
    (111, -1),   # bottom_right → A4 (210, 0)
    (-2, 67),    # top_left     → A4 (0, 297)
    (24, 67)     # top_right    → A4 (210, 297)
]

# Standard A4 coordinates
A4_CORNERS = [
    (0, 0),      # bottom_left
    (210, 0),    # bottom_right
    (0, 297),    # top_left
    (210, 297)   # top_right
]


def calculate_coordinate_transform_matrix():
    """Calculate transformation matrix from A4 coordinates to Andreas's GRBL coordinates"""

    # Create matrices for bilinear transformation
    # We need to solve: GRBL = M * A4 for transformation matrix M

    # For 4-point bilinear transformation, we can use a 2D affine transform as approximation
    # or implement proper bilinear interpolation

    a4_points = np.array(A4_CORNERS)
    grbl_points = np.array(ANDREAS_A4_CORNERS)

    # Use least squares to find best affine transformation
    # [x', y'] = [a b] [x] + [tx]
    #            [c d] [y]   [ty]

    # Convert to homogeneous coordinates
    a4_homo = np.column_stack([a4_points, np.ones(4)])  # [x, y, 1]

    # Solve for transformation matrix
    transform_x = np.linalg.lstsq(a4_homo, grbl_points[:, 0], rcond=None)[0]
    transform_y = np.linalg.lstsq(a4_homo, grbl_points[:, 1], rcond=None)[0]

    return transform_x, transform_y


def bilinear_interpolation(x, y, corner_values):
    """Proper bilinear interpolation for coordinate transformation"""

    # Normalize coordinates to [0,1] range for A4
    u = x / 210.0  # A4 width
    v = y / 297.0  # A4 height

    # Clamp to valid range
    u = max(0, min(1, u))
    v = max(0, min(1, v))

    # Get corner values
    bottom_left, bottom_right, top_left, top_right = corner_values

    # Bilinear interpolation
    bottom = (1 - u) * bottom_left[0] + u * bottom_right[0], (1 - u) * bottom_left[1] + u * bottom_right[1]
    top = (1 - u) * top_left[0] + u * top_right[0], (1 - u) * top_left[1] + u * top_right[1]

    result_x = (1 - v) * bottom[0] + v * top[0]
    result_y = (1 - v) * bottom[1] + v * top[1]

    return result_x, result_y


def theoretical_to_grbl_coordinates(x, y):
    """Convert from theoretical coordinate system to GRBL coordinate system"""

    # Method 1: Bilinear interpolation (most accurate for the A4 region)
    grbl_x, grbl_y = bilinear_interpolation(x, y, ANDREAS_A4_CORNERS)

    return grbl_x, grbl_y


def corrected_warp_transform(desired_x, desired_y):
    """Complete warp transform: desired coordinates → corrected GRBL coordinates"""

    # Step 1: Apply inverse kinematics to get robot parameters
    theta, l = inverse_original(desired_x, desired_y)

    # Step 2: Apply forward transform to get theoretical corrected position
    corrected_x, corrected_y = trans(theta, l)

    # Step 3: Convert from theoretical coordinates to GRBL coordinates
    grbl_x, grbl_y = theoretical_to_grbl_coordinates(corrected_x, corrected_y)

    return grbl_x, grbl_y


def test_square_drawing():
    """Test drawing a square with the corrected warp transform"""
    print("=== SQUARE DRAWING TEST ===")
    print("Testing corrected warp transform for square drawing")
    print()

    # Define a 50x50mm square
    square_size = 50
    square_points = [
        (0, 0),                    # bottom_left
        (square_size, 0),          # bottom_right
        (square_size, square_size), # top_right
        (0, square_size),          # top_left
        (0, 0)                     # back to start
    ]

    print("Original square coordinates:")
    for i, (x, y) in enumerate(square_points):
        print(f"  Point {i}: ({x:3}, {y:3})")

    print("\nCorrected GRBL coordinates:")
    corrected_points = []

    for i, (x, y) in enumerate(square_points):
        try:
            grbl_x, grbl_y = corrected_warp_transform(x, y)
            corrected_points.append((grbl_x, grbl_y))
            print(f"  Point {i}: ({x:3}, {y:3}) → ({grbl_x:6.1f}, {grbl_y:6.1f})")
        except Exception as e:
            print(f"  Point {i}: ({x:3}, {y:3}) → ERROR: {e}")
            corrected_points.append(None)

    # Analyze the corrected square
    print("\nSquare analysis:")
    if None not in corrected_points:
        # Calculate side lengths
        sides = []
        for i in range(4):
            p1 = corrected_points[i]
            p2 = corrected_points[i + 1]
            length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            sides.append(length)
            print(f"  Side {i+1} length: {length:.2f} GRBL units")

        # Calculate angles (should be ~90 degrees)
        angles = []
        for i in range(4):
            p1 = corrected_points[i]
            p2 = corrected_points[(i + 1) % 4]
            p3 = corrected_points[(i + 2) % 4]

            # Vectors
            v1 = (p1[0] - p2[0], p1[1] - p2[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])

            # Angle between vectors
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

            if mag1 > 0 and mag2 > 0:
                angle = math.degrees(math.acos(max(-1, min(1, dot_product / (mag1 * mag2)))))
                angles.append(angle)
                print(f"  Angle {i+1}: {angle:.1f}°")

        # Overall assessment
        avg_side = sum(sides) / len(sides)
        side_variation = max(sides) - min(sides)
        avg_angle = sum(angles) / len(angles) if angles else 0

        print(f"\n  Average side length: {avg_side:.2f} GRBL units")
        print(f"  Side length variation: {side_variation:.2f} GRBL units")
        print(f"  Average angle: {avg_angle:.1f}° (should be 90°)")

        # Quality assessment
        is_good_square = (side_variation < avg_side * 0.1) and (80 < avg_angle < 100)
        print(f"  Square quality: {'✓ GOOD' if is_good_square else '✗ POOR'}")


def generate_gcode_for_square():
    """Generate G-code for the corrected square"""
    print("\n=== G-CODE GENERATION ===")
    print("Generating G-code for corrected square")
    print()

    square_size = 50
    square_points = [
        (0, 0),
        (square_size, 0),
        (square_size, square_size),
        (0, square_size),
        (0, 0)
    ]

    gcode_lines = []

    # G-code header
    gcode_lines.extend([
        "G21 ; Set units to millimeters",
        "G90 ; Use absolute positioning",
        "G17 ; Select XY plane",
        ""
    ])

    # Generate movement commands
    for i, (x, y) in enumerate(square_points):
        try:
            grbl_x, grbl_y = corrected_warp_transform(x, y)

            if i == 0:
                # First point - rapid move with pen up
                gcode_lines.extend([
                    f"G0 X{grbl_x:.4f} Y{grbl_y:.4f} ; Move to start",
                    "M3 S50 ; Lower pen"
                ])
            else:
                # Drawing moves
                gcode_lines.append(f"G1 X{grbl_x:.4f} Y{grbl_y:.4f} F1500 ; Draw line")

        except Exception as e:
            gcode_lines.append(f"; ERROR at point ({x}, {y}): {e}")

    # G-code footer
    gcode_lines.extend([
        "",
        "M3 S30 ; Raise pen",
        "G0 X0 Y0 ; Return to origin",
        "M2 ; End program"
    ])

    # Print G-code
    print("Generated G-code:")
    for line in gcode_lines:
        print(f"  {line}")

    # Save to file
    gcode_file = "/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy/debug/corrected_square.gcode"
    try:
        with open(gcode_file, 'w') as f:
            f.write('\n'.join(gcode_lines))
        print(f"\nG-code saved to: {gcode_file}")
    except Exception as e:
        print(f"\nError saving G-code: {e}")


if __name__ == "__main__":
    print("Square Drawing Test with Corrected Warp Transform")
    print("=" * 60)

    # Test the square drawing
    test_square_drawing()

    # Generate G-code
    generate_gcode_for_square()

    print("\n" + "=" * 60)
    print("SUCCESS: Square drawing test completed!")
    print("Next step: Test the generated G-code on the actual robot")