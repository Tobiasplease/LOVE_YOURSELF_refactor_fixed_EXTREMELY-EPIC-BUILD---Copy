#!/usr/bin/env python3
"""
Warp Transform Mathematical Consistency Test
===========================================

Tests if the mathematician's inverse() and trans() functions are mathematically consistent.
If trans(inverse(x, y)) ≈ (x, y), then the math is sound and we have a coordinate system problem.
If not, then the mathematical functions themselves have bugs.
"""

import math

# === COPIED FROM warp_transform.py ===
# Robot arm dimensions - verified correct by user
biceps = 295  # Upper arm length (mm)
underarm = 325  # Lower arm length (mm)
tendon_biceps = 160  # Tendon length from biceps to elbow (mm)
tendon_underarm = 50  # Tendon length from elbow to attachment (mm)
mirror = -1  # Mirror factor (-1 or 1 depending on orientation)


def add_vectors(p1, p2):
    """Helper function for vector addition"""
    return (p1[0] + p2[0], p1[1] + p2[1])


def rotation(cos_v, sin_v):
    """Helper function for rotation matrix"""

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


def inverse(x, y):
    """Inverse transform: desired position → robot parameters"""
    # Mathematician's origin translation
    (x, y) = translate(x, y, 60, 60)

    cos_phi = (biceps**2 + underarm**2 - (x**2 + y**2)) / (2 * biceps * underarm)
    l = tendon_underarm * cos_phi + math.sqrt((tendon_underarm * cos_phi) ** 2 - (tendon_underarm**2 - tendon_biceps**2))
    theta = theta_calc(x, y)

    return theta, l


# === TEST FUNCTIONS ===


def test_mathematical_consistency():
    """Test if trans(inverse(x, y)) ≈ (x, y) for various points"""
    print("=== MATHEMATICAL CONSISTENCY TEST ===")
    print("Testing if trans(inverse(x, y)) ≈ (x, y)")
    print()

    # Test points including some that should work well within robot reach
    test_points = [(0, 0), (50, 50), (100, 100), (25, 75), (-50, 50), (0, 100), (100, 0)]

    consistent = True
    tolerance = 1.0  # 1mm tolerance

    print(f"{'Input':<12} {'→ Robot Params':<20} {'→ Back to Coords':<18} {'Error':<12} {'Status'}")
    print("-" * 80)

    for x, y in test_points:
        try:
            # Forward: desired position → robot parameters
            theta, l = inverse(x, y)

            # Back: robot parameters → actual position
            back_x, back_y = trans(theta, l)

            # Calculate error
            error_x = abs(back_x - x)
            error_y = abs(back_y - y)
            total_error = math.sqrt(error_x**2 + error_y**2)

            # Check if within tolerance
            is_consistent = total_error <= tolerance
            if not is_consistent:
                consistent = False

            status = "✓ OK" if is_consistent else "✗ FAIL"

            print(f"({x:4.0f},{y:4.0f})    θ={theta:6.3f} l={l:6.1f}   ({back_x:6.1f},{back_y:6.1f})    {total_error:6.2f}mm     {status}")

        except Exception as e:
            print(f"({x:4.0f},{y:4.0f})    ERROR: {e}")
            consistent = False

    print("-" * 80)

    if consistent:
        print("✅ MATHEMATICS IS CONSISTENT")
        print("   The inverse() and trans() functions are mathematically sound.")
        print("   The problem is likely in coordinate system conversion.")
    else:
        print("❌ MATHEMATICS IS INCONSISTENT")
        print("   The inverse() and trans() functions have mathematical errors.")
        print("   Need to debug the mathematical implementation.")

    return consistent


def test_robot_parameter_ranges():
    """Test what ranges of robot parameters are produced"""
    print("\n=== ROBOT PARAMETER RANGES ===")
    print("Analyzing theta and l ranges for typical coordinates")
    print()

    test_points = [(x, y) for x in range(0, 201, 50) for y in range(0, 201, 50)]

    thetas = []
    ls = []

    print(f"{'Coordinate':<12} {'Theta (rad)':<12} {'Theta (deg)':<12} {'L (mm)':<8}")
    print("-" * 50)

    for x, y in test_points:
        try:
            theta, l = inverse(x, y)
            thetas.append(theta)
            ls.append(l)

            theta_deg = math.degrees(theta)
            print(f"({x:3.0f},{y:3.0f})      {theta:8.4f}     {theta_deg:8.1f}°       {l:6.1f}")

        except Exception as e:
            print(f"({x:3.0f},{y:3.0f})      ERROR: {e}")

    if thetas and ls:
        print("-" * 50)
        print(f"Theta range: {min(thetas):.4f} to {max(thetas):.4f} rad ({math.degrees(min(thetas)):.1f}° to {math.degrees(max(thetas)):.1f}°)")
        print(f"L range:     {min(ls):.1f} to {max(ls):.1f} mm")

        # Check if ranges are reasonable for robot
        max_reach = biceps + underarm
        min_reach = abs(biceps - underarm)
        print(f"Robot reach: {min_reach:.1f} to {max_reach:.1f} mm (theoretical)")


if __name__ == "__main__":
    print("Warp Transform Mathematical Consistency Test")
    print("=" * 50)

    # Test if the math is consistent
    is_consistent = test_mathematical_consistency()

    # Analyze parameter ranges
    test_robot_parameter_ranges()

    print("\n" + "=" * 50)
    if is_consistent:
        print("CONCLUSION: Math is consistent, focus on coordinate system conversion")
    else:
        print("CONCLUSION: Math has errors, need to fix mathematical functions first")
