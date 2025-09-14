#!/usr/bin/env python3
"""
Debug Mathematical Errors in Warp Transform
==========================================

The consistency test showed that trans(inverse(x,y)) != (x,y).
This means the mathematical functions have fundamental errors.

Let's debug step by step what's going wrong.
"""

import math


# === COPIED FUNCTIONS ===
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
    """Forward transform: robot parameters → actual position"""
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


def inverse(x, y):
    """Inverse transform: desired position → robot parameters"""
    (x, y) = translate(x, y, 60, 60)
    cos_phi = (biceps**2 + underarm**2 - (x**2 + y**2)) / (2 * biceps * underarm)
    l = tendon_underarm * cos_phi + math.sqrt((tendon_underarm * cos_phi) ** 2 - (tendon_underarm**2 - tendon_biceps**2))
    theta = theta_calc(x, y)
    return theta, l


# === DEBUG FUNCTIONS ===

def debug_single_case(x, y):
    """Debug a single test case step by step"""
    print(f"\n=== DEBUGGING CASE: ({x}, {y}) ===")

    print("STEP 1: Input coordinates")
    print(f"  Input: ({x}, {y})")

    print("\nSTEP 2: Translation in inverse()")
    translated_x, translated_y = translate(x, y, 60, 60)
    print(f"  After translation (+60, +60): ({translated_x}, {translated_y})")

    print("\nSTEP 3: Calculate inverse kinematics")
    distance = math.sqrt(translated_x**2 + translated_y**2)
    print(f"  Distance from origin: {distance:.2f}mm")
    print(f"  Robot reach limits: {abs(biceps-underarm):.1f} to {biceps+underarm:.1f}mm")

    # Check if within robot reach
    if distance > biceps + underarm:
        print(f"  ⚠️  WARNING: Distance {distance:.1f}mm exceeds max reach {biceps+underarm:.1f}mm")
    elif distance < abs(biceps - underarm):
        print(f"  ⚠️  WARNING: Distance {distance:.1f}mm less than min reach {abs(biceps-underarm):.1f}mm")
    else:
        print(f"  ✓ Distance within robot reach")

    # Calculate cos_phi
    cos_phi = (biceps**2 + underarm**2 - (translated_x**2 + translated_y**2)) / (2 * biceps * underarm)
    print(f"  cos_phi = {cos_phi:.4f}")

    if abs(cos_phi) > 1:
        print(f"  ❌ ERROR: cos_phi = {cos_phi:.4f} is outside [-1, 1] range!")
        print(f"    This will cause math.acos() to fail or give invalid results")
        return None, None

    # Calculate l
    try:
        sqrt_term = (tendon_underarm * cos_phi) ** 2 - (tendon_underarm**2 - tendon_biceps**2)
        print(f"  Square root term: {sqrt_term:.4f}")
        if sqrt_term < 0:
            print(f"  ❌ ERROR: Negative square root term!")
            return None, None

        l = tendon_underarm * cos_phi + math.sqrt(sqrt_term)
        print(f"  Calculated l: {l:.2f}mm")

    except Exception as e:
        print(f"  ❌ ERROR calculating l: {e}")
        return None, None

    # Calculate theta
    try:
        theta = theta_calc(translated_x, translated_y)
        print(f"  Calculated theta: {theta:.4f} rad ({math.degrees(theta):.1f}°)")
    except Exception as e:
        print(f"  ❌ ERROR calculating theta: {e}")
        return None, None

    print(f"\nSTEP 4: Forward transform back")
    try:
        back_x, back_y = trans(theta, l)
        print(f"  trans({theta:.4f}, {l:.2f}) = ({back_x:.2f}, {back_y:.2f})")

        # Calculate error
        error_x = abs(back_x - x)
        error_y = abs(back_y - y)
        total_error = math.sqrt(error_x**2 + error_y**2)
        print(f"  Error: ({error_x:.2f}, {error_y:.2f}) = {total_error:.2f}mm total")

        return theta, l

    except Exception as e:
        print(f"  ❌ ERROR in forward transform: {e}")
        return None, None


def analyze_coordinate_system():
    """Analyze the coordinate system assumptions"""
    print("\n=== COORDINATE SYSTEM ANALYSIS ===")

    print("Robot configuration:")
    print(f"  Biceps length: {biceps}mm")
    print(f"  Underarm length: {underarm}mm")
    print(f"  Tendon biceps: {tendon_biceps}mm")
    print(f"  Tendon underarm: {tendon_underarm}mm")
    print(f"  Mirror factor: {mirror}")

    print(f"\nReach analysis:")
    max_reach = biceps + underarm
    min_reach = abs(biceps - underarm)
    print(f"  Theoretical max reach: {max_reach}mm")
    print(f"  Theoretical min reach: {min_reach}mm")

    print(f"\nTranslation analysis:")
    print(f"  inverse() adds (60, 60) to input coordinates")
    print(f"  This means input (0,0) becomes ({60},{60}) for calculations")
    print(f"  Distance from origin: {math.sqrt(60**2 + 60**2):.1f}mm")
    print(f"  This should be well within robot reach")


if __name__ == "__main__":
    print("Debugging Mathematical Errors in Warp Transform")
    print("=" * 60)

    # Analyze coordinate system first
    analyze_coordinate_system()

    # Debug specific failing cases
    test_cases = [(0, 0), (50, 50), (100, 100)]

    for x, y in test_cases:
        debug_single_case(x, y)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Fix mathematical errors identified above")
    print("2. Verify coordinate system assumptions")
    print("3. Test with corrected functions")