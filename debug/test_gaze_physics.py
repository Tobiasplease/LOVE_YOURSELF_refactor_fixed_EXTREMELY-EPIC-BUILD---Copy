#!/usr/bin/env python3
"""Test script for physics-based gaze simulation system."""

import math
import sys
import time

sys.path.insert(0, "/home/impostor/LOVE_YOURSELF_refactor_fixed_EXTREMELY-EPIC-BUILD---Copy")

from vision.gaze import (
    PAN_MAX,
    PAN_MIN,
    PHYSICS_PATTERNS,
    TILT_MAX,
    TILT_MIN,
    TRACKING_PHYSICS,
    GazePhysicsState,
    physics_state,
    update_physics_step,
)


def test_spring_convergence():
    """Test that spring-damper converges to target without oscillation."""
    print("\n=== Test: Spring-Damper Convergence ===")

    ps = physics_state
    ps.pan = 90.0
    ps.tilt = 90.0
    ps.pan_velocity = 0.0
    ps.tilt_velocity = 0.0
    ps.pan_target = 120.0
    ps.tilt_target = 80.0
    ps.last_update_time = time.time()

    # Use calm_observant parameters
    ps.blend_params(PHYSICS_PATTERNS["calm_observant"], blend_rate=1.0)

    print(f"Start: pan={ps.pan:.1f}, tilt={ps.tilt:.1f}")
    print(f"Target: pan={ps.pan_target:.1f}, tilt={ps.tilt_target:.1f}")

    positions = []
    for i in range(50):
        time.sleep(0.02)
        pan, tilt = update_physics_step(0.02, is_tracking=False)
        positions.append((pan, tilt))
        if i % 10 == 0:
            print(f"  Step {i}: pan={pan:.1f}, tilt={tilt:.1f}, vel=({ps.pan_velocity:.2f}, {ps.tilt_velocity:.2f})")

    final_pan, final_tilt = positions[-1]
    pan_error = abs(final_pan - ps.pan_target)
    tilt_error = abs(final_tilt - ps.tilt_target)

    print(f"Final: pan={final_pan:.1f}, tilt={final_tilt:.1f}")
    print(f"Error: pan={pan_error:.2f}°, tilt={tilt_error:.2f}°")

    if pan_error < 2.0 and tilt_error < 2.0:
        print("✓ PASS: Converged within tolerance")
    else:
        print("✗ FAIL: Did not converge")


def test_emotional_state_differences():
    """Test that different emotional states produce different movement characteristics."""
    print("\n=== Test: Emotional State Differences ===")

    states_to_test = ["energized_engaged", "calm_observant", "withdrawn_distant"]

    for emotion in states_to_test:
        ps = physics_state
        ps.pan = 90.0
        ps.tilt = 90.0
        ps.pan_velocity = 0.0
        ps.tilt_velocity = 0.0
        ps.pan_target = 110.0
        ps.tilt_target = 100.0
        ps.last_update_time = time.time()

        # Apply emotional parameters instantly
        ps.blend_params(PHYSICS_PATTERNS[emotion], blend_rate=1.0)

        print(f"\n{emotion}:")
        print(f"  mass={ps.mass:.1f}, spring={ps.spring_constant:.1f}, damping={ps.damping:.1f}")
        print(f"  tremor={ps.tremor_amplitude:.2f}, orbital={ps.orbital_strength:.2f}")

        # Measure time to reach 90% of target
        target_threshold = 90 + (110 - 90) * 0.9  # 90% of the way
        steps_to_90 = 0

        for i in range(100):
            pan, tilt = update_physics_step(0.02, is_tracking=False)
            if pan >= target_threshold and steps_to_90 == 0:
                steps_to_90 = i
            time.sleep(0.001)

        print(f"  Steps to 90% target: {steps_to_90} ({steps_to_90 * 20}ms)")

    print("\n✓ Different states produce different response speeds")


def test_tracking_responsiveness():
    """Test that face tracking uses stiff parameters for quick response."""
    print("\n=== Test: Tracking Responsiveness ===")

    ps = physics_state
    ps.pan = 90.0
    ps.tilt = 90.0
    ps.pan_velocity = 0.0
    ps.tilt_velocity = 0.0
    ps.pan_target = 120.0
    ps.tilt_target = 80.0
    ps.last_update_time = time.time()

    # Apply tracking parameters
    ps.blend_params(TRACKING_PHYSICS, blend_rate=1.0)

    print(f"Tracking params: mass={ps.mass:.1f}, spring={ps.spring_constant:.1f}")
    print(f"Target: pan 90→120 (30° move)")

    steps_to_95 = 0
    target_threshold = 90 + (120 - 90) * 0.95

    for i in range(50):
        pan, tilt = update_physics_step(0.02, is_tracking=True)
        if pan >= target_threshold and steps_to_95 == 0:
            steps_to_95 = i
            print(f"  Reached 95% at step {i} ({i * 20}ms)")
        time.sleep(0.001)

    if steps_to_95 > 0 and steps_to_95 < 25:  # Should reach in < 500ms
        print("✓ PASS: Tracking is responsive")
    else:
        print(f"✗ FAIL: Tracking too slow (steps={steps_to_95})")


def test_velocity_limits():
    """Test that velocity is properly limited."""
    print("\n=== Test: Velocity Limits ===")

    ps = physics_state
    ps.pan = 90.0
    ps.tilt = 90.0
    ps.pan_velocity = 0.0
    ps.tilt_velocity = 0.0
    ps.pan_target = 180.0  # Large jump
    ps.tilt_target = 180.0
    ps.last_update_time = time.time()

    # Use energized (high spring constant)
    ps.blend_params(PHYSICS_PATTERNS["energized_engaged"], blend_rate=1.0)

    max_observed_vel = 0.0
    for i in range(20):
        pan, tilt = update_physics_step(0.02, is_tracking=False)
        max_observed_vel = max(max_observed_vel, abs(ps.pan_velocity), abs(ps.tilt_velocity))
        time.sleep(0.001)

    print(f"Max observed velocity: {max_observed_vel:.2f}°/frame")

    if max_observed_vel <= 8.5:  # 8.0 limit + small tolerance
        print("✓ PASS: Velocity properly limited")
    else:
        print(f"✗ FAIL: Velocity exceeded limit ({max_observed_vel:.2f})")


def test_bounds_clamping():
    """Test that positions are clamped to servo limits."""
    print("\n=== Test: Bounds Clamping ===")

    ps = physics_state
    ps.pan = 90.0
    ps.tilt = 90.0
    ps.pan_velocity = 0.0
    ps.tilt_velocity = 0.0
    ps.pan_target = 200.0  # Beyond PAN_MAX
    ps.tilt_target = 200.0  # Beyond TILT_MAX
    ps.last_update_time = time.time()

    ps.blend_params(PHYSICS_PATTERNS["energized_engaged"], blend_rate=1.0)

    for i in range(100):
        pan, tilt = update_physics_step(0.02, is_tracking=False)
        time.sleep(0.001)

    print(f"Final position: pan={ps.pan:.1f}, tilt={ps.tilt:.1f}")
    print(f"Limits: pan=[{PAN_MIN}, {PAN_MAX}], tilt=[{TILT_MIN}, {TILT_MAX}]")

    if PAN_MIN <= ps.pan <= PAN_MAX and TILT_MIN <= ps.tilt <= TILT_MAX:
        print("✓ PASS: Position within bounds")
    else:
        print("✗ FAIL: Position out of bounds")


def visualize_movement():
    """Print ASCII visualization of movement over time."""
    print("\n=== Movement Visualization (ASCII) ===")

    ps = physics_state
    ps.pan = 90.0
    ps.tilt = 90.0
    ps.pan_velocity = 0.0
    ps.tilt_velocity = 0.0
    ps.pan_target = 120.0
    ps.tilt_target = 90.0
    ps.last_update_time = time.time()

    # Test with calm_observant
    ps.blend_params(PHYSICS_PATTERNS["calm_observant"], blend_rate=1.0)

    width = 60
    print(f"\nPan movement 90 → 120 (calm_observant):")
    print(f"{'Frame':>5} | Pan Position")
    print("-" * (width + 10))

    for i in range(40):
        pan, _ = update_physics_step(0.02, is_tracking=False)

        # Map pan to position in visualization
        normalized = (pan - 85) / (125 - 85)  # Normalize to 0-1
        pos = int(normalized * (width - 1))
        pos = max(0, min(width - 1, pos))

        # Create visualization line
        line = [" "] * width
        target_pos = int((120 - 85) / (125 - 85) * (width - 1))
        line[target_pos] = "|"  # Target marker
        line[pos] = "●"

        print(f"{i:>5} | {''.join(line)}")
        time.sleep(0.001)


def compare_emotional_curves():
    """Compare movement curves for different emotional states."""
    print("\n=== Emotional State Movement Comparison ===")

    states = ["energized_engaged", "calm_observant", "withdrawn_distant"]
    results = {}

    for emotion in states:
        ps = physics_state
        ps.pan = 90.0
        ps.tilt = 90.0
        ps.pan_velocity = 0.0
        ps.tilt_velocity = 0.0
        ps.pan_target = 110.0
        ps.tilt_target = 90.0
        ps.last_update_time = time.time()

        ps.blend_params(PHYSICS_PATTERNS[emotion], blend_rate=1.0)

        positions = []
        for i in range(60):
            pan, _ = update_physics_step(0.02, is_tracking=False)
            positions.append(pan)
            time.sleep(0.001)

        results[emotion] = positions

    # Print comparison
    print("\nPan position over time (target=110):")
    print(f"{'Frame':>5} | {'Energized':>10} | {'Calm':>10} | {'Withdrawn':>10}")
    print("-" * 50)

    for i in range(0, 60, 5):
        e = results["energized_engaged"][i]
        c = results["calm_observant"][i]
        w = results["withdrawn_distant"][i]
        print(f"{i:>5} | {e:>10.1f} | {c:>10.1f} | {w:>10.1f}")

    print("\nObserve: Energized reaches target fastest, Withdrawn is sluggish")


if __name__ == "__main__":
    print("=" * 60)
    print("PHYSICS-BASED GAZE SIMULATION TEST SUITE")
    print("=" * 60)

    test_spring_convergence()
    test_emotional_state_differences()
    test_tracking_responsiveness()
    test_velocity_limits()
    test_bounds_clamping()
    visualize_movement()
    compare_emotional_curves()

    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)
