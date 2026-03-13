#!/usr/bin/env python3
"""
Test drawing motivation calculation without full system initialization.
Simulates scores with various mood/novelty/boredom values.
"""
import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    DRAWING_BASE_THRESHOLD, DRAWING_NOVELTY_WEIGHT,
    DRAWING_BOREDOM_WEIGHT, DRAWING_MOOD_WEIGHT,
    DRAWING_MIN_INTERVAL, DRAWING_MAX_INTERVAL
)

def calculate_motivation(mood, novelty, boredom, time_since_last, max_interval):
    """Calculate drawing motivation score."""
    # Normalize mood from -1..1 to 0..1
    normalized_mood = max(0, min(1, (mood + 1) / 2))
    normalized_novelty = max(0, min(1, novelty))
    normalized_boredom = max(0, min(1, boredom))

    # Base motivation
    motivation_score = (
        normalized_novelty * DRAWING_NOVELTY_WEIGHT +
        normalized_boredom * DRAWING_BOREDOM_WEIGHT +
        normalized_mood * DRAWING_MOOD_WEIGHT
    )

    # Time pressure
    time_factor = min(1.0, time_since_last / max_interval)
    time_pressure = time_factor * 0.2

    total_motivation = motivation_score + time_pressure

    # Randomness range
    random_min = total_motivation - 0.1
    random_max = total_motivation + 0.1

    return {
        'mood_norm': normalized_mood,
        'novelty_norm': normalized_novelty,
        'boredom_norm': normalized_boredom,
        'base': motivation_score,
        'time_pressure': time_pressure,
        'total': total_motivation,
        'min_with_random': random_min,
        'max_with_random': random_max,
        'will_draw': random_max >= DRAWING_BASE_THRESHOLD
    }

def test_scenarios():
    print("=" * 80)
    print("DRAWING MOTIVATION SIMULATOR")
    print("=" * 80)

    print(f"\n⚙️ Configuration:")
    print(f"  Threshold: {DRAWING_BASE_THRESHOLD}")
    print(f"  Weights: Novelty={DRAWING_NOVELTY_WEIGHT}, Boredom={DRAWING_BOREDOM_WEIGHT}, Mood={DRAWING_MOOD_WEIGHT}")
    print(f"  Min interval: {DRAWING_MIN_INTERVAL}s ({DRAWING_MIN_INTERVAL//60}min)")
    print(f"  Max interval: {DRAWING_MAX_INTERVAL}s ({DRAWING_MAX_INTERVAL//60}min)")

    # Test various scenarios
    scenarios = [
        ("Neutral/Calm (typical idle)", 0.0, 0.2, 0.3),
        ("Low engagement", -0.2, 0.1, 0.5),
        ("High boredom", 0.0, 0.1, 0.8),
        ("High novelty (something new)", 0.3, 0.7, 0.4),
        ("Elevated mood + novelty", 0.5, 0.6, 0.5),
        ("Everything high", 0.7, 0.8, 0.9),
    ]

    test_times = [300, 900, 1500, 1800]  # 5, 15, 25, 30 minutes

    for scenario_name, mood, novelty, boredom in scenarios:
        print(f"\n{'='*80}")
        print(f"📊 Scenario: {scenario_name}")
        print(f"   Input: mood={mood:.1f}, novelty={novelty:.1f}, boredom={boredom:.1f}")
        print(f"{'='*80}")

        for time_since in test_times:
            result = calculate_motivation(mood, novelty, boredom, time_since, DRAWING_MAX_INTERVAL)
            mins = time_since // 60

            print(f"\n  ⏰ At {mins} minutes:")
            print(f"     Base motivation: {result['base']:.3f}")
            print(f"     + Time pressure: {result['time_pressure']:.3f}")
            print(f"     = Total: {result['total']:.3f}")
            print(f"     With randomness: {result['min_with_random']:.3f} - {result['max_with_random']:.3f}")

            if time_since >= DRAWING_MAX_INTERVAL:
                print(f"     ✅ FORCED DRAW (max interval)")
            elif result['will_draw']:
                print(f"     ✅ CAN DRAW (max score >= {DRAWING_BASE_THRESHOLD})")
            else:
                needed = DRAWING_BASE_THRESHOLD - result['max_with_random']
                print(f"     ❌ NO DRAW (need +{needed:.3f} more)")

    print(f"\n{'='*80}")
    print(f"💡 Analysis:")
    print(f"   - Threshold {DRAWING_BASE_THRESHOLD} requires elevated but achievable states")
    print(f"   - Time pressure helps as deadline approaches (+0.2 max)")
    print(f"   - Randomness adds ±0.1 variation")
    print(f"   - Max interval ({DRAWING_MAX_INTERVAL//60}min) ensures drawing happens eventually")

if __name__ == "__main__":
    test_scenarios()
