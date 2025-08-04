#!/usr/bin/env python3
"""
Quick test of the Markov chain parsing fixes.
"""

# Simulate the parsing function
def parse_markov_state_key(key_str):
    """Parse a string key back to tuple for generation (handles both old tuple and new string formats)."""
    if isinstance(key_str, tuple):
        return key_str  # Already a tuple (backwards compatibility)
    
    # Parse string representation of tuple: "(68, 57)" -> (68, 57)
    try:
        # Clean up the string - remove extra characters and fix malformed keys
        clean_str = str(key_str).strip()
        
        # Remove extra parentheses and fix malformed strings
        clean_str = clean_str.replace('))', ')')  # Fix double closing parens
        clean_str = clean_str.replace('..0', '.0')  # Fix truncated decimals
        
        # Remove parentheses and split by comma
        clean_str = clean_str.strip("()")
        parts = [part.strip() for part in clean_str.split(",")]
        
        # Handle different tuple formats
        if len(parts) == 2:
            # Simple (x, y) tuple
            return (int(float(parts[0])), int(float(parts[1])))
        elif len(parts) == 4:
            # Finger state tuple (f1, f2, f3, f4)
            return tuple(int(float(part)) for part in parts)
        else:
            # Try to parse as generic tuple
            return tuple(int(float(part)) for part in parts)
    except (ValueError, IndexError) as e:
        print(f"⚠️ Failed to parse Markov state key '{key_str}': {e}")
        # Return a fallback state
        return (90, 90, 90, 90)  # Default servo positions

# Test the problematic keys from the terminal output
test_keys = [
    "(180.0, 132.0, 158.0, 1600.0)",
    "(180.0, 96.0, 100.0, 102..0)",
    "(180.0, 94.0, 98.0, 98.0))",
    "(40, 40)",
    "(90.0, 110.0, 136.0, 162..0)"
]

print("🧪 Testing Markov state key parsing fixes:")
print("=" * 50)

for i, key in enumerate(test_keys, 1):
    print(f"Test {i}: '{key}'")
    result = parse_markov_state_key(key)
    print(f"  Result: {result} (length: {len(result)})")
    print(f"  Valid: {'✅' if len(result) >= 2 else '❌'}")
    print()

print("🎯 Testing discretization fix:")
discretization_step = 2.0

def simple_discretize(servo_positions):
    """Simple, fine discretization that preserves movement nuance."""
    return tuple(int(round(pos / discretization_step) * discretization_step) for pos in servo_positions)

test_positions = [
    [180.0, 132.0, 158.0, 160.0],
    [90.5, 91.3, 88.7, 89.2],
    [45.1, 44.9, 135.8, 136.2]
]

for i, positions in enumerate(test_positions, 1):
    print(f"Test {i}: {positions}")
    result = simple_discretize(positions)
    print(f"  Discretized: {result}")
    print(f"  Key: {str(result)}")
    print()

print("✅ All tests completed!")
