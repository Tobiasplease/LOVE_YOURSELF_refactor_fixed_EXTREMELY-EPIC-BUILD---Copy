# Gaze System Improvements

## Summary

The gaze pan/tilt system has been significantly improved to address aggressive, snappy movement and create more organic, lifelike behavior while maintaining responsive face tracking.

## Key Problems Addressed

### ❌ Before: Issues Identified
1. **Synchronous Movement**: Pan and tilt servos moved in lockstep with identical timing patterns
2. **Aggressive Idle**: High amplitude movements (35°/30°) with short pauses (1.5-6s) created restless behavior
3. **Limited Range**: Only ±25° pan, ±20° tilt felt constrained for natural head movement
4. **Binary Movement**: Idle system used harsh "big sweep vs small jitter" choices
5. **Low Update Rate**: 50Hz servo updates with coarse movement steps

### ✅ After: Improvements Implemented

## 1. Expanded Movement Range
- **Pan Range**: 55-125° (±35° from center) - was 65-115° (±25°)
- **Tilt Range**: 60-120° (±30° from center) - was 70-110° (±20°)
- **Pause Duration**: 3-12s (was 1.5-6s) for more contemplative idle behavior

## 2. High-Frequency Smooth Movement
- **Update Rate**: Increased to 100Hz (was 50Hz) for ultra-smooth servo motion
- **Angle Threshold**: Reduced to 0.3° (was 0.5°) for finer movement resolution
- **Command Interval**: 10ms between commands (was 20ms) for silk-smooth motion

## 3. Organic Idle Movement System
**Replaced binary sweep/jitter with Perlin noise-based organic patterns:**

- **Independent Pan/Tilt**: Separate frequency patterns (0.08Hz pan, 0.06Hz) create natural asynchronous movement
- **Perlin Noise**: Smooth, continuous curves replace jerky random movements
- **Emotional Scaling**: Movement intensity and frequency scale with emotional state
- **Natural Bias**: Slight downward tilt bias for natural head position

## 4. Responsive Face Tracking
**Maintained quick tracking response while adding wider range:**

- **Direct Mapping**: Face position → servo position without curves during tracking
- **Synchronized Movement**: Pan/tilt move together during tracking for responsiveness
- **Expanded Range**: Can track faces across much wider field of view
- **Clean Transitions**: Smooth handoff between idle ↔ tracking states

## 5. Clean State Machine
**Improved state transitions without movement artifacts:**

- **State Tracking**: Monitor transitions between idle/tracking/grace states
- **Position Freezing**: Lock current position during state changes
- **Smooth Handoff**: Gradual blend between idle organic movement and tracking

## Technical Implementation

### New Functions Added:
- `perlin_noise_1d()`: Generates smooth organic movement patterns
- `bezier_curve()`: For future curved movement interpolation
- `update_organic_movement()`: Replaces binary sweep system with noise-based patterns

### Movement Patterns:
```python
# Independent frequencies for natural asynchronous movement
pan_frequency = 0.08 * emotional_scale    # Pan moves at different rate than tilt
tilt_frequency = 0.06 * emotional_scale   # Creates complex, lifelike patterns

# Emotional scaling affects movement intensity
movement_range = base_range * emotional_pattern["movement_scale"]
```

### Servo Control Improvements:
```python
MIN_COMMAND_INTERVAL = 0.01  # 100Hz updates (was 50Hz)
ANGLE_THRESHOLD = 0.3        # Finer resolution (was 0.5°)
```

## Results

### Face Tracking:
- ✅ **Responsive**: Quick response to face movement across expanded range
- ✅ **Wide Range**: 55-125° pan, 60-120° tilt for natural head movement
- ✅ **Smooth**: 100Hz updates create silk-smooth tracking motion

### Idle Movement:
- ✅ **Organic**: Perlin noise creates natural, contemplative movement patterns
- ✅ **Independent**: Pan and tilt move asynchronously with different timing
- ✅ **Emotional**: Movement intensity scales appropriately with mood state
- ✅ **Contemplative**: Longer pauses (3-12s) feel more natural

### State Transitions:
- ✅ **Clean**: Smooth switching between idle and tracking without jerks
- ✅ **Responsive**: Immediate tracking response when face appears
- ✅ **Natural**: Gradual return to organic movement when face disappears

## Testing

Created comprehensive test suite in `debug/test_improved_gaze_system.py`:

1. **Organic Idle Test**: Verifies smooth, independent pan/tilt movement
2. **Face Tracking Test**: Confirms responsive tracking across expanded range
3. **State Transition Test**: Validates clean handoff between states

### Sample Results:
```
Face tracking now reaches:
- PAN: 63° to 117° (expanded from 74° to 106°)
- TILT: 72° to 107° (expanded from 84° to 96°)

Idle movement shows:
- Independent pan/tilt deltas (not synchronized)
- Smooth organic curves (not jerky jumps)
- Natural contemplative pauses
```

## Configuration Changes

### config/config.py:
```python
# Expanded movement range
PAN_MIN = 55    # was 65
PAN_MAX = 125   # was 115
TILT_MIN = 60   # was 70
TILT_MAX = 120  # was 110

# More contemplative idle timing
IDLE_PAUSE_MIN = 3.0   # was 1.5
IDLE_PAUSE_MAX = 12.0  # was 6.0
```

### servo_control/servo_control.py:
```python
# Ultra-smooth movement
ANGLE_THRESHOLD = 0.3       # was 0.5
MIN_COMMAND_INTERVAL = 0.01 # was 0.02 (100Hz vs 50Hz)
```

## Impact

The gaze system now feels **organic and lifelike** during idle periods while maintaining **responsive and accurate** face tracking. The movement patterns are **contemplative rather than restless**, with **natural curves instead of mechanical jerks**.

Key achievement: **Independent pan/tilt movement creates complex, natural patterns** that feel truly organic while **preserving instant tracking responsiveness** when faces appear.