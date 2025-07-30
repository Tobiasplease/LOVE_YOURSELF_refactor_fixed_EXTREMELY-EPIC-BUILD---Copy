# Hand Controller Setup Guide

This guide explains how to set up the robotic hand integration with your AI consciousness system.

## Hardware Setup

### Arduino Connections
- **Power**: 5V and GND to Arduino
- **Finger Servos**:
  - Index finger (servo 0): Pin 8
  - Middle finger (servo 1): Pin 9  
  - Ring finger (servo 2): Pin 10 (mirrored)
  - Pinky finger (servo 3): Pin 11 (mirrored)

### Serial Ports
- **COM10**: Main servos (gaze/breathing) - existing setup
- **COM3**: Hand controller - new addition

## Software Setup

### 1. Arduino Code
Upload `arduino_src/hand_controller.ino` to your Arduino connected to COM11.

The Arduino will:
- Accept commands from Python via serial
- Maintain your existing tapping behavior when not receiving consciousness commands
- Smoothly transition between autonomous and consciousness-driven movement
- Handle mirrored servos (pins 10 & 11) automatically

### 2. Python Configuration
The system automatically configures dual serial ports:

```python
# config/config.py
SERIAL_PORT = "COM10"        # Main servos
HAND_SERIAL_PORT = "COM11"   # Hand controller
```

### 3. Testing
Run the test script to verify everything works:

```bash
python debug/test_hand_controller.py
```

## Gesture Mapping

The AI consciousness maps to these hand gestures:

| Consciousness State | Hand Gesture | Description |
|-------------------|--------------|-------------|
| Low mood (< 0.2) | Withdrawn | Fingers curl inward, protective |
| High mood + novelty | Expressive | Animated, open gestures |
| Person detected + curiosity | Curious | Index finger extended, exploring |
| High boredom | Restless | Mid-range positions, ready to move |
| Contemplative | Thoughtful | Gentle curves, slow breathing motion |
| Freshly awakened | Awakening | Slow uncurling from rest position |
| Default | Idle | Your original tapping behavior |

## Serial Communication Protocol

Commands sent from Python to Arduino:
```
HAND,finger0,finger1,finger2,finger3\n
```

Example:
```
HAND,70,80,90,60\n  # Sets each finger to specified angle
```

## Troubleshooting

### No Serial Connection
- Verify Arduino is connected to COM3
- Check if COM3 is available in Device Manager
- Ensure Arduino IDE is closed (releases serial port)
- Try unplugging/reconnecting Arduino

### Hand Not Moving
- Check serial monitor in Arduino IDE for error messages
- Verify servo connections and power supply
- Run test script with debug output
- Check if consciousness timeout occurred (5 seconds without commands)

### Jerky Movement
- Adjust speed parameters in Arduino code
- Check servo power supply stability
- Verify servo mounting isn't binding

## Integration with Main System

The hand controller automatically integrates with your consciousness system:

1. **Mood changes** trigger appropriate gestures
2. **Person detection** makes hand more expressive  
3. **Temporal awareness** affects awakening sequences
4. **Reflection periods** may cause contemplative gestures
5. **Drawing decisions** can trigger expressive movements

The hand adds another layer of embodied consciousness to your mirror system, making the AI's internal states visible through organic finger movements! 🤖✋
