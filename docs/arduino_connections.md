# Arduino Connections Documentation

## Current Configuration (Linux)

### Connected Devices

1. **Lightbulb PWM Controller** - `/dev/ttyUSB1`
   - Protocol: Frame difference JSON (`{"frame_diff": 0.0-1.0}`)
   - Baud rate: 9600
   - Purpose: Controls lightbulb brightness based on motion detection
   - Startup message: "Frame diff lightbulb controller ready"

2. **Hand Controller** - `/dev/ttyUSB2`
   - Protocol: Pure Consciousness Mode (sends continuous data)
   - Baud rate: 9600
   - Purpose: Emotional hand gesture expression
   - Startup message: "Hand Controller Ready - Pure Consciousness Mode"
   - Auto-connects when available

3. **Servo/Lung System** - `/dev/ttyUSB3` (primary) or `/dev/ttyUSB0` (backup)
   - Protocol: Simple position commands (e.g., "90\n")
   - Baud rate: 9600
   - Purpose: Gaze tracking servos and breathing simulation
   - Note: Multiple servo controllers may be connected

### Future Devices (Reserved Ports)

4. **GRBL CNC Controller** - `/dev/ttyUSB4`
   - Purpose: CNC drawing/engraving operations
   - Status: Not yet connected

5. **uArm Swift Pro** - `/dev/ttyUSB5`
   - Purpose: Robotic arm control
   - Status: Not yet connected

## Testing Tools

### Identify All Arduinos
```bash
python debug/identify_arduinos.py
```

### Test Specific Devices
```bash
# Test lightbulb with frame difference protocol
python debug/test_frame_diff_debug.py

# Test all Arduino connections
python debug/test_all_arduinos.py

# Setup guide for configuring ports
python debug/setup_arduino_ports.py

# Direct Arduino protocol test
python debug/test_arduino_direct.py
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Add user to dialout group
   sudo usermod -a -G dialout $USER
   # Log out and back in, or use:
   newgrp dialout
   ```

2. **Port Not Found**
   - Check connections: `ls -la /dev/ttyUSB*`
   - Verify Arduino is powered and connected
   - Try unplugging and reconnecting

3. **Device Not Responding**
   - Check correct sketch is uploaded to Arduino
   - Verify baud rate matches (9600 for most devices)
   - Close any other programs using the port (Arduino IDE, etc.)

4. **Wrong Device on Port**
   - Run `python debug/identify_arduinos.py` to re-identify
   - Update config.py with correct port assignments

## Port Assignment Process

1. Connect all Arduino devices
2. Run `python debug/identify_arduinos.py`
3. Note which device responds on which port
4. Update config/config.py with correct assignments
5. Test with `python debug/test_all_arduinos.py`

## Config Variables

In `config/config.py`:
```python
SERIAL_PORT = "/dev/ttyUSB3"  # Servo/lung system
LIGHTBULB_SERIAL_PORT = "/dev/ttyUSB1"  # Lightbulb PWM
HAND_CONTROLLER_PORT = "/dev/ttyUSB2"  # Hand controller
GRBL_CNC_PORT = "/dev/ttyUSB4"  # Future: GRBL CNC
UARM_SWIFT_PORT = "/dev/ttyUSB5"  # Future: uArm
```