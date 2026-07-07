# Unified Servo Controller Plan

## Goal
Merge the legacy 4-finger hand controller with the standalone 8-servo architecture.
Two input layers during recording, one unified Markov chain output.

## Current State

### Legacy system (this repo)
- 4 finger servos via `HAND,f0,f1,f2,f3` protocol
- 2 arm servos (pins 4,5) via separate `SERVO,pin,angle` (organic_left_arm.py)
- Arm servos are "slave" — read finger controller state, adjust their own parameters
- Second-order Markov chains with timing distributions (the good stuff)
- 5400-line monolithic hand_control_interface.py

### Standalone system (servocontroller repo, ui-modernize branch)
- 8 servos via `HAND8,f0,f1,f2,f3,f4,arm0,arm1,arm2` protocol
- hardware_config.json for declarative servo definitions
- autonomous_controller.py for standalone emotion detection
- No second-order Markov, no timing distributions

## Architecture

### Recording: Two Input Layers, One Output

```
Layer 1: Wave/cursor physics ──→ [f0, f1, f2, f3]     (existing finger control)
Layer 2: Individual sliders   ──→ [shoulder, elbow, wrist] (new arm control)
                                        │
                                        ▼
Combined frame @ 40Hz: [f0, f1, f2, f3, shoulder, elbow, wrist]
                                        │
                                        ▼
                              Markov chain state tuple
```

During recording, both layers capture simultaneously into one unified stream.
During generation, the Markov chain replays the full coordinated gesture.

### Layer 2 Controls (arm servos)

Individual servo control during recording:
- Keyboard mappings (similar to finger WASD but for arm servos)
- Sliders in UI for precise positioning
- Smoothing/easing applied to manual input for organic feel
- Each arm servo independently controllable

### Servo Definition (hardware_config.json)

Adopt from standalone repo — declarative servo configuration:
```json
{
  "servos": [
    {"name": "Index",    "pin": 11, "min": 0, "max": 180, "default": 90, "group": "fingers"},
    {"name": "Middle",   "pin": 10, "min": 0, "max": 180, "default": 90, "group": "fingers", "reversed": true},
    {"name": "Ring",     "pin": 9,  "min": 0, "max": 180, "default": 90, "group": "fingers", "reversed": true},
    {"name": "Pinky",    "pin": 8,  "min": 0, "max": 180, "default": 90, "group": "fingers"},
    {"name": "Shoulder", "pin": 7,  "min": 0, "max": 180, "default": 90, "group": "arm"},
    {"name": "Elbow",    "pin": 6,  "min": 0, "max": 180, "default": 90, "group": "arm"},
    {"name": "Wrist",    "pin": 5,  "min": 0, "max": 180, "default": 90, "group": "arm"}
  ],
  "protocol": "HAND8",
  "baudrate": 9600
}
```

Adding a servo = add a line to config. No code changes.

### Serial Protocol

Switch from `HAND,f0,f1,f2,f3` + `SERVO,pin,angle` to unified:
`HAND8,f0,f1,f2,f3,arm0,arm1,arm2\n`

Single serial connection, single command per update cycle.
Arduino firmware: use the 8-servo listener from standalone repo.

### Markov Chain Changes

State tuple expands from 4D to 7D:
- Old: `"(f0, f1, f2, f3, phase)"`
- New: `"(f0, f1, f2, f3, shoulder, elbow, wrist, phase)"`

Everything else stays the same:
- Second-order chains with timing distributions
- Discretization at 2 degrees
- Diversity injection, stuck detection
- Dataset cycling, emotion switching

### What Gets Removed

- `hand_control/organic_left_arm.py` — arm servos integrated into main controller
- Separate serial connection for arm servos in machine.py
- `SERVO,pin,angle` protocol path in hand_expression.py
- The "slave" coupling between arm and finger controllers

### What Stays

- Wave/cursor physics for finger control (Layer 1)
- Second-order Markov chain engine
- Emotional state system (5 states)
- Dataset recording/loading/cycling
- Camera reactivity pause/resume
- Mood engine integration via direct_hand_control.py

## Implementation Steps

### Phase 1: Hardware & Protocol
1. Add `hardware_config.json` to hand_control/
2. Update `hand_expression.py` to send HAND8 commands (7 servos)
3. Flash Arduino with 8-servo listener firmware (from standalone repo)
4. Single serial connection in machine.py

### Phase 2: Layer 2 Input (Arm Servos)
5. Add arm servo state tracking to CleanCursorInterface
6. Add keyboard mappings for arm servos (y/h, u/j, i/k or similar)
7. Add arm servo sliders to UI
8. Apply smoothing/easing to arm servo input

### Phase 3: Unified Recording & Markov
9. Expand recording to capture 7-servo frames
10. Expand Markov state tuples to 7D
11. Update build_markov_chain() discretization for 7 servos
12. Update step_markov_generation() to output 7 servo positions
13. Update send_to_hand_controller() for 7-servo output

### Phase 4: Cleanup
14. Remove organic_left_arm.py
15. Remove separate arm serial connection from machine.py
16. Update machine.py init to use unified controller
17. Migrate or re-record datasets (old 4D datasets still loadable with arm defaults)

## Dataset Backwards Compatibility

Old 4-servo datasets: load normally, pad arm positions with defaults (90).
New 7-servo datasets: full fidelity.
Detection: check state tuple length on load.
