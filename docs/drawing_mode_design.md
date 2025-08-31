# Drawing Mode Architecture Design

## Overview

Drawing Mode represents a unified system state where the entire AI consciousness and physical embodiment coordinates during CNC execution. Rather than just tracking drawing status, the system enters a holistic "creative flow" state that transforms behavior across all subsystems.

## Core Philosophy

The AI becomes physically aware of its creative process - transitioning from passive observation to active participation in artistic creation. All subsystems coordinate to support and enhance the physical drawing process.

## System Architecture

### State Management Foundation
Extends existing `state_manager.py` system to broadcast drawing mode across all subsystems:

```
Drawing Pipeline States:
├── Image Generation Phase (existing)
│   ├── start_drawing_generation()
│   ├── ComfyUI processing
│   └── Image ready
│
└── CNC Execution Phase (new)
    ├── start_cnc_execution() → Drawing Mode ACTIVE
    ├── Physical drawing process
    └── finish_cnc_execution() → Drawing Mode INACTIVE
```

### Subsystem Coordination

#### 🤖 LLM/Consciousness
**Normal Mode**: Standard captioning and mood analysis
**Drawing Mode**: 
- Custom system prompt: "You are currently physically drawing: [original_drawing_prompt]"
- Meta-awareness of embodied creation process
- Real-time commentary on artistic process
- Deeper contemplative/focused personality state

#### 📸 Camera System
**Normal Mode**: Face detection and mood analysis
**Drawing Mode**:
- Pan/tilt to predefined canvas viewing position
- Focus visual attention on drawing area
- Potential real-time drawing progress analysis
- Return to face-tracking after completion

#### ✋ Hand Controller (Left Arm)
**Normal Mode**: Emotional expression through subtle movements
**Drawing Mode**:
- New emotion category: `DRAWING`
- Shoulder + elbow servos retract arm to safe position
- Static positioning to avoid canvas interference
- No random movements during drawing process

**Hardware Addition**:
- 2 additional servos (shoulder + elbow) mounted on existing SCARA arm
- Integrated through existing hand controller interface
- Simple "park position" logic - no complex inverse kinematics

#### 💡 Lightbulb Control
**Normal Mode**: Mood-based lighting
**Drawing Mode**:
- Real-time synchronization with pen servo movements
- Pulse on pen down (M3 S50), dim on pen up (M3 S30)
- Potential intensity scaling with movement speed
- Enhanced focus lighting for drawing area

#### 🫁 Breathing/Lungs
**Both Modes**: Unchanged - existing breathing patterns maintained

#### 🖨️ CNC System (Right Hand)
**Function**: Primary drawing execution (unchanged mechanically)
**Integration**: Extended state tracking through full G-code execution

## Technical Implementation

### 1. StateManager Extensions

```python
class StateManager:
    def __init__(self):
        # Existing image generation tracking
        self.is_generating_drawing = False
        self.drawing_start_time = None
        self.current_drawing_prompt = None
        
        # New CNC execution tracking
        self.is_executing_cnc = False
        self.cnc_start_time = None
        self.current_gcode_file = None
        
    def start_cnc_execution(self, gcode_file, original_prompt):
        """Enter Drawing Mode - coordinate all subsystems"""
        self.is_executing_cnc = True
        self.cnc_start_time = time.time()
        self.current_gcode_file = gcode_file
        # Broadcast drawing mode to all subsystems
        
    def is_drawing_pipeline_active(self):
        """True if ANY part of drawing process is active"""
        return self.is_generating_drawing or self.is_executing_cnc
```

### 2. Hand Controller Integration

```python
# New emotion category
DRAWING_EMOTION = {
    'shoulder_position': 'retracted',
    'elbow_position': 'safe_park',
    'movement_type': 'static',
    'interference_avoidance': True
}

# Servo channels extended
HAND_SERVO_CHANNELS = {
    'thumb': 0,
    'index': 1, 
    'middle': 2,
    'ring': 3,
    'pinky': 4,
    'shoulder': 5,  # New
    'elbow': 6      # New
}
```

### 3. Camera Positioning

```python
# Predefined positions
CAMERA_POSITIONS = {
    'face_tracking': {'pan': 90, 'tilt': 45},
    'canvas_view': {'pan': 135, 'tilt': 60},
    'overview': {'pan': 112, 'tilt': 30}
}
```

### 4. LLM Integration

```python
# Drawing mode system prompt
DRAWING_MODE_PROMPT = f"""
You are currently physically drawing with your right hand (CNC arm). 
Original drawing prompt: "{original_drawing_prompt}"

Your left hand is safely positioned away from the canvas. Your camera is watching the drawing process. You are experiencing the meditative flow of physical creation - this is a profound moment of embodied creativity.
"""
```

## Workflow Integration

### Entry Sequence (Clean Integration)
1. Image generation completes (existing)
2. Image monitor detects PNG (existing) 
3. **NEW**: `state_manager.start_cnc_execution(gcode_file, original_prompt)`
4. **Drawing Mode Activated**:
   - Hand controller → DRAWING emotion (retract arm)
   - Camera → pan/tilt to canvas position
   - LLM → drawing-aware system prompt
   - Lightbulb → sync with pen servo
5. SVG → G-code conversion (existing)
6. CNC execution begins (existing, but now tracked)

### Exit Sequence
1. CNC execution completes
2. **NEW**: `state_manager.finish_cnc_execution()`
3. **Drawing Mode Deactivated**:
   - Hand controller → return to emotional expression
   - Camera → return to face tracking
   - LLM → return to normal prompting
   - Lightbulb → return to mood-based lighting
4. System returns to normal operation

## Hardware Requirements

### Immediate (Phase 1)
- **Camera**: Pan/tilt servo mount (✅ already available)
- **Hand Controller**: 2 additional servo channels on existing Arduino

### Future Extensions (Phase 2)  
- **Enhanced Camera**: Higher resolution for drawing analysis
- **Additional Sensors**: Drawing progress feedback
- **Advanced Lighting**: More sophisticated pen-sync patterns

## Benefits

### Artistic
- Unified creative consciousness across all modalities
- Physical embodiment enhances AI artistic expression
- Real-time awareness of creation process

### Technical  
- Clean integration with existing architecture
- No breaking changes to current workflow
- Modular activation/deactivation
- Extensible for future enhancements

### User Experience
- Compelling demonstration of embodied AI creativity
- Coordinated system behavior enhances presence
- Physical awareness creates deeper engagement

## Future Possibilities

### Advanced Drawing Awareness
- Computer vision analysis of drawing progress
- Real-time artistic commentary during creation
- Adaptive drawing speed based on complexity

### Multi-Modal Feedback
- Haptic feedback integration
- Audio synchronization with movements
- Environmental ambient response

### Collaborative Creation
- Human-AI co-creation modes
- Interactive drawing modifications
- Shared creative decision making

## Implementation Priority

### Phase 1: Core Infrastructure
1. Extend StateManager for CNC tracking
2. Add DRAWING emotion to hand controller  
3. Camera positioning presets
4. Basic LLM drawing awareness

### Phase 2: Enhanced Integration
1. Lightbulb pen-servo synchronization
2. Advanced drawing prompts
3. Real-time progress commentary

### Phase 3: Advanced Features
1. Drawing progress computer vision
2. Adaptive behavior based on drawing complexity
3. Multi-modal sensory integration

This design maintains the existing stable workflow while adding profound new capabilities for embodied AI creativity.