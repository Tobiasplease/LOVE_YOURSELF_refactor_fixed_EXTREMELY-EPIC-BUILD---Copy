import time
import random
import math
from config.config import (
    SERVO_MIN,
    SERVO_MAX,
    FLIP_X,
    FLIP_Y,
    DEAD_ZONE,
    IDLE_CENTER_X,
    IDLE_CENTER_Y,
    IDLE_AMPLITUDE_X,
    IDLE_AMPLITUDE_Y,
    PHYSICS_FRICTION,
    PHYSICS_SPRING_FORCE,
    FACE_LOCK_DURATION,
    BLEND_SPEED,
    CONFIDENCE_THRESHOLD,
)

# Physics-based servo positions and velocities
servo_x = 90.0
servo_y = 90.0
velocity_x = 0.0
velocity_y = 0.0

# Face tracking state
face_target_x = 90.0
face_target_y = 90.0
face_lock_start = None
last_seen_time = time.time()

# Physics parameters for idle movement
physics_target_x = IDLE_CENTER_X
physics_target_y = IDLE_CENTER_Y
physics_time = 0.0


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def update_gaze(frame, face_box, current_mood=0.0, delta_time=None):
    global servo_x, servo_y, velocity_x, velocity_y
    global face_target_x, face_target_y, face_lock_start, last_seen_time
    global physics_target_x, physics_target_y, physics_time

    h, w = frame.shape[:2]
    person_present = face_box is not None
    now = time.time()
    
    # Use provided delta_time or default to 30 FPS
    if delta_time is None:
        delta_time = 1.0 / 30.0
    
    # Clamp delta_time to reasonable bounds to prevent physics explosions
    delta_time = max(0.001, min(0.1, delta_time))

    # === FACE TRACKING ===
    if person_present:
        (startX, startY, endX, endY) = face_box
        face_center_x = (startX + endX) // 2
        face_center_y = (startY + endY) // 2
        
        if FLIP_X:
            face_center_x = w - face_center_x
        if FLIP_Y:
            face_center_y = h - face_center_y

        # Convert face position to servo angles
        dx = face_center_x - (w // 2)
        dy = face_center_y - (h // 2)
        
        # Only update face targets if movement is significant
        face_movement = abs(dx) + abs(dy)
        if face_movement > DEAD_ZONE:
            # Calculate new face targets based on position (moderate sensitivity)
            face_target_x = clamp(90 + dx * 0.20, SERVO_MIN, SERVO_MAX)  # Reduced from 0.25
            face_target_y = clamp(90 + dy * 0.20, SERVO_MIN, SERVO_MAX)  # Reduced from 0.25
            
            if face_lock_start is None:
                face_lock_start = now
        
        last_seen_time = now

    # Calculate blend factor for smooth transition between idle and face tracking
    blend_factor = 0.0
    if face_lock_start is not None and now - face_lock_start < FACE_LOCK_DURATION:
        # Ramp up blend factor over time
        blend_progress = (now - face_lock_start) / BLEND_SPEED
        blend_factor = min(1.0, blend_progress)
    else:
        # Face lock expired or no face - reset
        if face_lock_start is not None:
            face_lock_start = None

    # === PHYSICS-BASED IDLE MOVEMENT ===
    physics_time += delta_time
    
    # Generate smooth, organic idle targets using sine waves
    idle_target_x = IDLE_CENTER_X + math.sin(physics_time * 0.3) * IDLE_AMPLITUDE_X + math.sin(physics_time * 0.7) * (IDLE_AMPLITUDE_X * 0.3)
    idle_target_y = IDLE_CENTER_Y + math.cos(physics_time * 0.4) * IDLE_AMPLITUDE_Y + math.cos(physics_time * 0.9) * (IDLE_AMPLITUDE_Y * 0.4)
    
    # Blend between idle and face tracking targets
    final_target_x = idle_target_x * (1.0 - blend_factor) + face_target_x * blend_factor
    final_target_y = idle_target_y * (1.0 - blend_factor) + face_target_y * blend_factor
    
    # === PHYSICS SIMULATION ===
    # Calculate spring forces toward targets
    force_x = (final_target_x - servo_x) * PHYSICS_SPRING_FORCE
    force_y = (final_target_y - servo_y) * PHYSICS_SPRING_FORCE
    
    # Apply forces to velocity
    velocity_x += force_x * delta_time
    velocity_y += force_y * delta_time
    
    # Apply friction
    velocity_x *= (1.0 - PHYSICS_FRICTION * delta_time)
    velocity_y *= (1.0 - PHYSICS_FRICTION * delta_time)
    
    # Update positions
    servo_x += velocity_x * delta_time
    servo_y += velocity_y * delta_time
    
    # Clamp to servo limits
    servo_x = clamp(servo_x, SERVO_MIN, SERVO_MAX)
    servo_y = clamp(servo_y, SERVO_MIN, SERVO_MAX)

    return person_present, int(round(servo_x)), int(round(servo_y))
