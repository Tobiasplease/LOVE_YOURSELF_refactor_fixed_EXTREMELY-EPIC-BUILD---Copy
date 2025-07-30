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
    CLEAN_CAPTION_OUTPUT,
)

# Physics-based servo positions and velocities
servo_x = 90.0
servo_y = 90.0
velocity_x = 0.0
velocity_y = 0.0

# Face tracking state with smoothing
face_target_x = 90.0
face_target_y = 90.0
face_lock_start = None
last_seen_time = time.time()
tracking_hold_until = None  # Hold last face position until this time
detection_loss_start = None  # Track when detection was first lost

# Face position smoothing to eliminate jitter
face_smooth_x = 90.0
face_smooth_y = 90.0
face_smoothing_factor = 0.15  # How much smoothing (0.1 = heavy, 0.3 = light)

# Detection loss tolerance - brief losses don't trigger hold period
DETECTION_LOSS_TOLERANCE = 0.5  # Allow 0.5s of detection loss before starting hold

# Physics parameters for idle movement
physics_target_x = IDLE_CENTER_X
physics_target_y = IDLE_CENTER_Y
physics_time = 0.0


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def update_gaze(frame, face_box, current_mood=0.0, delta_time=None):
    """
    Enhanced gaze tracking with improved stability for low-light conditions.
    
    Tracking flow:
    1. ACTIVE TRACKING: When face detected, smoothly track with jitter reduction
    2. DETECTION LOSS TOLERANCE: Brief losses (0.5s) don't trigger hold - handles flickering
    3. POSITION HOLD: After tolerance, hold last known position for FACE_LOCK_DURATION (8s)
    4. IDLE MOVEMENT: Finally transition to physics-based idle movement
    
    This prevents the "flimsy" tracking behavior where physics takes over immediately
    on detection loss, especially problematic in low lighting conditions.
    """
    global servo_x, servo_y, velocity_x, velocity_y
    global face_target_x, face_target_y, face_lock_start, last_seen_time
    global face_smooth_x, face_smooth_y, face_smoothing_factor
    global physics_target_x, physics_target_y, physics_time
    global tracking_hold_until, detection_loss_start  # Add tracking states

    h, w = frame.shape[:2]
    person_present = face_box is not None
    now = time.time()
    
    # Use provided delta_time or default to 30 FPS
    if delta_time is None:
        delta_time = 1.0 / 30.0
    
    # Clamp delta_time to reasonable bounds to prevent physics explosions
    delta_time = max(0.001, min(0.1, delta_time))

    # === SMOOTHED FACE TRACKING - JITTER REDUCTION ===
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
        
        # Only update if movement is significant to prevent micro-jitter
        face_movement = abs(dx) + abs(dy)
        if face_movement > DEAD_ZONE:
            # Calculate raw face targets with expanded range and higher sensitivity
            raw_target_x = clamp(90 + dx * 0.40, SERVO_MIN, SERVO_MAX)  # Increased from 0.25 for wider tracking
            raw_target_y = clamp(90 + dy * 0.40, SERVO_MIN, SERVO_MAX)  # Increased from 0.25 for wider tracking
            
            # Apply exponential smoothing to reduce jitter
            face_smooth_x += (raw_target_x - face_smooth_x) * face_smoothing_factor
            face_smooth_y += (raw_target_y - face_smooth_y) * face_smoothing_factor
            
            # Use smoothed targets for servo control
            servo_x = face_smooth_x
            servo_y = face_smooth_y
            
            # Zero out velocities when face tracking
            velocity_x = 0.0
            velocity_y = 0.0
            
            if face_lock_start is None:
                face_lock_start = now
                if not CLEAN_CAPTION_OUTPUT:
                    print(f"[GAZE] 🎯 SMOOTH FACE LOCK: Moving to ({servo_x:.1f},{servo_y:.1f})")
        
        # Update last seen time regardless of movement - PREVENTS BOUNCING
        last_seen_time = now
        
        # Clear tracking hold and detection loss tracking since we have active face detection
        tracking_hold_until = None
        detection_loss_start = None
        
        # Reset face lock timer to maintain tracking (prevents premature release)
        if face_lock_start is not None:
            face_lock_start = now
        
        # Debug face tracking every few seconds
        if int(now * 10) % 50 == 0 and not CLEAN_CAPTION_OUTPUT:  # Less frequent debug to reduce spam
            print(f"[GAZE] 👁️ SUSTAINED TRACKING: face=({face_center_x},{face_center_y}) servo=({servo_x:.1f},{servo_y:.1f})")
        
        # Return immediately when face tracking - skip all physics
        return person_present, int(round(servo_x)), int(round(servo_y))
        
        # Clear tracking hold and detection loss tracking since we have active face detection
        tracking_hold_until = None
        detection_loss_start = None
        
        # Reset face lock timer to maintain tracking (prevents premature release)
        if face_lock_start is not None:
            face_lock_start = now
        
        # Debug face tracking every few seconds
        if int(now * 10) % 50 == 0 and not CLEAN_CAPTION_OUTPUT:  # Less frequent debug to reduce spam
            print(f"[GAZE] 👁️ SUSTAINED TRACKING: face=({face_center_x},{face_center_y}) servo=({servo_x:.1f},{servo_y:.1f})")
        
        # Return immediately when face tracking - skip all physics
        return person_present, int(round(servo_x)), int(round(servo_y))

    # === DETECTION LOSS TOLERANCE - HANDLE BRIEF DETECTION GAPS ===
    # Don't immediately start hold period for brief detection losses (flickering in low light)
    if face_lock_start is not None:
        time_since_lost = now - last_seen_time
        
        # Start tracking detection loss time
        if detection_loss_start is None:
            detection_loss_start = now
            if not CLEAN_CAPTION_OUTPUT:
                print(f"[GAZE] ⚠️ Detection lost - tolerance period: {DETECTION_LOSS_TOLERANCE}s")
        
        # During tolerance period, maintain current position without starting hold
        if time_since_lost < DETECTION_LOSS_TOLERANCE:
            # Keep servos at last position, zero velocities
            velocity_x = 0.0
            velocity_y = 0.0
            
            # Debug tolerance status
            remaining_tolerance = DETECTION_LOSS_TOLERANCE - time_since_lost
            if int(now * 8) % 24 == 0 and not CLEAN_CAPTION_OUTPUT:  # Every 3 seconds
                print(f"[GAZE] ⌛ TOLERANCE: {remaining_tolerance:.1f}s remaining, maintaining ({servo_x:.1f},{servo_y:.1f})")
            
            # Return current position during tolerance period
            return False, int(round(servo_x)), int(round(servo_y))

    # === TRACKING HOLD PERIOD - MAINTAIN LAST FACE POSITION ===
    # After tolerance period expires, hold the last known position for FACE_LOCK_DURATION
    if face_lock_start is not None:
        # Set tracking hold period when tolerance period expires
        if tracking_hold_until is None:
            tracking_hold_until = now + FACE_LOCK_DURATION
            if not CLEAN_CAPTION_OUTPUT:
                print(f"[GAZE] ⏳ Tolerance expired - holding position for {FACE_LOCK_DURATION}s at ({servo_x:.1f},{servo_y:.1f})")
        
        # During hold period, maintain last face position
        if now < tracking_hold_until:
            # Keep servos at last face position, zero velocities
            velocity_x = 0.0
            velocity_y = 0.0
            
            # Debug hold status occasionally
            remaining_hold = tracking_hold_until - now
            if int(now * 4) % 20 == 0 and not CLEAN_CAPTION_OUTPUT:  # Every 5 seconds
                print(f"[GAZE] 🔒 HOLDING POSITION: {remaining_hold:.1f}s remaining at ({servo_x:.1f},{servo_y:.1f})")
            
            # Return current held position
            return False, int(round(servo_x)), int(round(servo_y))
        
        # Hold period expired - release to physics
        else:
            face_lock_start = None
            tracking_hold_until = None
            detection_loss_start = None
            if not CLEAN_CAPTION_OUTPUT:
                print(f"[GAZE] 🔓 Hold period expired after {FACE_LOCK_DURATION}s - transitioning to idle movement")

    # === IDLE MOVEMENT - ONLY AFTER TOLERANCE AND HOLD PERIODS ===

    # Physics-based idle movement with mood modulation
    physics_time += delta_time
    
    # Mood-based speed modulation
    base_speed_multiplier = 1.0 + (current_mood * 2.0)
    base_speed_multiplier = max(0.5, min(3.0, base_speed_multiplier))
    
    # Debug mood influence occasionally
    if int(physics_time * 10) % 50 == 0 and not CLEAN_CAPTION_OUTPUT:
        print(f"[GAZE] 😴 IDLE MODE - Mood: {current_mood:.2f} → Speed: {base_speed_multiplier:.2f}x")
    
    # Generate smooth idle targets
    idle_target_x = IDLE_CENTER_X + math.sin(physics_time * 0.3 * base_speed_multiplier) * IDLE_AMPLITUDE_X
    idle_target_y = IDLE_CENTER_Y + math.cos(physics_time * 0.4 * base_speed_multiplier) * IDLE_AMPLITUDE_Y
    
    # Physics simulation for idle movement
    force_x = (idle_target_x - servo_x) * PHYSICS_SPRING_FORCE
    force_y = (idle_target_y - servo_y) * PHYSICS_SPRING_FORCE
    
    velocity_x += force_x * delta_time
    velocity_y += force_y * delta_time
    
    velocity_x *= (1.0 - PHYSICS_FRICTION * delta_time)
    velocity_y *= (1.0 - PHYSICS_FRICTION * delta_time)
    
    servo_x += velocity_x * delta_time
    servo_y += velocity_y * delta_time
    
    # Clamp to servo limits
    servo_x = clamp(servo_x, SERVO_MIN, SERVO_MAX)
    servo_y = clamp(servo_y, SERVO_MIN, SERVO_MAX)

    return person_present, int(round(servo_x)), int(round(servo_y))
