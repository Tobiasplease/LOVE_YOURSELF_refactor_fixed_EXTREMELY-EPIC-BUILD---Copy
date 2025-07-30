"""
Consciousness-Driven Cursor System
==================================

Emotional puppeteering with dynamic behavioral parameters.
Features face tracking, gaze following, vibration, pulsation, direction changes,
wall bouncing, temporal behaviors, and multi-layer movement patterns.

Author: Advanced Emotional AI System
"""

import time
import math
import random


class ConsciousnessState:
    """Container for all consciousness data from machine.py"""
    
    def __init__(self):
        # Emotional State
        self.mood = 0.0  # -1.0 to 1.0 (negative=sad, positive=happy)
        self.novelty = 0.0  # 0.0 to 1.0 (how interesting/novel current situation)
        self.boredom = 0.0  # 0.0 to 1.0 (how bored the AI is)
        
        # Perception & Attention  
        self.person_present = False  # Human detected
        self.face_confidence = 0.0  # 0.0 to 1.0 (how confident face detection is)
        self.face_position = (0.5, 0.5)  # Normalized (x, y) of detected face
        
        # Physiological State
        self.breath_speed = 0.7  # Breathing rhythm
        self.breath_paused = False  # Held breath = stillness
        self.lung_angle = 0.0  # Breathing cycle position
        
        # Gaze & Attention (NEW - from your machine.py)
        self.gaze_pan = 90.0  # Where AI is "looking" horizontally  
        self.gaze_tilt = 90.0  # Where AI is "looking" vertically
        
        # YOLO Object Detection Data (NEW)
        self.detected_objects = []  # List of detected objects with positions
        self.primary_object = None  # Most interesting detected object
        self.object_confidence = 0.0  # Confidence in primary object
        
        # Caption Analysis (live consciousness stream)
        self.last_caption = ""  # Most recent AI thoughts
        self.caption_sentiment = 0.0  # Emotional sentiment of caption
        
        # Temporal Context
        self.time_since_change = 0.0  # Time since last significant change
        self.attention_focus = 0.5  # How focused vs scattered attention is
        
        # Special Events
        self.startle_triggered = False  # Recent startle reaction
        self.startle_time = 0.0  # When startle occurred


class ConsciousCursor:
    """
    Consciousness-driven cursor with dynamic behavioral parameters.
    
    Features:
    - Real-time parameter adjustment via sliders
    - Face tracking and object attention
    - Gaze following from AI's pan/tilt data
    - Multiple behavioral modes (exploring, focusing, vibrating, pulsating, lingering)
    - Direction changes with persistence and cooldowns
    - Wall bouncing with realistic physics
    - Temporal behaviors (pausing, bursting, rhythm sync)
    - Multi-layer noise and movement patterns
    - Emotional state modulation of all parameters
    """
    
    def __init__(self, canvas_width=None, canvas_height=None):
        # Position and velocity
        self.x = 0.5
        self.y = 0.5
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        
        # Canvas dimensions (optional, for compatibility)
        self.canvas_width = canvas_width or 563
        self.canvas_height = canvas_height or 304
        
        # Movement parameters (adjustable via sliders) - ENHANCED LIVELINESS
        self.base_speed = 2.5  # Much faster base movement
        self.dampening = 0.88  # Even less dampening for more activity
        self.emotional_influence = 1.5  # Stronger emotional response
        self.novelty_speed_multiplier = 4.0  # Bigger novelty boosts
        
        # Behavioral Mode Parameters - FASTER & MORE DYNAMIC
        self.behavioral_mode = "exploring"  # exploring, focusing, wandering, lingering, vibrating, pulsating
        self.behavior_timer = 0.0
        self.behavior_transition_interval = 2.0  # Even faster mode changes  
        self.lingering = False
        self.linger_timer = 0.0
        
        # Face Tracking Parameters
        self.face_tracking_enabled = True
        self.face_tracking_strength = 0.8
        self.face_tracking_smoothing = 0.9
        
        # Gaze Following Parameters  
        self.gaze_following_enabled = True
        self.gaze_following_strength = 0.6
        self.gaze_following_smoothing = 0.8
        
        # Object Attention Parameters
        self.object_attention_enabled = True
        self.object_attention_strength = 0.5
        self.object_attention_range = 0.3
        
        # Vibration Parameters
        self.vibration_active = False
        self.vibration_intensity = 0.05
        self.vibration_frequency = 10.0
        self.vibration_phase = 0.0
        
        # Pulsation Parameters
        self.pulsation_active = False
        self.pulsation_rate = 2.0
        self.pulsation_amplitude = 0.1
        self.pulsation_phase = 0.0
        
        # Direction Change Parameters
        self.direction_persistence = 0.7
        self.direction_change_chance = 0.1
        self.direction_change_cooldown = 0.0
        self.current_direction_x = 0.0
        self.current_direction_y = 0.0
        
        # Wall Bouncing Parameters - DISABLED TO PREVENT BOUNCE LOOPS
        self.wall_bounce_enabled = False  # Disabled - was causing excessive bouncing
        self.wall_bounce_strength = 0.0  # Disabled
        self.wall_collision_cooldown = 0.0
        self.wall_collision_active = False
        self._bounce_force_x = 0.0
        self._bounce_force_y = 0.0
        
        # Temporal Behavior Parameters - ENHANCED FOR LIVELINESS
        self.pause_probability = 0.03        # Slightly less pausing
        self.pause_duration = 0.7           # Shorter pauses
        self.burst_movement_chance = 0.08   # Much more bursts!
        self.burst_timer = 0.0
        self.rhythm_sync_strength = 0.6     # Stronger rhythm influence
        self._burst_direction_x = 0.0
        self._burst_direction_y = 0.0
        
        # Noise & Chaos Parameters - MUCH MORE DYNAMIC
        self.base_noise_level = 0.05        # More base noise
        self.chaos_multiplier = 0.25        # Much more chaos from novelty
        self.micro_jitter = 0.025           # More micro-movements
        
        # Internal state
        self.movement_history = []
        self.current_behavior = "exploring"
        self.time_accumulator = 0.0
        
        print("🚀 ConsciousCursor initialized - Ready for emotional puppeteering!")
    
    def update(self, consciousness, delta_time=0.016):
        """
        Consciousness-driven cursor update with dynamic behavioral patterns.
        
        Multi-layer movement calculation:
        1. Base emotional positioning (mood, novelty, boredom)
        2. Face tracking (if person present)
        3. Gaze following (AI's pan/tilt data)
        4. Behavioral patterns (vibration, pulsation, etc.)
        5. Direction changes and persistence
        6. Wall bouncing physics
        7. Temporal behaviors (pausing, bursting)
        8. Noise and micro-movements
        """
        self.time_accumulator += delta_time
        
        # Update behavior timers
        self._update_behavioral_mode(delta_time)
        self._update_cooldowns(delta_time)
        
        # Check for startle override (bypasses physics entirely)
        if hasattr(consciousness, 'startle_triggered') and consciousness.startle_triggered:
            self._handle_startle_override(consciousness)
            return (self.x, self.y)
        
        # 1. Base emotional forces
        emotional_x, emotional_y = self._calculate_emotional_forces(consciousness)
        
        # 2. Face tracking forces
        face_x, face_y = self._calculate_face_tracking(consciousness)
        
        # 3. Gaze following forces
        gaze_x, gaze_y = self._calculate_gaze_following(consciousness)
        
        # 4. Object attention forces
        object_x, object_y = self._calculate_object_attention(consciousness)
        
        # 5. Behavioral pattern forces
        behavior_x, behavior_y = self._calculate_behavioral_forces(consciousness, delta_time)
        
        # 6. Vibration forces
        vibration_x, vibration_y = self._calculate_vibration(delta_time)
        
        # 7. Pulsation forces
        pulsation_x, pulsation_y = self._calculate_pulsation(delta_time)
        
        # 8. Direction persistence and changes
        direction_x, direction_y = self._calculate_direction_changes(delta_time)
        
        # 9. Temporal behaviors (pausing, bursting)
        temporal_x, temporal_y = self._calculate_temporal_behaviors(consciousness, delta_time)
        
        # 10. Base noise and micro-jitter
        noise_x, noise_y = self._calculate_noise_forces(consciousness)
        
        # Combine all forces with weights
        total_force_x = (
            emotional_x * self.emotional_influence +
            face_x * self.face_tracking_strength +
            gaze_x * self.gaze_following_strength +
            object_x * self.object_attention_strength +
            behavior_x +
            vibration_x +
            pulsation_x +
            direction_x * self.direction_persistence +
            temporal_x +
            noise_x
        )
        
        total_force_y = (
            emotional_y * self.emotional_influence +
            face_y * self.face_tracking_strength +
            gaze_y * self.gaze_following_strength +
            object_y * self.object_attention_strength +
            behavior_y +
            vibration_y +
            pulsation_y +
            direction_y * self.direction_persistence +
            temporal_y +
            noise_y
        )
        
        # Apply speed multiplier based on consciousness state
        speed_multiplier = self._calculate_speed_multiplier(consciousness)
        total_force_x *= speed_multiplier
        total_force_y *= speed_multiplier
        
        # Update velocity
        self.velocity_x += total_force_x * delta_time
        self.velocity_y += total_force_y * delta_time
        
        # Apply wall bouncing forces
        if self.wall_bounce_enabled:
            self.velocity_x += self._bounce_force_x
            self.velocity_y += self._bounce_force_y
            self._bounce_force_x *= 0.9  # Decay bounce forces
            self._bounce_force_y *= 0.9
        
        # Apply dampening
        self.velocity_x *= self.dampening
        self.velocity_y *= self.dampening
        
        # Update position
        self.x += self.velocity_x * delta_time
        self.y += self.velocity_y * delta_time
        
        # Handle wall collisions
        if self.wall_bounce_enabled:
            if self.x <= 0.0 or self.x >= 1.0:
                self.x = max(0.0, min(1.0, self.x))
                self._trigger_wall_collision('x')
            
            if self.y <= 0.0 or self.y >= 1.0:
                self.y = max(0.0, min(1.0, self.y))
                self._trigger_wall_collision('y')
        else:
            # Simple boundary clamping
            self.x = max(0.0, min(1.0, self.x))
            self.y = max(0.0, min(1.0, self.y))
        
        # Update movement history
        self.movement_history.append((self.x, self.y, self.time_accumulator))
        if len(self.movement_history) > 50:
            self.movement_history.pop(0)
        
        return (self.x, self.y)
    
    def _update_behavioral_mode(self, delta_time):
        """Update current behavioral mode and transitions"""
        self.behavior_timer += delta_time
        
        if self.behavior_timer >= self.behavior_transition_interval:
            # Time to potentially switch behaviors
            behaviors = ["exploring", "focusing", "wandering", "lingering", "vibrating", "pulsating"]
            
            # Weight behaviors based on current state
            if self.behavioral_mode == "vibrating":
                # Less likely to stay vibrating
                new_behavior = random.choice(["exploring", "focusing", "wandering"])
            elif self.behavioral_mode == "pulsating":
                # Less likely to stay pulsating
                new_behavior = random.choice(["exploring", "focusing", "lingering"])
            else:
                new_behavior = random.choice(behaviors)
            
            if new_behavior != self.behavioral_mode:
                print(f"🎭 Behavioral mode changed: {self.behavioral_mode} → {new_behavior}")
                self.behavioral_mode = new_behavior
                
                # Enable/disable special behaviors
                self.vibration_active = (new_behavior == "vibrating")
                self.pulsation_active = (new_behavior == "pulsating")
                self.lingering = (new_behavior == "lingering")
            
            self.behavior_timer = 0.0
    
    def _update_cooldowns(self, delta_time):
        """Update various cooldown timers"""
        if self.direction_change_cooldown > 0:
            self.direction_change_cooldown -= delta_time
        
        if self.wall_collision_cooldown > 0:
            self.wall_collision_cooldown -= delta_time
            if self.wall_collision_cooldown <= 0:
                self.wall_collision_active = False
        
        if self.burst_timer > 0:
            self.burst_timer -= delta_time
        
        if self.linger_timer > 0:
            self.linger_timer -= delta_time
    
    def _handle_startle_override(self, consciousness):
        """Handle startle with immediate override (bypasses physics)"""
        # Immediate explosive movement
        explosive_force = 0.5
        random_angle = random.random() * 2 * math.pi
        
        # Direct position change (not velocity-based)
        self.x += math.cos(random_angle) * explosive_force * 0.3
        self.y += math.sin(random_angle) * explosive_force * 0.3
        
        # Clamp to bounds
        self.x = max(0.0, min(1.0, self.x))
        self.y = max(0.0, min(1.0, self.y))
        
        # Set high velocity for continued movement
        self.velocity_x = math.cos(random_angle) * explosive_force
        self.velocity_y = math.sin(random_angle) * explosive_force
        
        print("⚡ STARTLE OVERRIDE: Immediate explosive movement!")
    
    def _calculate_emotional_forces(self, consciousness):
        """Calculate base emotional forces - INCREASED FOR MORE VISIBLE MOVEMENT"""
        # Mood affects vertical tendency - INCREASED STRENGTH
        mood_force_y = consciousness.mood * 0.3  # Increased from 0.1
        
        # Novelty creates erratic movement - INCREASED CHAOS
        novelty_chaos = consciousness.novelty * 0.15  # Increased from 0.05
        novelty_x = (random.random() - 0.5) * novelty_chaos
        novelty_y = (random.random() - 0.5) * novelty_chaos
        
        # Boredom creates slow drift - INCREASED MOVEMENT
        boredom_drift = consciousness.boredom * 0.08  # Increased from 0.02
        boredom_x = (random.random() - 0.5) * boredom_drift
        boredom_y = (random.random() - 0.5) * boredom_drift
        
        return (novelty_x + boredom_x, mood_force_y + novelty_y + boredom_y)
    
    def _calculate_face_tracking(self, consciousness):
        """Calculate face tracking forces"""
        if not (self.face_tracking_enabled and consciousness.person_present):
            return (0.0, 0.0)
        
        face_x, face_y = consciousness.face_position
        
        # Calculate attraction to face
        diff_x = (face_x - self.x) * consciousness.face_confidence
        diff_y = (face_y - self.y) * consciousness.face_confidence
        
        # Apply smoothing
        force_x = diff_x * self.face_tracking_smoothing * 0.1
        force_y = diff_y * self.face_tracking_smoothing * 0.1
        
        return (force_x, force_y)
    
    def _calculate_gaze_following(self, consciousness):
        """Calculate gaze following forces"""
        if not (self.gaze_following_enabled and hasattr(consciousness, 'gaze_pan')):
            return (0.0, 0.0)
        
        # Convert gaze angles to normalized positions
        gaze_x = (consciousness.gaze_pan - 90.0) / 180.0 + 0.5  # Convert degrees to 0-1
        gaze_y = (consciousness.gaze_tilt - 90.0) / 180.0 + 0.5
        
        # Clamp to valid range
        gaze_x = max(0.0, min(1.0, gaze_x))
        gaze_y = max(0.0, min(1.0, gaze_y))
        
        # Calculate attraction to gaze point
        diff_x = (gaze_x - self.x)
        diff_y = (gaze_y - self.y)
        
        # Apply smoothing
        force_x = diff_x * self.gaze_following_smoothing * 0.08
        force_y = diff_y * self.gaze_following_smoothing * 0.08
        
        return (force_x, force_y)
    
    def _calculate_object_attention(self, consciousness):
        """Calculate object attention forces"""
        if not (self.object_attention_enabled and hasattr(consciousness, 'detected_objects')):
            return (0.0, 0.0)
        
        if not consciousness.detected_objects or not consciousness.primary_object:
            return (0.0, 0.0)
        
        # Get primary object position (assuming it has x, y attributes)
        obj = consciousness.primary_object
        if hasattr(obj, 'x') and hasattr(obj, 'y'):
            obj_x = obj.x
            obj_y = obj.y
            
            # Calculate attraction with confidence weighting
            diff_x = (obj_x - self.x) * consciousness.object_confidence
            diff_y = (obj_y - self.y) * consciousness.object_confidence
            
            # Only attract if within attention range
            distance = math.sqrt(diff_x**2 + diff_y**2)
            if distance <= self.object_attention_range:
                force_x = diff_x * 0.06
                force_y = diff_y * 0.06
                return (force_x, force_y)
        
        return (0.0, 0.0)
    
    def _calculate_behavioral_forces(self, consciousness, delta_time):
        """Calculate forces based on current behavioral mode - INCREASED FOR MORE VISIBLE MOVEMENT"""
        if self.behavioral_mode == "exploring":
            # Random exploration with stronger movement
            explore_x = (random.random() - 0.5) * 0.12  # Increased from 0.04
            explore_y = (random.random() - 0.5) * 0.12
            return (explore_x, explore_y)
        
        elif self.behavioral_mode == "focusing":
            # Focused attention - stronger pull toward center
            center_x = (0.5 - self.x) * 0.08  # Increased from 0.02
            center_y = (0.5 - self.y) * 0.08
            return (center_x, center_y)
        
        elif self.behavioral_mode == "wandering":
            # Stronger dreamy wandering
            wander_x = (random.random() - 0.5) * 0.08  # Increased from 0.02
            wander_y = (random.random() - 0.5) * 0.08
            return (wander_x, wander_y)
        
        elif self.behavioral_mode == "lingering":
            # Stay in place with slightly more visible movement
            if self.linger_timer <= 0:
                self.linger_timer = random.uniform(2.0, 5.0)
            
            linger_x = (random.random() - 0.5) * 0.02  # Increased from 0.005
            linger_y = (random.random() - 0.5) * 0.02
            return (linger_x, linger_y)
        
        return (0.0, 0.0)
    
    def _calculate_vibration(self, delta_time):
        """Calculate vibration forces"""
        if not self.vibration_active:
            return (0.0, 0.0)
        
        self.vibration_phase += self.vibration_frequency * delta_time
        
        vibration_x = math.sin(self.vibration_phase) * self.vibration_intensity
        vibration_y = math.cos(self.vibration_phase * 1.3) * self.vibration_intensity
        
        return (vibration_x, vibration_y)
    
    def _calculate_pulsation(self, delta_time):
        """Calculate pulsation forces"""
        if not self.pulsation_active:
            return (0.0, 0.0)
        
        self.pulsation_phase += self.pulsation_rate * delta_time
        
        # Pulsation radiates outward from center
        center_x = 0.5
        center_y = 0.5
        
        diff_x = self.x - center_x
        diff_y = self.y - center_y
        
        # Normalize direction
        distance = math.sqrt(diff_x**2 + diff_y**2)
        if distance > 0:
            norm_x = diff_x / distance
            norm_y = diff_y / distance
        else:
            norm_x = 1.0
            norm_y = 0.0
        
        pulse_strength = math.sin(self.pulsation_phase) * self.pulsation_amplitude
        
        pulsation_x = norm_x * pulse_strength
        pulsation_y = norm_y * pulse_strength
        
        return (pulsation_x, pulsation_y)
    
    def _calculate_direction_changes(self, delta_time):
        """Calculate direction persistence and changes"""
        # Check for direction change
        if (self.direction_change_cooldown <= 0 and 
            random.random() < self.direction_change_chance * delta_time):
            
            # Change direction
            angle = random.random() * 2 * math.pi
            self.current_direction_x = math.cos(angle)
            self.current_direction_y = math.sin(angle)
            self.direction_change_cooldown = 2.0
            print("🔄 Direction change triggered!")
        
        # Apply current direction with persistence
        direction_force = 0.03
        return (
            self.current_direction_x * direction_force,
            self.current_direction_y * direction_force
        )
    
    def _calculate_temporal_behaviors(self, consciousness, delta_time):
        """Calculate temporal behaviors (pausing, bursting, rhythm sync)"""
        # Burst movement
        if self.burst_timer > 0:
            burst_strength = 0.2
            burst_x = self._burst_direction_x * burst_strength
            burst_y = self._burst_direction_y * burst_strength
            return burst_x, burst_y
        
        # Check for new burst
        if random.random() < self.burst_movement_chance * delta_time:
            self.burst_timer = 0.3
            angle = random.random() * 2 * math.pi
            self._burst_direction_x = math.cos(angle)
            self._burst_direction_y = math.sin(angle)
            print("💥 Burst movement triggered!")
        
        return 0.0, 0.0
    
    def _trigger_wall_collision(self, axis):
        """Handle wall collision"""
        if self.wall_collision_cooldown > 0:
            return  # Already bouncing
        
        self.wall_collision_cooldown = 0.5
        bounce_strength = self.wall_bounce_strength * 0.1
        
        if axis == 'x':
            self._bounce_force_x = -self.velocity_x * bounce_strength
            self._bounce_force_y = self.velocity_y * 0.3  # Some perpendicular bounce
            self.velocity_x *= -0.8  # Reverse and dampen
        else:  # axis == 'y'
            self._bounce_force_y = -self.velocity_y * bounce_strength
            self._bounce_force_x = self.velocity_x * 0.3
            self.velocity_y *= -0.8
        
        print(f"🏀 Wall collision on {axis} axis - bouncing!")
    
    def _calculate_speed_multiplier(self, consciousness):
        """Calculate overall speed multiplier based on consciousness state"""
        base_multiplier = 1.0
        
        # Novelty increases speed
        novelty_boost = 1.0 + (consciousness.novelty * self.novelty_speed_multiplier)
        
        # Mood affects speed (extreme moods = more movement)
        mood_boost = 1.0 + (abs(consciousness.mood) * 0.5)
        
        # Boredom decreases speed
        boredom_reduction = 1.0 - (consciousness.boredom * 0.3)
        
        return base_multiplier * novelty_boost * mood_boost * boredom_reduction
    
    def _calculate_noise_forces(self, consciousness):
        """Calculate base noise and micro-jitter"""
        # Base noise level
        noise_x = (random.random() - 0.5) * self.base_noise_level
        noise_y = (random.random() - 0.5) * self.base_noise_level
        
        # Chaos multiplier based on novelty
        chaos_factor = 1.0 + (consciousness.novelty * self.chaos_multiplier)
        
        # Micro-jitter
        jitter_x = (random.random() - 0.5) * self.micro_jitter
        jitter_y = (random.random() - 0.5) * self.micro_jitter
        
        return (
            (noise_x + jitter_x) * chaos_factor,
            (noise_y + jitter_y) * chaos_factor
        )
    
    def get_position(self):
        """Get current cursor position"""
        return (self.x, self.y)
    
    def get_movement_speed(self):
        """Get current movement speed (magnitude of velocity vector)"""
        return math.sqrt(self.velocity_x**2 + self.velocity_y**2)
    
    def get_behavioral_state(self):
        """Get current behavioral state information"""
        return {
            'behavior': self.current_behavior,
            'behavior_timer': self.behavior_timer,
            'lingering': self.lingering,
            'wall_cooldown': self.wall_collision_cooldown,
            'burst_active': self.burst_timer > 0
        }
    
    def get_emotional_state_description(self, consciousness):
        """Get human-readable description of current emotional influence on movement"""
        descriptions = []
        
        # Current behavioral mode
        descriptions.append(f"Mode: {self.behavioral_mode}")
        
        # Emotional influences
        if consciousness.mood > 0.3:
            descriptions.append(f"Happy (upward pull: {consciousness.mood:.2f})")
        elif consciousness.mood < -0.3:
            descriptions.append(f"Sad (downward pull: {consciousness.mood:.2f})")
        
        if consciousness.novelty > 0.5:
            descriptions.append(f"Curious (fast erratic: {consciousness.novelty:.2f})")
        
        if consciousness.boredom > 0.4:
            descriptions.append(f"Bored (slow drift: {consciousness.boredom:.2f})")
        
        # Face tracking
        if consciousness.person_present and self.face_tracking_enabled:
            descriptions.append(f"Face tracking (confidence: {consciousness.face_confidence:.2f})")
        
        # Gaze following
        if hasattr(consciousness, 'gaze_pan') and self.gaze_following_enabled:
            descriptions.append(f"Gaze following (pan: {consciousness.gaze_pan:.0f}°)")
        
        # Object attention
        if hasattr(consciousness, 'detected_objects') and self.object_attention_enabled and consciousness.detected_objects:
            descriptions.append(f"Object attention ({len(consciousness.detected_objects)} objects)")
        
        # Physical state
        if consciousness.breath_paused:
            descriptions.append("Breath held (stillness)")
        
        if hasattr(consciousness, 'startle_triggered') and consciousness.startle_triggered:
            descriptions.append("STARTLE! (explosive movement)")
        
        # Behavioral states
        if self.vibration_active:
            descriptions.append(f"Vibrating (intensity: {self.vibration_intensity:.2f})")
        
        if self.pulsation_active:
            descriptions.append(f"Pulsating (rate: {self.pulsation_rate:.2f})")
        
        if self.wall_collision_active:
            descriptions.append("Wall collision active")
        
        if not descriptions:
            descriptions.append("Neutral contemplation")
        
        return " | ".join(descriptions)
    
    def reset_to_center(self):
        """Reset cursor to center position"""
        self.x = 0.5
        self.y = 0.5
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.movement_history.clear()
        print("🎯 Conscious cursor reset to center")
