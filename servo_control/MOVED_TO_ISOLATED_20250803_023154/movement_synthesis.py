#!/usr/bin/env python3
"""
Movement Synthesis System
========================

A hybrid AI system that learns from recorded human movements and then creatively
combines, refines, and improvises based on emotional consciousness inputs.

This is NOT a parrot system - it uses your movement DNA as a foundation and then
generates new, organic movement sequences that feel authentically "you" while
responding to emotional and behavioral inputs.

Key Features:
- Learns fundamental movement characteristics (your "signature")
- Establishes baseline behavioral patterns from recordings
- Creatively combines and morphs patterns based on consciousness state
- Improvises new sequences while maintaining your movement "DNA"
- Integrates emotional inputs as modulators of your learned patterns

Author: Movement Synthesis AI
"""

import random
import math
import json
import os
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MovementSignature:
    """Fundamental movement characteristics learned from a person's recordings"""
    # Core movement DNA
    typical_speed_range: Tuple[float, float]  # (min, max) speed
    preferred_sweep_distances: List[float]    # Common sweep magnitudes
    rhythm_patterns: List[float]              # Timing between movements
    directional_preferences: Dict[str, float] # Tendency toward directions
    
    # Behavioral patterns
    pause_characteristics: Dict[str, float]   # How and when pauses occur
    acceleration_patterns: List[float]        # Speed change behaviors
    spatial_coverage: float                   # How much of space is used
    
    # Style markers
    jitter_level: float                       # Amount of micro-movement
    smoothness_factor: float                  # How smooth vs jerky
    exploration_tendency: float               # Tendency to explore vs focus


class MovementSynthesizer:
    """
    Hybrid movement generation system that learns from recordings and improvises.
    
    This system learns your fundamental movement signature and then uses it as a 
    foundation for generating new, contextually appropriate movements that feel
    authentically "you" while responding to emotional and behavioral inputs.
    """
    
    def __init__(self):
        self.movement_signatures: Dict[str, MovementSignature] = {}
        self.base_signature: Optional[MovementSignature] = None
        self.current_pattern_memory: List[Tuple[float, float]] = []
        self.pattern_blend_weights: Dict[str, float] = {}
        
        # NEW: Store actual recorded movement sequences for pure replay
        self.recorded_sequences: Dict[str, List[Tuple[float, float]]] = {}
        self.sequence_playback_index: int = 0
        self.current_sequence: List[Tuple[float, float]] = []
        
        # Synthesis parameters
        self.creativity_level = 0.3  # How much to improvise vs stick to patterns
        self.emotional_influence = 0.5  # How much emotions modulate base patterns
        self.pattern_memory_length = 20  # How many recent moves to remember
        
        # Current synthesis state
        self.active_emotion_blend: Dict[str, float] = {}
        self.improvisation_momentum: float = 0.0
        self.last_synthesis_time: float = 0.0
        
        # Emotion cycling system
        self.emotion_cycling_enabled: bool = False
        self.emotion_cycle_interval: float = 120.0  # 2 minutes per emotion
        self.emotion_cycle_timer: float = 0.0
        
        print("🎭 Movement Synthesizer initialized - ready to learn and improvise!")
    
    def learn_movement_signature(self, emotion: str, movements: List[Dict]) -> bool:
        """
        Learn fundamental movement characteristics from recorded movements.
        Extracts the person's unique movement DNA rather than just copying sequences.
        """
        if len(movements) < 20:
            print(f"❌ Need at least 20 movements to learn signature for {emotion}")
            return False
        
        print(f"🧬 Learning movement signature for {emotion}...")
        
        # Extract positions and timing
        positions = [(m.get('x', 0.5), m.get('y', 0.5)) for m in movements]
        time_deltas = []
        
        for i in range(1, len(movements)):
            if 'time_delta' in movements[i]:
                time_deltas.append(movements[i]['time_delta'])
            else:
                time_deltas.append(0.016)  # Default 60fps
        
        # Analyze fundamental characteristics
        signature = self._extract_movement_signature(positions, time_deltas)
        self.movement_signatures[emotion] = signature
        
        # If this is the first signature, use it as the base
        if self.base_signature is None:
            self.base_signature = signature
            print(f"🏠 {emotion} established as base movement signature")
        
        # Save learned signatures
        self._save_signatures()
        
        print(f"✅ Learned movement signature for {emotion}")
        print(f"📊 Speed range: {signature.typical_speed_range[0]:.2f}-{signature.typical_speed_range[1]:.2f}")
        print(f"📏 Avg sweep distance: {sum(signature.preferred_sweep_distances)/len(signature.preferred_sweep_distances):.2f}")
        print(f"🎯 Spatial coverage: {signature.spatial_coverage:.2f}")
        
        return True
    
    def _extract_movement_signature(self, positions: List[Tuple[float, float]], 
                                  time_deltas: List[float]) -> MovementSignature:
        """Extract fundamental movement characteristics from position data"""
        
        # Calculate speeds and distances
        speeds = []
        sweep_distances = []
        directional_vectors = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
        
        for i in range(len(positions) - 1):
            dx = positions[i+1][0] - positions[i][0]
            dy = positions[i+1][1] - positions[i][1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if i < len(time_deltas) and time_deltas[i] > 0:
                speed = distance / time_deltas[i]
                speeds.append(speed)
            
            sweep_distances.append(distance)
            
            # Track directional preferences
            if abs(dx) > abs(dy):
                if dx > 0:
                    directional_vectors['right'] += abs(dx)
                else:
                    directional_vectors['left'] += abs(dx)
            else:
                if dy > 0:
                    directional_vectors['down'] += abs(dy)
                else:
                    directional_vectors['up'] += abs(dy)
        
        # Normalize directional preferences
        total_movement = sum(directional_vectors.values())
        if total_movement > 0:
            for direction in directional_vectors:
                directional_vectors[direction] /= total_movement
        
        # Analyze pauses (time deltas > threshold)
        long_pauses = [t for t in time_deltas if t > 0.1]
        pause_frequency = len(long_pauses) / len(time_deltas) if time_deltas else 0
        avg_pause_duration = sum(long_pauses) / len(long_pauses) if long_pauses else 0
        
        # Calculate spatial coverage (how much of the space is used)
        if positions:
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            spatial_coverage = (x_range + y_range) / 2
        else:
            spatial_coverage = 0
        
        # Analyze acceleration patterns
        acceleration_patterns = []
        if len(speeds) > 1:
            for i in range(1, len(speeds)):
                acceleration = speeds[i] - speeds[i-1]
                acceleration_patterns.append(acceleration)
        
        # Calculate jitter (micro-movements)
        micro_movements = sum(1 for d in sweep_distances if 0 < d < 0.02)
        jitter_level = micro_movements / len(sweep_distances) if sweep_distances else 0
        
        # Calculate smoothness (consistency of speed)
        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            speed_variance = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)
            smoothness_factor = 1.0 / (1.0 + speed_variance)  # Higher variance = less smooth
        else:
            smoothness_factor = 0.5
        
        # Calculate exploration tendency (how much they move around vs stay in one area)
        total_distance = sum(sweep_distances)
        straight_line_distance = math.sqrt((positions[-1][0] - positions[0][0])**2 + 
                                         (positions[-1][1] - positions[0][1])**2)
        exploration_tendency = total_distance / max(straight_line_distance, 0.01)
        
        return MovementSignature(
            typical_speed_range=(min(speeds) if speeds else 0, max(speeds) if speeds else 1),
            preferred_sweep_distances=sweep_distances[:20],  # Keep sample of preferred distances
            rhythm_patterns=time_deltas[:20],  # Keep sample of timing patterns
            directional_preferences=directional_vectors,
            pause_characteristics={
                'frequency': pause_frequency,
                'avg_duration': avg_pause_duration,
                'variance': speed_variance if speeds else 0
            },
            acceleration_patterns=acceleration_patterns[:10],
            spatial_coverage=spatial_coverage,
            jitter_level=jitter_level,
            smoothness_factor=smoothness_factor,
            exploration_tendency=exploration_tendency
        )
    
    def synthesize_movement(self, current_pos: Tuple[float, float], 
                          consciousness_state, delta_time: float) -> Tuple[float, float]:
        """
        Generate the next movement by creatively combining learned patterns
        with current emotional/behavioral context.
        
        This is the core AI that improvises movement while staying true to
        your learned movement signature.
        """
        if self.base_signature is None:
            # No learned signature yet - return small random movement
            return (random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01))
        
        # Handle emotion cycling if enabled
        self._update_emotion_cycling(delta_time)
        
        # Update pattern memory
        self.current_pattern_memory.append(current_pos)
        if len(self.current_pattern_memory) > self.pattern_memory_length:
            self.current_pattern_memory.pop(0)
        
        # Determine emotional context and how it should modulate movement
        emotion_modulation = self._calculate_emotional_modulation(consciousness_state)
        
        # Generate base movement from learned signature
        base_movement = self._generate_signature_based_movement(current_pos, delta_time)
        
        # Apply emotional modulation
        modulated_movement = self._apply_emotional_modulation(base_movement, emotion_modulation)
        
        # Add creative improvisation while maintaining signature
        final_movement = self._add_creative_improvisation(modulated_movement, consciousness_state)
        
        # Update synthesis state
        self.last_synthesis_time = time.time()
        
        return final_movement
    
    def _calculate_emotional_modulation(self, consciousness_state) -> Dict[str, float]:
        """Calculate how current emotional state should modulate base movement patterns"""
        
        modulation = {
            'speed_multiplier': 1.0,
            'sweep_scale': 1.0,
            'rhythm_change': 1.0,
            'exploration_boost': 1.0,
            'jitter_multiplier': 1.0,
            'directional_bias': {'x': 0.0, 'y': 0.0}
        }
        
        # Mood affects overall energy and vertical tendency
        if hasattr(consciousness_state, 'mood'):
            mood = consciousness_state.mood
            modulation['speed_multiplier'] *= (1.0 + mood * 0.5)  # Happy = faster
            modulation['directional_bias']['y'] = mood * 0.3  # Happy = upward tendency
            modulation['sweep_scale'] *= (1.0 + abs(mood) * 0.3)  # Strong emotions = bigger movements
        
        # Novelty affects exploration and chaos
        if hasattr(consciousness_state, 'novelty'):
            novelty = consciousness_state.novelty
            modulation['exploration_boost'] *= (1.0 + novelty * 0.8)
            modulation['jitter_multiplier'] *= (1.0 + novelty * 0.5)
            modulation['rhythm_change'] *= (1.0 + novelty * 0.4)  # More novel = more rhythm variation
        
        # Boredom affects exploration and reduces speed
        if hasattr(consciousness_state, 'boredom'):
            boredom = consciousness_state.boredom
            modulation['speed_multiplier'] *= (1.0 - boredom * 0.3)  # Bored = slower
            modulation['exploration_boost'] *= (1.0 + boredom * 0.6)  # Bored = more wandering
        
        # Face presence affects focus vs exploration
        if hasattr(consciousness_state, 'person_present') and consciousness_state.person_present:
            modulation['exploration_boost'] *= 0.7  # Focus more when person present
            modulation['sweep_scale'] *= 1.2  # But make movements more pronounced
        
        return modulation
    
    def _generate_signature_based_movement(self, current_pos: Tuple[float, float], 
                                         delta_time: float) -> Tuple[float, float]:
        """Generate movement based on learned movement signature"""
        
        signature = self.base_signature
        
        # Choose a characteristic speed from learned range
        speed_range = signature.typical_speed_range
        target_speed = random.uniform(speed_range[0], speed_range[1])
        
        # Choose a sweep distance similar to learned patterns
        if signature.preferred_sweep_distances:
            base_distance = random.choice(signature.preferred_sweep_distances)
        else:
            base_distance = 0.05
        
        # Apply directional preferences
        angle = random.uniform(0, 2 * math.pi)
        
        # Bias toward learned directional preferences
        dir_prefs = signature.directional_preferences
        if random.random() < 0.7:  # 70% chance to follow learned preferences
            # Choose direction based on learned preferences
            total_weight = sum(dir_prefs.values())
            if total_weight > 0:
                rand_val = random.uniform(0, total_weight)
                cumulative = 0
                for direction, weight in dir_prefs.items():
                    cumulative += weight
                    if rand_val <= cumulative:
                        if direction == 'right':
                            angle = 0 + random.uniform(-math.pi/4, math.pi/4)
                        elif direction == 'left':
                            angle = math.pi + random.uniform(-math.pi/4, math.pi/4)
                        elif direction == 'up':
                            angle = -math.pi/2 + random.uniform(-math.pi/4, math.pi/4)
                        elif direction == 'down':
                            angle = math.pi/2 + random.uniform(-math.pi/4, math.pi/4)
                        break
        
        # Calculate movement vector
        movement_distance = base_distance * target_speed * delta_time
        dx = math.cos(angle) * movement_distance
        dy = math.sin(angle) * movement_distance
        
        # Apply learned jitter level
        if signature.jitter_level > 0:
            jitter_x = random.uniform(-signature.jitter_level, signature.jitter_level) * 0.01
            jitter_y = random.uniform(-signature.jitter_level, signature.jitter_level) * 0.01
            dx += jitter_x
            dy += jitter_y
        
        # Apply spatial coverage tendency
        coverage_factor = signature.spatial_coverage
        dx *= coverage_factor
        dy *= coverage_factor
        
        return (dx, dy)
    
    def _apply_emotional_modulation(self, base_movement: Tuple[float, float], 
                                  modulation: Dict[str, float]) -> Tuple[float, float]:
        """Apply emotional modulation to base movement"""
        
        dx, dy = base_movement
        
        # Apply speed modulation
        speed_mult = modulation['speed_multiplier']
        dx *= speed_mult
        dy *= speed_mult
        
        # Apply sweep scale modulation
        scale_mult = modulation['sweep_scale']
        dx *= scale_mult
        dy *= scale_mult
        
        # Apply directional bias
        bias = modulation['directional_bias']
        dx += bias['x']
        dy += bias['y']
        
        # Apply jitter modulation
        jitter_mult = modulation['jitter_multiplier']
        if jitter_mult > 1.0:
            extra_jitter = (jitter_mult - 1.0) * 0.005
            dx += random.uniform(-extra_jitter, extra_jitter)
            dy += random.uniform(-extra_jitter, extra_jitter)
        
        return (dx, dy)
    
    def _add_creative_improvisation(self, modulated_movement: Tuple[float, float], 
                                  consciousness_state) -> Tuple[float, float]:
        """Add creative improvisation while maintaining movement signature"""
        
        dx, dy = modulated_movement
        
        # Creativity adds variation while respecting learned patterns
        creativity = self.creativity_level
        
        # Occasional burst movements (learned from signature acceleration patterns)
        # Only if creativity > 0 - when creativity is 0, use PURE learned patterns
        if (self.base_signature.acceleration_patterns and 
            creativity > 0.0 and 
            random.random() < (0.05 * creativity)):
            burst_factor = random.choice(self.base_signature.acceleration_patterns)
            dx *= (1.0 + burst_factor * creativity)
            dy *= (1.0 + burst_factor * creativity)
            print("💥 Creative burst movement!")
        
        # Occasional rhythm breaks (but based on learned timing patterns)
        # Only if creativity > 0 - when creativity is 0, stick to learned rhythms
        if (self.base_signature.rhythm_patterns and 
            creativity > 0.0 and 
            random.random() < (0.03 * creativity)):
            rhythm_factor = random.choice(self.base_signature.rhythm_patterns)
            pause_chance = rhythm_factor * creativity
            if random.random() < pause_chance:
                dx *= 0.1  # Slow down for improvised pause
                dy *= 0.1
                print("⏸️ Creative rhythm break!")
        
        # Pattern memory influence - sometimes repeat or mirror recent patterns
        # Only if creativity > 0 - when creativity is 0, follow patterns exactly
        if (len(self.current_pattern_memory) > 5 and 
            creativity > 0.0 and 
            random.random() < creativity * 0.3):
            # Occasionally echo a recent movement pattern
            recent_pos = self.current_pattern_memory[-3]
            current_pos = self.current_pattern_memory[-1]
            echo_dx = recent_pos[0] - current_pos[0]
            echo_dy = recent_pos[1] - current_pos[1]
            
            echo_strength = creativity * 0.2
            dx += echo_dx * echo_strength
            dy += echo_dy * echo_strength
        
        # Ensure movement respects learned spatial coverage
        if self.base_signature:
            max_single_move = self.base_signature.spatial_coverage * 0.1
            dx = max(-max_single_move, min(max_single_move, dx))
            dy = max(-max_single_move, min(max_single_move, dy))
        
        return (dx, dy)
    
    def set_creativity_level(self, level: float):
        """Set how much the system should improvise vs stick to learned patterns"""
        new_level = max(0.0, min(1.0, level))
        
        # Only print if the level actually changed significantly
        if not hasattr(self, 'creativity_level') or abs(new_level - self.creativity_level) > 0.01:
            self.creativity_level = new_level
            print(f"🎨 Creativity level: {self.creativity_level:.2f} (0=pure patterns, 1=maximum improvisation)")
            if self.creativity_level == 0.0:
                print("🧬 PURE PATTERN MODE: Using ONLY your recorded movements, no improvisation!")
            elif self.creativity_level < 0.3:
                print("🎯 LOW CREATIVITY: Mostly your patterns with minimal variation")
            elif self.creativity_level < 0.7:
                print("🎭 BALANCED: Your patterns + creative improvisation")  
            else:
                print("🚀 HIGH CREATIVITY: Maximum improvisation based on your movement DNA")
        else:
            self.creativity_level = new_level
    
    def set_emotional_influence(self, influence: float):
        """Set how much emotional inputs should modulate base patterns"""
        new_influence = max(0.0, min(1.0, influence))
        
        # Only print if the influence actually changed significantly
        if not hasattr(self, 'emotional_influence') or abs(new_influence - self.emotional_influence) > 0.01:
            self.emotional_influence = new_influence
            print(f"💭 Emotional modulation: {self.emotional_influence:.2f}")
        else:
            self.emotional_influence = new_influence
    
    def get_learned_emotions(self) -> List[str]:
        """Get list of emotions for which movement signatures have been learned"""
        return list(self.movement_signatures.keys())
    
    def blend_emotion_signatures(self, emotion_weights: Dict[str, float]):
        """Blend multiple learned emotional signatures for complex states"""
        # Normalize weights
        total_weight = sum(emotion_weights.values())
        if total_weight > 0:
            for emotion in emotion_weights:
                emotion_weights[emotion] /= total_weight
        
        self.active_emotion_blend = emotion_weights
        print(f"🎭 Blending emotions: {emotion_weights}")
    
    def _save_signatures(self):
        """Save learned movement signatures to disk"""
        try:
            signatures_dir = "movement_signatures"
            if not os.path.exists(signatures_dir):
                os.makedirs(signatures_dir)
            
            # Convert signatures to serializable format
            serializable_signatures = {}
            for emotion, signature in self.movement_signatures.items():
                serializable_signatures[emotion] = {
                    'typical_speed_range': signature.typical_speed_range,
                    'preferred_sweep_distances': signature.preferred_sweep_distances,
                    'rhythm_patterns': signature.rhythm_patterns,
                    'directional_preferences': signature.directional_preferences,
                    'pause_characteristics': signature.pause_characteristics,
                    'acceleration_patterns': signature.acceleration_patterns,
                    'spatial_coverage': signature.spatial_coverage,
                    'jitter_level': signature.jitter_level,
                    'smoothness_factor': signature.smoothness_factor,
                    'exploration_tendency': signature.exploration_tendency
                }
            
            with open(f"{signatures_dir}/learned_signatures.json", 'w') as f:
                json.dump(serializable_signatures, f, indent=2)
            
            print(f"💾 Saved {len(self.movement_signatures)} movement signatures")
            
        except Exception as e:
            print(f"❌ Error saving signatures: {e}")
    
    def load_signatures(self) -> bool:
        """Load previously learned movement signatures"""
        try:
            signatures_file = "movement_signatures/learned_signatures.json"
            if not os.path.exists(signatures_file):
                return False
            
            with open(signatures_file, 'r') as f:
                serializable_signatures = json.load(f)
            
            # Convert back to MovementSignature objects
            self.movement_signatures = {}
            for emotion, data in serializable_signatures.items():
                signature = MovementSignature(
                    typical_speed_range=tuple(data['typical_speed_range']),
                    preferred_sweep_distances=data['preferred_sweep_distances'],
                    rhythm_patterns=data['rhythm_patterns'],
                    directional_preferences=data['directional_preferences'],
                    pause_characteristics=data['pause_characteristics'],
                    acceleration_patterns=data['acceleration_patterns'],
                    spatial_coverage=data['spatial_coverage'],
                    jitter_level=data['jitter_level'],
                    smoothness_factor=data['smoothness_factor'],
                    exploration_tendency=data['exploration_tendency']
                )
                self.movement_signatures[emotion] = signature
            
            # Set first signature as base if no base exists
            if self.movement_signatures and self.base_signature is None:
                first_emotion = list(self.movement_signatures.keys())[0]
                self.base_signature = self.movement_signatures[first_emotion]
                print(f"🏠 Loaded {first_emotion} as base signature")
            
            print(f"📚 Loaded {len(self.movement_signatures)} movement signatures")
            return True
            
        except Exception as e:
            print(f"❌ Error loading signatures: {e}")
            return False
    
    def get_signature_description(self, emotion: str) -> str:
        """Get a human-readable description of a learned movement signature"""
        if emotion not in self.movement_signatures:
            return f"No signature learned for {emotion}"
        
        sig = self.movement_signatures[emotion]
        
        speed_desc = "fast" if sig.typical_speed_range[1] > 2.0 else "moderate" if sig.typical_speed_range[1] > 1.0 else "slow"
        coverage_desc = "wide-ranging" if sig.spatial_coverage > 0.5 else "focused"
        style_desc = "smooth" if sig.smoothness_factor > 0.7 else "jerky"
        exploration_desc = "exploratory" if sig.exploration_tendency > 2.0 else "direct"
        
        return f"{emotion}: {speed_desc}, {coverage_desc}, {style_desc}, {exploration_desc}"
    
    def set_active_emotion(self, emotion: str, weight: float = 1.0):
        """Set a specific emotion as the active pattern for synthesis"""
        if emotion in self.movement_signatures:
            self.active_emotion_blend = {emotion: weight}
            print(f"🎭 Active emotion set to: {emotion} (weight: {weight:.2f})")
        else:
            print(f"❌ Emotion '{emotion}' not found in learned patterns")
    
    def get_active_emotions(self) -> Dict[str, float]:
        """Get currently active emotion blend"""
        return self.active_emotion_blend.copy()
    
    def set_emotion_cycling_mode(self, enabled: bool, cycle_interval: float = 120.0):
        """Enable/disable automatic cycling through learned emotions"""
        self.emotion_cycling_enabled = enabled
        self.emotion_cycle_interval = cycle_interval
        self.emotion_cycle_timer = 0.0
        if enabled:
            print(f"🔄 Emotion cycling enabled: {cycle_interval:.0f}s intervals")
        else:
            print("⏸️ Emotion cycling disabled")
    
    def _update_emotion_cycling(self, delta_time: float):
        """Handle automatic cycling through learned emotions"""
        if not self.emotion_cycling_enabled or not self.movement_signatures:
            return
        
        self.emotion_cycle_timer += delta_time
        
        if self.emotion_cycle_timer >= self.emotion_cycle_interval:
            # Time to switch to next emotion
            emotions = list(self.movement_signatures.keys())
            if len(emotions) > 1:
                # Get current emotion or pick first one
                current_emotions = list(self.active_emotion_blend.keys())
                current_emotion = current_emotions[0] if current_emotions else emotions[0]
                
                # Find next emotion in cycle
                try:
                    current_index = emotions.index(current_emotion)
                    next_index = (current_index + 1) % len(emotions)
                except ValueError:
                    next_index = 0
                
                next_emotion = emotions[next_index]
                self.set_active_emotion(next_emotion)
                print(f"🔄 Auto-cycled from {current_emotion} → {next_emotion}")
            
            self.emotion_cycle_timer = 0.0
