#!/usr/bin/env python3
"""
Markov Chain Movement System
===========================

Direct movement pattern learning and replay system that creates
organic cursor movement by learning from actual recorded user movements.

Instead of abstract forces, this system learns the actual movement
transitions and sequences from recordings and replays them to create
truly organic movement that matches the user's recorded patterns.
"""

import random
import json
import os
import math
from typing import List, Dict, Tuple, Optional

class MovementChain:
    """Markov chain for movement patterns."""
    
    def __init__(self):
        # Movement state chains - these learn actual movement sequences
        self.position_transitions = {}  # position -> [next_positions]
        self.velocity_transitions = {}  # velocity -> [next_velocities] 
        self.direction_transitions = {}  # direction -> [next_directions]
        self.distance_transitions = {}  # distance -> [next_distances]
        
        # Movement characteristics learned from recordings
        self.typical_speeds = []
        self.typical_distances = []
        self.typical_pauses = []
        self.big_sweep_patterns = []  # Learn the big sweeping movements
        self.micro_patterns = []      # Learn small fidget movements
        
        # Current state for chain walking
        self.current_position = (0.5, 0.5)
        self.current_velocity = (0.0, 0.0)
        self.current_direction = 0.0
        self.last_distance = 0.0
        
        # Pattern replay state
        self.in_big_sweep = False
        self.sweep_target = None
        self.sweep_progress = 0.0
        
    def learn_from_movements(self, movements: List[Dict]) -> bool:
        """Learn movement patterns from recorded movements."""
        if len(movements) < 10:
            return False
            
        print(f"🧬 Learning Markov chains from {len(movements)} movement points...")
        
        # Extract movement sequences
        positions = [(m['x'], m['y']) for m in movements]
        
        # Learn position transitions (where do movements go next?)
        for i in range(len(positions) - 1):
            current_pos = self._discretize_position(positions[i])
            next_pos = self._discretize_position(positions[i + 1])
            
            if current_pos not in self.position_transitions:
                self.position_transitions[current_pos] = []
            self.position_transitions[current_pos].append(next_pos)
        
        # Learn velocity and direction patterns
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]
            
            # Calculate velocity
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            velocity = (dx, dy)
            
            # Calculate distance and direction
            distance = math.sqrt(dx*dx + dy*dy)
            direction = math.atan2(dy, dx) if distance > 0.001 else 0.0
            
            # Store for learning
            self.typical_speeds.append(distance)
            self.typical_distances.append(distance)
            
            # Learn velocity transitions
            if i > 1:
                prev_dx = positions[i-1][0] - positions[i-2][0]
                prev_dy = positions[i-1][1] - positions[i-2][1]
                prev_velocity = self._discretize_velocity((prev_dx, prev_dy))
                curr_velocity = self._discretize_velocity(velocity)
                
                if prev_velocity not in self.velocity_transitions:
                    self.velocity_transitions[prev_velocity] = []
                self.velocity_transitions[prev_velocity].append(curr_velocity)
            
            # Detect big sweeps (movements > threshold distance)
            if distance > 0.1:  # Big movement threshold
                sweep_pattern = {
                    'start': prev_pos,
                    'end': curr_pos,
                    'distance': distance,
                    'direction': direction,
                    'velocity': velocity
                }
                self.big_sweep_patterns.append(sweep_pattern)
            elif distance < 0.02:  # Micro movement threshold
                micro_pattern = {
                    'position': prev_pos,
                    'offset': (dx, dy),
                    'distance': distance
                }
                self.micro_patterns.append(micro_pattern)
        
        print(f"🎯 Learned {len(self.position_transitions)} position states")
        print(f"🏃 Learned {len(self.big_sweep_patterns)} big sweep patterns")
        print(f"🤏 Learned {len(self.micro_patterns)} micro movement patterns")
        
        return True
    
    def _discretize_position(self, pos: Tuple[float, float], resolution: int = 10) -> Tuple[int, int]:
        """Convert continuous position to discrete state."""
        x_discrete = int(pos[0] * resolution)
        y_discrete = int(pos[1] * resolution)
        return (x_discrete, y_discrete)
    
    def _discretize_velocity(self, vel: Tuple[float, float], resolution: int = 20) -> Tuple[int, int]:
        """Convert continuous velocity to discrete state."""
        # Clamp velocity to reasonable range
        vx = max(-0.5, min(0.5, vel[0]))
        vy = max(-0.5, min(0.5, vel[1]))
        
        vx_discrete = int((vx + 0.5) * resolution)
        vy_discrete = int((vy + 0.5) * resolution)
        return (vx_discrete, vy_discrete)
    
    def get_next_movement(self, delta_time: float = 0.016) -> Tuple[float, float]:
        """Generate next movement using learned patterns."""
        
        # Randomly decide between big sweep and normal movement
        if not self.in_big_sweep and random.random() < 0.1:  # 10% chance of big sweep
            if self.big_sweep_patterns:
                return self._start_big_sweep()
        
        # Continue big sweep if in progress
        if self.in_big_sweep:
            return self._continue_big_sweep(delta_time)
        
        # Normal Markov chain movement
        return self._get_markov_movement()
    
    def _start_big_sweep(self) -> Tuple[float, float]:
        """Start a big sweeping movement."""
        if not self.big_sweep_patterns:
            return (0.0, 0.0)
            
        # Pick a random big sweep pattern
        pattern = random.choice(self.big_sweep_patterns)
        
        # Set up sweep from current position toward pattern target
        self.sweep_target = pattern['end']
        self.in_big_sweep = True
        self.sweep_progress = 0.0
        
        # Scale the movement to match the learned pattern
        dx = pattern['velocity'][0] * 5.0  # Amplify for visibility
        dy = pattern['velocity'][1] * 5.0
        
        print(f"🏃 Starting big sweep: distance={pattern['distance']:.3f}")
        return (dx, dy)
    
    def _continue_big_sweep(self, delta_time: float) -> Tuple[float, float]:
        """Continue a big sweeping movement."""
        if not self.sweep_target:
            self.in_big_sweep = False
            return (0.0, 0.0)
        
        # Progress toward target
        self.sweep_progress += delta_time * 2.0  # Sweep speed
        
        if self.sweep_progress >= 1.0:
            # Sweep complete
            self.in_big_sweep = False
            self.sweep_target = None
            return (0.0, 0.0)
        
        # Calculate movement toward target
        target_dx = self.sweep_target[0] - self.current_position[0]
        target_dy = self.sweep_target[1] - self.current_position[1]
        
        # Smooth movement toward target
        move_x = target_dx * delta_time * 3.0
        move_y = target_dy * delta_time * 3.0
        
        return (move_x, move_y)
    
    def _get_markov_movement(self) -> Tuple[float, float]:
        """Get next movement from Markov chain."""
        current_discrete = self._discretize_position(self.current_position)
        
        # Look up next position from learned transitions
        if current_discrete in self.position_transitions:
            possible_next = self.position_transitions[current_discrete]
            if possible_next:
                next_discrete = random.choice(possible_next)
                
                # Convert back to continuous coordinates
                next_continuous = (
                    next_discrete[0] / 10.0,  # Convert from discrete
                    next_discrete[1] / 10.0
                )
                
                # Calculate movement
                dx = next_continuous[0] - self.current_position[0]
                dy = next_continuous[1] - self.current_position[1]
                
                # Scale movement for visibility
                return (dx * 3.0, dy * 3.0)
        
        # Fallback: use learned typical movements
        if self.big_sweep_patterns:
            pattern = random.choice(self.big_sweep_patterns)
            return (pattern['velocity'][0] * 2.0, pattern['velocity'][1] * 2.0)
        
        return (0.0, 0.0)
    
    def update_position(self, new_position: Tuple[float, float]):
        """Update current position for chain state."""
        self.current_position = new_position


class MarkovMovementSystem:
    """Main system that manages Markov chain movement learning and replay."""
    
    def __init__(self):
        self.chains = {}  # emotion -> MovementChain
        self.current_chain = None
        self.current_emotion = "neutral"
        
        # Load existing chains
        self.load_chains()
    
    def learn_emotion(self, emotion: str, movements: List[Dict]) -> bool:
        """Learn movement patterns for an emotion."""
        print(f"🧬 Learning Markov chain for emotion: {emotion}")
        
        chain = MovementChain()
        if chain.learn_from_movements(movements):
            self.chains[emotion] = chain
            self.save_chains()
            print(f"✅ Successfully learned {emotion} movement patterns!")
            return True
        else:
            print(f"❌ Failed to learn {emotion} - need more movement data")
            return False
    
    def apply_emotion(self, emotion: str) -> bool:
        """Switch to using a learned emotion's movement patterns."""
        if emotion in self.chains:
            self.current_chain = self.chains[emotion]
            self.current_emotion = emotion
            print(f"🎭 Now using {emotion} movement patterns!")
            return True
        else:
            print(f"❌ No learned patterns for {emotion}")
            return False
    
    def get_movement(self, current_position: Tuple[float, float], delta_time: float = 0.016) -> Tuple[float, float]:
        """Get next movement from current chain."""
        if not self.current_chain:
            return (0.0, 0.0)
        
        # Update chain's current position
        self.current_chain.update_position(current_position)
        
        # Get next movement
        return self.current_chain.get_next_movement(delta_time)
    
    def get_available_emotions(self) -> List[str]:
        """Get list of learned emotions."""
        return list(self.chains.keys())
    
    def save_chains(self):
        """Save learned chains to disk."""
        try:
            os.makedirs("markov_chains", exist_ok=True)
            
            for emotion, chain in self.chains.items():
                chain_data = {
                    'position_transitions': chain.position_transitions,
                    'velocity_transitions': chain.velocity_transitions,
                    'big_sweep_patterns': chain.big_sweep_patterns,
                    'micro_patterns': chain.micro_patterns,
                    'typical_speeds': chain.typical_speeds[:100],  # Limit size
                    'typical_distances': chain.typical_distances[:100]
                }
                
                filename = f"markov_chains/{emotion}_chain.json"
                with open(filename, 'w') as f:
                    json.dump(chain_data, f, indent=2)
                    
            print(f"💾 Saved {len(self.chains)} Markov chains")
        except Exception as e:
            print(f"❌ Error saving chains: {e}")
    
    def load_chains(self):
        """Load learned chains from disk."""
        try:
            if not os.path.exists("markov_chains"):
                return
            
            for filename in os.listdir("markov_chains"):
                if filename.endswith("_chain.json"):
                    emotion = filename.replace("_chain.json", "")
                    
                    with open(f"markov_chains/{filename}", 'r') as f:
                        chain_data = json.load(f)
                    
                    # Recreate chain
                    chain = MovementChain()
                    chain.position_transitions = chain_data.get('position_transitions', {})
                    chain.velocity_transitions = chain_data.get('velocity_transitions', {})
                    chain.big_sweep_patterns = chain_data.get('big_sweep_patterns', [])
                    chain.micro_patterns = chain_data.get('micro_patterns', [])
                    chain.typical_speeds = chain_data.get('typical_speeds', [])
                    chain.typical_distances = chain_data.get('typical_distances', [])
                    
                    # Convert string keys back to tuples for position_transitions
                    if isinstance(list(chain.position_transitions.keys())[0], str) if chain.position_transitions else False:
                        new_transitions = {}
                        for key_str, values in chain.position_transitions.items():
                            key_tuple = eval(key_str)  # Convert string back to tuple
                            new_transitions[key_tuple] = values
                        chain.position_transitions = new_transitions
                    
                    self.chains[emotion] = chain
                    
            print(f"📚 Loaded {len(self.chains)} Markov chains: {list(self.chains.keys())}")
            
        except Exception as e:
            print(f"⚠️ Error loading chains: {e}")
