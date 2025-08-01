"""
Movement Learning System
========================

Advanced system that learns from human movement patterns and translates them
into consciousness cursor parameters. This creates authentic emotional expressions
based on the user's own body language vocabulary.

Features:
- Real-time parameter extraction from cursor movements
- Emotional signature mapping to consciousness parameters  
- Adaptive intensity scaling for different emotional states
- Persistent storage of learned movement patterns

Author: Revolutionary AI Consciousness Team
"""

import json
import math
import time
from typing import Dict, List, Optional


class MovementLearning:
    """Revolutionary system that learns movement patterns from human input."""
    
    def __init__(self):
        self.emotional_profiles = {}  # Store learned emotional signatures
        
    def analyze_movement_signature(self, movements: List[Dict]) -> Dict:
        """Extract movement signature from recorded movements."""
        if len(movements) < 10:
            return {}
        
        # Calculate basic movement statistics
        speeds = []
        distances = []
        direction_changes = 0
        pauses = 0
        
        for i in range(1, len(movements)):
            prev = movements[i-1]
            curr = movements[i]
            
            # Calculate distance and speed
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            distance = math.sqrt(dx*dx + dy*dy)
            distances.append(distance)
            
            # Calculate time delta
            if 'time' in curr and 'time' in prev:
                time_delta = curr['time'] - prev['time']
                if time_delta > 0:
                    speed = distance / time_delta
                    speeds.append(speed)
            
            # Detect pauses (very small movement)
            if distance < 0.001:
                pauses += 1
        
        # Calculate signature metrics
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0
        speed_variance = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds) if speeds else 0
        
        # Movement characteristics
        signature = {
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'speed_variance': speed_variance,
            'total_distance': sum(distances),
            'pause_count': pauses,
            'movement_count': len(movements),
            'agitation_level': min(1.0, speed_variance * 10),  # How erratic
            'contemplation': min(1.0, pauses / len(movements)),  # How much pausing
            'explosiveness': min(1.0, max_speed / (avg_speed + 0.001)),  # Burst tendency
            'consistency': max(0.0, 1.0 - (speed_variance / (avg_speed + 0.001)))  # How steady
        }
        
        return signature
    
    def create_emotional_profile(self, emotion: str, signature: Dict) -> Dict:
        """Map movement signature to consciousness cursor parameters."""
        params = {}
        
        # Base movement speed
        base_speed = min(10.0, max(0.1, signature.get('avg_speed', 0.1) * 30))
        params['base_speed'] = base_speed
        
        # Behavioral transition timing
        agitation = signature.get('agitation_level', 0.1)
        params['behavior_transition_interval'] = max(1.0, 5.0 - (agitation * 4))
        
        # Burst movements
        explosiveness = signature.get('explosiveness', 0.1)
        params['burst_movement_chance'] = min(0.3, explosiveness * 0.5)
        
        # Noise and chaos
        consistency = signature.get('consistency', 0.5)
        params['noise_amplitude'] = max(0.05, (1.0 - consistency) * 0.5)
        params['macro_noise_amplitude'] = max(0.01, (1.0 - consistency) * 0.1)
        
        # Pausing behavior
        contemplation = signature.get('contemplation', 0.05)
        params['pause_probability'] = min(0.2, contemplation * 2)
        
        # Movement persistence
        params['direction_persistence'] = max(0.2, 1.0 - (agitation * 2))
        
        # Mood influence
        params['mood_influence'] = min(3.0, signature.get('avg_speed', 0.5) * 2)
        
        # Fine motor control
        params['micro_noise_amplitude'] = min(0.1, signature.get('speed_variance', 0.01) * 10)
        
        print(f"🧬 Created emotional profile for '{emotion}':")
        print(f"   Speed: {params['base_speed']:.2f} (from avg_speed: {signature.get('avg_speed', 0):.3f})")
        print(f"   Agitation: {agitation:.3f} → Transition interval: {params['behavior_transition_interval']:.1f}s")
        print(f"   Explosiveness: {explosiveness:.3f} → Burst chance: {params['burst_movement_chance']:.3f}")
        print(f"   Consistency: {consistency:.3f} → Noise: {params['noise_amplitude']:.3f}")
        print(f"   Contemplation: {contemplation:.3f} → Pause prob: {params['pause_probability']:.3f}")
        
        return params
    
    def learn_from_recording(self, emotion: str, movements: List[Dict]) -> bool:
        """Learn emotional parameters from recorded movements."""
        print(f"🎓 Learning movement signature for emotion: {emotion}")
        
        # Extract movement signature
        signature = self.analyze_movement_signature(movements)
        if not signature:
            print("❌ Not enough movement data to learn from")
            return False
        
        # Create consciousness parameters
        params = self.create_emotional_profile(emotion, signature)
        
        # Store the learned profile
        self.emotional_profiles[emotion] = {
            'signature': signature,
            'parameters': params,
            'sample_count': len(movements),
            'learned_at': movements[-1]['time'] if movements else time.time()
        }
        
        # Save to disk
        self._save_profiles()
        
        print(f"✅ Learned emotional profile for '{emotion}' from {len(movements)} movement points!")
        return True
    
    def apply_learned_parameters(self, consciousness_cursor, emotion: str, intensity: float = 1.0) -> bool:
        """Apply learned emotional parameters to consciousness cursor."""
        if emotion not in self.emotional_profiles:
            print(f"❌ No learned profile for emotion: {emotion}")
            return False
        
        params = self.emotional_profiles[emotion]['parameters']
        
        # Apply parameters with intensity scaling
        for param_name, value in params.items():
            if hasattr(consciousness_cursor, param_name):
                scaled_value = value * intensity
                setattr(consciousness_cursor, param_name, scaled_value)
                print(f"🎯 Applied {param_name} = {scaled_value:.3f}")
        
        print(f"🚀 Applied learned '{emotion}' movement style (intensity: {intensity:.1f})")
        return True
    
    def get_available_emotions(self) -> List[str]:
        """Get list of all learned emotions."""
        return list(self.emotional_profiles.keys())
    
    def _save_profiles(self):
        """Save learned profiles to disk."""
        try:
            import os
            profiles_dir = 'movement_profiles'
            if not os.path.exists(profiles_dir):
                os.makedirs(profiles_dir)
            
            profiles_file = os.path.join(profiles_dir, 'learned_profiles.json')
            with open(profiles_file, 'w') as f:
                json.dump(self.emotional_profiles, f, indent=2)
            print(f"💾 Saved {len(self.emotional_profiles)} emotional profiles to {profiles_file}")
        except Exception as e:
            print(f"⚠️ Could not save profiles: {e}")
    
    def load_profiles(self):
        """Load learned profiles from disk."""
        try:
            import os
            profiles_file = os.path.join('movement_profiles', 'learned_profiles.json')
            with open(profiles_file, 'r') as f:
                self.emotional_profiles = json.load(f)
            print(f"📚 Loaded {len(self.emotional_profiles)} emotional profiles")
            return True
        except FileNotFoundError:
            print("📚 No saved profiles found - starting fresh")
            return False
        except Exception as e:
            print(f"⚠️ Could not load profiles: {e}")
            return False

    def delete_emotion(self, emotion: str) -> bool:
        """Delete a learned emotion profile."""
        if emotion in self.emotional_profiles:
            del self.emotional_profiles[emotion]
            self._save_profiles()
            print(f"🗑️ Deleted emotion profile: {emotion}")
            return True
        else:
            print(f"❌ Emotion '{emotion}' not found in learned profiles")
            return False

    def clear_all_profiles(self) -> bool:
        """Clear all learned emotion profiles."""
        try:
            self.emotional_profiles = {}
            self._save_profiles()
            print("🧹 Cleared all emotion profiles")
            return True
        except Exception as e:
            print(f"❌ Error clearing profiles: {e}")
            return False

    def get_profile_info(self, emotion: str) -> Optional[Dict]:
        """Get detailed information about a learned emotion profile."""
        if emotion in self.emotional_profiles:
            profile = self.emotional_profiles[emotion]
            return {
                'emotion': emotion,
                'sample_count': profile.get('sample_count', 0),
                'learned_at': profile.get('learned_at', 0),
                'signature': profile.get('signature', {}),
                'parameters': profile.get('parameters', {})
            }
        return None
