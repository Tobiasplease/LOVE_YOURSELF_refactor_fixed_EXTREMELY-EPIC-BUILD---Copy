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
        return Noneptation from recorded patterns
- Emotional signature extraction and mapping
- Dynamic consciousness parameter calibration
- Movement DNA analysis and synthesis

Author: Revolutionary Consciousness System
"""

import json
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

class MovementLearning:
    """
    Learns from human movement patterns and translates them into consciousness parameters.
    Creates emotional signatures that the AI can use for authentic expression.
    """
    
    def __init__(self):
        self.emotional_profiles = {}  # Learned emotional movement profiles
        self.base_parameters = self._get_default_parameters()
        self.learning_rate = 0.3  # How much to adapt parameters
        
    def _get_default_parameters(self) -> Dict:
        """Default consciousness cursor parameters - CORRECTED NAMES to match ConsciousnessCursor attributes."""
        return {
            'base_speed': 2.5,
            'momentum_decay': 0.88,  # was 'dampening'
            'mood_influence': 1.5,   # was 'emotional_influence'
            'behavior_transition_interval': 2.0,
            'burst_movement_chance': 0.08,
            'noise_amplitude': 0.25,  # was 'chaos_multiplier'
            'macro_noise_amplitude': 0.05,  # was 'base_noise_level'
            'pause_probability': 0.03,
            'direction_persistence': 0.7,
            'micro_noise_amplitude': 0.025  # was 'micro_jitter'
        }
    
    def analyze_movement_signature(self, movements: List[Dict]) -> Dict:
        """
        Extract deep movement signature from recorded patterns.
        Goes beyond basic stats to find emotional DNA.
        """
        if len(movements) < 10:
            return {}
            
        # Basic movement metrics
        speeds = []
        accelerations = []
        direction_changes = 0
        micro_pauses = 0
        burst_episodes = []
        rhythm_patterns = []
        
        # Advanced analysis
        prev_x, prev_y = movements[0]['x'], movements[0]['y']
        prev_speed = 0
        prev_direction = None
        stillness_periods = []
        current_stillness = 0
        
        for i in range(1, len(movements)):
            curr_x, curr_y = movements[i]['x'], movements[i]['y']
            
            # Handle different timestamp field names
            if 'time_delta' in movements[i]:
                time_delta = movements[i]['time_delta']
            elif 'time' in movements[i] and 'time' in movements[i-1]:
                time_delta = movements[i]['time'] - movements[i-1]['time']
            else:
                time_delta = 0.1  # Default fallback
            
            # Movement velocity and acceleration
            distance = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            speed = distance / max(time_delta, 0.001)
            acceleration = abs(speed - prev_speed) / max(time_delta, 0.001)
            
            speeds.append(speed)
            accelerations.append(acceleration)
            
            # Detect directional changes (emotional agitation)
            if distance > 0.01:
                direction = math.atan2(curr_y - prev_y, curr_x - prev_x)
                if prev_direction is not None:
                    angle_diff = abs(direction - prev_direction)
                    if angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff
                    if angle_diff > math.pi / 6:  # 30 degree threshold
                        direction_changes += 1
                prev_direction = direction
                current_stillness = 0
            else:
                # Stillness detection
                current_stillness += time_delta
                if current_stillness > 0.5:  # Half second of stillness
                    micro_pauses += 1
            
            # Burst detection (sudden speed spikes)
            if speed > prev_speed * 2.0 and speed > 0.1:
                burst_episodes.append(speed)
            
            # Rhythm analysis (speed oscillations)
            rhythm_patterns.append(speed)
            
            prev_x, prev_y = curr_x, curr_y
            prev_speed = speed
        
        # Calculate signature metrics
        avg_speed = np.mean(speeds) if speeds else 0
        speed_variance = np.var(speeds) if speeds else 0
        avg_acceleration = np.mean(accelerations) if accelerations else 0
        burst_intensity = np.mean(burst_episodes) if burst_episodes else 0
        
        # Rhythm analysis (FFT for frequency detection)
        rhythm_frequency = self._analyze_rhythm(rhythm_patterns)
        
        # Movement personality traits
        agitation_level = direction_changes / len(movements)  # How often direction changes
        explosiveness = burst_intensity / max(avg_speed, 0.001)  # Burst vs normal speed ratio
        contemplation = micro_pauses / len(movements)  # How much pausing/thinking
        consistency = 1.0 - (speed_variance / max(avg_speed**2, 0.001))  # How consistent speed is
        
        return {
            'avg_speed': avg_speed,
            'speed_variance': speed_variance,
            'avg_acceleration': avg_acceleration,
            'direction_changes': direction_changes,
            'agitation_level': agitation_level,
            'burst_intensity': burst_intensity,
            'explosiveness': explosiveness,
            'contemplation': contemplation,
            'consistency': consistency,
            'rhythm_frequency': rhythm_frequency,
            'total_distance': sum([math.sqrt((movements[i]['x'] - movements[i-1]['x'])**2 + 
                                           (movements[i]['y'] - movements[i-1]['y'])**2) 
                                 for i in range(1, len(movements))]),
            'duration': movements[-1]['time'] - movements[0]['time'] if len(movements) > 1 else 0
        }
    
    def _analyze_rhythm(self, speed_pattern: List[float]) -> float:
        """Analyze rhythmic patterns in movement using FFT."""
        if len(speed_pattern) < 20:
            return 1.0
            
        try:
            # Simple rhythm detection - find dominant frequency
            fft = np.fft.fft(speed_pattern)
            freqs = np.fft.fftfreq(len(speed_pattern))
            dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            dominant_frequency = abs(freqs[dominant_freq_idx])
            return dominant_frequency * 10  # Scale for usability
        except:
            return 1.0
    
    def create_emotional_profile(self, emotion: str, signature: Dict) -> Dict:
        """
        Create consciousness parameters from movement signature.
        This is where the magic happens - translating human movement into AI parameters.
        """
        params = self.base_parameters.copy()
        
        # Speed mapping
        speed_multiplier = min(max(signature.get('avg_speed', 0.1) * 5, 0.5), 5.0)
        params['base_speed'] = speed_multiplier
        
        # Agitation → Behavioral transitions
        agitation = signature.get('agitation_level', 0.1)
        params['behavior_transition_interval'] = max(0.5, 3.0 - (agitation * 10))
        
        # Explosiveness → Burst movements
        explosiveness = signature.get('explosiveness', 0.1)
        params['burst_movement_chance'] = min(0.3, explosiveness * 0.5)
        
        # Consistency → Chaos/Noise
        consistency = signature.get('consistency', 0.5)
        params['noise_amplitude'] = max(0.05, (1.0 - consistency) * 0.5)  # was 'chaos_multiplier'
        params['macro_noise_amplitude'] = max(0.01, (1.0 - consistency) * 0.1)  # was 'base_noise_level'
        
        # Contemplation → Pausing
        contemplation = signature.get('contemplation', 0.05)
        params['pause_probability'] = min(0.2, contemplation * 2)
        
        # Direction changes → Persistence
        agitation = signature.get('agitation_level', 0.1)
        params['direction_persistence'] = max(0.2, 1.0 - (agitation * 2))
        
        # Rhythm → Mood influence
        rhythm = signature.get('rhythm_frequency', 1.0)
        params['mood_influence'] = min(3.0, rhythm)  # was 'emotional_influence'
        
        # Fine motor control
        params['micro_noise_amplitude'] = min(0.1, signature.get('speed_variance', 0.01) * 10)  # was 'micro_jitter'
        
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
            'learned_at': movements[-1]['time'] if movements else 0
        }
        
        # Save to disk
        self._save_profiles()
        
        print(f"✅ Learned emotional profile for '{emotion}' from {len(movements)} movement points!")
        return True
    
    def get_parameters_for_emotion(self, emotion: str, intensity: float = 1.0) -> Optional[Dict]:
        """Get consciousness parameters for a specific emotion."""
        if emotion not in self.emotional_profiles:
            return None
            
        params = self.emotional_profiles[emotion]['parameters'].copy()
        
        # Scale parameters by intensity
        scalable_params = ['base_speed', 'burst_movement_chance', 'noise_amplitude', 
                          'macro_noise_amplitude', 'mood_influence', 'micro_noise_amplitude']  # CORRECTED NAMES
        
        for param in scalable_params:
            if param in params:
                base_value = self.base_parameters[param]
                learned_value = params[param]
                # Interpolate between base and learned based on intensity
                params[param] = base_value + (learned_value - base_value) * intensity
        
        return params
    
    def apply_learned_parameters(self, consciousness_cursor, emotion: str, intensity: float = 1.0) -> bool:
        """Apply learned parameters to a consciousness cursor."""
        params = self.get_parameters_for_emotion(emotion, intensity)
        if not params:
            return False
        
        # Apply parameters to cursor
        for param_name, value in params.items():
            if hasattr(consciousness_cursor, param_name):
                setattr(consciousness_cursor, param_name, value)
                print(f"🎯 Applied {param_name} = {value:.3f}")
        
        print(f"🚀 Applied learned '{emotion}' movement style (intensity: {intensity:.1f})")
        return True
    
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
        except Exception as e:
            print(f"⚠️ Could not load profiles: {e}")
    
    def get_available_emotions(self) -> List[str]:
        """Get list of learned emotions."""
        return list(self.emotional_profiles.keys())
    
    def get_profile_summary(self, emotion: str) -> str:
        """Get human-readable summary of an emotional profile."""
        if emotion not in self.emotional_profiles:
            return f"No profile for '{emotion}'"
        
        profile = self.emotional_profiles[emotion]
        sig = profile['signature']
        
        # Categorize the movement style
        if sig.get('avg_speed', 0) > 0.3:
            speed_desc = "Fast"
        elif sig.get('avg_speed', 0) > 0.1:
            speed_desc = "Moderate"
        else:
            speed_desc = "Slow"
        
        if sig.get('agitation_level', 0) > 0.1:
            style_desc = "Agitated"
        elif sig.get('contemplation', 0) > 0.1:
            style_desc = "Contemplative"
        else:
            style_desc = "Fluid"
        
        if sig.get('explosiveness', 0) > 0.5:
            burst_desc = "Explosive bursts"
        elif sig.get('explosiveness', 0) > 0.2:
            burst_desc = "Occasional bursts"
        else:
            burst_desc = "Smooth movement"
        
        return f"{emotion}: {speed_desc}, {style_desc}, {burst_desc} ({profile['sample_count']} samples)"

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
