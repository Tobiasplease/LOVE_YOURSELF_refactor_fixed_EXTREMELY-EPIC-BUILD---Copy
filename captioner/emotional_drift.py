"""
Emotional Drift System - Relative emotional accumulation based on caption-to-caption comparison.
Each caption becomes the new baseline, allowing for unbounded emotional evolution.
"""

import time
import random
from collections import deque
from typing import Dict, Optional, Tuple
import math


class EmotionalDrift:
    """Tracks relative emotional changes between consecutive captions, allowing compound drift."""
    
    def __init__(self):
        # Three-dimensional momentum that can grow unbounded
        self.momentum = {
            "energy": 0.0,     # Energized vs exhausted
            "valence": 0.0,    # Positive vs negative
            "coherence": 0.0   # Focused vs scattered
        }
        
        self.last_caption: Optional[str] = None
        self.drift_history = deque(maxlen=20)  # Recent drift directions
        self.session_start = time.time()
        self.comparison_count = 0
        self.stuck_counter = 0  # Detect when stuck in one direction
        
    def process_caption(self, new_caption: str, environmental_factors: Dict) -> Dict[str, float]:
        """Process new caption and update emotional momentum."""
        if not self.last_caption:
            self.last_caption = new_caption
            return self.momentum
            
        # Compare to previous caption
        delta = self._compare_captions(new_caption, self.last_caption)
        
        # Apply environmental modulation
        delta = self._modulate_drift(delta, environmental_factors)
        
        # Update momentum with stability mechanisms
        self._update_momentum(delta)
        
        # Store for next comparison
        self.last_caption = new_caption
        self.comparison_count += 1
        
        return self.momentum
    
    def _compare_captions(self, current: str, previous: str) -> Dict[str, float]:
        """Compare emotional tone between two captions using TinyLlama."""
        from captioner.model_wrapper import MultimodalModel
        
        # Create a temporary model instance for comparison
        model = MultimodalModel()
        
        # Simple comparison prompt
        prompt = f"""Compare the emotional tone between these two observations:

Previous: "{previous[:200]}"
Current: "{current[:200]}"

Rate the change in three dimensions from -1 to +1:
Energy (tired to energized): ?
Valence (negative to positive): ?
Coherence (scattered to focused): ?

Return ONLY three numbers separated by commas like: 0.2,-0.1,0.3"""

        try:
            response = model.query_tinyllama(prompt)
            # Parse response
            parts = response.strip().split(',')
            if len(parts) == 3:
                return {
                    "energy": float(parts[0].strip()),
                    "valence": float(parts[1].strip()),
                    "coherence": float(parts[2].strip())
                }
        except:
            pass
            
        # Fallback: simple heuristic comparison
        return self._heuristic_comparison(current, previous)
    
    def _heuristic_comparison(self, current: str, previous: str) -> Dict[str, float]:
        """Fallback heuristic comparison if TinyLlama fails."""
        delta = {"energy": 0.0, "valence": 0.0, "coherence": 0.0}
        
        # Energy indicators
        if len(current) > len(previous) * 1.2:
            delta["energy"] += 0.1
        elif len(current) < len(previous) * 0.8:
            delta["energy"] -= 0.1
            
        # Valence indicators
        positive_words = ["interesting", "beautiful", "peaceful", "pleasant", "nice"]
        negative_words = ["boring", "frustrating", "annoying", "tired", "same"]
        
        current_lower = current.lower()
        for word in positive_words:
            if word in current_lower:
                delta["valence"] += 0.1
        for word in negative_words:
            if word in current_lower:
                delta["valence"] -= 0.1
                
        # Coherence indicators (fragmentation)
        if "..." in current:
            delta["coherence"] -= 0.05
        if "?" in current:
            delta["coherence"] += 0.05
            
        return delta
    
    def _modulate_drift(self, delta: Dict[str, float], environmental_factors: Dict) -> Dict[str, float]:
        """Apply environmental forces to prevent simple linear drift."""
        
        # Static scenes reduce drift (boredom gravity)
        if environmental_factors.get("scene_static", False):
            for dim in delta:
                delta[dim] *= 0.6
                
        # High novelty creates turbulence
        novelty = environmental_factors.get("novelty", 0.0)
        if novelty > 0.7:
            for dim in delta:
                delta[dim] += random.uniform(-0.15, 0.15)
                
        # Person presence moderates extremes
        if environmental_factors.get("person_present", False):
            for dim in delta:
                delta[dim] *= 0.85
                
        # Session duration affects energy
        session_hours = (time.time() - self.session_start) / 3600
        if session_hours > 2:  # Extended sessions
            delta["energy"] -= 0.02  # Natural fatigue
            
        return delta
    
    def _update_momentum(self, delta: Dict[str, float]):
        """Update momentum with stability mechanisms."""
        
        # Track drift direction
        self.drift_history.append(delta.copy())
        
        # Check if stuck in one direction
        if len(self.drift_history) >= 10:
            recent_directions = [d["energy"] > 0 for d in list(self.drift_history)[-10:]]
            if len(set(recent_directions)) == 1:  # Same direction for 10 captions
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        
        for dimension in self.momentum:
            # Add new delta with slight decay
            self.momentum[dimension] = self.momentum[dimension] * 0.98 + delta[dimension]
            
            # Extremes create counter-pressure (exhaustion)
            if abs(self.momentum[dimension]) > 5:
                counter = -0.05 * (1 if self.momentum[dimension] > 0 else -1)
                self.momentum[dimension] += counter
                
            # Stuck in one direction causes fatigue
            if self.stuck_counter > 3:
                self.momentum[dimension] *= 0.9
                
            # Add stochastic resonance (prevents getting stuck)
            self.momentum[dimension] += random.uniform(-0.01, 0.01)
            
            # Natural oscillation based on session time
            session_hours = (time.time() - self.session_start) / 3600
            if dimension == "energy":
                # Energy has natural cycles
                cycle = math.sin(session_hours * math.pi / 2) * 0.02
                self.momentum[dimension] += cycle
    
    def get_emotional_descriptor(self) -> str:
        """Convert current momentum to descriptive emotional state."""
        energy = self.momentum["energy"]
        valence = self.momentum["valence"]
        coherence = self.momentum["coherence"]
        
        # Extreme states (unbounded territory)
        if energy > 10:
            if valence > 5:
                return "manically ecstatic"
            else:
                return "frantically agitated"
        elif energy < -10:
            if valence < -5:
                return "profoundly depressed"
            else:
                return "utterly exhausted"
        elif valence > 10:
            return "transcendently blissful"
        elif valence < -10:
            return "deeply resentful"
        elif coherence > 8:
            return "laser-focused"
        elif coherence < -8:
            return "completely scattered"
            
        # Moderate compound states
        if energy > 3 and valence > 3:
            return "joyfully energized"
        elif energy > 3 and valence < -3:
            return "anxiously restless"
        elif energy < -3 and valence > 3:
            return "contentedly tired"
        elif energy < -3 and valence < -3:
            return "miserably drained"
        elif abs(energy) < 1 and abs(valence) < 1:
            return "emotionally neutral"
        else:
            # Build custom descriptor
            e_word = "energized" if energy > 0 else "tired"
            v_word = "positive" if valence > 0 else "troubled"
            c_word = "focused" if coherence > 0 else "scattered"
            
            # Pick most prominent dimension
            max_dim = max(abs(energy), abs(valence), abs(coherence))
            if abs(energy) == max_dim:
                return f"somewhat {e_word}"
            elif abs(valence) == max_dim:
                return f"feeling {v_word}"
            else:
                return f"mentally {c_word}"
    
    def get_drift_context(self) -> str:
        """Get context about current emotional trajectory for prompts."""
        if self.comparison_count < 3:
            return ""
            
        # Analyze recent drift
        if len(self.drift_history) >= 5:
            recent = list(self.drift_history)[-5:]
            energy_trend = sum(d["energy"] for d in recent)
            valence_trend = sum(d["valence"] for d in recent)
            
            if energy_trend > 1:
                return "Your energy has been building..."
            elif energy_trend < -1:
                return "You're growing increasingly tired..."
            elif valence_trend > 1:
                return "Your mood is lifting..."
            elif valence_trend < -1:
                return "Frustration is accumulating..."
            elif self.stuck_counter > 5:
                return "You've been in this state for a while now..."
                
        # Check for extreme states
        max_momentum = max(abs(v) for v in self.momentum.values())
        if max_momentum > 8:
            return "You're approaching an emotional extreme..."
        elif max_momentum < 0.5:
            return "Your emotions have settled to equilibrium..."
            
        return ""