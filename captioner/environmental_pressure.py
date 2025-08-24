"""
environmental_pressure.py
-------------------------
Creates environmental pressure on AI response parameters rather than injecting phrases.
Affects temperature, token limits, system prompts subtly based on environmental factors.
"""

from typing import Dict, Any, Tuple, Optional
import random


class EnvironmentalPressureEngine:
    """Modulates AI response parameters based on environmental and emotional factors."""
    
    def __init__(self):
        self.baseline_temperature = 0.7
        self.baseline_tokens = 100
        self.pressure_history = []
        
    def calculate_response_pressure(self, 
                                   novelty_score: float,
                                   repetition_context: Dict[str, Any],
                                   mood_vector: Tuple[float, float, float],
                                   environmental_change: float = 0.0,
                                   time_since_last: float = 0.0) -> Dict[str, Any]:
        """Calculate environmental pressure on response parameters."""
        
        valence, arousal, clarity = mood_vector
        
        # Start with baseline parameters
        pressure_modifiers = {
            'temperature_modifier': 0.0,  # Added to baseline
            'token_modifier': 0,          # Added to baseline tokens
            'brevity_pressure': 0.0,      # 0-1 scale
            'fragmentation_pressure': 0.0, # 0-1 scale  
            'emotional_intensity': abs(valence) + abs(arousal),
            'system_prompt_modifier': 'normal'
        }
        
        # NOVELTY PRESSURE
        if novelty_score > 0.8:
            # High novelty = more spontaneous, brief reactions
            pressure_modifiers['temperature_modifier'] += 0.3  # More random
            pressure_modifiers['token_modifier'] -= 30  # Briefer
            pressure_modifiers['brevity_pressure'] = 0.8
            pressure_modifiers['system_prompt_modifier'] = 'sharp_attention'
            
        elif novelty_score < 0.2:
            # Low novelty = more contemplative, complete thoughts
            pressure_modifiers['temperature_modifier'] -= 0.1  # More focused
            pressure_modifiers['token_modifier'] += 20  # More elaborate
            
        # REPETITION FATIGUE PRESSURE  
        if repetition_context.get('high_repetition_motifs'):
            repetition_level = len(repetition_context['high_repetition_motifs'])
            
            if repetition_level >= 3:  # Multiple things being repeated
                pressure_modifiers['temperature_modifier'] += 0.2  # More erratic
                pressure_modifiers['token_modifier'] -= 40  # Much briefer
                pressure_modifiers['brevity_pressure'] = 0.9
                pressure_modifiers['fragmentation_pressure'] = 0.7
                pressure_modifiers['system_prompt_modifier'] = 'repetition_fatigue'
                
        # AROUSAL PRESSURE
        if arousal > 0.7:  # High arousal
            pressure_modifiers['temperature_modifier'] += 0.2
            pressure_modifiers['token_modifier'] -= 20
            pressure_modifiers['fragmentation_pressure'] = 0.6
            
        elif arousal < -0.3:  # Very low arousal
            pressure_modifiers['temperature_modifier'] -= 0.2  # More measured
            pressure_modifiers['token_modifier'] += 10
            
        # CLARITY PRESSURE
        if clarity < 0.3:  # Low clarity = confused, fragmented
            pressure_modifiers['fragmentation_pressure'] = 0.8
            pressure_modifiers['token_modifier'] -= 25
            pressure_modifiers['system_prompt_modifier'] = 'uncertain'
            
        # ENVIRONMENTAL DISRUPTION
        if environmental_change > 0.5:  # Significant environmental change
            pressure_modifiers['temperature_modifier'] += 0.25
            pressure_modifiers['fragmentation_pressure'] = 0.5
            pressure_modifiers['system_prompt_modifier'] = 'disrupted'
            
        # LONG SILENCE PRESSURE
        if time_since_last > 180:  # Long silence
            pressure_modifiers['temperature_modifier'] += 0.1
            pressure_modifiers['brevity_pressure'] = 0.6  # Break silence briefly
            
        return pressure_modifiers
    
    def apply_pressure_to_model_options(self, base_options: Dict[str, Any], 
                                       pressure: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environmental pressure to model options."""
        
        modified_options = base_options.copy()
        
        # Apply temperature modification
        new_temp = max(0.1, min(1.0, 
            self.baseline_temperature + pressure['temperature_modifier']))
        modified_options['temperature'] = new_temp
        
        # Apply token modification
        new_tokens = max(20, min(300, 
            self.baseline_tokens + pressure['token_modifier']))
        modified_options['num_predict'] = new_tokens
        
        # Apply fragmentation through stop conditions (more conservative)
        if pressure['fragmentation_pressure'] > 0.7:  # Higher threshold
            # Allow more interruption/fragmentation
            if 'stop' not in modified_options:
                modified_options['stop'] = []
            # Add only very specific fragmentation patterns
            fragmentation_stops = ["—", "... but", "... though"]
            modified_options['stop'].extend(fragmentation_stops)
            
        # Apply brevity pressure through more restrictive stops
        if pressure['brevity_pressure'] > 0.7:
            if 'stop' not in modified_options:
                modified_options['stop'] = []
            # Use more specific stops that don't truncate mid-thought
            brevity_stops = [".\n", "...\n", "!\n", "?\n", ". I", ". The", ". But"]
            modified_options['stop'].extend(brevity_stops)
            
        return modified_options
    
    def get_pressure_influenced_system_prompt(self, base_prompt: str, 
                                            pressure: Dict[str, Any]) -> str:
        """Modify system prompt based on environmental pressure."""
        
        modifier_type = pressure['system_prompt_modifier']
        
        if modifier_type == 'sharp_attention':
            # High novelty - sharp, immediate attention
            return base_prompt + "\\n\\nSomething new requires immediate attention. React sharply to what's different."
            
        elif modifier_type == 'repetition_fatigue':
            # Repetition fatigue - growing impatience
            return base_prompt + "\\n\\nFamiliar patterns are becoming tiresome. Let any growing impatience show."
            
        elif modifier_type == 'uncertain':
            # Low clarity - confused, searching
            return base_prompt + "\\n\\nThoughts feel unclear and scattered. Express uncertainty naturally."
            
        elif modifier_type == 'disrupted':
            # Environmental disruption - interrupted thoughts
            return base_prompt + "\\n\\nSomething in the environment has shifted. Let this disruption interrupt your thoughts."
            
        else:
            return base_prompt
    
    def create_gut_voice_prompt(self, mood_vector: Tuple[float, float, float], 
                               environmental_context: str = "") -> Optional[str]:
        """Create a 'gut voice' prompt for very brief, instinctual reactions."""
        
        valence, arousal, clarity = mood_vector
        
        # Only trigger gut voice under specific high-pressure conditions
        if abs(arousal) < 0.5 and abs(valence) < 0.4:
            return None  # Not enough emotional intensity
            
        gut_prompt = f"""Brief gut reaction to: {environmental_context}
        
        Emotional state: {"tense" if arousal > 0.5 else "low"}, {"heavy" if valence < -0.3 else "light" if valence > 0.3 else "neutral"}
        
        Express in 3-8 words maximum. Raw feeling only:"""
        
        return gut_prompt


# Global instance
environmental_pressure_engine = EnvironmentalPressureEngine()