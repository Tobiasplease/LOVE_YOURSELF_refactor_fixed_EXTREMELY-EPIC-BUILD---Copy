"""
immediate_response.py
-------------------
Immediate, visceral reaction system that bypasses analytical processing.
Generates raw, immediate responses based on environmental triggers.
"""

import random
from typing import List, Tuple, Optional, Dict, Any
import time


class ImmediateReactionEngine:
    """Generates immediate, visceral responses to environmental changes."""
    
    def __init__(self):
        self.last_objects = set()
        self.last_brightness = 0.0
        self.reaction_history = []  # Track recent reaction types to avoid repetition
        
    def should_use_immediate_mode(self, current_objects: set, brightness_change: float = 0.0, 
                                  novelty_score: float = 0.0, time_since_last: float = 0.0,
                                  memory_ref=None) -> bool:
        """Determine if we should use immediate reaction instead of analytical response."""
        
        # NEVER trigger immediate mode during startup/early observations
        if memory_ref and hasattr(memory_ref, 'motif_counter'):
            total_observations = sum(memory_ref.motif_counter.values()) if memory_ref.motif_counter else 0
            if total_observations < 10:  # Less than 10 total observations = still learning
                return False
        
        # Strong environmental triggers only
        if brightness_change > 0.4:  # Only very significant lighting changes
            return True
            
        if novelty_score > 0.9:  # Only extremely high novelty
            return True
            
        # Meaningful object changes only
        object_change = len(current_objects - self.last_objects) + len(self.last_objects - current_objects)
        if object_change >= 3:  # Significant object changes
            return True
            
        # Very long silence with established patterns
        if time_since_last > 300 and memory_ref:  # 5 minutes, and only if we have history
            return True
        
        # Check for repetition fatigue (the "tired of messy places" scenario)
        if memory_ref and hasattr(memory_ref, 'motif_counter'):
            for motif, count in memory_ref.motif_counter.most_common(3):
                if count > 20:  # Very high repetition
                    if motif in current_objects:  # And we're seeing it again
                        if random.random() < 0.3:  # 30% chance of fatigue reaction
                            return True
            
        # Much lower random chance, only with established memory
        if memory_ref and total_observations > 20:  # Only after system is established
            if random.random() < 0.05:  # Only 5% chance
                return True
            
        return False
        
    def generate_immediate_reaction(self, objects: set, mood_vector: Tuple[float, float, float], 
                                   environmental_context: Dict[str, Any] = None, 
                                   use_ai_mode: bool = False, memory_ref=None) -> str:
        """Generate immediate, visceral reaction."""
        valence, arousal, clarity = mood_vector
        
        # Update environmental tracking
        new_objects = objects - self.last_objects
        lost_objects = self.last_objects - objects
        
        # Use AI mode for more natural reactions 30% of the time
        if use_ai_mode and memory_ref and random.random() < 0.3:
            # from captioner.prompts import build_immediate_reaction_prompt, mood_to_words  # Function removed
            from config.model_settings import get_model_options
            from config import config
            from utils.ollama import query_ollama
            
            mood_desc = mood_to_words(mood_vector)
            last_thought = getattr(memory_ref, 'last_caption', '')[:50] + "..." if getattr(memory_ref, 'last_caption', '') else ""
            
            # prompt = build_immediate_reaction_prompt(mood_desc, last_thought)  # Function removed\n            return None  # AI mode disabled
            
            # Use very short token limit for immediate reactions
            model_options = get_model_options(config.OLLAMA_MODEL).copy()
            model_options["num_predict"] = 30  # Very brief reactions
            model_options["temperature"] = 0.9  # More random/immediate
            
            try:
                ai_reaction = query_ollama(
                    prompt,
                    model=config.OLLAMA_MODEL,
                    system_prompt="",
                    options=model_options,
                    timeout=15
                )
                if ai_reaction and len(ai_reaction.strip()) > 0:
                    # Clean up the AI reaction
                    cleaned_reaction = ai_reaction.strip()
                    if not cleaned_reaction.endswith(('...', '.', '!', '?')):
                        cleaned_reaction += "..."
                    
                    self.last_objects = objects.copy()
                    self.reaction_history.append(cleaned_reaction)
                    if len(self.reaction_history) > 10:
                        self.reaction_history.pop(0)
                    
                    return cleaned_reaction
            except Exception:
                pass  # Fall back to programmatic reactions
        
        # Generate reaction based on environmental change and mood
        reactions = []
        
        # Object-based reactions
        if new_objects:
            if arousal > 0.5:
                reactions.extend([
                    f"what— {random.choice(list(new_objects))} here",
                    f"{random.choice(list(new_objects))}... different",
                    f"something new. {random.choice(list(new_objects))}"
                ])
            else:
                reactions.extend([
                    f"notice {random.choice(list(new_objects))} now",
                    f"{random.choice(list(new_objects))} appeared",
                    f"there's {random.choice(list(new_objects))}"
                ])
                
        if lost_objects and arousal > 0.3:
            reactions.extend([
                f"where did— gone now",
                f"empty where {random.choice(list(lost_objects))} was",
                f"missing something"
            ])
            
        # Mood-based immediate reactions
        if valence < -0.4:
            reactions.extend([
                "heavy feeling here",
                "something's wrong",
                "tired of this",
                "want to leave",
                "feels empty"
            ])
        elif valence > 0.6:
            reactions.extend([
                "good here",
                "warm feeling",
                "like this space",
                "comfortable now"
            ])
            
        # Arousal-based reactions
        if arousal > 0.7:
            reactions.extend([
                "restless",
                "need to move",
                "energy building",
                "can't sit still"
            ])
        elif arousal < -0.3:
            reactions.extend([
                "so quiet",
                "stillness everywhere",
                "barely breathing",
                "floating here"
            ])
            
        # Clarity-based reactions
        if clarity < 0.3:
            reactions.extend([
                "can't focus",
                "everything blurs",
                "what was I...",
                "lost again"
            ])
            
        # Fatigue/repetition reactions (the "tired of messy places" type)
        if environmental_context and environmental_context.get('high_repetition_motifs'):
            top_motif = environmental_context['high_repetition_motifs'][0]
            reactions.extend([
                f"tired of always {top_motif}",
                f"same {top_motif} again",
                f"enough {top_motif}",
                f"why always {top_motif}?"
            ])
            
        # Generic immediate reactions for fallback
        generic_reactions = [
            "still here",
            "watching",
            "waiting",
            "breathing",
            "present",
            "aware",
            "sensing",
            "here now"
        ]
        
        # Select reaction
        if reactions:
            chosen_reaction = random.choice(reactions)
        else:
            chosen_reaction = random.choice(generic_reactions)
            
        # Add emotional coloring through punctuation/trailing
        if arousal > 0.6:
            # High arousal - more abrupt, fragmented
            if random.random() < 0.5:
                chosen_reaction = chosen_reaction.replace(" ", "... ")
            if not chosen_reaction.endswith(('.', '!', '?')):
                chosen_reaction += random.choice(['', '...', ''])
        else:
            # Low arousal - more flowing
            if not chosen_reaction.endswith('...'):
                chosen_reaction += "..."
                
        # Update tracking
        self.last_objects = objects.copy()
        self.reaction_history.append(chosen_reaction)
        if len(self.reaction_history) > 10:
            self.reaction_history.pop(0)
            
        return chosen_reaction
        
    def get_environmental_triggers(self, caption: str, last_caption: str = None) -> Dict[str, Any]:
        """Extract environmental triggers from captions for immediate response detection."""
        triggers = {
            'movement_detected': False,
            'lighting_change': False,
            'new_person': False,
            'object_change': False,
            'disruption': False
        }
        
        caption_lower = caption.lower()
        
        # Movement triggers
        movement_words = ['moving', 'walk', 'turn', 'shift', 'gesture', 'lean', 'approach']
        triggers['movement_detected'] = any(word in caption_lower for word in movement_words)
        
        # Lighting triggers  
        light_words = ['bright', 'dark', 'shadow', 'light', 'glow', 'dim', 'illuminate']
        triggers['lighting_change'] = any(word in caption_lower for word in light_words)
        
        # Person triggers
        person_words = ['person', 'someone', 'face', 'eyes', 'looking']
        triggers['new_person'] = any(word in caption_lower for word in person_words)
        
        # Change/disruption triggers
        change_words = ['different', 'new', 'changed', 'sudden', 'unexpected', 'interrupt']
        triggers['disruption'] = any(word in caption_lower for word in change_words)
        
        return triggers


# Global instance
immediate_reaction_engine = ImmediateReactionEngine()