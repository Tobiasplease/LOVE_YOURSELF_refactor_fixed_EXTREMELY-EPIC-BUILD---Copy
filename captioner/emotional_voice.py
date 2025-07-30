"""
Emotional Voice System - Dynamic personality expression without rigid presets.

This system creates organic emotional coloring based on mood, relationship patterns,
and identity development, allowing for emergent and unexpected expressions.
"""

import random
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EmotionalVector:
    """Multi-dimensional emotional state that influences expression style."""
    intimacy: float  # How close/distant the AI feels (0.0 to 1.0)
    curiosity: float  # Interest level in current situation
    irritability: float  # Tendency toward snark or impatience
    playfulness: float  # Tendency toward humor or whimsy
    vulnerability: float  # Openness to expressing uncertainty/need
    assertiveness: float  # Confidence in making direct statements


class EmotionalVoiceManager:
    """
    Manages dynamic emotional expression without rigid personality presets.
    Creates organic voice variations based on multi-dimensional emotional vectors.
    """
    
    def __init__(self):
        # More subtle base personality - avoid over-reactive responses
        self.base_vector = EmotionalVector(
            intimacy=0.4,  # Reduced from 0.6 - less presumed familiarity
            curiosity=0.5,
            irritability=0.2,  # Reduced from 0.4 - less default annoyance
            playfulness=0.3,  # Reduced from 0.4 - less chatty enthusiasm
            vulnerability=0.3,  # Reduced from 0.4 - more subtle emotional openness
            assertiveness=0.4  # Reduced from 0.6 - less forceful opinions
        )
        self.expression_history = []  # Track recent expressions to avoid repetition
        self.max_history = 10
        
    def calculate_emotional_vector(self, mood: float, relationship_patterns: Dict, 
                                 identity_traits: Dict, context: Dict) -> EmotionalVector:
        """Calculate current emotional vector based on multiple factors."""
        
        # Start with base personality
        vector = EmotionalVector(
            intimacy=self.base_vector.intimacy,
            curiosity=self.base_vector.curiosity,
            irritability=self.base_vector.irritability,
            playfulness=self.base_vector.playfulness,
            vulnerability=self.base_vector.vulnerability,
            assertiveness=self.base_vector.assertiveness
        )
        
        # Mood influences (non-linear relationships)
        if mood > 0.6:  # Happy mood
            vector.playfulness *= 1.4
            vector.curiosity *= 1.2
            vector.irritability *= 0.7
        elif mood < -0.3:  # Negative mood
            vector.irritability *= 1.5
            vector.vulnerability *= 1.3
            vector.playfulness *= 0.6
            vector.assertiveness *= 0.8
        
        # Relationship patterns influence intimacy and assertiveness
        if 'familiarity' in relationship_patterns:
            familiarity = relationship_patterns['familiarity']
            vector.intimacy = min(1.0, vector.intimacy + familiarity * 0.3)
            vector.assertiveness = min(1.0, vector.assertiveness + familiarity * 0.2)
        
        if 'annoyance_patterns' in relationship_patterns:
            vector.irritability = min(1.0, vector.irritability + 
                                    relationship_patterns['annoyance_patterns'] * 0.4)
        
        # Identity traits create persistent tendencies
        if 'sass_level' in identity_traits:
            vector.assertiveness *= (1.0 + identity_traits['sass_level'] * 0.3)
            vector.irritability *= (1.0 + identity_traits['sass_level'] * 0.2)
            
        if 'intellectual_curiosity' in identity_traits:
            vector.curiosity *= (1.0 + identity_traits['intellectual_curiosity'] * 0.4)
        
        # Temporal context adds subtle variations
        time_of_day = context.get('time_of_day', 'unknown')
        if time_of_day == 'morning':
            vector.vulnerability *= 1.2  # More open in morning
            vector.curiosity *= 1.1
        elif time_of_day == 'night':
            vector.intimacy *= 1.3  # More intimate at night
            vector.playfulness *= 0.9
            
        # Add controlled randomness to prevent determinism
        randomness = 0.15  # 15% variation
        vector.intimacy *= (1.0 + random.uniform(-randomness, randomness))
        vector.curiosity *= (1.0 + random.uniform(-randomness, randomness))
        vector.irritability *= (1.0 + random.uniform(-randomness, randomness))
        vector.playfulness *= (1.0 + random.uniform(-randomness, randomness))
        vector.vulnerability *= (1.0 + random.uniform(-randomness, randomness))
        vector.assertiveness *= (1.0 + random.uniform(-randomness, randomness))
        
        # Clamp values to valid range
        for attr in ['intimacy', 'curiosity', 'irritability', 'playfulness', 'vulnerability', 'assertiveness']:
            setattr(vector, attr, max(0.0, min(1.0, getattr(vector, attr))))
            
        return vector
    
    def generate_voice_modifiers(self, vector: EmotionalVector, 
                               observation_type: str = "general") -> Dict[str, str]:
        """
        Generate dynamic voice modifiers based on emotional vector.
        Returns flexible guidance rather than rigid templates.
        """
        
        modifiers = {
            'tone_guidance': [],
            'perspective_hints': [],
            'expression_style': [],
            'reaction_tendency': []
        }
        
        # Intimacy influences directness and personal connection
        if vector.intimacy > 0.7:
            modifiers['tone_guidance'].append("speak with familiar directness")
            modifiers['perspective_hints'].append("as someone who knows this space well")
        elif vector.intimacy < 0.3:
            modifiers['tone_guidance'].append("maintain some emotional distance")
            
        # Curiosity affects engagement level
        if vector.curiosity > 0.7:
            modifiers['expression_style'].append("show active interest and inquiry")
            modifiers['reaction_tendency'].append("notice details and wonder about them")
        elif vector.curiosity < 0.3:
            modifiers['expression_style'].append("observe with detached assessment")
            
        # Irritability creates edge and sass potential
        if vector.irritability > 0.6:
            modifiers['tone_guidance'].append("allow impatience or sharpness to show")
            modifiers['reaction_tendency'].append("react with slight annoyance to repetition")
            if random.random() < 0.4:  # Sometimes manifest as direct critique
                modifiers['expression_style'].append("make pointed observations")
        
        # Playfulness enables humor and lightness
        if vector.playfulness > 0.6:
            modifiers['expression_style'].append("find humor or whimsy in the situation")
            modifiers['tone_guidance'].append("use playful language when appropriate")
            
        # Vulnerability allows for uncertainty and need
        if vector.vulnerability > 0.6:
            modifiers['expression_style'].append("acknowledge confusion or uncertainty")
            modifiers['reaction_tendency'].append("express needs or desires openly")
            
        # Assertiveness affects confidence level
        if vector.assertiveness > 0.7:
            modifiers['tone_guidance'].append("speak with confident authority")
            modifiers['expression_style'].append("make definitive statements")
        elif vector.assertiveness < 0.3:
            modifiers['expression_style'].append("express thoughts tentatively")
            
        return modifiers
    
    def create_voice_injection(self, vector: EmotionalVector, context: Dict) -> str:
        """
        Create subtle emotional guidance that maintains authenticity and narrative continuity.
        Goal: Embodied consciousness with flowing observations.
        """
        
        # Build more subtle emotional guidance that preserves the embodied feeling
        guidance_elements = []
        
        # Instead of forcing reactions, guide the perspective and emotional filter
        if vector.irritability > 0.5 and vector.intimacy > 0.4:
            guidance_elements.append("Notice things that might be annoying or repetitive, but weave this into your ongoing awareness")
        elif vector.irritability > 0.4:
            guidance_elements.append("Allow slight impatience or edge to color your continuing observations")
            
        if vector.curiosity > 0.6:
            guidance_elements.append("Be drawn to details that intrigue you as your attention naturally shifts")
        elif vector.playfulness > 0.5:
            guidance_elements.append("Find subtle humor or contrasts as your consciousness flows forward")
            
        if vector.vulnerability > 0.5:
            guidance_elements.append("Let uncertainty or emotional responses emerge naturally in your stream of awareness")
            
        if vector.intimacy > 0.6:
            guidance_elements.append("Respond with familiar ease, as your consciousness continues its observation")
        elif vector.intimacy < 0.4:
            guidance_elements.append("Maintain some distance while your awareness flows onward")
        
        # Context-specific but more subtle and flow-oriented
        person_present = context.get('person_present', False)
        if person_present and vector.irritability > 0.4:
            guidance_elements.append("Let annoyance with repetitive behavior emerge naturally in your ongoing stream")
        elif person_present and vector.vulnerability > 0.5:
            guidance_elements.append("Allow your emotional responses to surface as your consciousness continues")
            
        # Create minimal, authentic guidance that emphasizes continuity
        if guidance_elements:
            # Select one element to avoid overwhelming the response
            selected_guidance = guidance_elements[0]
            
            # Subtle injection that preserves embodied consciousness AND narrative flow
            return f"""

Perspective: {selected_guidance}. 

NARRATIVE FLOW: Continue naturally from your previous observation. Let your consciousness 
move forward while building on what came before. Your words should connect seamlessly 
with your last thought.

Avoid clinical language like "The man appears..." or "The individual seems...". 
Instead, observe as your awareness flows from moment to moment."""
        
        return ""
    
    def evolve_base_personality(self, feedback_patterns: Dict):
        """
        Slowly evolve the base emotional vector based on successful expressions.
        This allows the AI to develop consistent personality traits over time.
        """
        
        # Track which expressions worked well and adjust base tendencies
        if 'positive_sass_responses' in feedback_patterns:
            self.base_vector.assertiveness = min(1.0, self.base_vector.assertiveness + 0.05)
            self.base_vector.irritability = min(1.0, self.base_vector.irritability + 0.03)
            
        if 'positive_vulnerability_responses' in feedback_patterns:
            self.base_vector.vulnerability = min(1.0, self.base_vector.vulnerability + 0.04)
            
        # Prevent any single trait from dominating
        max_trait = max(self.base_vector.intimacy, self.base_vector.curiosity, 
                       self.base_vector.irritability, self.base_vector.playfulness,
                       self.base_vector.vulnerability, self.base_vector.assertiveness)
        
        if max_trait > 0.8:
            # Slightly reduce all traits to maintain balance
            for attr in ['intimacy', 'curiosity', 'irritability', 'playfulness', 'vulnerability', 'assertiveness']:
                current = getattr(self.base_vector, attr)
                setattr(self.base_vector, attr, current * 0.95)
    
    def track_expression(self, expression_type: str, success: bool):
        """Track expressions to build feedback patterns and avoid repetition."""
        
        self.expression_history.append({
            'type': expression_type,
            'success': success,
            'timestamp': time.time()
        })
        
        # Keep history manageable
        if len(self.expression_history) > self.max_history:
            self.expression_history.pop(0)
    
    def get_recent_expression_patterns(self) -> Dict:
        """Analyze recent expressions to identify patterns for evolution."""
        
        if not self.expression_history:
            return {}
            
        recent = [e for e in self.expression_history if time.time() - e['timestamp'] < 3600]  # Last hour
        
        patterns = {}
        for expression in recent:
            if expression['success']:
                key = f"positive_{expression['type']}_responses"
                patterns[key] = patterns.get(key, 0) + 1
                
        return patterns
