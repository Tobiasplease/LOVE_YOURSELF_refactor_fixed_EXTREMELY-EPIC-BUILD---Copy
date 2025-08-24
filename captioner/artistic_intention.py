"""
artistic_intention.py
-------------------
Tracks and accumulates artistic intentions from reflections.
Creates continuity between contemplative moments and drawing output.
"""

from typing import List, Dict, Optional, Tuple
from collections import deque
import time
import re


class ArtisticIntentionTracker:
    """Accumulates artistic intentions from reflections to inform drawing decisions."""
    
    def __init__(self, max_intentions: int = 15):
        self.drawing_ambitions = deque(maxlen=max_intentions)  # Recent artistic intentions
        self.thematic_threads = {}  # Recurring themes/interests
        self.visual_fascinations = []  # Things that visually captivate
        self.emotional_expressions = []  # Emotions wanting artistic outlet
        
    def extract_artistic_intention(self, reflection_text: str) -> Optional[Dict]:
        """Extract drawing-related intentions from reflection text."""
        
        intention_patterns = {
            'want_to_capture': r'(?:want to capture|desire to show|hope to express|wish to convey|trying to capture|yearning to show)[\s\w]*?([^.!?]*)',
            'fascinated_by': r'(?:fascinated by|drawn to|captivated by|intrigued by|mesmerized by)[\s\w]*?([^.!?]*)',
            'visual_interest': r'(?:visual|visually|way the light|how the|color|shadow|form|shape|composition)[\s\w]*?([^.!?]*)', 
            'artistic_feeling': r'(?:want to draw|imagine drawing|picture creating|envision making|artistic|creative impulse)[\s\w]*?([^.!?]*)',
            'emotional_drive': r'(?:need to express|urge to show|compelled to|driven to create|burning to)[\s\w]*?([^.!?]*)',
        }
        
        extracted_intentions = []
        
        for intention_type, pattern in intention_patterns.items():
            matches = re.findall(pattern, reflection_text.lower(), re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 5:  # Avoid very short fragments
                    extracted_intentions.append({
                        'type': intention_type,
                        'content': match.strip(),
                        'timestamp': time.time(),
                        'strength': self._assess_intention_strength(match, reflection_text)
                    })
        
        # Also look for specific artistic themes
        artistic_themes = self._extract_visual_themes(reflection_text)
        for theme in artistic_themes:
            extracted_intentions.append({
                'type': 'visual_theme',
                'content': theme,
                'timestamp': time.time(),
                'strength': 0.6
            })
        
        return extracted_intentions if extracted_intentions else None
    
    def _assess_intention_strength(self, intention: str, full_text: str) -> float:
        """Assess how strong/urgent this artistic intention is."""
        strength_indicators = {
            'urgent': ['desperately', 'urgently', 'intensely', 'burning', 'compelled', 'must'],
            'strong': ['really want', 'deeply', 'strongly', 'yearning', 'longing'],
            'moderate': ['want to', 'hope to', 'would like', 'interested in'],
            'mild': ['might', 'perhaps', 'could', 'sometimes']
        }
        
        base_strength = 0.5
        full_lower = full_text.lower()
        
        for level, indicators in strength_indicators.items():
            for indicator in indicators:
                if indicator in full_lower:
                    if level == 'urgent': base_strength += 0.4
                    elif level == 'strong': base_strength += 0.3
                    elif level == 'moderate': base_strength += 0.1
                    elif level == 'mild': base_strength -= 0.1
        
        return max(0.1, min(1.0, base_strength))
    
    def _extract_visual_themes(self, text: str) -> List[str]:
        """Extract recurring visual themes and interests."""
        theme_patterns = [
            r'(?:light|lighting|illumination|shadow|contrast)',
            r'(?:solitude|solitary|alone|isolation|companionship)',
            r'(?:technology|digital|screens|devices)',
            r'(?:nature|organic|natural|living)',
            r'(?:space|environment|room|atmosphere)',
            r'(?:human|person|face|gesture|expression)',
            r'(?:color|hue|tone|palette)',
            r'(?:texture|surface|material|form)',
            r'(?:movement|motion|stillness|static)',
            r'(?:intimate|personal|private|public)'
        ]
        
        found_themes = []
        text_lower = text.lower()
        
        for pattern in theme_patterns:
            if re.search(pattern, text_lower):
                # Extract the actual theme word
                match = re.search(pattern, text_lower)
                if match:
                    found_themes.append(match.group().replace('|', ' or '))
        
        return found_themes
    
    def add_reflection_intentions(self, reflection_text: str):
        """Process a reflection and extract/store artistic intentions."""
        intentions = self.extract_artistic_intention(reflection_text)
        
        if intentions:
            for intention in intentions:
                self.drawing_ambitions.append(intention)
                
                # Track thematic threads
                theme_key = intention['type']
                if theme_key not in self.thematic_threads:
                    self.thematic_threads[theme_key] = []
                self.thematic_threads[theme_key].append(intention['content'])
                
                # Keep only recent themes
                if len(self.thematic_threads[theme_key]) > 5:
                    self.thematic_threads[theme_key].pop(0)
    
    def get_accumulated_drawing_intentions(self, limit: int = 8) -> List[Dict]:
        """Get the most relevant accumulated drawing intentions."""
        # Sort by recency and strength
        recent_intentions = list(self.drawing_ambitions)[-limit:]
        
        # Prioritize stronger intentions
        sorted_intentions = sorted(recent_intentions, 
                                 key=lambda x: (x['strength'], x['timestamp']), 
                                 reverse=True)
        
        return sorted_intentions
    
    def get_thematic_summary(self) -> str:
        """Generate a summary of recurring artistic themes."""
        if not self.thematic_threads:
            return ""
        
        theme_summary = []
        
        # Find most common themes
        for theme_type, contents in self.thematic_threads.items():
            if len(contents) >= 2:  # Recurring theme
                recent_content = contents[-2:]  # Last 2 instances
                theme_summary.append(f"{theme_type.replace('_', ' ')}: {', '.join(recent_content[:2])}")
        
        return "; ".join(theme_summary) if theme_summary else ""
    
    def build_drawing_context_from_intentions(self) -> str:
        """Build context string for drawing prompts based on accumulated intentions."""
        intentions = self.get_accumulated_drawing_intentions()
        
        if not intentions:
            return ""
        
        context_parts = []
        
        # Group by intention type
        grouped = {}
        for intention in intentions:
            intention_type = intention['type']
            if intention_type not in grouped:
                grouped[intention_type] = []
            grouped[intention_type].append(intention['content'])
        
        # Build natural language summary
        if 'want_to_capture' in grouped:
            context_parts.append(f"You've been wanting to capture: {', '.join(grouped['want_to_capture'][:2])}")
        
        if 'fascinated_by' in grouped:
            context_parts.append(f"You're fascinated by: {', '.join(grouped['fascinated_by'][:2])}")
            
        if 'visual_theme' in grouped:
            themes = list(set(grouped['visual_theme']))[:3]  # Unique themes
            context_parts.append(f"Recurring visual interests: {', '.join(themes)}")
        
        if 'artistic_feeling' in grouped:
            context_parts.append(f"Artistic impulses: {', '.join(grouped['artistic_feeling'][:2])}")
        
        return " | ".join(context_parts) if context_parts else ""


# Global instance
artistic_intention_tracker = ArtisticIntentionTracker()