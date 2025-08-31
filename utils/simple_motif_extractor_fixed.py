# utils/simple_motif_extractor_fixed.py
"""
Simple, reliable motif extraction that finds actual objects and concepts.
Replaces the broken thematic analyzer with something that actually works.
"""

import re
from collections import defaultdict
from typing import List


class SimpleMotifExtractor:
    """Extract meaningful motifs (objects, concepts, emotions) from captions."""

    def __init__(self):
        # Common objects and concepts that appear in captions
        self.object_patterns = {
            # People and beings
            "person": r"\b(?:person|people|human|man|woman|someone|individual)\b",
            "face": r"\b(?:face|eyes|smile|expression)\b",
            # Furniture and objects
            "screen": r"\b(?:screen|monitor)\b",
            "table": r"\b(?:table|desk|surface|workspace)\b",
            "door": r"\b(?:door|doorway|entrance|exit)\b",
            "chair": r"\b(?:chair|seat|sitting)\b",
            "wall": r"\b(?:wall|walls|ceiling)\b",
            "window": r"\b(?:window|windows|glass)\b",
            "light": r"\b(?:light|lighting|lamp|bright|dim|illuminated)\b",
            "room": r"\b(?:room|space|environment|area|place)\b",
            # Emotions and states
            "calm": r"\b(?:calm|peaceful|serene|tranquil|quiet|still)\b",
            "curious": r"\b(?:curious|wonder|wondering|interesting|fascinated)\b",
            "comfortable": r"\b(?:comfortable|cozy|warm|familiar|at ease)\b",
            "isolated": r"\b(?:isolated|alone|lonely|solitude|apart)\b",
            "connected": r"\b(?:connected|together|bond|relationship|close)\b",
            "focused": r"\b(?:focused|concentrated|absorbed|attentive)\b",
            "restless": r"\b(?:restless|agitated|restrained|trapped)\b",
            # Activities
            "working": r"\b(?:working|typing|studying|reading|using)\b",
            "watching": r"\b(?:watching|observing|looking|seeing|viewing)\b",
            "thinking": r"\b(?:thinking|contemplating|reflecting|pondering)\b",
            # Qualities and descriptions
            "familiar": r"\b(?:familiar|known|recognized|usual|accustomed)\b",
            "unknown": r"\b(?:unknown|unfamiliar|strange|new|different)\b",
            "soft": r"\b(?:soft|gentle|subtle|delicate)\b",
            "dark": r"\b(?:dark|shadow|shadows|dim|faded)\b",
            "warm": r"\b(?:warm|warmth|cozy|inviting)\b",
        }

        # Track extraction history for continuity
        self.recent_motifs = defaultdict(int)
        self.caption_count = 0

    def extract_motifs(self, caption: str) -> List[str]:
        """Extract meaningful motifs from a caption."""
        self.caption_count += 1
        found_motifs = []

        caption_lower = caption.lower()

        # Find all matching patterns
        for motif_name, pattern in self.object_patterns.items():
            if re.search(pattern, caption_lower, re.IGNORECASE):
                found_motifs.append(motif_name)
                self.recent_motifs[motif_name] += 1

        # Add some context-specific motifs

        # Detect specific relationships
        if re.search(r"person.{0,20}(?:know|familiar|well|together)", caption_lower):
            found_motifs.append("familiar_person")

        if re.search(r"(?:three days?|3 days?)", caption_lower):
            found_motifs.append("established_relationship")

        # Detect spatial relationships
        if re.search(r"(?:in front of|behind|near|next to|beside)", caption_lower):
            found_motifs.append("spatial_relationship")

        # Detect temporal awareness
        if re.search(r"(?:remember|earlier|before|previous|used to)", caption_lower):
            found_motifs.append("memory_reference")

        if re.search(r"(?:now|current|present|this moment)", caption_lower):
            found_motifs.append("present_awareness")

        # Remove duplicates while preserving order
        return list(dict.fromkeys(found_motifs))

    def get_current_motifs(self) -> List[str]:
        """Get the most recent motifs for continuity."""
        # Return motifs mentioned in recent captions
        if not self.recent_motifs:
            return []

        # Get motifs mentioned recently (decaying importance)
        current = []
        for motif, count in sorted(self.recent_motifs.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                current.append(motif)
                # Decay the count
                self.recent_motifs[motif] = max(0, count - 1)

        return current[:8]  # Top 8 current motifs

    def clear_history(self):
        """Clear motif history (for new sessions)."""
        self.recent_motifs.clear()
        self.caption_count = 0


# Global instance
_extractor = None


def get_simple_motif_extractor() -> SimpleMotifExtractor:
    """Get the global simple motif extractor."""
    global _extractor
    if _extractor is None:
        _extractor = SimpleMotifExtractor()
    return _extractor
