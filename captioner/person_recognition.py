"""
person_recognition.py
--------------------
Visual-based person recognition and tracking system.

Instead of generic "primary person" IDs, this system:
- Observes visual characteristics (hair, age, clothing, build)
- Tracks individuals based on these features
- Builds familiarity over time with specific people
- Only activates when people are actually present
"""

import time
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class VisualProfile:
    """Visual characteristics of a person."""
    hair_color: Optional[str] = None
    hair_length: Optional[str] = None  # "short", "medium", "long"
    apparent_age: Optional[str] = None  # "young", "middle-aged", "older"
    build: Optional[str] = None  # "slim", "average", "tall", "broad"
    common_clothing: List[str] = field(default_factory=list)
    distinctive_features: List[str] = field(default_factory=list)
    
    def similarity_score(self, other: 'VisualProfile') -> float:
        """Calculate how similar two visual profiles are (0-1)."""
        score = 0.0
        comparisons = 0
        
        # Hair characteristics (high weight - usually consistent)
        if self.hair_color and other.hair_color:
            comparisons += 2
            if self.hair_color == other.hair_color:
                score += 2
        
        if self.hair_length and other.hair_length:
            comparisons += 2
            if self.hair_length == other.hair_length:
                score += 2
        
        # Age (medium weight)
        if self.apparent_age and other.apparent_age:
            comparisons += 1
            if self.apparent_age == other.apparent_age:
                score += 1
        
        # Build (medium weight)
        if self.build and other.build:
            comparisons += 1
            if self.build == other.build:
                score += 1
        
        # Clothing overlap (lower weight - changes frequently)
        if self.common_clothing and other.common_clothing:
            clothing_overlap = len(set(self.common_clothing) & set(other.common_clothing))
            clothing_total = len(set(self.common_clothing) | set(other.common_clothing))
            if clothing_total > 0:
                comparisons += 0.5
                score += 0.5 * (clothing_overlap / clothing_total)
        
        return score / comparisons if comparisons > 0 else 0.0


@dataclass 
class PersonRecord:
    """Record of an individual person over time."""
    person_id: str
    visual_profile: VisualProfile
    first_seen: float
    last_seen: float
    total_encounters: int = 0
    session_encounters: int = 0  # Encounters in current session
    familiarity_level: float = 0.0  # 0-1, builds over time
    
    def update_encounter(self, current_time: float, new_observations: VisualProfile):
        """Update this person record with a new sighting."""
        self.last_seen = current_time
        self.total_encounters += 1
        self.session_encounters += 1
        
        # Merge visual observations (keep most common/recent)
        if new_observations.hair_color:
            self.visual_profile.hair_color = new_observations.hair_color
        if new_observations.hair_length:
            self.visual_profile.hair_length = new_observations.hair_length
        if new_observations.apparent_age:
            self.visual_profile.apparent_age = new_observations.apparent_age
        if new_observations.build:
            self.visual_profile.build = new_observations.build
            
        # Add new clothing/features (keep recent ones)
        for item in new_observations.common_clothing:
            if item not in self.visual_profile.common_clothing:
                self.visual_profile.common_clothing.append(item)
        for feature in new_observations.distinctive_features:
            if feature not in self.visual_profile.distinctive_features:
                self.visual_profile.distinctive_features.append(feature)
        
        # Trim clothing/features lists to keep them manageable
        self.visual_profile.common_clothing = self.visual_profile.common_clothing[-3:]
        self.visual_profile.distinctive_features = self.visual_profile.distinctive_features[-2:]
        
        # Update familiarity (builds slowly over many encounters)
        self.familiarity_level = min(1.0, self.total_encounters / 20.0)
    
    def get_description(self) -> str:
        """Get a natural description of this person."""
        parts = []
        
        # Build basic description
        if self.visual_profile.apparent_age:
            parts.append(f"the {self.visual_profile.apparent_age} person")
        else:
            parts.append("the person")
            
        if self.visual_profile.hair_color or self.visual_profile.hair_length:
            hair_desc = []
            if self.visual_profile.hair_length:
                hair_desc.append(self.visual_profile.hair_length)
            if self.visual_profile.hair_color:
                hair_desc.append(self.visual_profile.hair_color)
            if hair_desc:
                parts.append(f"with {' '.join(hair_desc)} hair")
        
        description = " ".join(parts)
        
        # Add familiarity context
        if self.familiarity_level > 0.7:
            return f"{description} (someone I know well)"
        elif self.familiarity_level > 0.3:
            return f"{description} (someone familiar)" 
        elif self.total_encounters > 1:
            return f"{description} (I've seen them before)"
        else:
            return f"{description} (someone new)"


class PersonRecognitionSystem:
    """Lightweight visual-based person recognition system."""
    
    def __init__(self):
        self.known_people: Dict[str, PersonRecord] = {}
        self.next_person_id = 1
        self.session_start = time.time()
        
        # Recognition thresholds
        self.SIMILARITY_THRESHOLD = 0.4  # How similar profiles need to be to match (lowered for better matching)
        self.MIN_TIME_BETWEEN_UPDATES = 5   # Don't update same person too frequently (reduced for testing)
    
    def reset_session(self):
        """Reset session counters (call when starting new session)."""
        self.session_start = time.time()
        for person in self.known_people.values():
            person.session_encounters = 0
    
    def extract_visual_features(self, caption: str) -> Optional[VisualProfile]:
        """Extract visual characteristics from a caption mentioning a person."""
        if not any(keyword in caption.lower() for keyword in ["person", "man", "woman", "individual", "someone"]):
            return None
        
        profile = VisualProfile()
        caption_lower = caption.lower()
        
        # Hair color detection
        hair_colors = ["black", "brown", "blonde", "blond", "red", "gray", "grey", "white", "dark", "light"]
        for color in hair_colors:
            if f"{color} hair" in caption_lower or f"with {color}" in caption_lower:
                profile.hair_color = color
                break
        
        # Hair length detection  
        if any(term in caption_lower for term in ["long hair", "lengthy hair", "with long"]):
            profile.hair_length = "long"
        elif any(term in caption_lower for term in ["short hair", "brief hair", "with short"]):
            profile.hair_length = "short"
        elif any(term in caption_lower for term in ["medium hair", "with medium"]):
            profile.hair_length = "medium"
        
        # Age detection
        if any(term in caption_lower for term in ["young", "youthful", "teen"]):
            profile.apparent_age = "young"
        elif any(term in caption_lower for term in ["middle-aged", "adult", "mature"]):
            profile.apparent_age = "middle-aged"  
        elif any(term in caption_lower for term in ["older", "elderly", "senior"]):
            profile.apparent_age = "older"
        
        # Build detection
        if any(term in caption_lower for term in ["tall", "height"]):
            profile.build = "tall"
        elif any(term in caption_lower for term in ["slim", "thin", "slender"]):
            profile.build = "slim"
        elif any(term in caption_lower for term in ["broad", "wide", "stocky"]):
            profile.build = "broad"
        
        # Clothing detection (simple items)
        clothing_items = ["shirt", "jacket", "sweater", "dress", "glasses", "hat", "cap"]
        for item in clothing_items:
            if item in caption_lower:
                profile.common_clothing.append(item)
        
        # Return profile only if we detected something useful
        if any([profile.hair_color, profile.hair_length, profile.apparent_age, 
                profile.build, profile.common_clothing]):
            return profile
        return None
    
    def recognize_or_create_person(self, caption: str) -> Optional[str]:
        """Recognize a person from visual description, or create new record."""
        visual_profile = self.extract_visual_features(caption)
        if not visual_profile:
            return None
        
        current_time = time.time()
        
        # Try to match against known people
        best_match_id = None
        best_similarity = 0.0
        
        for person_id, person_record in self.known_people.items():
            # Don't update same person too frequently (skip for testing if time is very recent)
            time_since_last_seen = current_time - person_record.last_seen
            if time_since_last_seen < self.MIN_TIME_BETWEEN_UPDATES and time_since_last_seen > 1:
                continue
                
            similarity = person_record.visual_profile.similarity_score(visual_profile)
            if similarity > best_similarity and similarity >= self.SIMILARITY_THRESHOLD:
                best_similarity = similarity
                best_match_id = person_id
        
        if best_match_id:
            # Update existing person
            self.known_people[best_match_id].update_encounter(current_time, visual_profile)
            return best_match_id
        else:
            # Create new person
            new_id = f"person_{self.next_person_id}"
            self.next_person_id += 1
            
            self.known_people[new_id] = PersonRecord(
                person_id=new_id,
                visual_profile=visual_profile,
                first_seen=current_time,
                last_seen=current_time,
                total_encounters=1,
                session_encounters=1
            )
            return new_id
    
    def get_person_context(self, person_id: str) -> str:
        """Get contextual description of a known person."""
        if person_id not in self.known_people:
            return "someone I'm just noticing"
        
        person = self.known_people[person_id]
        return person.get_description()
    
    def get_recognition_summary(self) -> str:
        """Get a summary of people known in this session (for debugging)."""
        if not self.known_people:
            return "No people recognized yet"
        
        summaries = []
        for person_id, person in self.known_people.items():
            if person.session_encounters > 0:
                summaries.append(f"{person_id}: {person.get_description()}")
        
        return f"Known people this session: {'; '.join(summaries)}" if summaries else "No people seen this session"


# Global instance
person_recognition = PersonRecognitionSystem()

# Reset function for memory clearing
def reset_person_recognition():
    """Reset the global person recognition system completely."""
    global person_recognition
    person_recognition = PersonRecognitionSystem()