# mood/emotional_memory.py
"""
Emotional Memory System - Memories and motifs carry emotional weight

This system tracks the emotional valence of memories and motifs, allowing for
organic emotional responses based on recall, nostalgia, missing, and rediscovery.
"""

from __future__ import annotations
import time
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class EmotionalMemory:
    """A memory with emotional coloring"""
    content: str
    timestamp: float
    valence: float  # -1 (negative) to 1 (positive)
    arousal: float  # -1 (calm) to 1 (excited)
    significance: float  # 0 (mundane) to 1 (profound)
    emotion_label: str
    context: Dict[str, Any]
    recall_count: int = 0
    last_recalled: Optional[float] = None


@dataclass
class MotifEmotion:
    """Emotional association with a recurring motif"""
    motif: str
    cumulative_valence: float  # Average emotional valence when this appears
    appearance_emotions: List[float]  # History of emotions when seen
    significance_events: List[float]  # Times when this was emotionally significant
    comfort_level: float  # How comforting/familiar this has become (0-1)
    nostalgia_potential: float  # How much nostalgia this can evoke (0-1)


class EmotionalMemoryBank:
    """
    Tracks emotional associations with memories and motifs.
    Enables organic emotional responses based on memory and recognition.
    """
    
    def __init__(self):
        # Memory storage with emotional metadata
        self.memories: deque = deque(maxlen=500)  # Last 500 memories with full emotional context
        self.significant_memories: List[EmotionalMemory] = []  # Highly significant memories never forgotten
        
        # Motif emotional associations
        self.motif_emotions: Dict[str, MotifEmotion] = {}
        
        # Emotional patterns and associations
        self.emotional_associations: defaultdict = defaultdict(list)  # object -> list of emotional responses
        self.comfort_objects: set = set()  # Objects that bring comfort through familiarity
        self.trigger_objects: set = set()  # Objects that trigger strong emotions
        
        # Nostalgia and missing tracking
        self.last_seen_times: Dict[str, float] = {}  # When we last saw each motif
        self.missing_threshold = 1800  # 30 minutes before we start missing things
        self.nostalgia_memories: deque = deque(maxlen=50)  # Memories that evoke nostalgia
        
        # Emotional state tracking for context
        self.current_emotional_context = {
            "dominant_emotion": "neutral",
            "emotional_stability": 0.5,
            "memory_mood_influence": 0.0
        }
    
    def store_memory(self, content: str, mood_vector: Tuple[float, float, float], 
                    emotion_label: str, objects: List[str], significance: float = None) -> EmotionalMemory:
        """Store a memory with its emotional coloring"""
        
        valence, arousal, clarity = mood_vector
        
        # Calculate significance if not provided
        if significance is None:
            # Significance based on emotional intensity and clarity
            significance = abs(valence) * 0.4 + abs(arousal) * 0.3 + clarity * 0.3
            significance = np.clip(significance, 0.0, 1.0)
        
        # Create emotional memory
        memory = EmotionalMemory(
            content=content,
            timestamp=time.time(),
            valence=valence,
            arousal=arousal,
            significance=significance,
            emotion_label=emotion_label,
            context={"objects": objects, "mood_vector": mood_vector}
        )
        
        # Store in appropriate places
        self.memories.append(memory)
        
        # Keep highly significant memories forever
        if significance > 0.7:
            self.significant_memories.append(memory)
            # Limit significant memories to prevent unbounded growth
            if len(self.significant_memories) > 100:
                # Keep only the most significant
                self.significant_memories.sort(key=lambda m: m.significance, reverse=True)
                self.significant_memories = self.significant_memories[:75]
        
        # Update object emotional associations
        for obj in objects:
            self.emotional_associations[obj].append(valence)
            self.last_seen_times[obj] = time.time()
            
            # Update motif emotions
            if obj not in self.motif_emotions:
                self.motif_emotions[obj] = MotifEmotion(
                    motif=obj,
                    cumulative_valence=valence,
                    appearance_emotions=[valence],
                    significance_events=[],
                    comfort_level=0.0,
                    nostalgia_potential=0.0
                )
            else:
                motif_emotion = self.motif_emotions[obj]
                motif_emotion.appearance_emotions.append(valence)
                # Update cumulative valence (weighted average favoring recent)
                motif_emotion.cumulative_valence = (
                    motif_emotion.cumulative_valence * 0.7 + valence * 0.3
                )
                
                # Track if this was emotionally significant
                if abs(valence) > 0.5 or significance > 0.6:
                    motif_emotion.significance_events.append(time.time())
                
                # Update comfort level (increases with positive repetition)
                if valence > 0:
                    motif_emotion.comfort_level = min(1.0, motif_emotion.comfort_level + 0.02)
                elif valence < -0.3:
                    motif_emotion.comfort_level = max(0.0, motif_emotion.comfort_level - 0.05)
                
                # Update nostalgia potential (builds over time with positive associations)
                if len(motif_emotion.appearance_emotions) > 10 and motif_emotion.cumulative_valence > 0.2:
                    motif_emotion.nostalgia_potential = min(1.0, motif_emotion.nostalgia_potential + 0.01)
        
        return memory
    
    def recall_memory(self, memory: EmotionalMemory) -> Dict[str, float]:
        """Recall a memory and get its emotional impact"""
        memory.recall_count += 1
        memory.last_recalled = time.time()
        
        # Calculate emotional impact of recall
        time_since = time.time() - memory.timestamp
        
        # Nostalgia factor increases with time for positive memories
        nostalgia_factor = 0.0
        if memory.valence > 0.2 and time_since > 3600:  # Positive and over an hour old
            # Logarithmic nostalgia growth
            nostalgia_factor = min(0.5, np.log(time_since / 3600) * 0.2)
        
        # Memory fading (emotions become gentler over time)
        fade_factor = max(0.3, 1.0 - (time_since / 86400))  # Fades to 30% over a day
        
        # Recall impact
        emotional_impact = {
            "valence_shift": memory.valence * fade_factor + nostalgia_factor,
            "arousal_shift": memory.arousal * fade_factor * 0.5,  # Arousal fades faster
            "nostalgia": nostalgia_factor,
            "significance": memory.significance * fade_factor
        }
        
        # Add to nostalgia memories if applicable
        if nostalgia_factor > 0.2:
            self.nostalgia_memories.append(memory)
        
        return emotional_impact
    
    def get_motif_emotional_response(self, motif: str) -> Dict[str, float]:
        """Get emotional response to encountering a motif"""
        
        if motif not in self.motif_emotions:
            # First encounter - neutral to slightly curious
            return {
                "valence_shift": 0.0,
                "arousal_shift": 0.1,
                "response_type": "novel",
                "comfort": 0.0,
                "nostalgia": 0.0
            }
        
        motif_emotion = self.motif_emotions[motif]
        current_time = time.time()
        
        # Check if we've been missing this
        time_since_seen = current_time - self.last_seen_times.get(motif, current_time)
        missing_factor = 0.0
        
        if time_since_seen > self.missing_threshold and motif_emotion.cumulative_valence > 0.1:
            # We've been missing this positive thing
            missing_factor = min(0.3, (time_since_seen / 7200) * 0.3)  # Max at 2 hours
            response_type = "reunion"
        elif time_since_seen < 60:
            # Very recently seen
            response_type = "familiar"
        else:
            response_type = "recognition"
        
        # Calculate emotional response
        base_valence = motif_emotion.cumulative_valence
        
        # Comfort from familiarity (only for positive or neutral associations)
        comfort_boost = 0.0
        if base_valence >= -0.1:
            comfort_boost = motif_emotion.comfort_level * 0.2
        
        # Nostalgia possibility
        nostalgia_boost = 0.0
        if motif_emotion.nostalgia_potential > 0.3 and random.random() < motif_emotion.nostalgia_potential:
            nostalgia_boost = motif_emotion.nostalgia_potential * 0.3
            response_type = "nostalgic"
        
        return {
            "valence_shift": base_valence * 0.3 + comfort_boost + nostalgia_boost + missing_factor,
            "arousal_shift": -motif_emotion.comfort_level * 0.1 if response_type == "familiar" else 0.1,
            "response_type": response_type,
            "comfort": motif_emotion.comfort_level,
            "nostalgia": nostalgia_boost,
            "missing": missing_factor
        }
    
    def get_missing_objects_mood_effect(self) -> Dict[str, float]:
        """Calculate mood effect from objects we haven't seen in a while"""
        
        current_time = time.time()
        missing_effects = {
            "valence_shift": 0.0,
            "arousal_shift": 0.0,
            "missing_objects": []
        }
        
        for motif, last_seen in self.last_seen_times.items():
            time_since = current_time - last_seen
            
            # Only miss things with positive associations that we haven't seen recently
            if (time_since > self.missing_threshold and 
                motif in self.motif_emotions and 
                self.motif_emotions[motif].cumulative_valence > 0.2):
                
                # Melancholy from missing positive things
                missing_intensity = min(0.2, (time_since / 7200) * 0.1)
                missing_effects["valence_shift"] -= missing_intensity
                missing_effects["missing_objects"].append(motif)
        
        # Cap the total missing effect
        missing_effects["valence_shift"] = max(-0.3, missing_effects["valence_shift"])
        
        return missing_effects
    
    def find_similar_memories(self, current_objects: List[str], current_emotion: str) -> List[EmotionalMemory]:
        """Find memories similar to current situation"""
        
        similar_memories = []
        
        for memory in self.memories:
            memory_objects = memory.context.get("objects", [])
            
            # Check object overlap
            if memory_objects:
                overlap = len(set(current_objects) & set(memory_objects)) / len(set(current_objects) | set(memory_objects))
            else:
                overlap = 0
            
            # Check emotional similarity
            emotion_match = 1.0 if memory.emotion_label == current_emotion else 0.3
            
            # Combined similarity score
            similarity = overlap * 0.6 + emotion_match * 0.4
            
            if similarity > 0.5:
                similar_memories.append(memory)
        
        # Return top 3 most similar
        similar_memories.sort(key=lambda m: m.significance, reverse=True)
        return similar_memories[:3]
    
    def get_emotional_memory_context(self) -> str:
        """Generate natural language context about emotional memories"""
        
        context_parts = []
        
        # Recent emotional highs and lows
        recent_memories = list(self.memories)[-20:] if len(self.memories) >= 20 else list(self.memories)
        if recent_memories:
            high_valence = max(recent_memories, key=lambda m: m.valence)
            low_valence = min(recent_memories, key=lambda m: m.valence)
            
            if high_valence.valence > 0.5:
                time_ago = (time.time() - high_valence.timestamp) / 60
                context_parts.append(f"Felt joy {int(time_ago)}min ago: {high_valence.content[:30]}")
            
            if low_valence.valence < -0.3:
                time_ago = (time.time() - low_valence.timestamp) / 60
                context_parts.append(f"Felt troubled {int(time_ago)}min ago: {low_valence.content[:30]}")
        
        # Comfort objects
        comfort_objects = [
            motif for motif, emotion in self.motif_emotions.items() 
            if emotion.comfort_level > 0.5
        ]
        if comfort_objects:
            context_parts.append(f"Find comfort in: {', '.join(comfort_objects[:3])}")
        
        # Missing objects
        missing_effect = self.get_missing_objects_mood_effect()
        if missing_effect["missing_objects"]:
            context_parts.append(f"Missing: {', '.join(missing_effect['missing_objects'][:3])}")
        
        # Nostalgic memories
        if self.nostalgia_memories:
            recent_nostalgia = list(self.nostalgia_memories)[-1]
            context_parts.append(f"Nostalgic about: {recent_nostalgia.content[:30]}")
        
        return " | ".join(context_parts) if context_parts else "Building emotional memories"
    
    def calculate_memory_mood_influence(self, current_objects: List[str], current_emotion: str) -> Dict[str, float]:
        """Calculate how memories should influence current mood"""
        
        influence = {
            "valence_shift": 0.0,
            "arousal_shift": 0.0,
            "dominant_influence": "neutral"
        }
        
        # Get motif emotional responses
        for obj in current_objects:
            motif_response = self.get_motif_emotional_response(obj)
            influence["valence_shift"] += motif_response["valence_shift"] * 0.3
            influence["arousal_shift"] += motif_response["arousal_shift"] * 0.2
        
        # Add missing objects effect
        missing_effect = self.get_missing_objects_mood_effect()
        influence["valence_shift"] += missing_effect["valence_shift"] * 0.5
        
        # Find and process similar memories
        similar_memories = self.find_similar_memories(current_objects, current_emotion)
        for memory in similar_memories:
            recall_impact = self.recall_memory(memory)
            influence["valence_shift"] += recall_impact["valence_shift"] * 0.2
            influence["arousal_shift"] += recall_impact["arousal_shift"] * 0.1
        
        # Determine dominant influence
        if abs(influence["valence_shift"]) < 0.1:
            influence["dominant_influence"] = "neutral"
        elif influence["valence_shift"] > 0.2:
            influence["dominant_influence"] = "positive_memories"
        elif influence["valence_shift"] < -0.2:
            influence["dominant_influence"] = "melancholy"
        else:
            influence["dominant_influence"] = "mixed"
        
        # Clamp to reasonable ranges
        influence["valence_shift"] = np.clip(influence["valence_shift"], -0.5, 0.5)
        influence["arousal_shift"] = np.clip(influence["arousal_shift"], -0.3, 0.3)
        
        return influence