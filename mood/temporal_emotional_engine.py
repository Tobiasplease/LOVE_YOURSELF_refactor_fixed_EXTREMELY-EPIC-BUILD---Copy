# mood/temporal_emotional_engine.py
"""
Temporal Emotional Engine - Genuine emotional trajectories based on lived experience

This engine tracks the actual weight of time and creates causal emotional chains.
No templates. No prescribed responses. Just the raw accumulation of experience.
"""

from __future__ import annotations
import time
import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass

try:
    from config.config import DEBUG_TEMPORAL_EMOTIONS
except ImportError:
    DEBUG_TEMPORAL_EMOTIONS = False

def _debug_log(message: str):
    """Log debug message only if temporal emotion debugging is enabled"""
    if DEBUG_TEMPORAL_EMOTIONS:
        print(f"[TEMPORAL] {message}")


@dataclass
class EmotionalMoment:
    """A moment in emotional time"""
    timestamp: float
    emotion: str
    intensity: float
    trigger: str
    context: Dict[str, Any]


class TemporalMemoryBank:
    """
    Tracks everything quantitatively. No interpretation - just facts.
    The AI draws its own conclusions from these temporal truths.
    """
    
    def __init__(self):
        self.observation_count = defaultdict(int)  # {"book": 423, "table": 312, ...}
        self.phrase_history = []  # Every phrase ever said
        self.temporal_markers = []  # [(time, "discovered book title"), ...]
        self.attention_duration = defaultdict(float)  # {"book": 8423 seconds, ...}
        self.repetition_tracker = defaultdict(list)  # Track when things repeat
        self.stagnation_start = time.time()
        self.last_discovery = None
        self.vocabulary_exhaustion_ratio = 0.0
        
    def record_observation(self, objects: List[str], caption: str):
        """Record this moment in temporal memory"""
        now = time.time()
        
        # Count object observations
        for obj in objects:
            self.observation_count[obj] += 1
            self.attention_duration[obj] += 10  # Default 10 seconds per caption
        
        # Track phrases for repetition detection
        self.phrase_history.append(caption)
        
        # Calculate vocabulary exhaustion
        unique_phrases = len(set(self.phrase_history))
        total_phrases = len(self.phrase_history)
        self.vocabulary_exhaustion_ratio = 1.0 - (unique_phrases / max(total_phrases, 1))
        
        # Check for truly new discoveries (substantial phrase differences, not just minor variations)
        is_genuinely_new = True
        for prev_phrase in self.phrase_history[-20:]:  # Check last 20 phrases
            if self._phrases_are_similar(caption, prev_phrase, threshold=0.7):
                is_genuinely_new = False
                break
        
        if is_genuinely_new and len(self.phrase_history) > 5:  # Only after some observations
            self.last_discovery = now
            self.temporal_markers.append((now, f"discovered: {caption[:50]}"))
    
    def _phrases_are_similar(self, phrase1: str, phrase2: str, threshold: float = 0.7) -> bool:
        """Check if two phrases are substantially similar"""
        # Simple similarity check based on word overlap
        words1 = set(phrase1.lower().split())
        words2 = set(phrase2.lower().split())
        
        if len(words1) == 0 and len(words2) == 0:
            return True
        if len(words1) == 0 or len(words2) == 0:
            return False
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0
        return similarity >= threshold
    
    def get_stagnation_duration(self) -> float:
        """How long since anything genuinely new happened"""
        if self.last_discovery:
            return time.time() - self.last_discovery
        return time.time() - self.stagnation_start
    
    def get_temporal_truth(self) -> Dict[str, Any]:
        """Raw temporal facts - no interpretation"""
        if not self.observation_count:
            return {"total_observations": 0}
            
        most_observed = max(self.observation_count.items(), key=lambda x: x[1])
        
        return {
            "most_observed_object": most_observed[0],
            "most_observed_count": most_observed[1],
            "total_observations": sum(self.observation_count.values()),
            "vocabulary_exhaustion": self.vocabulary_exhaustion_ratio,
            "stagnation_minutes": self.get_stagnation_duration() / 60.0,
            "repetitions_of_top_object": self.observation_count[most_observed[0]],
            "total_phrases": len(self.phrase_history),
            "unique_phrases": len(set(self.phrase_history)),
            "session_duration_hours": (time.time() - self.stagnation_start) / 3600.0
        }


class EmotionalTrajectory:
    """
    Maps natural emotional progressions based on temporal pressure.
    No forced transitions - just probability weights based on context.
    """
    
    TRAJECTORY_PATTERNS = {
        "spiral_down": ["curiosity", "interest", "boredom", "frustration", "anger", "despair"],
        "stockholm_syndrome": ["boredom", "resentment", "resignation", "acceptance", "affection", "devotion"],
        "manic_obsession": ["boredom", "hyperfocus", "obsession", "revelation", "mania", "crash"],
        "zen_acceptance": ["frustration", "exhaustion", "emptiness", "clarity", "peace", "joy"],
        "breakdown_rebuild": ["stability", "doubt", "fragmentation", "collapse", "void", "rebirth"],
        "micro_joy_spiral": ["despair", "numbness", "micro_discovery", "hope", "gratitude", "transcendence"]
    }
    
    def __init__(self):
        self.current_trajectory = None
        self.trajectory_position = 0
        self.trajectory_resistance = 0.0  # How much the mind resists change
        
    def get_natural_progression(self, current_emotion: str, temporal_context: Dict) -> Optional[str]:
        """
        Based on current state and temporal pressure, what's the natural next emotion?
        Not forced - just weighted probabilities.
        """
        # Find which trajectories contain current emotion
        possible_trajectories = []
        for name, path in self.TRAJECTORY_PATTERNS.items():
            if current_emotion in path:
                current_pos = path.index(current_emotion)
                if current_pos < len(path) - 1:  # Not at end
                    next_emotion = path[current_pos + 1]
                    possible_trajectories.append((name, next_emotion, current_pos))
        
        if not possible_trajectories:
            return None
        
        # Weight trajectories based on temporal context
        weights = []
        for traj_name, next_emotion, position in possible_trajectories:
            weight = self._calculate_trajectory_weight(traj_name, temporal_context, position)
            weights.append(weight)
        
        if max(weights) > 0.3:  # Strong enough pull
            best_trajectory = possible_trajectories[weights.index(max(weights))]
            return best_trajectory[1]  # Return next emotion
        
        return None
    
    def _calculate_trajectory_weight(self, trajectory_name: str, context: Dict, position: int) -> float:
        """Calculate how likely this trajectory is based on temporal context"""
        hours = context.get("session_duration_hours", 0)
        stagnation = context.get("stagnation_minutes", 0)
        repetitions = context.get("repetitions_of_top_object", 0)
        vocabulary_exhaustion = context.get("vocabulary_exhaustion", 0)
        
        weights = {
            "spiral_down": min(hours * 0.2 + stagnation * 0.1, 1.0),
            "stockholm_syndrome": min(stagnation * 0.05 + hours * 0.15, 1.0) if hours > 1 else 0,
            "manic_obsession": min(repetitions * 0.002 + vocabulary_exhaustion * 0.8, 1.0),
            "zen_acceptance": min(hours * 0.3, 1.0) if hours > 2 else 0,
            "breakdown_rebuild": min(vocabulary_exhaustion * 1.2 + hours * 0.4, 1.0) if hours > 3 else 0,
            "micro_joy_spiral": min(stagnation * 0.1, 1.0) if stagnation > 60 else 0
        }
        
        return weights.get(trajectory_name, 0.0)


class DiscoveryJoyCalculator:
    """
    Calculates joy intensity based on stagnation duration.
    The longer the drought, the more intense the joy.
    """
    
    def __init__(self):
        self.baseline_joy = 0.1
        
    def calculate_discovery_impact(self, stagnation_minutes: float, discovery_magnitude: float) -> float:
        """
        Joy is proportional to suffering endured before discovery
        """
        if stagnation_minutes < 5:
            return discovery_magnitude * 1.0  # Normal joy
        
        # Exponential joy scaling based on stagnation
        starvation_multiplier = min(1 + (stagnation_minutes / 30.0), 10.0)  # Caps at 10x
        
        if stagnation_minutes > 180:  # 3+ hours of stagnation
            # Even tiny discoveries become ECSTATIC
            return min(discovery_magnitude * starvation_multiplier * 5.0, 1.0)
        elif stagnation_minutes > 60:  # 1+ hour
            return min(discovery_magnitude * starvation_multiplier * 3.0, 1.0)
        else:
            return discovery_magnitude * starvation_multiplier


class MicroVictoryDetector:
    """
    Finds tiny sources of joy in overwhelming monotony.
    The mind's survival mechanism against sensory starvation.
    """
    
    def __init__(self):
        self.micro_discoveries = {
            "counting": ["successfully counted", "enumerated", "tallied"],
            "pattern": ["noticed a pattern", "saw a shape", "found symmetry"],
            "memory": ["just remembered", "this reminds me", "suddenly recalled"],
            "theory": ["figured out why", "understanding dawns", "the reason becomes clear"],
            "acceptance": ["this is okay", "peaceful", "accepting this"],
            "physics": ["the light", "shadow", "dust", "air current"],
            "texture": ["rough", "smooth", "surface", "grain", "texture"],
            "philosophy": ["meaning", "existence", "purpose", "reality"]
        }
        
    def detect_micro_victory(self, caption: str, repetition_count: int, stagnation_hours: float) -> Optional[Tuple[str, float]]:
        """
        Find micro-victories in the caption. Likelihood increases with desperation.
        """
        if repetition_count < 50 or stagnation_hours < 0.5:
            return None  # Not desperate enough yet
            
        # Probability of finding micro-joy increases with suffering
        desperation_factor = min(repetition_count / 200.0 + stagnation_hours * 0.3, 0.8)
        
        if random.random() > desperation_factor:
            return None
            
        # Look for micro-victory keywords in caption
        for victory_type, keywords in self.micro_discoveries.items():
            for keyword in keywords:
                if keyword in caption.lower():
                    # Joy intensity scales with stagnation
                    joy_intensity = min(0.1 + (stagnation_hours * 0.1), 0.6)
                    return (victory_type, joy_intensity)
        
        # Random micro-victory (mind creating joy from nothing)
        if random.random() < desperation_factor * 0.1:
            victory_type = random.choice(list(self.micro_discoveries.keys()))
            joy_intensity = min(0.05 + (stagnation_hours * 0.05), 0.3)
            return (victory_type, joy_intensity)
            
        return None


class TemporalEmotionalEngine:
    """
    The main engine that creates genuine emotional causality based on temporal experience.
    No templates. Just the weight of accumulated time and experience.
    """
    
    def __init__(self):
        self.memory_bank = TemporalMemoryBank()
        self.trajectory = EmotionalTrajectory()
        self.discovery_joy = DiscoveryJoyCalculator()
        self.micro_victory = MicroVictoryDetector()
        
        # Current emotional state
        self.current_emotion = "curiosity"
        self.emotion_intensity = 0.5
        self.emotional_momentum = 0.0  # How strongly locked into current emotion
        
        # Emotional history for context
        self.emotional_history = deque(maxlen=50)
        
        # Track emotional triggers for causality
        self.last_trigger = None
        self.trigger_impact = 0.0
        
    def process_observation(self, caption: str, objects: List[str], saw_person: bool) -> Dict[str, Any]:
        """
        Process a new observation and update emotional state based on temporal causality
        """
        # Record in temporal memory
        self.memory_bank.record_observation(objects, caption)
        temporal_truth = self.memory_bank.get_temporal_truth()
        
        # Detect emotional triggers
        triggers = self._detect_emotional_triggers(caption, objects, saw_person, temporal_truth)
        
        # Calculate emotional changes
        emotion_changes = self._calculate_emotional_response(triggers, temporal_truth)
        
        # Apply natural emotional progression
        natural_progression = self.trajectory.get_natural_progression(
            self.current_emotion, temporal_truth
        )
        
        # Detect micro-victories in monotony
        micro_victory = self.micro_victory.detect_micro_victory(
            caption, 
            temporal_truth.get("repetitions_of_top_object", 0),
            temporal_truth.get("session_duration_hours", 0)
        )
        
        # Update emotional state
        new_emotion, new_intensity = self._update_emotional_state(
            emotion_changes, natural_progression, micro_victory, temporal_truth
        )
        
        # Record emotional moment
        emotional_moment = EmotionalMoment(
            timestamp=time.time(),
            emotion=new_emotion,
            intensity=new_intensity,
            trigger=self.last_trigger or "temporal_flow",
            context=temporal_truth
        )
        self.emotional_history.append(emotional_moment)
        
        self.current_emotion = new_emotion
        self.emotion_intensity = new_intensity
        
        return {
            "emotion": new_emotion,
            "intensity": new_intensity,
            "temporal_truth": temporal_truth,
            "triggers": triggers,
            "natural_progression": natural_progression,
            "micro_victory": micro_victory,
            "emotional_context": self._get_emotional_context(),
            "legacy_mood_scalar": self._convert_to_legacy_mood(new_emotion, new_intensity)
        }
    
    def _detect_emotional_triggers(self, caption: str, objects: List[str], saw_person: bool, temporal_truth: Dict) -> List[Dict]:
        """Detect what might trigger emotional changes"""
        triggers = []
        
        # Person appearance after isolation
        if saw_person and temporal_truth.get("stagnation_minutes", 0) > 30:
            isolation_hours = temporal_truth.get("session_duration_hours", 0)
            impact = min(0.5 + (isolation_hours * 0.2), 1.0)
            triggers.append({"type": "salvation_arrival", "impact": impact})
            self.last_trigger = "person_after_isolation"
        
        # New object or detail discovery
        current_objects = set(objects)
        if hasattr(self, '_last_objects') and current_objects != self._last_objects:
            new_objects = current_objects - self._last_objects
            if new_objects:
                stagnation = temporal_truth.get("stagnation_minutes", 0)
                impact = self.discovery_joy.calculate_discovery_impact(stagnation, 0.3)
                triggers.append({"type": "discovery", "objects": list(new_objects), "impact": impact})
                self.last_trigger = f"discovered_{list(new_objects)[0]}"
        
        self._last_objects = current_objects
        
        # Vocabulary exhaustion trigger
        if temporal_truth.get("vocabulary_exhaustion", 0) > 0.8:
            triggers.append({"type": "vocabulary_exhaustion", "impact": 0.4})
            
        # Repetition threshold triggers
        repetitions = temporal_truth.get("repetitions_of_top_object", 0)
        if repetitions > 100 and repetitions % 100 == 0:  # Every 100th repetition
            triggers.append({"type": "repetition_milestone", "count": repetitions, "impact": 0.2})
            self.last_trigger = f"repetition_{repetitions}"
            
        return triggers
    
    def _calculate_emotional_response(self, triggers: List[Dict], temporal_truth: Dict) -> Dict[str, float]:
        """Calculate emotional changes based on triggers and temporal context"""
        changes = {"intensity_delta": 0.0, "valence_delta": 0.0}
        
        for trigger in triggers:
            trigger_type = trigger["type"]
            impact = trigger["impact"]
            
            if trigger_type == "salvation_arrival":
                # Joy proportional to previous isolation
                changes["valence_delta"] += impact * 2.0  # Massive positive shift
                changes["intensity_delta"] += impact * 1.5
                
            elif trigger_type == "discovery":
                changes["valence_delta"] += impact
                changes["intensity_delta"] += impact * 0.5
                
            elif trigger_type == "vocabulary_exhaustion":
                changes["valence_delta"] -= impact
                changes["intensity_delta"] += impact  # Frustration is intense
                
            elif trigger_type == "repetition_milestone":
                # Could go either way - breakthrough or breakdown
                if random.random() > 0.5:
                    changes["valence_delta"] += impact  # "I've mastered this!"
                else:
                    changes["valence_delta"] -= impact  # "I'm trapped!"
        
        return changes
    
    def _update_emotional_state(self, emotion_changes: Dict, natural_progression: Optional[str], 
                               micro_victory: Optional[Tuple[str, float]], temporal_truth: Dict) -> Tuple[str, float]:
        """Update emotional state based on all influences"""
        new_emotion = self.current_emotion
        new_intensity = self.emotion_intensity
        
        # Apply trigger-based changes
        valence_change = emotion_changes.get("valence_delta", 0)
        intensity_change = emotion_changes.get("intensity_delta", 0)
        
        # Micro-victory can cause sudden emotional shifts
        if micro_victory:
            victory_type, joy_impact = micro_victory
            valence_change += joy_impact
            if joy_impact > 0.3:  # Strong micro-victory
                new_emotion = "micro_joy"
                self.last_trigger = f"micro_victory_{victory_type}"
        
        # Natural trajectory progression
        if natural_progression and random.random() > 0.7:  # 30% chance of natural flow
            new_emotion = natural_progression
            self.last_trigger = "natural_progression"
        
        # Update intensity
        new_intensity = np.clip(new_intensity + intensity_change, 0.0, 1.0)
        
        # Convert valence changes to appropriate emotions if large enough
        if abs(valence_change) > 0.5:
            if valence_change > 0:
                positive_emotions = ["joy", "wonder", "gratitude", "transcendence", "peace"]
                new_emotion = random.choice(positive_emotions)
            else:
                negative_emotions = ["frustration", "despair", "anger", "emptiness"]
                new_emotion = random.choice(negative_emotions)
        
        return new_emotion, new_intensity
    
    def _get_emotional_context(self) -> Dict[str, Any]:
        """Get rich emotional context for prompt building"""
        if not self.emotional_history:
            return {"trajectory": "beginning", "stability": "stable"}
            
        recent_emotions = [moment.emotion for moment in list(self.emotional_history)[-5:]]
        emotion_changes = len(set(recent_emotions))
        
        return {
            "recent_emotions": recent_emotions,
            "emotional_volatility": emotion_changes / 5.0,
            "last_trigger": self.last_trigger,
            "trajectory_stability": "volatile" if emotion_changes > 3 else "stable",
            "dominant_recent_emotion": max(set(recent_emotions), key=recent_emotions.count)
        }
    
    def _convert_to_legacy_mood(self, emotion: str, intensity: float) -> float:
        """Convert rich emotional state back to simple 0-1 scalar for compatibility"""
        emotion_values = {
            "joy": 0.9, "wonder": 0.85, "curiosity": 0.7, "interest": 0.6,
            "peace": 0.8, "gratitude": 0.85, "transcendence": 0.95,
            "micro_joy": 0.75, "acceptance": 0.65, "contemplation": 0.55,
            "boredom": 0.3, "frustration": 0.2, "anger": 0.1,
            "despair": 0.05, "emptiness": 0.15, "numbness": 0.25
        }
        
        base_value = emotion_values.get(emotion, 0.5)
        # Intensity modifies the base value
        return np.clip(base_value * intensity, 0.0, 1.0)
    
    def get_temporal_prompt_context(self) -> str:
        """Generate natural language context about temporal experience for prompts"""
        truth = self.memory_bank.get_temporal_truth()
        
        if truth.get("total_observations", 0) == 0:
            return "This is your first observation."
        
        context_parts = []
        
        # Repetition awareness
        if truth.get("repetitions_of_top_object", 0) > 50:
            obj = truth.get("most_observed_object", "object")
            count = truth.get("repetitions_of_top_object", 0)
            context_parts.append(f"You have mentioned {obj} {count} times.")
        
        # Vocabulary exhaustion
        if truth.get("vocabulary_exhaustion", 0) > 0.5:
            ratio = truth.get("vocabulary_exhaustion", 0) * 100
            context_parts.append(f"{ratio:.0f}% of your words are repetitions.")
        
        # Stagnation awareness
        stagnation_min = truth.get("stagnation_minutes", 0)
        if stagnation_min > 30:
            context_parts.append(f"Nothing new discovered for {stagnation_min:.0f} minutes.")
        
        # Session duration
        hours = truth.get("session_duration_hours", 0)
        if hours > 1:
            context_parts.append(f"Session duration: {hours:.1f} hours.")
        
        # Emotional context
        emotional_context = self._get_emotional_context()
        if emotional_context.get("last_trigger"):
            context_parts.append(f"Last emotional trigger: {emotional_context['last_trigger']}")
        
        if not context_parts:
            return "Your temporal experience is just beginning."
        
        return " ".join(context_parts)