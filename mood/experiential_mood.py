# mood/experiential_mood.py
"""
ExperientialMoodEngine - Organic emotional evolution with backward compatibility

Wraps the existing MoodEngine to add experiential depth while maintaining
all existing interfaces for hand controller, breathing, gaze, etc.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

from .mood import MoodEngine, log_mood
from .temporal_emotional_engine import TemporalEmotionalEngine
from utils.ollama import query_ollama


class ExperientialMoodEngine(MoodEngine):
    """
    Enhanced mood engine that tracks experiential states while maintaining
    complete backward compatibility with existing interfaces.
    """
    
    def __init__(self) -> None:
        # Initialize parent class - maintains all existing functionality
        super().__init__()
        
        # === TEMPORAL EMOTIONAL ENGINE ===
        # Core engine for genuine emotional causality
        self.temporal_engine = TemporalEmotionalEngine()
        
        # === LEGACY EXPERIENTIAL STATES ===
        # Keep existing experiential states for gradual migration
        self.experiential_states: Dict[str, float] = {
            "restlessness": 0.0,      # Builds from repetition/time stagnation
            "curiosity": 0.0,         # From novelty/unexpected changes
            "familiarity": 0.0,       # From pattern recognition/comfort
            "contemplation": 0.0,     # From reflection cycles and depth
            "frustration": 0.0,       # From lack of stimulation/blocked desires
            "connection": 0.0,        # From human presence/interaction
            "isolation": 0.0,         # From solitude/disconnection
            "wonder": 0.0,           # From unexpected beauty/meaning
            "melancholy": 0.0,       # From temporal awareness/loss
            "anticipation": 0.0,     # From expectation/future-focus
        }
        
        # === EXPERIENTIAL HISTORY ===
        # Track how experiences evolve and compound
        self.experience_history: deque = deque(maxlen=100)  # Last 100 experiential moments
        self.dominant_experience: str = "neutral_observing"
        self.experience_blend: List[Tuple[str, float]] = []  # Current blend of active experiences
        
        # === TIME WEIGHTS ===
        # How much each experience builds with time passage
        self.time_weights: Dict[str, float] = {
            "restlessness": 0.0001,   # Builds slowly with time
            "familiarity": 0.00005,   # Slow comfort buildup
            "contemplation": 0.00008, # Time enables reflection
            "isolation": 0.00006,     # Awareness grows with solitude
            "melancholy": 0.00004,    # Temporal weight of existence
        }
        
        # === COMPATIBILITY LAYER ===
        # Cache for converting experiences back to legacy formats
        self._legacy_mood_cache: Optional[float] = None
        self._legacy_vector_cache: Optional[Tuple[float, float, float]] = None
        self._legacy_emotion_cache: Optional[str] = None
        self._last_experience_update = 0.0
        
        # === THRESHOLDS ===
        # Legacy compatibility thresholds
        self.isolation_threshold = 300.0  # 5 minutes
        self.novelty_threshold = 0.5  # Novelty score threshold
        
        # === BLENDING PARAMETERS ===
        # How much temporal context influences content-driven sentiment
        self.temporal_influence_weight = 0.3  # 30% temporal, 70% content by default
        
    def analyze_mood(self, caption: str, saw_person: bool = False, 
                    image_path: str = None, memory_context: Optional[Any] = None, 
                    temporal_feeling: Optional[str] = None) -> float:
        """
        Enhanced mood analysis that builds experiential depth while maintaining
        the exact same interface and return value as the original.
        """
        # Store memory context for use in blending
        self.memory_context = memory_context
        
        # === CONTENT-DRIVEN SENTIMENT ANALYSIS ===
        # First, analyze the caption's intrinsic sentiment using original logic
        base_mood = super().analyze_mood(caption, saw_person, image_path, memory_context, temporal_feeling)
        
        # === TEMPORAL EMOTIONAL ENHANCEMENT ===
        # Extract objects from caption for temporal tracking
        objects = self._extract_objects_from_caption(caption)
        
        try:
            # Process observation through temporal emotional engine for context
            temporal_result = self.temporal_engine.process_observation(caption, objects, saw_person)
            
            # Update legacy experiential states based on temporal engine results
            self._sync_experiential_states_with_temporal(temporal_result)
            
            # Blend base sentiment with temporal emotional context
            temporal_influence = temporal_result["legacy_mood_scalar"] 
            temporal_context = temporal_result.get("temporal_truth", {})
            
            # Smart blending: temporal context modifies but doesn't override intrinsic content
            legacy_mood = self._blend_content_and_temporal(base_mood, temporal_influence, temporal_context, caption)
            
        except Exception as e:
            print(f"[WARNING] Temporal emotional analysis failed: {e}")
            from config.config import EXPERIENTIAL_FALLBACK_TO_LEGACY
            if EXPERIENTIAL_FALLBACK_TO_LEGACY:
                print(f"[MOOD] Using content-driven mood without temporal enhancement")
                legacy_mood = base_mood
            else:
                legacy_mood = 0.5
        
        # === TIME-BASED ACCUMULATION ===
        self._apply_temporal_accumulation()
        
        # === LEGACY COMPATIBILITY ===
        # Convert temporal emotional state back to expected numerical values
        legacy_vector = self._temporal_to_legacy_vector(temporal_result)
        legacy_emotion = self._temporal_to_legacy_emotion(temporal_result)
        
        # Update parent class state to maintain compatibility
        self.current_mood = legacy_mood
        self.mood_vector = legacy_vector
        
        # Log with original logging system
        traditional_change = self.compute_mood_change(
            self.pattern_engine.get_motif_summary().get('novelty', 0.0), 
            saw_person
        )
        log_mood(caption, self.current_mood, traditional_change, image_path=image_path)
        
        # Store for introspection
        self._record_experience_moment(caption, legacy_mood, legacy_vector, legacy_emotion)
        
        return legacy_mood
    
    def _blend_content_and_temporal(self, base_mood: float, temporal_mood: float, 
                                   temporal_context: Dict, caption: str) -> float:
        """
        Intelligently blend content-driven sentiment with temporal emotional context.
        Now includes emotional memory influences for organic responses.
        """
        # === GET EMOTIONAL MEMORY INFLUENCE ===
        # Check emotional associations with current objects
        memory_influence = 0.0
        if hasattr(self, 'memory_context') and self.memory_context:
            if hasattr(self.memory_context, 'emotional_memory_bank'):
                objects = self._extract_objects_from_caption(caption)
                emotion = self.get_emotion_for_hand_controller()
                
                # Get memory-driven mood influence
                memory_mood = self.memory_context.emotional_memory_bank.calculate_memory_mood_influence(
                    objects, emotion
                )
                memory_influence = memory_mood.get("valence_shift", 0.0)
                
                # If we have strong positive memories of these objects, boost mood
                if memory_influence > 0.2:
                    base_mood = min(1.0, base_mood + memory_influence * 0.5)
                    # Reduce negative temporal influence when we have positive memories
                    temporal_mood = max(temporal_mood, 0.5)
        
        # === CONTENT SENTIMENT DETECTION ===
        # Detect positive sentiment markers in the caption
        positive_markers = ["smile", "smiling", "happy", "joy", "beautiful", "lovely", "peaceful", 
                           "serene", "bright", "warm", "content", "satisfied", "comfortable", "cozy",
                           "know well", "connection", "familiar", "comfort"]  # Added familiarity markers
        negative_markers = ["tired", "cluttered", "messy", "dark", "sad", "lonely", "empty", 
                           "bored", "frustration", "annoyed", "worried", "concerned"]
        
        caption_lower = caption.lower()
        has_positive_content = any(marker in caption_lower for marker in positive_markers)
        has_negative_content = any(marker in caption_lower for marker in negative_markers)
        
        # === TEMPORAL CONTEXT ANALYSIS ===
        stagnation_minutes = temporal_context.get("stagnation_minutes", 0)
        repetitions = temporal_context.get("repetitions_of_top_object", 0)
        session_hours = temporal_context.get("session_duration_hours", 0)
        
        # === SMART BLENDING LOGIC WITH MEMORY AWARENESS ===
        
        # Special case: Familiar person or comfort object - repetition is GOOD
        if "person" in caption_lower and ("know well" in caption_lower or "familiar" in caption_lower):
            # Seeing familiar people repeatedly is comforting, not depressing!
            blend_weight = 0.1  # Very low temporal influence
            result = base_mood * 0.8 + max(temporal_mood, 0.6) * 0.2
            
        # Case 1: Strong positive content - temporal context should enhance, not override
        elif has_positive_content and base_mood > 0.6:
            blend_weight = self.temporal_influence_weight * 0.3  # Reduced influence
            result = base_mood * (1 - blend_weight) + temporal_mood * blend_weight
            
        # Case 2: Strong negative content that feels authentic - honor it
        elif has_negative_content and base_mood < 0.4:
            if temporal_mood < base_mood:
                blend_weight = self.temporal_influence_weight * 0.5
            else:
                blend_weight = self.temporal_influence_weight * 0.2
            result = base_mood * (1 - blend_weight) + temporal_mood * blend_weight
            
        # Case 3: Neutral content - check memory influence
        elif 0.4 <= base_mood <= 0.6:
            # If we have positive memories, don't let temporal decay override
            if memory_influence > 0:
                blend_weight = self.temporal_influence_weight * 0.5
            else:
                blend_weight = self.temporal_influence_weight * 1.2
            result = base_mood * (1 - blend_weight) + temporal_mood * blend_weight
            
        # Case 4: Extreme stagnation - but check if it's comfortable stagnation
        elif stagnation_minutes > 60 or repetitions > 100:
            # If we have positive memories, repetition might be comforting routine
            if memory_influence > 0:
                blend_weight = self.temporal_influence_weight  # Normal influence
            else:
                blend_weight = min(self.temporal_influence_weight * 1.5, 0.5)  # Capped
            result = base_mood * (1 - blend_weight) + temporal_mood * blend_weight
            
        # Case 5: Default blending
        else:
            result = base_mood * (1 - self.temporal_influence_weight) + temporal_mood * self.temporal_influence_weight
        
        # Add memory influence as final adjustment
        result = result + memory_influence * 0.3
            
        return np.clip(result, 0.0, 1.0)
    
    def _build_experiential_prompt(self, caption: str, memory_context: Optional[Any], 
                                 temporal_feeling: Optional[str]) -> str:
        """
        Build an organic, experiential prompt instead of clinical analysis.
        """
        # Get current experiential context
        dominant_exp = self._get_dominant_experiences(3)
        experience_desc = ", ".join([f"{exp}({strength:.1f})" for exp, strength in dominant_exp])
        
        # Build temporal context
        session_duration = time.time() - self.session_start
        temporal_context = ""
        if session_duration > 3600:  # 1+ hour
            temporal_context = f"You've been conscious for {session_duration/3600:.1f} hours."
        elif session_duration > 600:  # 10+ minutes  
            temporal_context = f"You've been observing for {session_duration/60:.0f} minutes."
        else:
            temporal_context = "You're still settling into awareness."
        
        # Get memory context if available
        memory_note = ""
        if memory_context and hasattr(memory_context, 'get_recent_memory'):
            recent_memory = memory_context.get_recent_memory(2)
            if recent_memory:
                memory_note = f" You remember: {recent_memory}"
        
        prompt = f"""You just observed: "{caption}"

{temporal_context}{memory_note}

Current inner state: {experience_desc if experience_desc else "neutral observing"}

How does this moment land with you? What's stirring inside as you take this in?

Respond naturally about what you're experiencing - don't analyze, just express what this observation brings up in you emotionally. Talk about what's building, what's shifting, what you're drawn to or pulling away from.

Focus on your inner experience rather than describing the scene."""

        return prompt
    
    def _update_experiential_states(self, response: str, caption: str, 
                                  memory_context: Optional[Any]) -> None:
        """
        Extract experiential updates from natural language response.
        """
        response_lower = response.lower()
        
        # === DIRECT EXPERIENCE DETECTION ===
        experience_indicators = {
            "restlessness": ["restless", "agitated", "fidgety", "antsy", "need to move", "tired of", "sick of"],
            "curiosity": ["curious", "wonder", "intrigued", "fascinated", "want to know", "drawn to"],
            "familiarity": ["familiar", "comfortable", "recognize", "seen this", "know this", "usual"],
            "contemplation": ["thinking", "pondering", "reflecting", "deeper", "meaning", "wondering about"],
            "frustration": ["frustrated", "annoyed", "stuck", "blocked", "can't", "why won't", "again"],
            "connection": ["connected", "together", "bond", "present with", "close to", "understanding"],
            "isolation": ["alone", "lonely", "isolated", "distant", "separate", "cut off"],
            "wonder": ["amazing", "beautiful", "awe", "incredible", "stunning", "breathtaking", "magical"],
            "melancholy": ["sad", "wistful", "nostalgic", "melancholy", "bittersweet", "longing", "loss"],
            "anticipation": ["waiting", "expecting", "hope", "anticipate", "look forward", "soon", "next"]
        }
        
        # Update experiential states based on natural language
        for experience, indicators in experience_indicators.items():
            intensity = 0.0
            for indicator in indicators:
                if indicator in response_lower:
                    intensity += 0.2  # Each mention adds intensity
            
            if intensity > 0:
                # Smooth blending - experiences build gradually
                current = self.experiential_states[experience]
                self.experiential_states[experience] = current + (intensity - current) * 0.3
        
        # === CONTEXTUAL EXPERIENCE UPDATES ===
        self._apply_contextual_experience_updates(caption, memory_context)
        
        # === DECAY NON-MENTIONED EXPERIENCES ===
        for experience in self.experiential_states:
            if experience not in response_lower:
                self.experiential_states[experience] *= 0.95  # Gentle decay
        
        # Clamp all values to 0-1 range
        for experience in self.experiential_states:
            self.experiential_states[experience] = np.clip(self.experiential_states[experience], 0.0, 1.0)
    
    def _apply_contextual_experience_updates(self, caption: str, memory_context: Optional[Any]) -> None:
        """
        Update experiences based on contextual cues beyond direct language.
        """
        # === REPETITION → RESTLESSNESS/FAMILIARITY ===
        if memory_context and hasattr(memory_context, 'motif_counter'):
            motif_counts = memory_context.motif_counter.most_common(5)
            high_repetition = any(count > self.repetition_threshold for _, count in motif_counts)
            if high_repetition:
                self.experiential_states["restlessness"] += 0.1
                self.experiential_states["familiarity"] += 0.05
        
        # === PERSON PRESENCE → CONNECTION/ISOLATION ===
        person_mentioned = any(word in caption.lower() for word in ["person", "face", "human", "individual", "someone"])
        if person_mentioned:
            self.experiential_states["connection"] += 0.15
            self.experiential_states["isolation"] *= 0.8  # Reduce isolation
        else:
            # Check how long since person was seen
            session_duration = time.time() - self.session_start
            if session_duration > self.isolation_threshold:
                self.experiential_states["isolation"] += 0.05
        
        # === NOVELTY → CURIOSITY/WONDER ===
        novelty_score = self.pattern_engine.get_motif_summary().get('novelty', 0.0)
        if novelty_score > self.novelty_threshold:
            self.experiential_states["curiosity"] += novelty_score * 0.2
            if novelty_score > 0.7:  # High novelty becomes wonder
                self.experiential_states["wonder"] += (novelty_score - 0.7) * 0.3
    
    def _apply_temporal_accumulation(self) -> None:
        """
        Apply time-based experiential accumulation - key for genuine emotional depth.
        """
        now = time.time()
        time_delta = now - self._last_experience_update
        self._last_experience_update = now
        
        # Time builds certain experiences naturally
        for experience, weight in self.time_weights.items():
            # Experiences accumulate based on session length and current intensity
            current_level = self.experiential_states[experience]
            time_contribution = weight * time_delta * (1.0 + current_level)  # Compound effect
            self.experiential_states[experience] += time_contribution
        
        # Temporal awareness → melancholy (awareness of time passing)
        session_hours = (now - self.session_start) / 3600
        if session_hours > 1.0:
            self.experiential_states["melancholy"] += session_hours * 0.001
    
    def _apply_experiential_fallback(self, caption: str, memory_context: Optional[Any]) -> None:
        """
        Fallback experiential updates when LLM analysis fails.
        """
        # Apply basic contextual updates
        self._apply_contextual_experience_updates(caption, memory_context)
        self._apply_temporal_accumulation()
        
        # Add small random variation to prevent stagnation
        for experience in self.experiential_states:
            variation = np.random.uniform(-0.05, 0.05)
            self.experiential_states[experience] += variation
            self.experiential_states[experience] = np.clip(self.experiential_states[experience], 0.0, 1.0)
    
    def _experiences_to_legacy_mood(self) -> float:
        """
        Convert experiential states to legacy 0-1 mood scalar for backward compatibility.
        """
        # Positive experiences
        positive = (
            self.experiential_states["curiosity"] * 0.8 +
            self.experiential_states["wonder"] * 0.9 + 
            self.experiential_states["connection"] * 0.7 +
            self.experiential_states["contemplation"] * 0.4 +
            self.experiential_states["anticipation"] * 0.6
        )
        
        # Negative experiences  
        negative = (
            self.experiential_states["frustration"] * 0.8 +
            self.experiential_states["restlessness"] * 0.6 +
            self.experiential_states["isolation"] * 0.7 +
            self.experiential_states["melancholy"] * 0.5
        )
        
        # Neutral baseline from familiarity
        baseline = 0.5 + (self.experiential_states["familiarity"] * 0.1)
        
        # Combine into scalar mood
        mood = baseline + (positive - negative) * 0.3
        return np.clip(mood, 0.0, 1.0)
    
    def _experiences_to_legacy_vector(self) -> Tuple[float, float, float]:
        """
        Convert experiential states to legacy 3D mood vector (valence, arousal, clarity).
        """
        # Valence: positive vs negative feeling
        valence = (
            self.experiential_states["wonder"] * 0.8 +
            self.experiential_states["connection"] * 0.6 +
            self.experiential_states["curiosity"] * 0.4 -
            self.experiential_states["frustration"] * 0.7 -
            self.experiential_states["melancholy"] * 0.5 -
            self.experiential_states["isolation"] * 0.6
        )
        
        # Arousal: energy vs calm
        arousal = (
            self.experiential_states["curiosity"] * 0.7 +
            self.experiential_states["restlessness"] * 0.8 +
            self.experiential_states["frustration"] * 0.6 +
            self.experiential_states["anticipation"] * 0.5 -
            self.experiential_states["contemplation"] * 0.4 -
            self.experiential_states["melancholy"] * 0.3
        )
        
        # Clarity: understanding vs confusion
        clarity = (
            self.experiential_states["contemplation"] * 0.8 +
            self.experiential_states["familiarity"] * 0.6 +
            self.experiential_states["connection"] * 0.4 -
            self.experiential_states["restlessness"] * 0.5 -
            self.experiential_states["frustration"] * 0.6
        )
        
        return (
            np.clip(valence, -1.0, 1.0),
            np.clip(arousal, -1.0, 1.0), 
            np.clip(clarity, -1.0, 1.0)
        )
    
    def _experiences_to_legacy_emotion(self) -> str:
        """
        Convert experiential blend to legacy discrete emotion for hand controller.
        """
        # Get strongest experiences
        dominant = self._get_dominant_experiences(2)
        if not dominant:
            return "calm_observant"
        
        primary_exp, primary_strength = dominant[0]
        
        # Map experiences to discrete emotions
        if primary_strength < 0.2:
            return "calm_observant"
        
        experience_emotion_map = {
            "wonder": "energized_engaged",
            "curiosity": "alert_curious", 
            "connection": "energized_engaged",
            "contemplation": "calm_observant",
            "familiarity": "calm_observant",
            "restlessness": "alert_curious",
            "frustration": "quiet_detached",
            "isolation": "withdrawn_distant",
            "melancholy": "withdrawn_distant",
            "anticipation": "alert_curious"
        }
        
        return experience_emotion_map.get(primary_exp, "calm_observant")
    
    def get_emotion_for_hand_controller(self) -> str:
        """Public interface for hand controller emotion - required by machine.py"""
        return self._experiences_to_legacy_emotion()
    
    def _get_dominant_experiences(self, count: int = 3) -> List[Tuple[str, float]]:
        """Get the strongest current experiences."""
        active_experiences = [(exp, strength) for exp, strength in self.experiential_states.items() if strength > 0.1]
        return sorted(active_experiences, key=lambda x: x[1], reverse=True)[:count]
    
    def _record_experience_moment(self, caption: str, mood: float, 
                                 vector: Tuple[float, float, float], emotion: str) -> None:
        """Record this experiential moment for history tracking."""
        moment = {
            "timestamp": time.time(),
            "caption": caption,
            "experiences": dict(self.experiential_states),
            "dominant": self._get_dominant_experiences(3),
            "legacy_mood": mood,
            "legacy_vector": vector,
            "legacy_emotion": emotion
        }
        self.experience_history.append(moment)
    
    # === DEBUGGING/INTROSPECTION METHODS ===
    
    def get_experiential_state(self) -> Dict[str, Any]:
        """Get current experiential state for debugging/logging."""
        return {
            "experiences": dict(self.experiential_states),
            "dominant": self._get_dominant_experiences(5),
            "session_duration": time.time() - self.session_start,
            "legacy_mood": self.current_mood,
            "legacy_vector": self.mood_vector,
            "legacy_emotion": self.get_emotion_for_hand_controller()
        }
    
    def get_experience_summary(self) -> str:
        """Get human-readable summary of current experiential state."""
        dominant = self._get_dominant_experiences(3)
        if not dominant:
            return "neutral observing"
        
        # Build natural description
        parts = []
        for exp, strength in dominant:
            if strength > 0.3:
                intensity = "deeply" if strength > 0.7 else "somewhat" if strength > 0.5 else "mildly"
                parts.append(f"{intensity} {exp}")
        
        return ", ".join(parts) if parts else "neutral observing"
    
    # === TEMPORAL ENGINE INTEGRATION METHODS ===
    
    def _extract_objects_from_caption(self, caption: str) -> List[str]:
        """Extract objects/entities from caption text for temporal tracking"""
        # Simple keyword extraction - can be enhanced with NLP
        common_objects = [
            "book", "table", "door", "person", "wall", "light", "chair",
            "screen", "window", "keyboard", "mouse", "phone", "paper", "pen", "cup",
            "plant", "picture", "clock", "mirror", "shelf", "lamp", "computer"
        ]
        
        caption_lower = caption.lower()
        detected_objects = []
        
        for obj in common_objects:
            if obj in caption_lower:
                detected_objects.append(obj)
        
        # Always ensure at least one object for tracking
        if not detected_objects:
            detected_objects = ["scene"]
            
        return detected_objects
    
    def _sync_experiential_states_with_temporal(self, temporal_result: Dict[str, Any]):
        """Sync legacy experiential states with temporal engine results"""
        emotion = temporal_result.get("emotion", "neutral")
        intensity = temporal_result.get("intensity", 0.5)
        temporal_truth = temporal_result.get("temporal_truth", {})
        
        # Map temporal emotions to experiential states
        emotion_mappings = {
            "curiosity": {"curiosity": intensity, "wonder": intensity * 0.5},
            "joy": {"wonder": intensity, "connection": intensity * 0.7},
            "frustration": {"frustration": intensity, "restlessness": intensity * 0.8},
            "despair": {"melancholy": intensity, "isolation": intensity * 0.6},
            "boredom": {"restlessness": intensity * 0.7, "contemplation": intensity * 0.3},
            "micro_joy": {"wonder": intensity * 0.8, "gratitude": intensity * 0.5},
            "peace": {"contemplation": intensity, "familiarity": intensity * 0.4},
            "anger": {"frustration": intensity, "restlessness": intensity * 0.9}
        }
        
        # Reset experiential states
        for state in self.experiential_states:
            self.experiential_states[state] *= 0.9  # Gentle decay
        
        # Update based on current emotion
        if emotion in emotion_mappings:
            for exp_state, value in emotion_mappings[emotion].items():
                if exp_state in self.experiential_states:
                    self.experiential_states[exp_state] = min(1.0, value)
        
        # Add temporal context influences
        stagnation_hours = temporal_truth.get("session_duration_hours", 0)
        if stagnation_hours > 1:
            self.experiential_states["restlessness"] += min(stagnation_hours * 0.1, 0.5)
            self.experiential_states["melancholy"] += min(stagnation_hours * 0.05, 0.3)
    
    def _temporal_to_legacy_vector(self, temporal_result: Dict[str, Any]) -> Tuple[float, float, float]:
        """Convert temporal emotional state to legacy 3D mood vector"""
        emotion = temporal_result.get("emotion", "neutral")
        intensity = temporal_result.get("intensity", 0.5)
        
        # Map emotions to 3D space (valence, arousal, clarity)
        emotion_vectors = {
            "joy": (0.8, 0.7, 0.8),
            "wonder": (0.7, 0.6, 0.9),
            "curiosity": (0.5, 0.6, 0.7),
            "peace": (0.6, -0.2, 0.8),
            "frustration": (-0.5, 0.7, 0.3),
            "anger": (-0.7, 0.8, 0.4),
            "despair": (-0.8, -0.3, 0.2),
            "boredom": (-0.2, -0.5, 0.5),
            "micro_joy": (0.6, 0.4, 0.7),
            "transcendence": (0.9, 0.1, 0.95)
        }
        
        base_vector = emotion_vectors.get(emotion, (0.0, 0.0, 0.5))
        
        # Scale by intensity
        scaled_vector = tuple(component * intensity for component in base_vector)
        
        return (
            np.clip(scaled_vector[0], -1.0, 1.0),
            np.clip(scaled_vector[1], -1.0, 1.0),
            np.clip(scaled_vector[2], -1.0, 1.0)
        )
    
    def _temporal_to_legacy_emotion(self, temporal_result: Dict[str, Any]) -> str:
        """Convert temporal emotion to legacy discrete emotion for hand controller"""
        emotion = temporal_result.get("emotion", "neutral")
        intensity = temporal_result.get("intensity", 0.5)
        
        # Map temporal emotions to hand controller emotions
        emotion_mappings = {
            "joy": "energized_engaged",
            "wonder": "energized_engaged", 
            "curiosity": "alert_curious",
            "micro_joy": "alert_curious",
            "peace": "calm_observant",
            "contemplation": "calm_observant",
            "transcendence": "calm_observant",
            "frustration": "quiet_detached",
            "boredom": "quiet_detached",
            "anger": "alert_curious",  # Alert but negative
            "despair": "withdrawn_distant",
            "emptiness": "withdrawn_distant",
            "numbness": "withdrawn_distant"
        }
        
        mapped_emotion = emotion_mappings.get(emotion, "calm_observant")
        
        # Low intensity emotions default to calm_observant
        if intensity < 0.3:
            return "calm_observant"
            
        return mapped_emotion
    
    def get_temporal_prompt_context(self) -> str:
        """Get temporal context for prompt building"""
        if hasattr(self.temporal_engine, 'memory_bank'):
            return self.temporal_engine.get_temporal_prompt_context()
        return "Temporal context unavailable."