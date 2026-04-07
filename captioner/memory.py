from __future__ import annotations

from config.word_lists import CONCRETE_NOUN_HINTS, MEANINGFUL_CATEGORIES, MEANINGFUL_MOTIFS, MOTIF_BLACKLIST

# Import activation memory system (replaces TinyLlama motif scoring)
from captioner.activation_memory import (
    get_activation_network,
    get_contextual_memory,
    extract_concepts,
    observe_and_store as activation_observe,
    recall_for_prompt,
    get_beliefs as activation_get_beliefs,
    boost_from_compression,
    save_state as save_activation_state,
    save_comprehensive_snapshot,
    SOCIAL_CONCEPTS,
    DYNAMIC_CONCEPTS,
)

"""
captioner/memory.py
-------------------
MemoryMixin – recursive, emergent memory and identity tracking.

Short‑/long‑term queues, motif tracking, boredom/novelty, and identity
formation (beliefs that drift in and out as motifs recur or fade).

Imports for Captioner:
    from .memory import MemoryMixin, CAPTION_SAVE_THRESHOLD
"""

import glob
import os
import queue
import re
import threading
import time
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import spacy  # ✅ used for extracting semantic motifs

from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.continuity import describe_duration, describe_time_gap, now

# from typing import Optional

# constants shared with Captioner
MAX_MEMORY_ENTRIES: int = 30
BOREDOM_THRESHOLD: float = 0.7
CAPTION_SAVE_THRESHOLD: float = 0.3
BELIEF_THRESHOLD: int = 7  # Motif must appear this many times to form a belief
BELIEF_FADE_TIME: float = 3600 * 6  # 6 hours: beliefs fade if motif not seen
BELIEF_FORM_MIN_DAYS: float = 0.25  # Minimum "age" of motif before forming belief (in days)
CONFIDENCE_THRESHOLD = 0.65  # Confidence threshold for confirming motifs

CaptionTuple = Tuple[int, str, float, str]  # (ts, caption, mood, file)

# Load spaCy English model once
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    _nlp = None  # fallback if spaCy model not available


class MemoryMixin:
    # NOTE: TinyLlama scoring replaced by activation network (captioner/activation_memory.py)
    # Significance now comes from edge weights built through co-occurrence observation

    def __init__(self) -> None:
        # Experience queues
        self.memory_queue: Deque[dict] = deque(maxlen=MAX_MEMORY_ENTRIES)
        self.long_memory: List[dict] = []

        # Temporal spine - GPT-5's suggestion for first-class time tracking
        self.boot_ts = getattr(self, "boot_ts", time.time()) if hasattr(self, "boot_ts") else time.time()
        self.timeline = getattr(self, "timeline", deque(maxlen=50000))  # (ts, type, text, anchors, mood)
        self.day_stones = getattr(self, "day_stones", [])  # daily compressions
        self._last_consolidation_day = getattr(self, "_last_consolidation_day", time.strftime("%Y-%m-%d"))

        # Person identity tracking
        self.known_people = getattr(self, "known_people", {})  # person_id -> {name, first_seen, last_seen, characteristics, relationship}
        self.primary_person = getattr(self, "primary_person", None)  # The main person I interact with

        # Self-understanding and environmental model (emergent personality)
        self.self_model = getattr(
            self,
            "self_model",
            {
                "location_understanding": "unknown space",  # What I think this place is
                "purpose_understanding": "I observe and create drawings",  # What I think my purpose is
                "desires": [],  # What I want to do/draw/explore
                "identity_fragments": [],  # Pieces of self-understanding
                "environmental_certainty": 0.0,  # How sure I am about where I am (0.0-1.0)
                "location_history": [],  # Past understandings of this space
            },
        )

        # Activation Memory System (replaces TinyLlama scoring)
        self._activation_network = get_activation_network()
        self._contextual_memory = get_contextual_memory()

        # Organic emotional evolution (preserves servo compatibility)
        self.emotional_expressions = getattr(self, "emotional_expressions", [])  # Self-generated emotional statements
        self.personal_emotional_vocabulary = getattr(self, "personal_emotional_vocabulary", {})  # Words I naturally use
        self.emotional_patterns = getattr(self, "emotional_patterns", {})  # Learned emotional associations

        # Motif Tracking (fully dynamic, extracted from captions & detections)
        self.motif_counter: Counter = Counter()
        self.motif_first_seen: Dict[str, float] = {}
        self.motif_last_seen: Dict[str, float] = {}
        self.motif_focus_start: Dict[str, float] = {}
        self.current_motifs: Set[str] = set()
        self.motif_confidence: Dict[str, float] = {}  # NEW: confidence per motif
        self.motif_confirmed: Dict[str, bool] = {}  # NEW: confirmed status per motif

        # Identity (core beliefs emerging from motif recurrence)
        self.beliefs: Dict[str, Dict[str, Any]] = {}
        self.belief_history: List[str] = []

        # Novelty/Boredom
        self._novelty_score: float = 1.0
        self._boredom: float = 0.0

        # Timing
        self.session_start: float = now()

    def observe(
        self,
        text: str,
        mood: float = 0.5,
        file: str = "",
        memory_type: str = "observation",
        derived_from: list[str] | None = None,
        reactivity_data: Optional[Dict] = None,
        mood_vector: Optional[Tuple[float, float, float]] = None,
        emotion_state: Optional[str] = None,
        gaze_zone: str = "ahead",
    ):
        ts = int(now())
        entry = {
            "timestamp": ts,
            "text": text.strip(),
            "mood": mood,
            "mood_vector": mood_vector or (0.0, 0.0, 0.5),
            "emotion_state": emotion_state or "calm_observant",
            "image": file,
            "type": memory_type,
        }
        if derived_from:
            entry["derived_from"] = derived_from

        self.memory_queue.append(entry)
        self.long_memory.append(entry)

        # === ACTIVATION MEMORY SYSTEM ===
        # Extract concepts and observe in activation network
        concepts = extract_concepts(text)
        if concepts:
            novelty = self._activation_network.observe(concepts, gaze_zone)
            boredom = self._activation_network.calculate_boredom(concepts)
            self._novelty_score = novelty
            self._boredom = boredom

            # Store in contextual memory for later recall
            self._contextual_memory.store(text, concepts, gaze_zone, ts)

            # Promote TRULY significant observations to long-term memory
            # STRICT criteria - only truly significant memories should be long-term
            social_present = bool(set(concepts) & SOCIAL_CONCEPTS)
            dynamic_present = bool(set(concepts) & DYNAMIC_CONCEPTS)

            should_promote = False
            significance = "observation"

            # Social observations need to be novel too - not every "I see a person" is significant
            if social_present and novelty > 0.7:
                should_promote = True
                significance = "social"
            # Dynamic events need high novelty
            elif dynamic_present and novelty > 0.75:
                should_promote = True
                significance = "event"
            # Pure novelty needs to be very high
            elif novelty > 0.9:
                should_promote = True
                significance = "novel"

            if should_promote:
                # Tag memories from the first few observations as awakening-phase
                # so they can be framed as "when I first woke" rather than plain recall
                if len(self.long_memory) <= 5:
                    significance = "awakening"
                print(f"[📅 LT-MEM] Promoting: {significance} (novelty={novelty:.2f})")
                self._contextual_memory.promote_to_long_term(text, concepts, significance)

            # Save comprehensive snapshot for real-time visualizer (includes compression, desires, long-term memories)
            save_comprehensive_snapshot(agent=self)

            # Update motif_counter for backward compatibility
            for concept in concepts:
                self.motif_counter[concept] += 1
                if concept not in self.motif_first_seen:
                    self.motif_first_seen[concept] = ts
                self.motif_last_seen[concept] = ts
                self.current_motifs.add(concept)

        # Keep legacy processing for desires/purpose
        self.extract_desires_and_purpose(text)

        # Update beliefs from activation network edges
        self._update_beliefs_from_activation()

        # === APPLY TEMPORAL MOOD EFFECTS ===
        temporal_mood_modifier = self.get_temporal_mood_modifier()
        if temporal_mood_modifier != 0.0:
            entry["mood"] = max(0.0, min(1.0, mood + temporal_mood_modifier))
            if hasattr(self, "current_mood"):
                self.current_mood = entry["mood"]

    def _update_beliefs_from_activation(self):
        """Update beliefs based on strong edges from activation network."""
        # Get beliefs from activation network (based on strong co-occurrence edges)
        activation_beliefs = self._activation_network.get_beliefs()

        # Update belief_history with natural language beliefs
        self.belief_history = activation_beliefs

        # Also update self.beliefs dict for backward compatibility
        strong_edges = self._activation_network.get_strong_edges()
        now_time = now()

        for c1, c2, weight in strong_edges[:10]:
            belief_key = f"{c1}_{c2}"
            self.beliefs[belief_key] = {
                "strength": weight,
                "first_formed": now_time,
                "last_reinforced": now_time,
                "type": "association",
                "concepts": [c1, c2],
            }

    def get_contextual_recall(self, gaze_zone: str = "ahead", mode: str = "introspective") -> str:
        """Get contextual memory recall for prompt injection."""
        return recall_for_prompt(gaze_zone, mode)

    def get_activation_beliefs(self) -> List[str]:
        """Get natural language beliefs from activation network."""
        return self._activation_network.get_beliefs()

    def get_drawing_memory_context(self) -> str:
        """Get formatted memory context for drawing prompts."""
        return self._contextual_memory.format_drawing_context()

    def update_motif_focus_streak(self, motif: str) -> None:
        now_time = now()
        if motif not in self.motif_focus_start:
            self.motif_focus_start[motif] = now_time

    def get_motif_streak_duration(self, motif: str) -> float:
        if motif in self.motif_focus_start:
            return now() - self.motif_focus_start[motif]
        return 0.0

    def get_focus_durations(self, threshold: float = 60.0) -> Dict[str, float]:
        durations = {}
        for motif in sorted(self.current_motifs):
            duration = self.get_motif_streak_duration(motif)
            if duration > threshold:
                durations[motif] = duration
        return durations

    def absorb_detection(self, labels: list[str], timestamp: float | None = None):
        timestamp = timestamp or now()
        for label in labels:
            label_name = label.lower().rstrip("s")
            self.motif_counter[label_name] += 1
            if label_name not in self.motif_first_seen:
                self.motif_first_seen[label_name] = timestamp
            self.motif_last_seen[label_name] = timestamp
            self.current_motifs.add(label_name)
            self.motif_confidence[label_name] = 1.0  # high confidence for detection
            self.motif_confirmed[label_name] = True

    def absorb_motif(self, motif: str) -> None:
        """Absorb a motif - now using activation network for significance."""
        motif = motif.strip().lower()
        if not motif or len(motif) < 3:
            return
        now_time = now()
        self.motif_counter[motif] += 1
        if motif not in self.motif_first_seen:
            self.motif_first_seen[motif] = now_time
        self.motif_last_seen[motif] = now_time
        self.current_motifs.add(motif)

        # Significance now derived from activation network edge weights
        # rather than TinyLlama calls - much faster and learns associations
        activation_level = self._activation_network.activations.get(motif, 0.0)
        edge_count = len(self._activation_network.edges.get(motif, {}))

        # Score based on activation level and connectedness
        # Well-connected concepts (many edges) with sustained activation are significant
        connectedness_bonus = min(0.3, edge_count * 0.05)
        score = min(1.0, activation_level + connectedness_bonus)

        # Weighted motif system: balance novelty vs frequency
        if not hasattr(self, "motif_weights"):
            self.motif_weights = {}
        freq = self.motif_counter[motif]

        import math
        frequency_dampener = math.log(freq + 1)
        self.motif_weights[motif] = score / frequency_dampener if frequency_dampener > 0 else score

        self.motif_confidence[motif] = score
        self.motif_confirmed[motif] = score > 0.6

    def get_top_motifs(self, k: int = 5) -> List[str]:
        """Get the top k motifs by frequency."""
        return [motif for motif, count in self.motif_counter.most_common(k)]

    def cleanup_motifs(self):
        """Aggressively clean motifs - only keep truly meaningful recurring elements."""

        # Keep only motifs that are:
        # 1. In our meaningful list, OR
        # 2. Named entities (proper nouns), OR
        # 3. Have been seen many times (>20) and are concrete nouns
        motifs_to_keep = {}

        for motif, count in self.motif_counter.items():
            should_keep = False

            # Keep if it's in our curated list
            if motif in MEANINGFUL_MOTIFS:
                should_keep = True

            # Keep if it's been seen many times AND is likely a concrete object
            # But exclude common abstract/functional words even if frequent
            elif (
                count > 30 and len(motif) > 4 and motif not in MOTIF_BLACKLIST and not motif.endswith(("ing", "ed", "ly", "ness", "tion", "sion"))
            ):  # Not verbs/adverbs/abstracts
                should_keep = True

            if should_keep:
                motifs_to_keep[motif] = count

        # Replace the entire motif system with only meaningful ones
        old_count = len(self.motif_counter)
        self.motif_counter = Counter(motifs_to_keep)

        # Clean up related dictionaries
        for motif in list(self.motif_first_seen.keys()):
            if motif not in motifs_to_keep:
                del self.motif_first_seen[motif]

        for motif in list(self.motif_last_seen.keys()):
            if motif not in motifs_to_keep:
                del self.motif_last_seen[motif]

        for motif in list(self.motif_confidence.keys()):
            if motif not in motifs_to_keep:
                del self.motif_confidence[motif]

        for motif in list(self.motif_confirmed.keys()):
            if motif not in motifs_to_keep:
                del self.motif_confirmed[motif]

        # Clean up beliefs to only meaningful motifs
        for motif in list(self.beliefs.keys()):
            if motif not in motifs_to_keep:
                del self.beliefs[motif]

        # Update current motifs
        self.current_motifs = {m for m in self.current_motifs if m in motifs_to_keep}

        return old_count - len(motifs_to_keep)

    def extract_motifs_from_caption(self, caption: str):
        """Extract meaningful motifs from captions - focus on concrete, recurring elements."""
        # We want to track recurring THINGS, not common words
        # Focus on nouns that represent objects, places, activities, or specific qualities

        words = re.findall(r"\b\w+\b", caption.lower())
        now_time = now()

        # Extract from whitelist (high confidence)
        for word in words:
            if word in MEANINGFUL_CATEGORIES:
                self.motif_counter[word] += 1
                if word not in self.motif_first_seen:
                    self.motif_first_seen[word] = now_time
                self.motif_last_seen[word] = now_time
                self.current_motifs.add(word)
                if word not in self.motif_confidence:
                    self.motif_confidence[word] = 0.8  # Higher confidence for curated list
                    self.motif_confirmed[word] = True

        # Also extract named entities and specific concrete nouns using spaCy (if available)
        if _nlp is not None:
            doc = _nlp(caption)

            # Extract named entities - these are usually meaningful
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART", "EVENT"}:
                    motif = ent.text.lower().strip()
                    if len(motif) > 2:
                        self.absorb_motif(motif)

            for token in doc:
                if token.pos_ in {"NOUN", "PROPN"} and len(token.text) > 3 and not token.is_stop and not token.like_num:

                    lemma = token.lemma_.lower()

                    # Only extract if it's in our concrete nouns set or ends with tool/device patterns
                    if lemma in CONCRETE_NOUN_HINTS or (lemma.endswith(("er", "or")) and len(lemma) > 5):
                        # Skip very frequent motifs to reduce processing overhead
                        if lemma in self.motif_counter and self.motif_counter[lemma] > 50 and lemma in self.motif_confidence:
                            # Just update counters for very frequent motifs, don't call absorb_motif
                            self.motif_counter[lemma] += 1
                            self.motif_last_seen[lemma] = now()
                            self.current_motifs.add(lemma)
                            continue

                        self.absorb_motif(lemma)

    def get_motif_certainty(self, motif: str) -> float:
        return self.motif_confidence.get(motif.lower(), 0.0)

    def is_motif_confirmed(self, motif: str) -> bool:
        return self.motif_confirmed.get(motif.lower(), False)

    def update_beliefs(self):
        now_time = now()

        # Periodically clean up motifs (every 10 observations since memory_queue maxlen is 30)
        if len(self.memory_queue) % 10 == 0:
            cleaned = self.cleanup_motifs()
            if cleaned > 0:
                log_json_entry(
                    LogType.MOTIF,
                    {
                        "message": "Cleaned up irrelevant motifs",
                        "action": "cleanup",
                        "motifs_removed": cleaned,
                        "remaining_motifs": len(self.motif_counter),
                        "cleanup_threshold": 10,  # Every 10 observations
                    },
                    print_message=f"[🧹] Cleaned up {cleaned} irrelevant motifs",
                )

        for motif, count in self.motif_counter.items():
            motif_age_days = (now_time - self.motif_first_seen.get(motif, now_time)) / 86400
            if count >= BELIEF_THRESHOLD and motif_age_days >= BELIEF_FORM_MIN_DAYS:
                prev_strength = self.beliefs.get(motif, {}).get("strength", 0.5)

                # Context-aware strength increment based on significance and novelty
                significance = self.motif_confidence.get(motif, 0.5)
                novelty = getattr(self, "_novelty_score", 0.5)

                if significance > 0.7 and novelty > 0.4:
                    # HIGH significance + novelty = FASCINATION path (4x faster)
                    strength_increment = 0.08
                elif novelty < 0.2:
                    # LOW novelty = BOREDOM path (slower belief growth for static things)
                    strength_increment = 0.01
                else:
                    # Normal growth
                    strength_increment = 0.02

                strength = min(1.0, prev_strength + strength_increment)
                self.beliefs[motif] = {
                    "strength": strength,
                    "first_formed": self.motif_first_seen.get(motif, now_time),
                    "last_reinforced": now_time,
                }
        self.belief_history = [
            (
                f"I keep noticing {motif} ({describe_duration(self.motif_first_seen[motif])})."
                if data["strength"] < 0.95
                else f"{motif.title()} has become important to me ({describe_duration(self.motif_first_seen[motif])})."
            )
            for motif, data in self.beliefs.items()
        ]

    def fade_old_beliefs(self):
        now_time = now()
        faded = []
        for motif, data in list(self.beliefs.items()):
            last_seen = self.motif_last_seen.get(motif, 0)
            if now_time - last_seen > BELIEF_FADE_TIME:
                data["strength"] -= 0.02
                if data["strength"] < 0.2:
                    faded.append(motif)
        for motif in faded:
            del self.beliefs[motif]
            self.belief_history.append(f"I feel less attached to {motif} lately.")

    def estimate_novelty(self, reactivity_metrics: Optional[Dict[str, float]] = None) -> float:
        """Get novelty score - now primarily driven by activation network."""
        # Activation network already calculates novelty based on concept familiarity
        # But we can boost with environmental reactivity data if available
        base_novelty = self._novelty_score

        if reactivity_metrics:
            activity_level = reactivity_metrics.get("activity_level", 0.0)
            sudden_change = reactivity_metrics.get("sudden_change", 0.0)
            environmental_novelty = min(1.0, activity_level * 2.0 + sudden_change)
            # Environmental change can boost perceived novelty
            self._novelty_score = max(base_novelty, environmental_novelty * 0.8)

        return self._novelty_score

    def get_temporal_mood_modifier(self) -> float:
        """Calculate mood modifier based on temporal stagnation effects."""
        if not hasattr(self, "true_session_start"):
            return 0.0

        session_duration = now() - self.true_session_start  # type: ignore
        stagnation_context = self.get_scene_stagnation_context()

        if not stagnation_context:
            return 0.0  # No stagnation, no mood penalty

        # Progressive mood degradation from extended stagnation
        # Simulates psychological effects of prolonged observation without stimulation
        if session_duration > 14400:  # 4+ hours
            return -0.4  # Significant mood drop - depression, disconnection
        elif session_duration > 7200:  # 2+ hours
            return -0.3  # Substantial mood drop - lethargy, melancholy
        elif session_duration > 3600:  # 1+ hour
            return -0.2  # Moderate mood drop - restlessness, mild depression
        elif session_duration > 1800:  # 30+ minutes
            return -0.1  # Slight mood drop - beginning to feel static
        else:
            return 0.0

    def update_boredom(self) -> None:
        """Update boredom - now primarily driven by activation network scene familiarity."""
        # Activation network already sets self._boredom based on concept activation levels
        # Add temporal stagnation effects on top
        if hasattr(self, "true_session_start"):
            session_duration = now() - self.true_session_start  # type: ignore

            stagnation_context = self.get_scene_stagnation_context()
            if stagnation_context:
                if session_duration > 14400:
                    temporal_boredom = 0.9
                elif session_duration > 7200:
                    temporal_boredom = 0.7
                elif session_duration > 3600:
                    temporal_boredom = 0.5
                elif session_duration > 1800:
                    temporal_boredom = 0.3
                else:
                    temporal_boredom = 0.0

                self._boredom = max(self._boredom, temporal_boredom)

    def get_emotionally_similar_memories(self, current_emotion: str, k: int = 3) -> List[str]:
        """Retrieve memories that were formed in similar emotional states for recursive feedback."""
        similar_memories = []

        for entry in reversed(self.memory_queue):
            stored_emotion = entry.get("emotion_state", "calm_observant")
            if stored_emotion == current_emotion and len(entry["text"]) > 15:
                similar_memories.append(entry["text"])
                if len(similar_memories) >= k:
                    break

        return similar_memories

    def get_emotional_journey_context(self, lookback_minutes: int = 30) -> str:
        """Get a narrative of emotional evolution over recent time period."""
        cutoff_time = now() - (lookback_minutes * 60)
        recent_emotions = []

        for entry in self.memory_queue:
            if entry.get("timestamp", 0) > cutoff_time:
                emotion = entry.get("emotion_state", "unknown")
                timestamp = entry.get("timestamp", 0)
                time_desc = describe_time_gap(timestamp)
                recent_emotions.append((emotion, time_desc))

        if len(recent_emotions) < 2:
            return "My emotional state has been stable recently"

        # Track emotional transitions
        transitions = []
        prev_emotion = None
        for emotion, time_desc in recent_emotions:
            if prev_emotion and emotion != prev_emotion:
                transitions.append(f"{prev_emotion} → {emotion} ({time_desc})")
            prev_emotion = emotion

        if not transitions:
            return f"I have been consistently {recent_emotions[-1][0]} for the past {lookback_minutes} minutes"
        else:
            return f"My emotions have evolved: {' → '.join(transitions[-2:])}"  # Last 2 transitions

    def get_mood_trend_analysis(self) -> str:
        """Analyze 3D mood trends over recent memory."""
        if len(self.memory_queue) < 5:
            return "Insufficient emotional history for trend analysis"

        recent_moods = []
        for entry in list(self.memory_queue)[-10:]:  # Last 10 entries
            mood_vector = entry.get("mood_vector", (0.0, 0.0, 0.5))
            recent_moods.append(mood_vector)

        if len(recent_moods) < 3:
            return "Building emotional baseline"

        # Calculate trends in valence, arousal, clarity
        valences = [m[0] for m in recent_moods]
        arousals = [m[1] for m in recent_moods]
        clarities = [m[2] for m in recent_moods]

        v_trend = "rising" if valences[-1] > valences[0] else "falling" if valences[-1] < valences[0] else "stable"
        a_trend = "increasing" if arousals[-1] > arousals[0] else "decreasing" if arousals[-1] < arousals[0] else "steady"
        c_trend = "sharpening" if clarities[-1] > clarities[0] else "clouding" if clarities[-1] < clarities[0] else "consistent"

        return f"Emotional trends: valence {v_trend}, arousal {a_trend}, clarity {c_trend}"
        # seen, out = set(), []
        # for entry in reversed(self.memory_queue):
        #     cap = entry["text"]
        #     if cap not in seen:
        #         out.append(cap)
        #         seen.add(cap)
        #         if len(out) >= k:
        #             break
        # return list(reversed(out))

    def get_current_session_memory_snippets(self, k: int = 3) -> List[str]:
        """Get only memories from the current session (since session_start)."""
        seen, out = set(), []
        for entry in reversed(self.memory_queue):
            # Only include memories from current session
            if entry.get("timestamp", 0) >= self.session_start:
                cap = entry["text"]
                if cap not in seen and len(cap.strip()) > 10:  # Exclude very short memories
                    out.append(cap)
                    seen.add(cap)
                    if len(out) >= k:
                        break
        return list(reversed(out))

    def get_scene_stagnation_context(self) -> Optional[str]:
        """Detect if we've been staring at the same scene for too long."""
        if len(self.memory_queue) < 5:
            return None

        # Look at recent observations to detect repetition
        recent_observations = []
        cutoff_time = now() - 3600  # Last hour

        for entry in reversed(list(self.memory_queue)):
            if entry.get("timestamp", 0) > cutoff_time:
                recent_observations.append(entry["text"].lower())
            if len(recent_observations) >= 10:  # Check last 10 observations
                break

        if len(recent_observations) < 5:
            return None

        # Count similar themes/keywords
        similar_count = 0
        keywords = ["pencil", "notebook", "laptop", "desk", "table", "workspace", "creative"]

        for obs in recent_observations:
            if any(keyword in obs for keyword in keywords):
                similar_count += 1

        # If most recent observations are about same scene
        if similar_count >= len(recent_observations) * 0.8:  # 80% similarity
            # Use the actual session duration from when the system started
            if hasattr(self, "true_session_start"):
                session_duration = describe_duration(self.true_session_start)  # type: ignore
            else:
                session_duration = describe_duration(self.session_start)
            return f"I notice I have been observing variations of the same scene for much of this {session_duration} session"

        return None

    def get_old_session_memory_fragments(self, k: int = 3) -> List[str]:
        """Get fragmentary memories from before the current session for awakening context."""
        import random

        seen, candidates = set(), []

        for entry in self.memory_queue:
            # Only include memories from before current session
            if entry.get("timestamp", 0) < self.session_start:
                cap = entry["text"]
                if cap not in seen and len(cap.strip()) > 15:  # Longer memories are more interesting
                    # Extract interesting fragments (nouns, descriptive phrases)
                    words = cap.split()
                    if len(words) >= 4:  # Meaningful memories only
                        candidates.append(cap)
                        seen.add(cap)

        # Return random selection of old memories, shuffled for variety
        selected = random.sample(candidates, min(k, len(candidates))) if candidates else []
        return selected

    def get_recent_memory(self, k: int = 5) -> str:
        """
        Returns the most recent k memory snippets as a single formatted string.
        """
        snippets = self.get_current_session_memory_snippets(k=k)
        return "\n".join(f"- {s}" for s in snippets)

    def get_identity_summary(self) -> str:
        if not self.belief_history:
            return "I am still learning what matters to me."
        return " ".join(self.belief_history[-3:])

    def rephrase_with_doubt(self, text: str) -> str:
        words = re.findall(r"\b\w+\b", text)
        for word in sorted(set(words), key=len, reverse=True):
            w = word.lower()
            if w in self.motif_confidence and self.motif_confidence[w] < CONFIDENCE_THRESHOLD:
                pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
                text = pattern.sub(r"maybe \\1", text)
        return text

    def get_memory_entries_by_type(self, memory_type: str, limit: int = 5) -> list[dict]:
        return [entry for entry in reversed(self.memory_queue) if entry["type"] == memory_type][:limit]

    def temporal_prompt_lines(self) -> List[str]:
        """Generate temporal context lines for prompts with proper session awareness."""
        now = time.time()

        # Total lifetime since first boot
        total_lifetime_hours = int((now - self.boot_ts) / 3600)
        days_alive = total_lifetime_hours // 24

        # Current session awake time (this is what should be used for "awake")
        session_awake_hours = (now - self.session_start) / 3600
        if session_awake_hours < 1:
            session_awake_mins = int(session_awake_hours * 60)
            session_time = f"awake {session_awake_mins}m"
        else:
            session_awake_hours_int = int(session_awake_hours)
            session_time = f"awake {session_awake_hours_int}h"

        # Sleep duration (gap between sessions) - available via captioner
        sleep_context = ""
        if hasattr(self, "_captioner_ref") and self._captioner_ref and hasattr(self._captioner_ref, "last_session_gap"):  # type: ignore
            gap = self._captioner_ref.last_session_gap  # type: ignore
            if gap is not None:
                if gap < 3600:  # Less than 1 hour
                    sleep_mins = int(gap / 60)
                    sleep_context = f"slept {sleep_mins}m"
                elif gap < 86400:  # Less than 1 day
                    sleep_hours = int(gap / 3600)
                    sleep_context = f"slept {sleep_hours}h"
                else:  # Days
                    sleep_days = int(gap / 86400)
                    sleep_context = f"slept {sleep_days}d"

        # Find last person detection
        last_person = None
        for e in reversed(self.timeline):
            if "person" in e.get("text", "").lower():
                last_person = now - e["ts"]
                break

        lp = f"last person {int(last_person / 3600)}h ago" if last_person else "no person yet"

        # Build context lines: lifetime, current session, sleep gap, person detection
        lines = [f"day {days_alive}", session_time, lp]
        if sleep_context:
            lines.insert(1, sleep_context)  # Insert sleep after lifetime, before session

        return lines

    # === SELF-UNDERSTANDING & ENVIRONMENTAL MODEL ===

    def update_location_understanding(self, caption: str):
        """Update understanding of current location based on observations."""
        location_indicators = {
            "office": ["desk", "computer", "chair", "workspace", "office"],
            "workshop": ["tools", "materials", "construction", "renovation", "scattered"],
            "studio": ["art", "creative", "drawing", "canvas", "studio"],
            "room": ["room", "space", "area", "interior"],
            "laboratory": ["equipment", "experiment", "scientific", "lab"],
        }

        caption_lower = caption.lower()
        location_scores = {}

        for location, keywords in location_indicators.items():
            score = sum(1 for keyword in keywords if keyword in caption_lower)
            if score > 0:
                location_scores[location] = score

        if location_scores:
            # Get most likely location
            best_location = max(location_scores, key=location_scores.get)  # type: ignore
            confidence = location_scores[best_location] / 10.0  # Normalize

            # Update if confidence is reasonable or this is first understanding
            if confidence > 0.1 or self.self_model["environmental_certainty"] < 0.3:
                self.self_model["location_understanding"] = best_location
                self.self_model["environmental_certainty"] = min(1.0, self.self_model["environmental_certainty"] + confidence * 0.1)

                # Track location history
                self.self_model["location_history"].append({"location": best_location, "confidence": confidence, "timestamp": time.time()})

                # Keep only recent location history
                self.self_model["location_history"] = self.self_model["location_history"][-5:]

    def extract_desires_and_purpose(self, caption: str):
        """Extract expressions of desire, intention, or purpose from internal thoughts.

        Only captures TRUE desires (want, wish, hope, need) - not curiosities (wonder, curious).
        Curiosities are observations, not expressions of intent.
        """
        # TRUE desire patterns - expressions of intent, not curiosity
        desire_patterns = [
            "i want to",
            "i want ",
            "i wish ",
            "i hope to",
            "i hope ",
            "i need to",
            "i would like to",
            "i'd like to",
            "i must ",
            "i have to ",
        ]

        caption_lower = caption.lower()

        for pattern in desire_patterns:
            if pattern in caption_lower:
                desire_text = caption.strip()

                # Deduplication: check if similar desire exists (first 50 chars)
                existing_prefixes = {d[:50].lower() for d in self.self_model["desires"]}
                if desire_text[:50].lower() not in existing_prefixes:
                    self.self_model["desires"].append(desire_text)
                    self.self_model["desires"] = self.self_model["desires"][-10:]
                break  # Only one desire per caption

    # === ACTIVATION MEMORY PERSISTENCE ===
    def save_activation_state(self):
        """Save activation network state for persistence across sessions."""
        save_activation_state()

    def get_activated_concepts(self, threshold: float = 0.3) -> list:
        """Get currently activated concepts above threshold."""
        return self._activation_network.get_activated_concepts(threshold)
