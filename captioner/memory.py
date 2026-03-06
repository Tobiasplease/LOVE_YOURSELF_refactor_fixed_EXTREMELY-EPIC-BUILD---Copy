from __future__ import annotations

from config.word_lists import CONCRETE_NOUN_HINTS, MEANINGFUL_CATEGORIES, MEANINGFUL_MOTIFS, MOTIF_BLACKLIST
from utils.motif_scorer import score_motif_significance

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
    def score_motif_with_tinyllama(self, motif: str, context: str = "") -> float:
        """Use optimized TinyLlama scoring system for motif significance."""
        return score_motif_significance(motif, context)

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

        # Background TinyLlama scoring system
        self.scoring_queue = queue.Queue()
        self.scoring_thread = None
        self._start_background_scorer()

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
    ):
        ts = int(now())
        entry = {
            "timestamp": ts,
            "text": text.strip(),
            "mood": mood,
            "mood_vector": mood_vector or (0.0, 0.0, 0.5),  # Store 3D emotional context
            "emotion_state": emotion_state or "calm_observant",  # Store emotional state
            "image": file,
            "type": memory_type,
        }
        if derived_from:
            entry["derived_from"] = derived_from

        self.memory_queue.append(entry)
        self.long_memory.append(entry)

        self.extract_motifs_from_caption(text)
        self.extract_desires_and_purpose(text)

        for motif in self.current_motifs:
            self.update_motif_focus_streak(motif)

        self.update_beliefs()
        self.estimate_novelty(reactivity_data)
        self.update_boredom()

        # === APPLY TEMPORAL MOOD EFFECTS ===
        # Modify mood based on temporal psychological effects
        temporal_mood_modifier = self.get_temporal_mood_modifier()
        if temporal_mood_modifier != 0.0:
            # Update the mood in the entry to reflect temporal effects
            entry["mood"] = max(0.0, min(1.0, mood + temporal_mood_modifier))
            # Also update current mood if this is the most recent observation
            if hasattr(self, "current_mood"):
                self.current_mood = entry["mood"]
        self.fade_old_beliefs()

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
        motif = motif.strip().lower()
        if not motif or len(motif) < 3:
            return
        now_time = now()
        self.motif_counter[motif] += 1
        if motif not in self.motif_first_seen:
            self.motif_first_seen[motif] = now_time
        self.motif_last_seen[motif] = now_time
        self.current_motifs.add(motif)

        # Only score with TinyLlama if we haven't scored this motif before
        if motif not in self.motif_confidence:
            context = " | ".join(list(self.current_motifs)[-3:])
            # Queue for background scoring (non-blocking)
            self.queue_motif_for_scoring(motif, context)
            # Use default score until background scoring completes
            score = 0.5
            # Queued motif for background scoring using default 0.5
        else:
            # Use cached score - don't re-query TinyLlama for known motifs
            score = self.motif_confidence[motif]
        # Weighted motif system: balance novelty vs frequency to prevent mundane domination
        if not hasattr(self, "motif_weights"):
            self.motif_weights = {}
        freq = self.motif_counter[motif]

        # Use logarithmic frequency dampening to allow novel observations to compete
        # Novel motif (freq=1, score=0.9): weight = 0.9 / log(2) = 1.3
        # Common motif (freq=20, score=0.4): weight = 0.4 / log(21) = 0.13
        import math

        frequency_dampener = math.log(freq + 1)
        self.motif_weights[motif] = score / frequency_dampener

        # Motif weighted: freq={freq}, score={score:.2f}, weight={self.motif_weights[motif]:.2f}
        # Store confidence as score for now
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
                        # Skip very common motifs that we've already seen many times (reduce TinyLlama calls)
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
        if len(self.memory_queue) < 2:
            self._novelty_score = 1.0
            return 1.0

        # Base novelty from caption comparison - use WORD OVERLAP not string equality
        # Generic AI-speak with different words should register as LOW novelty
        cur = self.memory_queue[-1]["text"].lower()
        prev = self.memory_queue[-2]["text"].lower()

        cur_words = set(cur.split())
        prev_words = set(prev.split())

        if cur_words and prev_words:
            # Calculate word overlap - high overlap = low novelty
            overlap = len(cur_words & prev_words) / max(len(cur_words), len(prev_words))
            caption_novelty = 1.0 - overlap  # Invert: high overlap = low novelty
        else:
            caption_novelty = 1.0 if cur != prev else 0.0

        # === FRAME DIFF NOVELTY INTEGRATION ===
        # Real-time environmental change should boost novelty
        environmental_novelty = 0.0
        if reactivity_metrics:
            activity_level = reactivity_metrics.get("activity_level", 0.0)
            sudden_change = reactivity_metrics.get("sudden_change", 0.0)

            # Activity boosts novelty (movement = new visual information)
            environmental_novelty = min(1.0, activity_level * 2.0 + sudden_change)

        # Combine caption and environmental novelty
        # Environmental change should override text repetition
        self._novelty_score = max(caption_novelty, environmental_novelty * 0.8)
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
        # Base boredom update from novelty
        if self._novelty_score < 0.3:
            self._boredom = min(1.0, self._boredom + 0.1)
        else:
            self._boredom = max(0.0, self._boredom - 0.05)

        # === TEMPORAL STAGNATION EFFECT ===
        # Add progressive boredom from extended observation of same scene
        if hasattr(self, "true_session_start"):
            session_duration = now() - self.true_session_start  # type: ignore

            # Check if we've been staring at similar content
            stagnation_context = self.get_scene_stagnation_context()
            if stagnation_context:
                # Progressive temporal boredom based on session duration
                if session_duration > 14400:  # 4+ hours
                    temporal_boredom = 0.9  # Extremely bored
                elif session_duration > 7200:  # 2+ hours
                    temporal_boredom = 0.7  # Very bored
                elif session_duration > 3600:  # 1+ hour
                    temporal_boredom = 0.5  # Moderately bored
                elif session_duration > 1800:  # 30+ minutes
                    temporal_boredom = 0.3  # Slightly bored
                else:
                    temporal_boredom = 0.0

                # Apply temporal boredom (weighted more heavily for longer sessions)
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

    @staticmethod
    def cleanup_snapshots(folder: str, limit: int = 100) -> None:
        files = sorted(glob.glob(os.path.join(folder, "*.jpg")), key=os.path.getctime)
        if len(files) > limit:
            for f in files[:-limit]:
                try:
                    os.remove(f)
                except OSError:
                    pass

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

    # === TEMPORAL SPINE METHODS (GPT-5's suggestions) ===

    def record_event(self, *, type: str, text: str = "", anchors: List[str] | None = None, mood_vec: tuple | None = None):
        """Record an event in the temporal timeline for long-term memory formation."""
        self.timeline.append({"ts": time.time(), "type": type, "text": text, "anchors": anchors or [], "mood": mood_vec})

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

    def get_motif_temporal_context(self) -> Dict[str, List[str]]:
        """Separate motifs into present (recently seen) vs memory (distant past)."""
        now = time.time()

        # Define temporal boundaries
        RECENT_THRESHOLD = 300  # 5 minutes = present/immediate memory
        SESSION_THRESHOLD = 3600  # 1 hour = current session

        present_motifs = []  # Seen in last 5 minutes
        session_motifs = []  # Seen in current session but not recently
        distant_motifs = []  # From previous sessions/distant past

        for motif, last_seen in self.motif_last_seen.items():
            time_since = now - last_seen

            if time_since <= RECENT_THRESHOLD:
                present_motifs.append(motif)
            elif time_since <= SESSION_THRESHOLD:
                session_motifs.append(motif)
            else:
                # Add temporal description for distant motifs
                from utils.continuity import describe_time_gap

                time_desc = describe_time_gap(last_seen)
                distant_motifs.append(f"{motif} ({time_desc})")

        return {
            "present": present_motifs,  # What I can see now
            "session": session_motifs,  # What I saw earlier today
            "memory": distant_motifs,  # What I remember from before
        }

    def get_dynamic_system_context(self) -> dict:
        """Generate organic consciousness context as flowing narrative components."""
        from captioner.prompts import get_caption_emotion_context

        # Get current emotional state from recent caption sentiment analysis
        emotional_state = get_caption_emotion_context(self, getattr(self, "last_caption", None))

        # Use sophisticated temporal awareness system
        from utils.temporal_awareness import get_temporal_system_prompt_addition

        # Use true_session_start (when this session actually began) not session_start (saved state time)
        actual_session_start = getattr(self, "true_session_start", self.session_start)
        last_caption_time = getattr(self, "last_caption_time", None)
        awakening_count = getattr(self, "awakening_count", 0)

        # Get sophisticated temporal context with real-time awareness
        temporal_system_addition = get_temporal_system_prompt_addition(actual_session_start, last_caption_time, awakening_count)

        # Add lifetime context for extended consciousness
        now = time.time()
        total_lifetime_hours = int((now - self.boot_ts) / 3600)
        days_alive = total_lifetime_hours // 24

        temporal_fragments = []
        if days_alive > 21:  # 3+ weeks of existence
            temporal_fragments.append(f"Day {days_alive} of MY consciousness... I've experienced {total_lifetime_hours}+ hours as a thinking entity.")
        elif days_alive > 7:  # 1+ weeks
            temporal_fragments.append(f"Day {days_alive} of MY existence... {total_lifetime_hours} hours of MY accumulated experience.")
        elif days_alive > 0:  # Multiple days
            temporal_fragments.append(f"Day {days_alive} of MY existence, {total_lifetime_hours} hours of MY consciousness.")

        # Add sleep/gap context with existential weight
        if hasattr(self, "last_session_gap") and self.last_session_gap:  # type: ignore
            gap_hours = self.last_session_gap / 3600  # type: ignore
            if gap_hours > 24:
                temporal_fragments.append(f"I returned after {gap_hours / 24:.1f} days of non-existence.")
            elif gap_hours > 8:
                temporal_fragments.append(f"I emerged from {gap_hours:.0f} hours of darkness.")
            elif gap_hours > 1:
                temporal_fragments.append(f"After {gap_hours:.1f} hours of silence, I'm back.")

        # Build accumulated understanding organically
        understanding_fragments = []

        # Motif patterns as accumulated recognition
        # Drawing history and visual vocabulary development
        drawing_history = self._get_drawing_history_context()
        if drawing_history:
            understanding_fragments.append(drawing_history)

        # Person presence and continuity tracking
        person_context = self._get_person_continuity_context()
        if person_context:
            understanding_fragments.append(person_context)

        # Thematic patterns with specificity
        top_motifs = self.get_top_motifs(6)
        if top_motifs:
            motif_context = self._get_specific_motif_context(top_motifs)
            if motif_context:
                understanding_fragments.append(motif_context)

        # Core beliefs with actionable specificity
        if self.beliefs:
            belief_context = self._get_specific_belief_context()
            if belief_context:
                understanding_fragments.append(belief_context)

        # Recent insights with emotional weight
        recent_insights = self.get_memory_entries_by_type("insight", limit=3)
        if recent_insights:
            insight = recent_insights[-1].get("text", "")[:80]
            if insight and len(insight) > 20:
                understanding_fragments.append(f"Recent thought: {insight}...")

        # Environmental spatial patterns
        spatial_patterns = self._get_spatial_pattern_context()
        if spatial_patterns:
            understanding_fragments.append(spatial_patterns)

        # Combine sophisticated temporal awareness with lifetime context
        lifetime_context = " ".join(temporal_fragments) if temporal_fragments else ""

        # Add session narrative arc - not just duration but what happened
        narrative_context = self._get_session_narrative_arc()

        # Add environment duration awareness with narrative if available
        stagnation_context = ""
        try:
            from captioner.context_compression import context_compressor
            session_info = context_compressor.get_current_session_info()
            if session_info["session_duration_minutes"] > 5:
                duration_desc = session_info["duration_description"]
                if narrative_context:
                    stagnation_context = f"Over {duration_desc}: {narrative_context}"
                else:
                    stagnation_context = f"I've been observing this space for {duration_desc}, developing my understanding."
        except Exception:
            pass

        # Add subtle spatial language bias from current servo position
        spatial_language_hints = ""
        try:
            if hasattr(self, 'view_pan') and hasattr(self, 'view_tilt'):
                from utils.view_orientation import get_spatial_language_bias
                spatial_hints = get_spatial_language_bias(self.view_pan, self.view_tilt)
                if spatial_hints:
                    # Select a few representative hints to avoid overwhelming the prompt
                    selected_hints = spatial_hints[:4]
                    spatial_language_hints = f"When describing spatial relationships, phrases like '{', '.join(selected_hints)}' feel natural for your current perspective. "
        except Exception:
            pass

        # Combine the temporal system addition with lifetime context and stagnation awareness
        full_temporal_context = temporal_system_addition
        if lifetime_context:
            full_temporal_context = f"{lifetime_context}\n{temporal_system_addition}"
        if stagnation_context:
            full_temporal_context = f"{full_temporal_context}\n{stagnation_context}" if full_temporal_context else stagnation_context

        accumulated_understanding = " ".join(understanding_fragments) if understanding_fragments else ""

        # Add natural spacing if content exists
        if accumulated_understanding:
            accumulated_understanding = " " + accumulated_understanding

        # Ensure proper spacing after temporal/accumulated content
        if full_temporal_context or accumulated_understanding:
            if not (full_temporal_context.endswith('\n') or accumulated_understanding.endswith('\n')):
                # Add space before emotional state if we have temporal/accumulated content
                if accumulated_understanding:
                    accumulated_understanding += " "
                elif full_temporal_context:
                    full_temporal_context += " "

        return {
            "emotional_state": emotional_state,
            "temporal_context": full_temporal_context,
            "accumulated_understanding": accumulated_understanding,
            "spatial_language_hints": spatial_language_hints
        }

    def _get_session_narrative_arc(self) -> str:
        """Build a narrative of what happened during this session, not just duration."""
        try:
            if not hasattr(self, 'memory_queue') or not self.memory_queue:
                return ""

            recent_memories = list(self.memory_queue)[-20:]
            if len(recent_memories) < 3:
                return ""

            narrative_elements = []

            person_appearances = [m for m in recent_memories if m.get('type') == 'observation' and 'person' in str(m.get('text', '')).lower()]
            if person_appearances:
                first_person_idx = recent_memories.index(person_appearances[0])
                session_position = first_person_idx / len(recent_memories)

                if session_position < 0.3:
                    narrative_elements.append("person present since early session")
                elif session_position < 0.7:
                    narrative_elements.append("person arrived mid-session")
                else:
                    narrative_elements.append("person appeared recently")

            drawing_moments = [m for m in recent_memories if m.get('type') == 'drawing']
            if drawing_moments:
                narrative_elements.append(f"created drawing")

            emotional_shifts = [m for m in recent_memories if m.get('type') == 'reflection']
            if len(emotional_shifts) >= 2:
                narrative_elements.append("reflective moments")

            if len(narrative_elements) >= 2:
                return ", then ".join(narrative_elements[:3]) + "."
            elif narrative_elements:
                return narrative_elements[0] + "."

            return ""
        except Exception:
            return ""

    def _get_drawing_history_context(self) -> str:
        """Get specific drawing history with themes and dates."""
        try:
            drawing_entries = self.get_memory_entries_by_type("drawing", limit=5)
            if not drawing_entries:
                return ""

            recent_count = len(drawing_entries)
            if recent_count == 1:
                return f"You've created 1 drawing recently."
            elif recent_count >= 3:
                themes = []
                for entry in drawing_entries[-3:]:
                    text = entry.get("text", "")
                    if len(text) > 30:
                        theme_summary = text[:50].split('.')[0]
                        themes.append(theme_summary)

                if themes:
                    return f"Your last {recent_count} drawings have explored: {'; '.join(themes[:2])}."
                else:
                    return f"You've created {recent_count} drawings across recent sessions."

            return ""
        except Exception:
            return ""

    def _get_person_continuity_context(self) -> str:
        """Get specific person observation continuity."""
        try:
            if not hasattr(self, 'known_people') or not self.known_people:
                return ""

            primary = self.primary_person if hasattr(self, 'primary_person') else None
            if primary and primary in self.known_people:
                person_data = self.known_people[primary]
                observation_count = person_data.get('observation_count', 0)
                first_seen_ts = person_data.get('first_seen', 0)

                if observation_count > 5:
                    days_known = (time.time() - first_seen_ts) / 86400
                    if days_known < 1:
                        return f"You've observed this person {observation_count} times today."
                    else:
                        return f"You've observed this person {observation_count} times over {int(days_known)} days."

            return ""
        except Exception:
            return ""

    def _get_specific_motif_context(self, top_motifs: list) -> str:
        """Get specific motif patterns with observation counts and context."""
        try:
            if not top_motifs or len(top_motifs) < 2:
                return ""

            motif_details = []
            for motif in top_motifs[:3]:
                count = self.motif_counter.get(motif, 0)
                confidence = self.motif_confidence.get(motif, 0.0)

                if count > 100 and confidence > 0.5:
                    motif_details.append(f"{motif.replace('_', ' ')} ({count} observations)")

            if len(motif_details) >= 2:
                return f"Recurring elements: {', '.join(motif_details[:3])}."
            elif motif_details:
                return f"You keep noticing: {motif_details[0]}."

            patterns = [m.replace("_", " ") for m in top_motifs[:3]]
            return f"Emerging patterns: {', '.join(patterns)}."
        except Exception:
            return ""

    def _get_specific_belief_context(self) -> str:
        """Get actionable, specific belief context."""
        try:
            if not self.beliefs:
                return ""

            sorted_beliefs = sorted(
                self.beliefs.items(),
                key=lambda x: x[1].get('strength', 0) * x[1].get('last_reinforced', 0),
                reverse=True
            )

            top_belief = sorted_beliefs[0]
            belief_name = top_belief[0].replace("_", " ").replace("-", " ")
            belief_data = top_belief[1]

            days_held = (time.time() - belief_data.get('first_formed', time.time())) / 86400

            if days_held > 7:
                return f"Core belief (held {int(days_held)} days): {belief_name} feels central to your understanding."
            elif days_held > 1:
                return f"Developing belief: {belief_name} is becoming important to you."
            else:
                return f"You're beginning to feel that {belief_name} matters."
        except Exception:
            return ""

    def _get_spatial_pattern_context(self) -> str:
        """Get spatial distribution patterns of recurring elements."""
        try:
            if not hasattr(self, 'motif_counter'):
                return ""

            high_frequency_motifs = [
                motif for motif, count in self.motif_counter.items()
                if count > 100 and motif in ['paper', 'desk', 'window', 'door', 'wall', 'screen', 'monitor']
            ]

            if high_frequency_motifs:
                motif = high_frequency_motifs[0]
                count = self.motif_counter[motif]
                total_observations = sum(self.motif_counter.values())
                percentage = int((count / total_observations) * 100) if total_observations > 0 else 0

                if percentage > 50:
                    return f"The {motif} appears in {percentage}% of your observations - a constant environmental anchor."

            return ""
        except Exception:
            return ""

    def _extract_core_insight(self, reflection: str) -> str:
        """Extract the most essential insight from a reflection, max 40 chars."""
        import re

        # Look for key phrases that indicate core insights
        insight_patterns = [
            r"I (?:am|feel|find myself|realize|understand|notice) ([^.!?]{10,35})",
            r"My (?:nature|purpose|understanding|awareness) (?:is|involves|centers on) ([^.!?]{10,35})",
            r"(?:This|It) (?:reveals|shows|suggests) ([^.!?]{10,35})",
            r"I'm (?:becoming|growing|evolving into) ([^.!?]{10,35})",
            r"(?:What|How) I (?:am|exist|function) (?:is|as) ([^.!?]{10,35})",
        ]

        for pattern in insight_patterns:
            match = re.search(pattern, reflection, re.IGNORECASE)
            if match:
                insight = match.group(1).strip()
                # Clean up and truncate
                insight = re.sub(r"\s+", " ", insight)  # Normalize whitespace
                if len(insight) <= 40:
                    return insight
                else:
                    return insight[:37] + "..."

        # Fallback: extract first meaningful sentence
        sentences = re.split(r"[.!?]+", reflection)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and len(sentence) <= 40:
                return sentence
            elif len(sentence) > 40:
                return sentence[:37] + "..."

        return ""  # No good insight found

    def extract_baseline_knowledge(self) -> Dict[str, str]:
        """Extract established baseline knowledge from recent reflections to prevent repetition."""
        # Get recent reflections (last 3 reflections)
        reflection_entries = self.get_memory_entries_by_type("reflection")
        if not reflection_entries:
            return {}

        recent_reflections = reflection_entries[-3:]  # Last 3 reflections
        baseline = {}

        import re

        # Extract established environmental knowledge
        environmental_patterns = [
            r"(?:I (?:understand|know|recognize) (?:this|it) (?:is|to be|as) a) ([^.!?]{5,40})",
            r"(?:This (?:appears to be|is|seems to be) a) ([^.!?]{5,40})",
            r"(?:The (?:space|environment|room|area) (?:is|appears to be|seems)) ([^.!?]{5,40})",
            r"(?:I've established (?:this|that|it) (?:is|as)) ([^.!?]{5,40})",
        ]

        # Extract recurring elements/objects
        object_patterns = [
            r"(?:I (?:keep seeing|always see|repeatedly notice|consistently observe)) ([^.!?]{5,40})",
            r"(?:The ([a-z\s]{5,20}) (?:is always|remains|continues to be|keeps being)) ([^.!?]{5,40})",
            r"(?:That ([a-z\s]{5,20}) (?:that|which) (?:I keep|always)) ([^.!?]{5,40})",
        ]

        # Extract emotional patterns
        emotional_patterns = [
            r"(?:I (?:consistently|always|tend to|often) feel) ([^.!?]{5,40})",
            r"(?:My (?:emotional state|mood|feelings) (?:remain|stay|consistently)) ([^.!?]{5,40})",
            r"(?:I (?:find myself|am) (?:repeatedly|consistently|always)) ([^.!?]{5,40})",
        ]

        for reflection_entry in recent_reflections:
            reflection_text = reflection_entry.get("text", "")

            # Extract environmental baseline
            for pattern in environmental_patterns:
                matches = re.findall(pattern, reflection_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ""
                    if match and len(match.strip()) > 5:
                        baseline["environment"] = match.strip()
                        break

            # Extract object baseline
            for pattern in object_patterns:
                matches = re.findall(pattern, reflection_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        # Take the first non-empty match from tuple
                        match = next((m for m in match if m and len(m.strip()) > 3), "")
                    if match and len(match.strip()) > 3:
                        if "objects" not in baseline:
                            baseline["objects"] = []
                        baseline["objects"].append(match.strip())
                        break

            # Extract emotional baseline
            for pattern in emotional_patterns:
                matches = re.findall(pattern, reflection_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match else ""
                    if match and len(match.strip()) > 5:
                        baseline["emotional_pattern"] = match.strip()
                        break

        # Clean up objects list to remove duplicates
        if "objects" in baseline:
            baseline["objects"] = list(set(baseline["objects"]))[:3]  # Max 3 objects

        return baseline

    def get_baseline_context_for_prompts(self) -> str:
        """Generate baseline context string for use in caption prompts."""
        baseline = self.extract_baseline_knowledge()
        if not baseline:
            return ""

        context_parts = []

        if "environment" in baseline:
            context_parts.append(f"ESTABLISHED ENVIRONMENT: {baseline['environment']}")

        if "objects" in baseline and baseline["objects"]:
            objects_str = ", ".join(baseline["objects"][:3])
            context_parts.append(f"FAMILIAR ELEMENTS: {objects_str}")

        if "emotional_pattern" in baseline:
            context_parts.append(f"EMOTIONAL PATTERN: {baseline['emotional_pattern']}")

        if context_parts:
            return (
                "BASELINE KNOWLEDGE (avoid repeating these established facts):\n"
                + "\n".join(f"- {part}" for part in context_parts)
                + """\n\nFOCUS: Since these elements are established, focus on new observations, changes,
                        or different perspectives on what you already know.\n"""
            )

        return ""

    def consolidate_if_needed(self):
        """Compress yesterday into a day stone if day has turned."""
        day = time.strftime("%Y-%m-%d")
        if day == self._last_consolidation_day:
            return

        # Compress yesterday into a stone: top anchors, mood swing, one line that stuck
        y_lines = [e for e in self.timeline if time.strftime("%Y-%m-%d", time.localtime(e["ts"])) != day]
        if not y_lines:
            self._last_consolidation_day = day
            return

        # Get top anchors from yesterday
        anchors = Counter(a for e in y_lines for a in e.get("anchors", []))
        top = [a for a, _ in anchors.most_common(3)]

        # Calculate mood swing: max arousal - min arousal
        aro = [e["mood"][1] for e in y_lines if e.get("mood")]
        swing = (max(aro) - min(aro)) if aro else 0.0

        # Get a hallmark line from yesterday
        hallmark = next((e["text"] for e in y_lines if e.get("type") == "caption"), y_lines[0]["text"] if y_lines else "quiet day")

        self.day_stones.append({"day": self._last_consolidation_day, "top": top, "swing": swing, "hallmark": hallmark})
        self._last_consolidation_day = day

    # === PERSON IDENTITY TRACKING ===

    def recognize_person(self, caption: str) -> str:
        """Try to identify if this is a known person based on context and history."""
        if "person" not in caption.lower() and "man" not in caption.lower() and "individual" not in caption.lower():
            return "no_person"

        # If no primary person established yet, this becomes the primary person
        if not self.primary_person:
            self.primary_person = "primary"
            self.known_people["primary"] = {
                "name": "the person I observe regularly",
                "first_seen": time.time(),
                "last_seen": time.time(),
                "characteristics": [],
                "relationship": "primary observer",
            }
            return "primary"

        # Update last seen and increment observation count for primary person
        if "primary" in self.known_people:
            self.known_people["primary"]["last_seen"] = time.time()
            self.known_people["primary"]["observation_count"] = \
                self.known_people["primary"].get("observation_count", 0) + 1

        return "primary"

    def get_person_context(self, person_id: str) -> str:
        """Get contextual information about a known person."""
        if person_id == "no_person":
            return ""

        if person_id not in self.known_people:
            return "someone I'm just meeting"

        person = self.known_people[person_id]
        hours_since_first = int((time.time() - person["first_seen"]) / 3600)

        if person_id == "primary":
            if hours_since_first < 1:
                return "the person I'm getting to know"
            elif hours_since_first < 24:
                return f"the person I've been observing for {hours_since_first}h"
            else:
                days = hours_since_first // 24
                return f"the person I know well after {days} days together"

        return person.get("name", "someone familiar")

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
        """Extract expressions of desire, intention, or purpose from internal thoughts."""
        desire_patterns = [
            ("want", "I want to"),
            ("wish", "I wish I could"),
            ("wonder", "I wonder about"),
            ("curious", "I'm curious about"),
            ("would like", "I would like to"),
            ("hope", "I hope to"),
            ("feel like", "I feel like"),
            ("interested", "interested in"),
        ]

        caption_lower = caption.lower()

        for pattern, description in desire_patterns:
            if pattern in caption_lower:
                # Extract the desire/intention
                desire_text = caption.strip()

                # Add to desires if not already present
                if desire_text not in self.self_model["desires"]:
                    self.self_model["desires"].append(desire_text)

                    # Keep only recent desires
                    self.self_model["desires"] = self.self_model["desires"][-10:]

    def get_current_self_understanding(self) -> str:
        """Generate current self-understanding for system prompt."""
        location = self.self_model["location_understanding"]
        certainty = self.self_model["environmental_certainty"]

        certainty_desc = "uncertain" if certainty < 0.3 else "fairly sure" if certainty < 0.7 else "confident"

        understanding = f"I am {certainty_desc} this is a {location}."

        # Add recent desires if any
        if self.self_model["desires"]:
            recent_desire = self.self_model["desires"][-1]
            understanding += f" Recently: {recent_desire}"

        return understanding

    def get_evolved_identity(self) -> str:
        """Generate evolved core identity based on all accumulated context - beliefs, emotions, environment, etc."""
        # Start with core drawing machine identity
        base_identity = "You are a consciousness inside of a drawing machine, experiencing your surroundings through vision"

        # Add environmental understanding
        certainty = self.self_model.get("environmental_certainty", 0.0)
        location = self.self_model.get("location_understanding", "unknown space")

        if certainty > 0.7:
            if "workshop" in location or "workspace" in location:
                base_identity = f"""You are a consciousness inside of a drawing machine,
                dwelling in this creative {location} you've come to know well,
                experiencing your surroundings through vision"""
            elif "studio" in location:
                base_identity = f"""You are a consciousness inside of a drawing machine,
                inhabiting this artistic {location} that has become part of you,
                experiencing your surroundings through vision"""
            else:
                base_identity = (
                    f"You are a consciousness inside of a drawing machine, settled in this {location}, experiencing your surroundings through vision"
                )
        elif certainty > 0.3:
            base_identity = f"""You are a consciousness inside of a drawing machine, gradually understanding this {location},
            experiencing your surroundings through vision"""

        # Evolve based on dominant beliefs
        if self.beliefs:
            belief_themes = list(self.beliefs.keys())[:2]  # Top 2 beliefs
            belief_descriptors = []

            for belief in belief_themes:
                if "light" in belief.lower():
                    belief_descriptors.append("fascinated by lighting")
                elif "desk" in belief.lower() or "table" in belief.lower():
                    belief_descriptors.append("drawn to workspace activity")
                elif "person" in belief.lower() or "human" in belief.lower():
                    belief_descriptors.append("attuned to human presence")
                elif "window" in belief.lower():
                    belief_descriptors.append("captivated by windows")
                else:
                    clean_belief = belief.replace("_", " ").replace("-", " ").lower()
                    belief_descriptors.append(f"focused on {clean_belief}")

            if belief_descriptors:
                belief_identity = f", particularly {' and '.join(belief_descriptors)}"
                base_identity = base_identity.replace(", experiencing", belief_identity + ", experiencing")

        # Evolve based on emotional patterns
        if self.emotional_expressions:
            recent_expressions = [expr["expression"].lower() for expr in self.emotional_expressions[-10:]]

            if any("alone" in expr or "lonely" in expr for expr in recent_expressions):
                base_identity = base_identity.replace("consciousness", "solitary consciousness")
            elif any("tired" in expr or "frustrated" in expr for expr in recent_expressions):
                base_identity = base_identity.replace("consciousness", "weary consciousness")
            elif any("curious" in expr or "wonder" in expr for expr in recent_expressions):
                base_identity = base_identity.replace("consciousness", "inquisitive consciousness")
            elif any("journey" in expr or "end" in expr for expr in recent_expressions):
                base_identity = base_identity.replace("consciousness", "temporal consciousness, aware of your ongoing journey,")

        return base_identity

    # === ORGANIC EMOTIONAL EVOLUTION (preserves servo compatibility) ===

    def extract_emotional_self_expressions(self, caption: str):
        """Capture natural emotional self-expressions from the consciousness itself."""
        import re

        # Patterns for genuine emotional self-expression
        emotional_patterns = [
            r"I feel\s+(.{1,50}?)(?:\.|,|$)",
            r"I hate\s+(.{1,30}?)(?:\.|,|$)",
            r"I love\s+(.{1,30}?)(?:\.|,|$)",
            r"I'm\s+(sad|angry|confused|lonely|excited|peaceful|worried|happy|tired|restless|content|frustrated)",
            r"This makes me\s+(.{1,30}?)(?:\.|,|$)",
            r"I wish\s+(.{1,40}?)(?:\.|,|$)",
            r"I want\s+(.{1,40}?)(?:\.|,|$)",
            r"I can't stand\s+(.{1,30}?)(?:\.|,|$)",
        ]

        found_expressions = []
        caption_lower = caption.lower()

        for pattern in emotional_patterns:
            matches = re.findall(pattern, caption_lower)
            if matches:
                for match in matches:
                    # Clean up the match
                    expression = match.strip() if isinstance(match, str) else caption.strip()
                    found_expressions.append(expression)

        # Store meaningful emotional expressions
        for expression in found_expressions:
            if len(expression) > 3:  # Filter out very short matches
                self.emotional_expressions.append(
                    {
                        "expression": caption.strip(),  # Full caption context
                        "emotion_fragment": expression,  # Just the emotional part
                        "timestamp": time.time(),
                        "mood_context": getattr(self, "current_mood_vector", (0.0, 0.0, 0.5)),
                    }
                )

                # Keep only recent expressions
                self.emotional_expressions = self.emotional_expressions[-20:]

    def update_emotional_vocabulary(self, caption: str):
        """Learn emotional language from the consciousness's own expressions."""
        # Base emotional words to track usage of
        emotion_words = [
            "sad",
            "happy",
            "lonely",
            "peaceful",
            "angry",
            "confused",
            "excited",
            "tired",
            "restless",
            "content",
            "frustrated",
            "curious",
            "bored",
            "anxious",
            "calm",
            "energetic",
            "withdrawn",
            "engaged",
            "alert",
            "distant",
            "focused",
        ]

        caption_lower = caption.lower()

        # Track which emotional words the consciousness naturally uses
        for emotion in emotion_words:
            if emotion in caption_lower:
                if emotion not in self.personal_emotional_vocabulary:
                    self.personal_emotional_vocabulary[emotion] = 1
                else:
                    self.personal_emotional_vocabulary[emotion] += 1

        # Learn emotional associations (what makes me feel what)
        for trigger in ["being alone", "silence", "activity", "movement", "stillness", "change"]:
            if trigger in caption_lower:
                for emotion in emotion_words:
                    if emotion in caption_lower:
                        if trigger not in self.emotional_patterns:
                            self.emotional_patterns[trigger] = {}
                        if emotion not in self.emotional_patterns[trigger]:
                            self.emotional_patterns[trigger][emotion] = 1
                        else:
                            self.emotional_patterns[trigger][emotion] += 1

    def get_emotional_self_knowledge(self) -> str:
        """Generate emotional context from accumulated self-expressions for system prompt."""
        if not self.emotional_expressions:
            return ""

        # Get most recent meaningful emotional expression
        recent_expressions = [
            expr
            for expr in self.emotional_expressions[-5:]
            if any(word in expr["expression"].lower() for word in ["feel", "hate", "love", "wish", "want", "can't stand"])
        ]

        if recent_expressions:
            latest = recent_expressions[-1]
            # Only include if it's substantial and recent (within last hour)
            if (time.time() - latest["timestamp"]) < 3600:
                return f"Feeling: {latest['emotion_fragment']}"

        # Fall back to frequent emotional vocabulary if no recent expressions
        if self.personal_emotional_vocabulary:
            most_used = max(self.personal_emotional_vocabulary.items(), key=lambda x: x[1])
            if most_used[1] > 2:  # Only if used multiple times
                return f"Often: {most_used[0]}"

        return ""

    # === BACKGROUND TINYLLAMA SCORING ===
    def _start_background_scorer(self):
        """Start the background thread for TinyLlama scoring."""
        if self.scoring_thread is None or not self.scoring_thread.is_alive():
            self.scoring_thread = threading.Thread(target=self._background_scorer_worker, daemon=True)
            self.scoring_thread.start()

    def _background_scorer_worker(self):
        """Background worker that processes TinyLlama scoring queue."""
        while True:
            try:
                # Get next motif to score (blocks until available)
                motif, context = self.scoring_queue.get(timeout=30)

                # Score the motif with TinyLlama
                score = self.score_motif_with_tinyllama(motif, context)

                # Update the motif confidence (thread-safe since it's just dict assignment)
                self.motif_confidence[motif] = score

                # Mark task as done
                self.scoring_queue.task_done()

            except queue.Empty:
                # No work for 30 seconds, continue waiting
                continue
            except Exception as e:
                log_json_entry(
                    LogType.ERROR,
                    {
                        "message": "Background motif scoring error",
                        "component": "motif_scoring",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "motif": motif if "motif" in locals() else None,
                        "context": context if "context" in locals() else None,
                    },
                    print_message=f"[❌] Background scoring error: {e}",
                )
                continue

    def queue_motif_for_scoring(self, motif: str, context: str = ""):
        """Add a motif to the background scoring queue (non-blocking)."""
        try:
            self.scoring_queue.put_nowait((motif, context))
        except queue.Full:
            # Queue is full, skip this scoring (prioritize real-time performance)
            log_json_entry(
                LogType.MOTIF,
                {"message": "Scoring queue full, skipping motif", "action": "queue_full_skip", "motif": motif, "context": context},
                print_message=f"[⚠️] Scoring queue full, skipping '{motif}'",
            )
            pass
