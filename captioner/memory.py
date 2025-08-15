from __future__ import annotations

from config.word_lists import CONCRETE_NOUN_HINTS, MEANINGFUL_CATEGORIES, MEANINGFUL_MOTIFS, MOTIF_BLACKLIST

"""
captioner/memory.py
-------------------
MemoryMixin – recursive, emergent memory and identity tracking.

Short‑/long‑term queues, motif tracking, boredom/novelty, and identity
formation (beliefs that drift in and out as motifs recur or fade).

Imports for Captioner:
    from .memory import MemoryMixin, CAPTION_SAVE_THRESHOLD
"""

import re
import os
import glob
from collections import deque, Counter
from typing import Deque, List, Tuple, Set, Dict, Any, Optional

import spacy  # ✅ used for extracting semantic motifs
from utils.continuity import now, describe_duration, describe_time_gap
from typing import Optional

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
    def __init__(self) -> None:
        # Experience queues
        self.memory_queue: Deque[dict] = deque(maxlen=MAX_MEMORY_ENTRIES)
        self.long_memory: List[dict] = []

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
        self.novelty_score: float = 1.0
        self.boredom: float = 0.0

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
            if hasattr(self, 'current_mood'):
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
        if motif not in self.motif_confidence:
            self.motif_confidence[motif] = 0.4  # default to low confidence
            self.motif_confirmed[motif] = False

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
                print(f"[🧹] Cleaned up {cleaned} irrelevant motifs")

        for motif, count in self.motif_counter.items():
            motif_age_days = (now_time - self.motif_first_seen.get(motif, now_time)) / 86400
            if count >= BELIEF_THRESHOLD and motif_age_days >= BELIEF_FORM_MIN_DAYS:
                prev_strength = self.beliefs.get(motif, {}).get("strength", 0.5)
                strength = min(1.0, prev_strength + 0.02)
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
            self.novelty_score = 1.0
            return 1.0
            
        # Base novelty from caption comparison
        cur = self.memory_queue[-1]["text"].lower()
        prev = self.memory_queue[-2]["text"].lower()
        caption_novelty = 1.0 if cur != prev else 0.0
        
        # === FRAME DIFF NOVELTY INTEGRATION ===
        # Real-time environmental change should boost novelty
        environmental_novelty = 0.0
        if reactivity_metrics:
            activity_level = reactivity_metrics.get('activity_level', 0.0)
            sudden_change = reactivity_metrics.get('sudden_change', 0.0)
            
            # Activity boosts novelty (movement = new visual information)
            environmental_novelty = min(1.0, activity_level * 2.0 + sudden_change)
            
        # Combine caption and environmental novelty
        # Environmental change should override text repetition
        self.novelty_score = max(caption_novelty, environmental_novelty * 0.8)
        return self.novelty_score

    def get_temporal_mood_modifier(self) -> float:
        """Calculate mood modifier based on temporal stagnation effects."""
        if not hasattr(self, 'true_session_start'):
            return 0.0
            
        session_duration = now() - self.true_session_start
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
        if self.novelty_score < 0.3:
            self.boredom = min(1.0, self.boredom + 0.1)
        else:
            self.boredom = max(0.0, self.boredom - 0.05)
            
        # === TEMPORAL STAGNATION EFFECT ===
        # Add progressive boredom from extended observation of same scene
        if hasattr(self, 'true_session_start'):
            session_duration = now() - self.true_session_start
            
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
                self.boredom = max(self.boredom, temporal_boredom)

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
        seen, out = set(), []
        for entry in reversed(self.memory_queue):
            cap = entry["text"]
            if cap not in seen:
                out.append(cap)
                seen.add(cap)
                if len(out) >= k:
                    break
        return list(reversed(out))

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
            if hasattr(self, 'true_session_start'):
                session_duration = describe_duration(self.true_session_start)
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
        snippets = self.get_clean_memory_snippets(k=k)
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
