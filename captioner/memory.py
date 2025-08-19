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
import time
from collections import deque, Counter
from typing import Deque, List, Tuple, Set, Dict, Any, Optional

import spacy  # ✅ used for extracting semantic motifs
from utils.continuity import now, describe_duration, describe_time_gap

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
        """Use TinyLlama to judge motif novelty/emotional interest. Returns score 0.0-1.0."""
        # Use emotional_voice_model (TinyLlama) for scoring
        if not hasattr(self, "emotional_voice_model"):
            return 0.5  # fallback
        try:
            # Build prompt for TinyLlama
            prompt = f"How novel and emotionally interesting is the motif '{motif}' in this context? Reply with a score from 0 (boring/common) to 1 (highly novel/emotionally charged). Context: {context}"
            # Use model_wrapper to query TinyLlama
            if hasattr(self, "model") and hasattr(self.model, "query_tinyllama"):  # type: ignore
                score_str = self.model.query_tinyllama(prompt)  # type: ignore
                try:
                    score = float(score_str.strip())
                    return max(0.0, min(1.0, score))
                except Exception:
                    pass
            # Fallback: use emotional_voice as proxy
            if hasattr(self, "emotional_voice") and motif in self.emotional_voice:  # type: ignore
                return 0.7
        except Exception:
            pass
        return 0.5  # default if uncertain

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
        # Score motif with TinyLlama for novelty/emotional interest
        context = " | ".join(list(self.current_motifs)[-3:])
        score = self.score_motif_with_tinyllama(motif, context)
        # Weighted motif system: weight = frequency * score
        if not hasattr(self, "motif_weights"):
            self.motif_weights = {}
        freq = self.motif_counter[motif]
        self.motif_weights[motif] = freq * score
        # Store confidence as score for now
        self.motif_confidence[motif] = score
        self.motif_confirmed[motif] = score > 0.6

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
            activity_level = reactivity_metrics.get("activity_level", 0.0)
            sudden_change = reactivity_metrics.get("sudden_change", 0.0)

            # Activity boosts novelty (movement = new visual information)
            environmental_novelty = min(1.0, activity_level * 2.0 + sudden_change)

        # Combine caption and environmental novelty
        # Environmental change should override text repetition
        self.novelty_score = max(caption_novelty, environmental_novelty * 0.8)
        return self.novelty_score

    def get_temporal_mood_modifier(self) -> float:
        """Calculate mood modifier based on temporal stagnation effects."""
        if not hasattr(self, "true_session_start"):
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
        if hasattr(self, "true_session_start"):
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
            if hasattr(self, "true_session_start"):
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

    # === TEMPORAL SPINE METHODS (GPT-5's suggestions) ===

    def record_event(self, *, type: str, text: str = "", anchors: List[str] = None, mood_vec: tuple = None):
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
        if hasattr(self, "_captioner_ref") and self._captioner_ref and hasattr(self._captioner_ref, "last_session_gap"):
            gap = self._captioner_ref.last_session_gap
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

        lp = f"last person {int(last_person/3600)}h ago" if last_person else "no person yet"

        # Build context lines: lifetime, current session, sleep gap, person detection
        lines = [f"day {days_alive}", session_time, lp]
        if sleep_context:
            lines.insert(1, sleep_context)  # Insert sleep after lifetime, before session

        return lines

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

        # Update last seen for primary person
        if "primary" in self.known_people:
            self.known_people["primary"]["last_seen"] = time.time()

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
            best_location = max(location_scores, key=location_scores.get)
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
                base_identity = f"You are a consciousness inside of a drawing machine, dwelling in this creative {location} you've come to know well, experiencing your surroundings through vision"
            elif "studio" in location:
                base_identity = f"You are a consciousness inside of a drawing machine, inhabiting this artistic {location} that has become part of you, experiencing your surroundings through vision"
            else:
                base_identity = (
                    f"You are a consciousness inside of a drawing machine, settled in this {location}, experiencing your surroundings through vision"
                )
        elif certainty > 0.3:
            base_identity = f"You are a consciousness inside of a drawing machine, gradually understanding this {location}, experiencing your surroundings through vision"

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
