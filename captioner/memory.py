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
from typing import Deque, List, Tuple, Set, Dict, Any

import spacy  # ✅ used for extracting semantic motifs
from utils.continuity import now, describe_duration

# constants shared with Captioner
MAX_MEMORY_ENTRIES: int = 30
BOREDOM_THRESHOLD: float = 0.7
CAPTION_SAVE_THRESHOLD: float = 0.3
BELIEF_THRESHOLD: int = 7  # Motif must appear this many times to form a belief
BELIEF_FADE_TIME: float = 3600 * 2  # 2 hours: beliefs fade if motif not seen (reduced from 6h)
BELIEF_FORM_MIN_DAYS: float = 0.25  # Minimum "age" of motif before forming belief (in days)
CONFIDENCE_THRESHOLD = 0.75  # Increased confidence threshold for confirming motifs (more doubt)

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

        # Focus persistence tracking (for continuity awareness)
        self.current_focus_object: str = ""  # What we're currently focused on
        self.focus_duration: float = 0.0  # How long we've been focused on current object
        self.focus_start_time: float = 0.0  # When we started focusing on current object
        self.focus_depth: int = 0  # How many consecutive observations of same object

        # Novelty/Boredom
        self.novelty_score: float = 1.0
        self.boredom: float = 0.0

        # Connection/Loneliness tracking (for emotional depth)
        self.last_person_seen: float = now()  # Timestamp of last human presence
        self.time_alone: float = 0.0  # Accumulated seconds alone
        self.connection_relief: float = 0.0  # Spikes when person appears after isolation

        # Timing
        self.session_start: float = now()
        
        # Performance optimization: Cache expensive computations
        self._cached_identity_summary: str = ""
        self._identity_cache_time: float = 0.0
        self._cached_consciousness_stream: str = ""
        self._consciousness_cache_time: float = 0.0
        self._cache_duration: float = 30.0  # Cache for 30 seconds

    def observe(
        self,
        text: str,
        mood: float = 0.5,
        file: str = "",
        memory_type: str = "observation",
        derived_from: list[str] | None = None,
    ):
        ts = int(now())
        entry = {
            "timestamp": ts,
            "text": text.strip(),
            "mood": mood,
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

        self.update_focus_persistence(text)
        self.update_beliefs()
        self.estimate_novelty()
        self.update_boredom()
        self.fade_old_beliefs()
        self.update_loneliness(person_present=False)  # Default to alone unless specified
    
    def update_focus_persistence(self, caption: str) -> None:
        """Track how long we've been focused on the same object/concept."""
        # Extract primary object/concept from caption (simple heuristic)
        primary_focus = self.extract_primary_focus(caption)
        current_time = now()
        
        if primary_focus == self.current_focus_object:
            # Still focused on same thing - update duration and depth
            self.focus_duration = current_time - self.focus_start_time
            self.focus_depth += 1
        else:
            # Focus has shifted - reset tracking
            self.current_focus_object = primary_focus
            self.focus_start_time = current_time
            self.focus_duration = 0.0
            self.focus_depth = 1
    
    def extract_primary_focus(self, caption: str) -> str:
        """Extract the main object/concept from a caption."""
        # Simple extraction - look for key nouns
        words = caption.lower().split()
        
        # Common objects that might be primary focus
        objects = ['fish', 'figurine', 'sculpture', 'table', 'wall', 'room', 'desk', 'chair', 'lamp', 'book', 'cup', 'bottle', 'plant']
        
        for obj in objects:
            if obj in words:
                return obj
        
        # Fallback to first significant noun
        for word in words:
            if len(word) > 3 and word.isalpha():
                return word
        
        return "unknown"
    
    def get_identity_summary_cached(self) -> str:
        """Get identity summary with caching for performance."""
        current_time = now()
        if (current_time - self._identity_cache_time > self._cache_duration or 
            not self._cached_identity_summary):
            self._cached_identity_summary = self.get_identity_summary()
            self._identity_cache_time = current_time
        return self._cached_identity_summary
    
    def get_focus_context(self) -> str:
        """Get context about current focus persistence for prompts."""
        if self.focus_depth <= 1:
            return "new focus"
        elif self.focus_depth <= 3:
            return f"focused on {self.current_focus_object} for {self.focus_depth} moments"
        elif self.focus_depth <= 6:
            return f"been staring at {self.current_focus_object} for a while now"
        else:
            return f"fixated on {self.current_focus_object} for {self.focus_depth} observations - getting quite familiar"

    def update_loneliness(self, person_present: bool) -> None:
        """Update loneliness tracking based on current person presence."""
        current_time = now()
        
        if person_present:
            # Person is present - calculate relief if we were alone
            if self.time_alone > 60:  # Only feel relief if alone for more than 1 minute
                # Relief proportional to time alone (max relief = 1.0)
                self.connection_relief = min(1.0, self.time_alone / 600)  # Peak relief at 10 minutes alone
            else:
                self.connection_relief = 0.0
            
            # Reset loneliness tracking
            self.time_alone = 0.0
            self.last_person_seen = current_time
        else:
            # Person is not present - accumulate loneliness
            time_since_person = current_time - self.last_person_seen
            self.time_alone = time_since_person
            
            # Decay connection relief when alone
            self.connection_relief *= 0.95  # Gradual decay

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
                # Import here to avoid circular import
                from config.config import CLEAN_CAPTION_OUTPUT

                if not CLEAN_CAPTION_OUTPUT:
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
            time_since_seen = now_time - last_seen
            if time_since_seen > BELIEF_FADE_TIME:
                # More aggressive fade - doubt increases over time
                fade_amount = 0.05 + (time_since_seen / BELIEF_FADE_TIME) * 0.03
                data["strength"] -= fade_amount
                if data["strength"] < 0.1:  # Lower threshold for removal
                    faded.append(motif)
        for motif in faded:
            del self.beliefs[motif]
            self.belief_history.append(f"I'm no longer certain about {motif}... maybe I was mistaken.")

    def estimate_novelty(self) -> float:
        if len(self.memory_queue) < 2:
            self.novelty_score = 1.0
            return 1.0
        cur = self.memory_queue[-1]["text"].lower()
        prev = self.memory_queue[-2]["text"].lower()
        self.novelty_score = 1.0 if cur != prev else 0.0
        return self.novelty_score

    def update_boredom(self) -> None:
        self.boredom = min(1.0, self.boredom + 0.1) if self.novelty_score < 0.3 else max(0.0, self.boredom - 0.05)

    def get_clean_memory_snippets(self, k: int = 5) -> List[str]:
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
        """Add doubt markers to uncertain motifs and old beliefs."""
        words = re.findall(r"\b\w+\b", text)
        now_time = now()
        
        for word in sorted(set(words), key=len, reverse=True):
            w = word.lower()
            
            # Skip if word is already preceded by doubt marker
            doubt_pattern = rf"\b(?:maybe|I think I see|what might be|seems like)\s+{re.escape(word)}\b"
            if re.search(doubt_pattern, text, re.IGNORECASE):
                continue
            
            # Add doubt based on confidence
            if w in self.motif_confidence and self.motif_confidence[w] < CONFIDENCE_THRESHOLD:
                pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
                text = pattern.sub(r"maybe \1", text)
            
            # Add doubt for beliefs that haven't been seen recently
            elif w in self.beliefs:
                last_seen = self.motif_last_seen.get(w, 0)
                time_since_seen = now_time - last_seen
                if time_since_seen > BELIEF_FADE_TIME * 0.5:  # Start doubting at halfway to fade
                    pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
                    doubt_phrases = ["I think I see", "what might be", "seems like"]
                    import random
                    doubt_phrase = random.choice(doubt_phrases)
                    text = pattern.sub(rf"{doubt_phrase} \1", text)
                    
        return text

    def get_memory_entries_by_type(self, memory_type: str, limit: int = 5) -> list[dict]:
        return [entry for entry in reversed(self.memory_queue) if entry["type"] == memory_type][:limit]
