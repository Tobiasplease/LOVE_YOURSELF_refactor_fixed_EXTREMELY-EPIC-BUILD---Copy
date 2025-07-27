from __future__ import annotations

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

        self.update_beliefs()
        self.estimate_novelty()
        self.update_boredom()
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
        
        # Define what we actually want to track
        MEANINGFUL_MOTIFS = {
            # Visual objects that matter
            'camera', 'phone', 'computer', 'laptop', 'monitor', 'screen', 'desk', 'table', 'chair',
            'window', 'door', 'mirror', 'artwork', 'picture', 'book', 'paper', 'notebook',
            'pen', 'pencil', 'brush', 'lamp', 'plant', 'bottle', 'glass', 'cup',
            'clothes', 'shirt', 'glasses', 'watch', 'necklace',
            
            # Spaces and environments
            'kitchen', 'bedroom', 'office', 'studio', 'workspace', 'bathroom', 'garden',
            'restaurant', 'cafe', 'library',
            
            # Activities worth tracking
            'writing', 'reading', 'drawing', 'painting', 'cooking', 'eating', 'drinking',
            'typing', 'working', 'studying', 'creating', 'cleaning',
            
            # Creative/professional
            'artist', 'writer', 'programmer', 'designer', 'musician', 'teacher', 'student',
            'painting', 'sketch', 'drawing', 'music', 'violin', 'piano', 'guitar',
            'canvas', 'easel', 'palette', 'instrument',
            
            # Specific environmental qualities
            'cluttered', 'organized', 'messy', 'bright', 'shadowy', 'sunlit', 'colorful',
            'vintage', 'modern', 'minimalist',
            
            # Natural elements
            'sunlight', 'shadow', 'reflection', 'tree', 'flowers', 'water', 'laundry'
        }
        
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
            elif (count > 30 and 
                  len(motif) > 4 and
                  motif not in {'thought', 'about', 'appears', 'might', 'there', 'where', 'their', 'seems', 
                               'front', 'right', 'various', 'perhaps', 'notice', 'important', 'minute', 
                               'moment', 'sense', 'thoughts', 'remember', 'visible', 'individual', 'young', 
                               'short', 'neutral', 'natural', 'quiet', 'personal', 'domestic', 'indoor', 
                               'activity', 'items', 'object', 'scene', 'image', 'quite', 'really', 'pretty',
                               'little', 'small', 'large', 'good', 'nice', 'simple', 'basic', 'special',
                               'general', 'normal', 'usual', 'common', 'different', 'similar', 'other',
                               'feeling', 'looking', 'getting', 'making', 'doing', 'being', 'having'} and
                  not motif.endswith(('ing', 'ed', 'ly', 'ness', 'tion', 'sion'))):  # Not verbs/adverbs/abstracts
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
        
        MEANINGFUL_CATEGORIES = {
            # Objects and tools
            'camera', 'phone', 'computer', 'laptop', 'monitor', 'screen', 'desk', 'table', 'chair',
            'window', 'door', 'mirror', 'picture', 'artwork', 'book', 'paper', 'pen', 'pencil',
            'lamp', 'light', 'plant', 'flower', 'bottle', 'glass', 'cup', 'plate', 'bowl',
            'clothes', 'shirt', 'jacket', 'hat', 'glasses', 'watch', 'necklace', 'ring',
            
            # Places and environments  
            'kitchen', 'bedroom', 'office', 'studio', 'workspace', 'bathroom', 'garden', 
            'restaurant', 'cafe', 'library', 'park', 'street', 'building',
            
            # Activities and skills
            'writing', 'reading', 'drawing', 'painting', 'cooking', 'eating', 'drinking',
            'typing', 'working', 'studying', 'playing', 'exercising', 'sleeping', 'talking',
            'listening', 'watching', 'thinking', 'creating', 'building', 'cleaning',
            
            # Creative and professional terms
            'artist', 'writer', 'programmer', 'designer', 'musician', 'teacher', 'student',
            'painting', 'sketch', 'drawing', 'music', 'instrument', 'violin', 'piano', 'guitar',
            'canvas', 'brush', 'pencil', 'marker', 'easel', 'palette',
            
            # Specific descriptive qualities (not generic emotions)
            'cluttered', 'organized', 'messy', 'clean', 'bright', 'shadowy', 'sunlit',
            'colorful', 'monochrome', 'vintage', 'modern', 'rustic', 'elegant', 'minimalist',
            
            # Natural elements
            'sunlight', 'shadow', 'reflection', 'tree', 'leaves', 'flowers', 'water', 'sky'
        }
        
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
            
            # Extract only very specific, concrete nouns that represent things we can see/interact with
            CONCRETE_NOUN_HINTS = {
                'desk', 'chair', 'table', 'window', 'door', 'wall', 'floor', 'ceiling',
                'kitchen', 'bedroom', 'bathroom', 'office', 'studio', 'workspace',
                'computer', 'laptop', 'phone', 'camera', 'monitor', 'screen',
                'book', 'paper', 'pen', 'pencil', 'brush', 'canvas', 'easel',
                'plant', 'flower', 'tree', 'sunlight', 'shadow', 'reflection'
            }
            
            for token in doc:
                if (token.pos_ in {"NOUN", "PROPN"} and 
                    len(token.text) > 3 and 
                    not token.is_stop and 
                    not token.like_num):
                    
                    lemma = token.lemma_.lower()
                    
                    # Only extract if it's in our concrete nouns set or ends with tool/device patterns
                    if (lemma in CONCRETE_NOUN_HINTS or 
                        (lemma.endswith(('er', 'or')) and len(lemma) > 5)):
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
            if now_time - last_seen > BELIEF_FADE_TIME:
                data["strength"] -= 0.02
                if data["strength"] < 0.2:
                    faded.append(motif)
        for motif in faded:
            del self.beliefs[motif]
            self.belief_history.append(f"I feel less attached to {motif} lately.")

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
        words = re.findall(r"\b\w+\b", text)
        for word in sorted(set(words), key=len, reverse=True):
            w = word.lower()
            if w in self.motif_confidence and self.motif_confidence[w] < CONFIDENCE_THRESHOLD:
                pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
                text = pattern.sub(r"maybe \\1", text)
        return text

    def get_memory_entries_by_type(self, memory_type: str, limit: int = 5) -> list[dict]:
        return [entry for entry in reversed(self.memory_queue) if entry["type"] == memory_type][:limit]
