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
        self.extract_semantic_motifs(text)

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

    def extract_motifs_from_caption(self, caption: str):
        words = re.findall(r"\b\w+\b", caption.lower())
        now_time = now()
        for word in words:
            if len(word) > 3:
                self.motif_counter[word] += 1
                if word not in self.motif_first_seen:
                    self.motif_first_seen[word] = now_time
                self.motif_last_seen[word] = now_time
                self.current_motifs.add(word)
                if word not in self.motif_confidence:
                    self.motif_confidence[word] = 0.4
                    self.motif_confirmed[word] = False

    def extract_semantic_motifs(self, caption: str):
        if _nlp is None:
            return
        doc = _nlp(caption)
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN", "ADJ"} and len(token.text) > 2:
                self.absorb_motif(token.lemma_)

    def get_motif_certainty(self, motif: str) -> float:
        return self.motif_confidence.get(motif.lower(), 0.0)

    def is_motif_confirmed(self, motif: str) -> bool:
        return self.motif_confirmed.get(motif.lower(), False)

    def update_beliefs(self):
        now_time = now()
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
