"""
Unified pattern recognition engine combining motif extraction and novelty detection.
"""

from __future__ import annotations
import time
import spacy
import re
from typing import Dict, List, Set, Tuple
from collections import Counter, deque
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType

# Load spacy model once
nlp = spacy.load("en_core_web_sm")

# Curated meaningful motifs (from memory.py)
MEANINGFUL_MOTIFS = {
    "desk",
    "keyboard",
    "screen",
    "monitor",
    "computer",
    "laptop",
    "workspace",
    "chair",
    "window",
    "light",
    "shadow",
    "books",
    "papers",
    "coffee",
    "mug",
    "plant",
    "headphones",
    "mouse",
    "phone",
    "tablet",
    "notebook",
    "pen",
    "wall",
    "door",
    "ceiling",
    "floor",
    "shelf",
    "bottle",
    "glass",
    "clock",
    "lamp",
    "cable",
    "cord",
    "hand",
    "face",
    "person",
    "eye",
    "hair",
    "shirt",
    "clothing",
    "skin",
    "movement",
    "typing",
    "writing",
    "reading",
    "sitting",
    "standing",
    "walking",
    "looking",
    "thinking",
    "focus",
    "concentration",
    "attention",
    "awareness",
    "presence",
    "calm",
    "energy",
    "stillness",
}

MOTIF_BLACKLIST = {
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "up",
    "about",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "among",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "who",
    "whom",
    "whose",
    "there",
    "here",
    "now",
    "then",
    "today",
    "yesterday",
    "tomorrow",
    "always",
    "never",
    "sometimes",
    "often",
    "usually",
    "rarely",
    "hardly",
    "almost",
    "quite",
    "very",
    "really",
    "pretty",
    "rather",
    "somewhat",
    "fairly",
    "extremely",
    "incredibly",
    "absolutely",
    "completely",
    "totally",
    "entirely",
    "perfectly",
    "exactly",
    "precisely",
    "generally",
    "basically",
    "essentially",
    "particularly",
    "especially",
    "specifically",
    "probably",
    "possibly",
    "maybe",
    "perhaps",
    "definitely",
    "certainly",
    "surely",
    "obviously",
    "clearly",
    "apparently",
    "seemingly",
    "supposedly",
    "allegedly",
    "actually",
    "really",
    "truly",
    "indeed",
    "however",
    "moreover",
    "furthermore",
    "therefore",
    "thus",
    "hence",
    "consequently",
    "meanwhile",
    "otherwise",
    "nevertheless",
    "nonetheless",
    "besides",
    "instead",
    "rather",
    "alternatively",
    "similarly",
    "likewise",
    "accordingly",
    "subsequently",
    "previously",
    "currently",
    "recently",
    "lately",
    "soon",
    "eventually",
    "finally",
    "initially",
    "originally",
    "basically",
    "mainly",
    "mostly",
    "largely",
    "primarily",
    "chiefly",
    "particularly",
    "especially",
    "specifically",
    "notably",
    "remarkably",
    "surprisingly",
    "interestingly",
    "fortunately",
    "unfortunately",
    "hopefully",
    "thankfully",
    "regrettably",
    "sadly",
    "happily",
    "luckily",
}


class PatternRecognitionEngine:
    """Unified engine for motif extraction, novelty detection, and temporal tracking."""

    def __init__(self):
        self.recent_motifs = deque(maxlen=5)  # Last 5 caption motif sets for novelty calculation
        self.current_motifs: Set[str] = set()
        self.motif_history: List[Tuple[float, Set[str]]] = []  # (timestamp, motifs) pairs
        self.last_caption = ""

    def analyze_caption(self, caption: str) -> Dict:
        """
        Unified analysis of caption for motifs and novelty.
        Returns comprehensive pattern analysis.
        """
        timestamp = time.time()

        # Single SpaCy parse for efficiency
        doc = nlp(caption)

        # Extract current motifs
        current_motifs = self._extract_motifs_from_doc(doc)

        # Calculate novelty based on motif overlap (more semantic than word-based)
        novelty = self._calculate_motif_novelty(current_motifs)

        # Determine new vs recurring motifs for temporal awareness
        new_motifs = current_motifs - self.current_motifs
        recurring_motifs = current_motifs & self.current_motifs

        # Update state
        self.recent_motifs.append(current_motifs.copy())
        self.current_motifs = current_motifs
        self.motif_history.append((timestamp, current_motifs.copy()))
        self.last_caption = caption

        # Log temporal motif information
        self._log_motif_temporality(timestamp, new_motifs, recurring_motifs, novelty)

        return {
            "motifs": list(current_motifs),
            "new_motifs": list(new_motifs),
            "recurring_motifs": list(recurring_motifs),
            "novelty": novelty,
            "motif_count": len(current_motifs),
            "timestamp": timestamp,
        }

    def _extract_motifs_from_doc(self, doc) -> Set[str]:
        """Extract meaningful motifs using SpaCy analysis."""
        motifs = set()

        # 1. Noun chunks (filtered)
        for chunk in doc.noun_chunks:
            clean_chunk = chunk.text.lower().strip()
            if self._is_significant_motif(clean_chunk):
                motifs.add(clean_chunk)

        # 2. Named entities (meaningful ones)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                clean_ent = ent.text.lower().strip()
                if self._is_significant_motif(clean_ent):
                    motifs.add(clean_ent)

        # 3. Adjective + noun combinations
        for token in doc:
            if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                combo = f"{token.text.lower()} {token.head.text.lower()}"
                if self._is_significant_motif(combo):
                    motifs.add(combo)

        # 4. Individual meaningful nouns
        for token in doc:
            if token.pos_ == "NOUN" and not token.is_stop:
                word = token.lemma_.lower()
                if word in MEANINGFUL_MOTIFS:
                    motifs.add(word)

        return motifs

    def _is_significant_motif(self, motif: str) -> bool:
        """Determine if a motif is significant enough to track."""
        motif = motif.strip().lower()

        # Skip if empty, too short, or blacklisted
        if not motif or len(motif) < 3 or motif in MOTIF_BLACKLIST:
            return False

        # Skip pure punctuation or numbers
        if re.match(r"^[^\w\s]+$", motif) or motif.isdigit():
            return False

        # Accept if in meaningful motifs list
        if motif in MEANINGFUL_MOTIFS:
            return True

        # Accept if it looks like a concrete noun (not too abstract)
        if len(motif) > 4 and not motif.endswith(("ing", "ed", "ly", "ness", "tion", "sion", "ment", "ful", "less")):
            return True

        return False

    def _calculate_motif_novelty(self, current_motifs: Set[str]) -> float:
        """Calculate novelty based on motif overlap with recent observations."""
        if not self.recent_motifs:
            return 0.3  # Moderate novelty for first observation

        # Calculate overlap with recent motif sets
        total_overlap = 0
        total_possible = 0

        for recent_set in self.recent_motifs:
            if recent_set:  # Skip empty sets
                intersection = len(current_motifs & recent_set)
                union = len(current_motifs | recent_set)
                if union > 0:
                    overlap = intersection / union
                    total_overlap += overlap
                    total_possible += 1

        if total_possible == 0:
            return 0.3

        # Average similarity across recent observations
        avg_similarity = total_overlap / total_possible

        # Convert to novelty (inverse relationship)
        novelty = 1.0 - avg_similarity

        # Scale and cap novelty to prevent extreme values
        return min(0.5, novelty * 0.7)

    def _log_motif_temporality(self, timestamp: float, new_motifs: Set[str], recurring_motifs: Set[str], novelty: float):
        """Log motif temporal information for better memory/perception distinction."""

        # Only log when there are meaningful changes
        if new_motifs or len(recurring_motifs) > 2:
            log_entry = {
                "timestamp": timestamp,
                "new_motifs": list(new_motifs),
                "recurring_motifs": list(recurring_motifs),
                "novelty": round(novelty, 3),
                "total_motifs": len(new_motifs) + len(recurring_motifs),
            }

            # Create human-readable message
            message_parts = []
            if new_motifs:
                message_parts.append(f"New: {', '.join(list(new_motifs)[:3])}")
            if recurring_motifs:
                message_parts.append(f"Recurring motifs: {', '.join(list(recurring_motifs)[:3])}")

            message = f"[🔍] {' | '.join(message_parts)} (novelty: {novelty:.2f})"

            log_json_entry(LogType.MOTIF, log_entry, print_message=message)

    def get_motif_summary(self, minutes_back: int = 5) -> Dict:
        """Get summary of motifs seen in the last N minutes."""
        cutoff_time = time.time() - (minutes_back * 60)
        recent_entries = [(ts, motifs) for ts, motifs in self.motif_history if ts > cutoff_time]

        if not recent_entries:
            return {"period_motifs": [], "frequency": {}, "timeline": []}

        # Aggregate motifs from the period
        all_motifs = set()
        frequency = Counter()

        for _, motifs in recent_entries:
            all_motifs.update(motifs)
            frequency.update(motifs)

        return {
            "period_motifs": list(all_motifs),
            "frequency": dict(frequency.most_common(10)),
            "timeline": [(ts, list(motifs)) for ts, motifs in recent_entries[-5:]],
        }

    def clear_old_history(self, hours_back: int = 2):
        """Clean up old motif history to prevent memory bloat."""
        cutoff_time = time.time() - (hours_back * 3600)
        self.motif_history = [(ts, motifs) for ts, motifs in self.motif_history if ts > cutoff_time]
