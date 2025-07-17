from __future__ import annotations
"""
captioner/memory.py
-------------------
MemoryMixin – short‑/long‑term queues, novelty & boredom tracking, motif stats,
and snapshot cleanup.  Captioner imports:

    from .memory import MemoryMixin, CAPTION_SAVE_THRESHOLD
"""

import time, re, os, glob
from collections import deque
from typing import Deque, List, Tuple, Set

# constants shared with Captioner
MAX_MEMORY_ENTRIES: int = 30
BOREDOM_THRESHOLD: float = 0.7
CAPTION_SAVE_THRESHOLD: float = 0.3

CaptionTuple = Tuple[int, str, float, str]    # (ts, caption, mood, file)

class MemoryMixin:
    def __init__(self) -> None:
        # queues
        self.memory_queue: Deque[CaptionTuple] = deque(maxlen=MAX_MEMORY_ENTRIES)
        self.long_memory:   List[CaptionTuple] = []
        # motif tracking
        self.meta_memory = {"motifs": {}, "last_caption_objects": set()}
        self.motif_presence: dict[str, dict[str, float | int | str]] = {}
        # novelty / boredom
        self.novelty_score: float = 1.0
        self.boredom: float = 0.0
        self.memory_anchor: str = ""
        # timer
        self.session_start: float = time.time()

    # ---------- absorb_detection ----------
    def absorb_detection(self, labels: list[str], timestamp: float | None = None):
        """Store recurring objects as symbolic motifs with confidence and count."""
        timestamp = timestamp or time.time()
        for label in labels:
            entry = self.motif_presence.setdefault(label, {
                "count": 0,
                "confidence": 0.0,
                "last_seen": 0
            })
            entry["count"] += 1
            entry["confidence"] = min(entry["confidence"] + 0.1, 1.0)
            entry["last_seen"] = timestamp

    # ---------- motif helpers ----------
    def update_meta_memory(self, caption: str) -> None:
        text = caption.lower()
        words = re.findall(r"\\b\\w+\\b", text)
        tracked = ["pizza","laptop","lamp","curtain","chair","person","monitor"]
        seen_now: Set[str] = set()
        now = time.time()
        for word in tracked:
            if word in words:
                seen_now.add(word)
                self.meta_memory["motifs"][word] = self.meta_memory["motifs"].get(word,0)+1
                if word not in self.motif_presence:
                    self.motif_presence[word] = {"first_seen":now,"last_seen":now,"count":1}
                else:
                    rec = self.motif_presence[word]
                    rec["last_seen"] = now
                    rec["count"] += 1
        self.meta_memory["last_caption_objects"] = seen_now

    def format_detected_labels(self) -> str:
        try:
            from perception.detection_memory import DetectionMemory  # type: ignore
        except ImportError:
            return ""
        labels = DetectionMemory.get_labels()
        if not labels:
            return ""
        out = []
        for l in labels:
            out.append("I think that's a person" if "person" in l.lower() else f"That might be a {l}")
        return " ".join(out)

    # ---------- novelty / boredom ----------
    def estimate_novelty(self) -> float:
        if len(self.memory_queue) < 2:
            self.novelty_score = 1.0
            return 1.0
        cur = self.memory_queue[-1][1].lower()
        prev = self.memory_queue[-2][1].lower()
        self.novelty_score = 1.0 if cur != prev else 0.0
        return self.novelty_score

    def update_boredom(self) -> None:
        self.boredom = min(1.0, self.boredom + 0.1) if self.novelty_score < 0.3 else max(0.0, self.boredom - 0.05)

    # ---------- snippets & compression ----------
    def get_clean_memory_snippets(self, k:int=5) -> List[str]:
        seen, out = set(), []
        for ts, cap, *_ in reversed(self.memory_queue):
            if cap not in seen:
                out.append(cap); seen.add(cap)
                if len(out) >= k: break
        return list(reversed(out))

    def compress_long_memory(self) -> None:
        recent = [e[1] for e in self.long_memory[-MAX_MEMORY_ENTRIES:]]
        if recent:
            self.memory_anchor = "So far, I have observed: " + ", ".join(recent[:5]) + "."

    # ---------- cleanup ----------
    @staticmethod
    def cleanup_snapshots(folder:str, limit:int=100) -> None:
        files = sorted(glob.glob(os.path.join(folder, "*.jpg")), key=os.path.getctime)
        if len(files) > limit:
            for f in files[:-limit]:
                try: os.remove(f)
                except OSError: pass
