from __future__ import annotations

"""
captioner/memory.py
-------------------
MemoryMixin – session memory, identity tracking, and activation network integration.

Concept tracking and cross-session persistence are handled by SemanticMemory (ChromaDB).
Motif extraction and spaCy-based processing have been removed — concepts flow from
SemanticMemory.match_or_create_concepts() into the activation network.
"""

import os
import re
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.continuity import describe_duration, now

# constants shared with Captioner
MAX_MEMORY_ENTRIES: int = 30


class MemoryMixin:
    def __init__(self) -> None:
        # Experience queue (used by emotional journey, session snippets)
        self.memory_queue: Deque[dict] = deque(maxlen=MAX_MEMORY_ENTRIES)

        # Temporal spine
        self.boot_ts = getattr(self, "boot_ts", time.time()) if hasattr(self, "boot_ts") else time.time()
        self.timeline = getattr(self, "timeline", deque(maxlen=50000))
        self.day_stones = getattr(self, "day_stones", [])
        self._last_consolidation_day = getattr(self, "_last_consolidation_day", time.strftime("%Y-%m-%d"))

        # Person identity tracking
        self.known_people = getattr(self, "known_people", {})
        self.primary_person = getattr(self, "primary_person", None)

        # Self-understanding and environmental model
        self.self_model = getattr(
            self,
            "self_model",
            {
                "location_understanding": "unknown space",
                "environmental_certainty": 0.0,
            },
        )

        # Boredom (the activation network's one surviving output — retired
        # Aug 30 2026, memory-effectiveness-audit-aug30.md §1: novelty had
        # zero behavioral effect, edges never persisted, and the compression
        # boost inflated the concepts ledger's times_seen. What boredom needs
        # is a per-concept recent-attention level, kept here directly.)
        self._boredom: float = 0.0
        # concept_id -> (attention_level, last_seen_ts); +0.3 per sighting,
        # decayed 0.95 per 10s — the same boost/decay the network applied.
        self._concept_attention: Dict[str, Tuple[float, float]] = {}

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
        matched_concepts: Optional[List[Dict]] = None,
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

        # Boredom from concept metadata + recent attention (see __init__ note).
        # Weights preserved from the retired network's calculate_boredom:
        # new concepts and people barely count; the more familiar a concept,
        # the more its sustained recent attention reads as a stale scene.
        if matched_concepts:
            now_ts = float(ts)
            total_weighted = 0.0
            total_weight = 0.0
            for cd in matched_concepts:
                level, seen = self._concept_attention.get(cd["id"], (0.0, now_ts))
                level *= 0.95 ** ((now_ts - seen) / 10.0)
                level = min(1.0, level + 0.3)
                self._concept_attention[cd["id"]] = (level, now_ts)

                label = cd.get("label", "").lower()
                if cd.get("is_new", False):
                    weight = 0.1
                elif any(w in label for w in ["person", "someone", "man", "woman", "people", "figure", "sitting", "standing"]):
                    weight = 0.1
                else:
                    weight = min(1.0, cd.get("times_seen", 1) / 20.0)

                total_weighted += level * weight
                total_weight += weight
            self._boredom = total_weighted / total_weight if total_weight > 0 else 0.0
            if len(self._concept_attention) > 500:
                cutoff = now_ts - 3600
                self._concept_attention = {k: v for k, v in self._concept_attention.items() if v[1] > cutoff}

        # Apply temporal mood effects
        temporal_mood_modifier = self.get_temporal_mood_modifier()
        if temporal_mood_modifier != 0.0:
            entry["mood"] = max(0.0, min(1.0, mood + temporal_mood_modifier))
            if hasattr(self, "current_mood"):
                self.current_mood = entry["mood"]

    def get_temporal_mood_modifier(self) -> float:
        """Calculate mood modifier based on temporal stagnation effects."""
        if not hasattr(self, "true_session_start"):
            return 0.0

        session_duration = now() - self.true_session_start  # type: ignore
        stagnation_context = self.get_scene_stagnation_context()

        if not stagnation_context:
            return 0.0

        if session_duration > 14400:
            return -0.4
        elif session_duration > 7200:
            return -0.3
        elif session_duration > 3600:
            return -0.2
        elif session_duration > 1800:
            return -0.1
        else:
            return 0.0

    def get_current_session_memory_snippets(self, k: int = 3) -> List[str]:
        """Get only memories from the current session."""
        seen, out = set(), []
        for entry in reversed(self.memory_queue):
            if entry.get("timestamp", 0) >= self.session_start:
                cap = entry["text"]
                if cap not in seen and len(cap.strip()) > 10:
                    out.append(cap)
                    seen.add(cap)
                    if len(out) >= k:
                        break
        return list(reversed(out))

    def get_scene_stagnation_context(self) -> Optional[str]:
        """Detect if we've been staring at the same scene for too long."""
        if len(self.memory_queue) < 5:
            return None

        recent_observations = []
        cutoff_time = now() - 3600

        for entry in reversed(list(self.memory_queue)):
            if entry.get("timestamp", 0) > cutoff_time:
                recent_observations.append(entry["text"].lower())
            if len(recent_observations) >= 10:
                break

        if len(recent_observations) < 5:
            return None

        similar_count = 0
        keywords = ["pencil", "notebook", "laptop", "desk", "table", "workspace", "creative"]

        for obs in recent_observations:
            if any(keyword in obs for keyword in keywords):
                similar_count += 1

        if similar_count >= len(recent_observations) * 0.8:
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
            if entry.get("timestamp", 0) < self.session_start:
                cap = entry["text"]
                if cap not in seen and len(cap.strip()) > 15:
                    words = cap.split()
                    if len(words) >= 4:
                        candidates.append(cap)
                        seen.add(cap)

        selected = random.sample(candidates, min(k, len(candidates))) if candidates else []
        return selected

    def get_memory_entries_by_type(self, memory_type: str, limit: int = 5) -> list[dict]:
        return [entry for entry in reversed(self.memory_queue) if entry["type"] == memory_type][:limit]

    def temporal_prompt_lines(self) -> List[str]:
        """Generate temporal context lines for prompts with proper session awareness."""
        now_time = time.time()

        total_lifetime_hours = int((now_time - self.boot_ts) / 3600)
        days_alive = total_lifetime_hours // 24

        session_awake_hours = (now_time - self.session_start) / 3600
        if session_awake_hours < 1:
            session_awake_mins = int(session_awake_hours * 60)
            session_time = f"awake {session_awake_mins}m"
        else:
            session_awake_hours_int = int(session_awake_hours)
            session_time = f"awake {session_awake_hours_int}h"

        sleep_context = ""
        if hasattr(self, "_captioner_ref") and self._captioner_ref and hasattr(self._captioner_ref, "last_session_gap"):  # type: ignore
            gap = self._captioner_ref.last_session_gap  # type: ignore
            if gap is not None:
                if gap < 3600:
                    sleep_mins = int(gap / 60)
                    sleep_context = f"slept {sleep_mins}m"
                elif gap < 86400:
                    sleep_hours = int(gap / 3600)
                    sleep_context = f"slept {sleep_hours}h"
                else:
                    sleep_days = int(gap / 86400)
                    sleep_context = f"slept {sleep_days}d"

        last_person = None
        for e in reversed(self.timeline):
            if "person" in e.get("text", "").lower():
                last_person = now_time - e["ts"]
                break

        lp = f"last person {int(last_person / 3600)}h ago" if last_person else "no person yet"

        lines = [f"day {days_alive}", session_time, lp]
        if sleep_context:
            lines.insert(1, sleep_context)

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
            best_location = max(location_scores, key=location_scores.get)  # type: ignore
            confidence = location_scores[best_location] / 10.0

            if confidence > 0.1 or self.self_model["environmental_certainty"] < 0.3:
                self.self_model["location_understanding"] = best_location
                self.self_model["environmental_certainty"] = min(1.0, self.self_model["environmental_certainty"] + confidence * 0.1)

