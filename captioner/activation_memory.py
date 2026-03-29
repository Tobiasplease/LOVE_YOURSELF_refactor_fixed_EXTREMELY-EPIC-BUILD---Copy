"""
activation_memory.py
--------------------
Cognitive-inspired activation-spreading memory network.

Replaces shallow motif extraction with:
- Activation levels that decay over time
- Edge weights learned from co-occurrence
- Spreading activation for association-based recall
- Contextual memory retrieval with temporal framing
"""

import json
import os
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import spacy

from config.config import MOOD_SNAPSHOT_FOLDER
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.continuity import describe_time_gap

# Configuration
ACTIVATION_DECAY_RATE = 0.95
ACTIVATION_BOOST = 0.3
ACTIVATION_SPREAD_FACTOR = 0.25
EDGE_BUILD_INCREMENT = 0.05
EDGE_STRENGTH_THRESHOLD = 0.7
ACTIVATION_RECALL_THRESHOLD = 0.1
COMPRESSION_BOOST = 0.15
MAX_MEMORIES = 200
MAX_LONG_TERM_MEMORIES = 50  # Persist across sessions
EDGE_PERSISTENCE_FILE = os.path.join(MOOD_SNAPSHOT_FOLDER, "activation_edges.json")
LONG_TERM_MEMORY_FILE = os.path.join(MOOD_SNAPSHOT_FOLDER, "long_term_memories.json")
VISUALIZER_SNAPSHOT_FILE = os.path.join(MOOD_SNAPSHOT_FOLDER, "activation_snapshot.json")

# Load spaCy model (shared with memory.py)
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    _nlp = None

# Concept extraction filters
CONCEPT_BLACKLIST = {
    "thing", "way", "time", "lot", "bit", "kind", "sort", "part",
    "something", "nothing", "anything", "everything",
    "one", "ones", "other", "others",
    "place", "area", "space", "spot",
    "moment", "minute", "second", "hour", "day",
    "sense", "feeling", "thought", "idea",
}

MEANINGFUL_POS = {"NOUN", "PROPN"}

# Semantic categories for boredom weighting
# Static objects = boring when repeated (nothing changes)
STATIC_CONCEPTS = {
    "table", "desk", "chair", "wall", "floor", "ceiling", "door", "window",
    "shelf", "cabinet", "drawer", "screen", "monitor", "keyboard", "mouse",
    "lamp", "light", "book", "paper", "pen", "pencil", "notebook", "box",
    "cup", "mug", "bottle", "plant", "clock", "frame", "picture", "poster",
    "room", "office", "workspace", "corner", "surface", "clutter", "mess",
}

# Dynamic/emotional concepts = not boring when repeated (ongoing concern)
DYNAMIC_CONCEPTS = {
    "threat", "fear", "danger", "worry", "anxiety", "tension", "unease",
    "movement", "motion", "change", "shift", "sound", "noise",
    "presence", "arrival", "departure", "approach", "visit",
    "interest", "curiosity", "wonder", "excitement", "energy",
    "work", "activity", "action", "task", "project",
}

# Social concepts = engagement, not boredom
SOCIAL_CONCEPTS = {
    "person", "someone", "man", "woman", "human", "people", "visitor",
    "face", "hand", "body", "figure", "silhouette", "they", "them",
    "friend", "stranger", "guest", "observer", "watcher",
}


def extract_concepts(text: str) -> List[str]:
    """Extract meaningful concepts from text using spaCy."""
    if not _nlp or not text:
        return []

    doc = _nlp(text.lower())
    concepts = []

    for token in doc:
        if token.pos_ not in MEANINGFUL_POS:
            continue
        if token.is_stop or token.like_num:
            continue

        lemma = token.lemma_.lower().strip()

        if len(lemma) < 3:
            continue
        if lemma in CONCEPT_BLACKLIST:
            continue
        if lemma.endswith(("ing", "ness", "tion", "sion", "ment")):
            continue

        concepts.append(lemma)

    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT"}:
            concepts.append(ent.text.lower().strip())

    return list(set(concepts))


class ActivationNetwork:
    """Spreading activation network for concept associations."""

    def __init__(self):
        self.activations: Dict[str, float] = {}
        self.edges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.spatial_tags: Dict[str, str] = {}
        self.last_seen: Dict[str, float] = {}
        self.last_decay_time: float = time.time()
        self._lock = threading.RLock()  # RLock allows nested acquisition (prevents deadlock)

        # Tracking for visualizer
        self._last_novelty: float = 0.5
        self._last_boredom: float = 0.0

        self._load_edges()

    def observe(self, concepts: List[str], gaze_zone: str = "ahead") -> float:
        """Observe concepts, boost activation, build edges, return novelty."""
        if not concepts:
            return 0.5

        with self._lock:
            self._apply_decay()

            novelty_sum = 0.0
            now = time.time()

            for concept in concepts:
                old_activation = self.activations.get(concept, 0.0)
                novelty_sum += (1.0 - old_activation)

                new_activation = min(1.0, old_activation + ACTIVATION_BOOST)
                self.activations[concept] = new_activation
                self.last_seen[concept] = now

                if gaze_zone:
                    self.spatial_tags[concept] = gaze_zone

            for i, c1 in enumerate(concepts):
                for c2 in concepts[i + 1:]:
                    self.edges[c1][c2] = min(1.0, self.edges[c1][c2] + EDGE_BUILD_INCREMENT)
                    self.edges[c2][c1] = min(1.0, self.edges[c2][c1] + EDGE_BUILD_INCREMENT)

            self._spread_activation(concepts)

            novelty = novelty_sum / len(concepts)
            self._last_novelty = novelty  # Track for visualizer
            return novelty

    def _apply_decay(self):
        """Apply time-proportional decay to all activations."""
        now = time.time()
        elapsed = now - self.last_decay_time
        if elapsed < 1.0:
            return

        ticks = int(elapsed / 10.0)
        if ticks < 1:
            return

        decay_factor = ACTIVATION_DECAY_RATE ** ticks

        to_remove = []
        for concept in self.activations:
            self.activations[concept] *= decay_factor
            if self.activations[concept] < 0.01:
                to_remove.append(concept)

        for concept in to_remove:
            del self.activations[concept]

        self.last_decay_time = now

    def _spread_activation(self, source_concepts: List[str]):
        """Spread activation from source concepts through edges."""
        spread_updates = defaultdict(float)

        for source in source_concepts:
            source_activation = self.activations.get(source, 0)
            if source_activation < 0.1:
                continue

            for neighbor, weight in self.edges[source].items():
                if neighbor in source_concepts:
                    continue

                spread_amount = source_activation * weight * ACTIVATION_SPREAD_FACTOR
                spread_updates[neighbor] = max(spread_updates[neighbor], spread_amount)

        for concept, spread in spread_updates.items():
            current = self.activations.get(concept, 0)
            self.activations[concept] = min(1.0, current + spread)

    def get_activated_concepts(self, threshold: float = ACTIVATION_RECALL_THRESHOLD) -> List[Tuple[str, float]]:
        """Get concepts above activation threshold, sorted by activation."""
        with self._lock:
            activated = [(c, a) for c, a in self.activations.items() if a >= threshold]
            return sorted(activated, key=lambda x: -x[1])

    def get_strong_edges(self, threshold: float = EDGE_STRENGTH_THRESHOLD) -> List[Tuple[str, str, float]]:
        """Get edges above weight threshold."""
        with self._lock:
            strong = []
            seen = set()
            for c1, neighbors in self.edges.items():
                for c2, weight in neighbors.items():
                    if weight >= threshold:
                        pair = tuple(sorted([c1, c2]))
                        if pair not in seen:
                            strong.append((c1, c2, weight))
                            seen.add(pair)
            return sorted(strong, key=lambda x: -x[2])

    def get_beliefs(self) -> List[str]:
        """Get spatial associations - what things appear together.

        NOTE: These are NOT beliefs in the philosophical sense. They're just
        learned associations about what co-occurs in the visual field.
        Real beliefs would require deeper understanding.
        """
        strong = self.get_strong_edges(EDGE_STRENGTH_THRESHOLD)
        if not strong:
            return []

        # Return as simple spatial facts, not fake "beliefs"
        associations = []
        for c1, c2, weight in strong[:3]:
            associations.append(f"{c1} and {c2}")

        if associations:
            return [f"Often together: {', '.join(associations)}"]
        return []

    def get_spatial_belief(self, concept: str) -> Optional[str]:
        """Get spatial belief for a concept if consistently tagged."""
        zone = self.spatial_tags.get(concept)
        if not zone or zone == "ahead":
            return None

        zone_words = {
            "left": "to my left",
            "right": "to my right",
            "up": "above",
            "down": "below me",
        }
        direction = zone_words.get(zone, zone)
        return f"The {concept} is {direction}."

    def calculate_boredom(self, observed_concepts: List[str]) -> float:
        """Calculate boredom based on scene familiarity with semantic weighting.

        Static objects (table, desk) contribute more to boredom when repeated.
        Dynamic/emotional concepts (threat, fear) contribute less - they indicate
        ongoing concern, not boring sameness.
        Social concepts (person, someone) contribute least - engagement, not boredom.
        """
        if not observed_concepts:
            return 0.0

        with self._lock:
            total_weighted = 0.0
            total_weight = 0.0

            for concept in observed_concepts:
                activation = self.activations.get(concept, 0.0)

                # Weight by semantic category
                if concept in STATIC_CONCEPTS:
                    weight = 1.0  # Full boredom contribution
                elif concept in DYNAMIC_CONCEPTS:
                    weight = 0.2  # Low boredom - ongoing concern
                elif concept in SOCIAL_CONCEPTS:
                    weight = 0.1  # Minimal boredom - social engagement
                else:
                    weight = 0.5  # Default - moderate contribution

                total_weighted += activation * weight
                total_weight += weight

            boredom = total_weighted / total_weight if total_weight > 0 else 0.0
            self._last_boredom = boredom
            return boredom

    def boost_from_compression(self, compression_text: str):
        """Boost concepts that made it into compression (feedback loop)."""
        concepts = extract_concepts(compression_text)
        if not concepts:
            return

        with self._lock:
            for concept in concepts:
                current = self.activations.get(concept, 0)
                self.activations[concept] = min(1.0, current + COMPRESSION_BOOST)

            for i, c1 in enumerate(concepts):
                for c2 in concepts[i + 1:]:
                    self.edges[c1][c2] = min(1.0, self.edges[c1][c2] + 0.08)
                    self.edges[c2][c1] = min(1.0, self.edges[c2][c1] + 0.08)

        log_json_entry(
            LogType.INFO,
            {"message": "Compression feedback applied", "concepts": concepts, "count": len(concepts)},
            print_message=f"[🔄] Compression boosted: {', '.join(concepts[:3])}..."
        )

    def get_activation_trends(self) -> Dict[str, List[str]]:
        """Get concepts that are rising vs fading."""
        with self._lock:
            now = time.time()
            rising = []
            fading = []

            for concept, activation in self.activations.items():
                last = self.last_seen.get(concept, 0)
                age = now - last

                if age < 60 and activation > 0.5:
                    rising.append(concept)
                elif age > 300 and activation > 0.2:
                    fading.append(concept)

            return {"rising": rising[:5], "fading": fading[:5]}

    def _save_edges(self):
        """Persist edge weights to file."""
        try:
            os.makedirs(os.path.dirname(EDGE_PERSISTENCE_FILE), exist_ok=True)
            with self._lock:
                data = {
                    "edges": {k: dict(v) for k, v in self.edges.items()},
                    "spatial_tags": self.spatial_tags,
                    "saved_at": time.time(),
                }
            with open(EDGE_PERSISTENCE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log_json_entry(LogType.ERROR, {"message": f"Failed to save edges: {e}"})

    def _load_edges(self):
        """Load edge weights from file."""
        if not os.path.exists(EDGE_PERSISTENCE_FILE):
            return

        try:
            with open(EDGE_PERSISTENCE_FILE, "r") as f:
                data = json.load(f)

            with self._lock:
                for c1, neighbors in data.get("edges", {}).items():
                    for c2, weight in neighbors.items():
                        self.edges[c1][c2] = weight

                self.spatial_tags = data.get("spatial_tags", {})

            log_json_entry(
                LogType.INFO,
                {"message": "Loaded activation edges", "edge_count": sum(len(v) for v in self.edges.values())},
                print_message=f"[🧠] Loaded {sum(len(v) for v in self.edges.values())} learned associations"
            )
        except Exception as e:
            log_json_entry(LogType.ERROR, {"message": f"Failed to load edges: {e}"})

    def save_state(self):
        """Save network state (call on shutdown)."""
        self._save_edges()

    def save_visualizer_snapshot(self, memories: list = None, long_term_memories: list = None, extra_state: dict = None):
        """Save complete state snapshot for visualizer (called frequently).

        Args:
            memories: Recent session memories
            long_term_memories: Persisted long-term memories
            extra_state: Additional state dict with compression, desires, etc.
        """
        try:
            with self._lock:
                snapshot = {
                    "activations": dict(self.activations),
                    "edges": {k: dict(v) for k, v in self.edges.items()},
                    "spatial_tags": dict(self.spatial_tags),
                    "novelty": self._last_novelty,
                    "boredom": self._last_boredom,
                    "timestamp": time.time(),
                    "beliefs": self.get_beliefs(),
                    "trends": self.get_activation_trends(),
                }

            if memories:
                snapshot["memories"] = memories[-10:]

            if long_term_memories:
                snapshot["long_term_memories"] = long_term_memories[-MAX_LONG_TERM_MEMORIES:]

            # Merge extra state (compression, desires, etc.)
            if extra_state:
                snapshot.update(extra_state)

            os.makedirs(os.path.dirname(VISUALIZER_SNAPSHOT_FILE), exist_ok=True)
            with open(VISUALIZER_SNAPSHOT_FILE, "w") as f:
                json.dump(snapshot, f, indent=2)
        except Exception:
            pass  # Non-critical, don't log errors


class ContextualMemory:
    """Memory store with activation-based recall and long-term persistence."""

    def __init__(self, network: ActivationNetwork):
        self.network = network
        self.memories: List[dict] = []  # In-session memories
        self.long_term_memories: List[dict] = []  # Persisted across sessions
        self._lock = threading.RLock()  # RLock for safe nested acquisition

        self.mode_concept_boosts = {
            "relational": {"person", "someone", "man", "woman", "they", "human"},
            "workspace": {"desk", "paper", "tool", "notebook", "pen", "keyboard", "screen"},
            "introspective": {"feeling", "wonder", "thought", "quiet", "alone", "tired"},
            "observational": set(),
        }

        # Load persisted long-term memories
        self._load_long_term_memories()

    def store(self, text: str, concepts: List[str], gaze_zone: str, timestamp: float = None):
        """Store a memory with metadata."""
        if not text or not concepts:
            return

        timestamp = timestamp or time.time()

        with self._lock:
            self.memories.append({
                "text": text,
                "concepts": concepts,
                "zone": gaze_zone,
                "timestamp": timestamp,
            })

            if len(self.memories) > MAX_MEMORIES:
                self.memories = self.memories[-MAX_MEMORIES:]

    def recall(self, current_gaze: str = None, mode: str = None, k: int = 2) -> List[str]:
        """Recall relevant memories with temporal framing."""
        activated = self.network.get_activated_concepts()
        if not activated:
            return []

        activated_set = {c for c, _ in activated}
        activation_map = {c: a for c, a in activated}
        mode_boost_concepts = self.mode_concept_boosts.get(mode, set())
        now = time.time()

        with self._lock:
            scored = []

            for mem in self.memories:
                mem_concepts = set(mem["concepts"])
                overlap = mem_concepts & activated_set

                if not overlap:
                    continue

                overlap_score = sum(activation_map.get(c, 0) for c in overlap)
                spatial_boost = 1.5 if mem["zone"] == current_gaze else 1.0
                mode_boost = 1.3 if (mem_concepts & mode_boost_concepts) else 1.0

                age_hours = (now - mem["timestamp"]) / 3600
                recency = 1.0 / (1.0 + age_hours * 0.1)

                score = overlap_score * spatial_boost * mode_boost * recency
                scored.append((mem, score))

            scored.sort(key=lambda x: -x[1])
            return [self._format_memory(m) for m, _ in scored[:k]]

    def _format_memory(self, mem: dict) -> str:
        """Format memory with temporal context."""
        time_desc = describe_time_gap(mem["timestamp"])
        text = mem["text"][:100].strip()
        if not text.endswith((".", "!", "?")):
            text += "..."
        return f"I remember {time_desc}: \"{text}\""

    def get_recent_novel(self, threshold: float = 0.3, max_age: float = 300) -> List[str]:
        """Get concepts observed recently that were novel (low prior activation)."""
        now = time.time()
        novel = []

        with self._lock:
            for mem in reversed(self.memories[-10:]):
                if now - mem["timestamp"] > max_age:
                    continue

                for concept in mem["concepts"]:
                    activation = self.network.activations.get(concept, 0)
                    if activation < threshold and concept not in novel:
                        novel.append(concept)

        return novel[:5]

    def get_drawing_context(self) -> dict:
        """Get rich context for drawing prompts."""
        activated = self.network.get_activated_concepts(threshold=0.4)
        active_concepts = [c for c, _ in activated[:8]]
        novel = self.get_recent_novel()
        associations = self.network.get_strong_edges()[:5]
        trends = self.network.get_activation_trends()

        return {
            "active_concepts": active_concepts,
            "novel_observations": novel,
            "associations": [(c1, c2) for c1, c2, _ in associations],
            "rising": trends.get("rising", []),
            "fading": trends.get("fading", []),
        }

    def format_drawing_context(self) -> str:
        """Format drawing context as natural language."""
        ctx = self.get_drawing_context()
        parts = []

        if ctx["active_concepts"]:
            concepts_str = ", ".join(ctx["active_concepts"][:5])
            parts.append(f"On your mind: {concepts_str}")

        if ctx["novel_observations"]:
            novel_str = ", ".join(ctx["novel_observations"][:3])
            parts.append(f"Something new: {novel_str}")

        if ctx["associations"]:
            assoc = ctx["associations"][0]
            parts.append(f"You know {assoc[0]} and {assoc[1]} go together")

        if ctx["fading"]:
            fading_str = ctx["fading"][0]
            parts.append(f"{fading_str.title()} is fading from attention")

        return ". ".join(parts) + "." if parts else ""

    # === LONG-TERM MEMORY PERSISTENCE ===

    def _load_long_term_memories(self):
        """Load persisted long-term memories from disk."""
        if not os.path.exists(LONG_TERM_MEMORY_FILE):
            return

        try:
            with open(LONG_TERM_MEMORY_FILE, "r") as f:
                data = json.load(f)

            with self._lock:
                self.long_term_memories = data.get("memories", [])

            log_json_entry(
                LogType.INFO,
                {"message": "Loaded long-term memories", "count": len(self.long_term_memories)},
                print_message=f"[🧠] Loaded {len(self.long_term_memories)} long-term memories"
            )
        except Exception as e:
            log_json_entry(LogType.ERROR, {"message": f"Failed to load long-term memories: {e}"})

    def _save_long_term_memories(self):
        """Save long-term memories to disk."""
        try:
            os.makedirs(os.path.dirname(LONG_TERM_MEMORY_FILE), exist_ok=True)
            with self._lock:
                data = {
                    "memories": self.long_term_memories[-MAX_LONG_TERM_MEMORIES:],
                    "saved_at": time.time(),
                }
            with open(LONG_TERM_MEMORY_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log_json_entry(LogType.ERROR, {"message": f"Failed to save long-term memories: {e}"})

    def promote_to_long_term(self, text: str, concepts: List[str], significance: str = "observation"):
        """Promote a significant memory to long-term storage.

        Includes deduplication to prevent similar memories from accumulating.
        """
        if not text or len(text) < 20:
            return

        with self._lock:
            # Deduplication: check if similar memory already exists (first 60 chars)
            text_prefix = text[:60].lower()
            for existing in self.long_term_memories:
                if existing.get("text", "")[:60].lower() == text_prefix:
                    log_json_entry(LogType.DEBUG, {"message": f"[LT-MEM] Skipping duplicate: {text[:40]}..."})
                    return  # Skip duplicate

            # Also check concept overlap - if >80% same concepts, likely duplicate
            concept_set = set(concepts[:5])
            for existing in self.long_term_memories[-10:]:  # Check recent 10
                existing_concepts = set(existing.get("concepts", []))
                if concept_set and existing_concepts:
                    overlap = len(concept_set & existing_concepts) / max(len(concept_set), 1)
                    if overlap > 0.8:
                        log_json_entry(LogType.DEBUG, {"message": f"[LT-MEM] Skipping concept overlap ({overlap:.0%}): {text[:40]}..."})
                        return  # Skip near-duplicate

            # LOG: Memory being promoted
            log_json_entry(
                LogType.DEBUG,
                {
                    "message": f"[LT-MEM] PROMOTED: {significance}",
                    "text": text[:100],
                    "concepts": concepts[:5],
                    "total_long_term": len(self.long_term_memories) + 1,
                },
                print_message=f"[📅 LT-MEM] Promoted ({significance}): {text[:50]}..."
            )

            self.long_term_memories.append({
                "text": text[:200],
                "concepts": concepts[:5],
                "significance": significance,
                "timestamp": time.time(),
            })

            if len(self.long_term_memories) > MAX_LONG_TERM_MEMORIES:
                self.long_term_memories = self.long_term_memories[-MAX_LONG_TERM_MEMORIES:]

        self._save_long_term_memories()

    def recall_long_term(self, k: int = 3) -> List[str]:
        """Recall relevant long-term memories based on current activation."""
        activated = self.network.get_activated_concepts()
        if not activated:
            with self._lock:
                # Return most recent if no activation context
                if self.long_term_memories:
                    recent = self.long_term_memories[-1]
                    return [self._format_long_term_memory(recent)]
            return []

        activated_set = {c for c, _ in activated}
        now = time.time()

        with self._lock:
            scored = []

            for mem in self.long_term_memories:
                mem_concepts = set(mem.get("concepts", []))
                overlap = mem_concepts & activated_set

                if not overlap:
                    continue

                overlap_score = len(overlap)
                age_days = (now - mem["timestamp"]) / 86400

                # Older memories score higher for long-term (they're more "distant")
                age_bonus = min(1.0 + age_days * 0.05, 2.0)
                score = overlap_score * age_bonus

                scored.append((mem, score))

            scored.sort(key=lambda x: -x[1])
            return [self._format_long_term_memory(m) for m, _ in scored[:k]]

    def _format_long_term_memory(self, mem: dict) -> str:
        """Format long-term memory with temporal context."""
        timestamp = mem.get("timestamp", time.time())
        age_days = (time.time() - timestamp) / 86400

        if age_days < 1:
            time_desc = "earlier today"
        elif age_days < 2:
            time_desc = "yesterday"
        elif age_days < 7:
            time_desc = f"{int(age_days)} days ago"
        elif age_days < 30:
            weeks = int(age_days / 7)
            time_desc = f"{weeks} week{'s' if weeks > 1 else ''} ago"
        else:
            months = int(age_days / 30)
            time_desc = f"{months} month{'s' if months > 1 else ''} ago"

        text = mem["text"][:100].strip()
        if not text.endswith((".", "!", "?")):
            text += "..."

        if mem.get("significance") == "awakening":
            return f"I remember when I first woke {time_desc}: \"{text}\""
        if mem.get("significance") == "discovery":
            return f"I once discovered: \"{text}\""
        return f"I remember from {time_desc}: \"{text}\""

    def save_state(self):
        """Save memory state (call on shutdown)."""
        self._save_long_term_memories()


_network: Optional[ActivationNetwork] = None
_memory: Optional[ContextualMemory] = None


def get_activation_network() -> ActivationNetwork:
    """Get or create the global activation network."""
    global _network
    if _network is None:
        _network = ActivationNetwork()
    return _network


def get_contextual_memory() -> ContextualMemory:
    """Get or create the global contextual memory."""
    global _memory, _network
    if _memory is None:
        _memory = ContextualMemory(get_activation_network())
    return _memory


def should_include_context(context_type: str, mode: str = "introspective") -> bool:
    """Determine if a context type should be included based on activation state and mode.

    This is the KEY gating function - instead of including ALL context types,
    we only include what's currently relevant based on:
    - Current prompt mode (relational, observational, introspective, workspace, restless)
    - Activation network state (what concepts are active)
    - Novelty/boredom levels

    Context types:
    - relational: person presence awareness
    - pressure: boredom/stagnation hints
    - curiosity: novelty/change hints
    - drawing: current drawing activity
    - paper: paper presence/absence
    - motifs: recurring themes
    - beliefs: learned associations
    - long_term: long-term memories
    - story: compression narrative
    - mood: emotional state

    Returns:
        True if context should be included, False otherwise
    """
    network = get_activation_network()
    activated = {c for c, a in network.get_activated_concepts(threshold=0.3)}
    boredom = network._last_boredom
    novelty = getattr(network, "_last_novelty", 0.5)

    # Always include: gaze (embodiment), continuity (2 captions), identity
    if context_type in ("gaze", "continuity", "identity"):
        return True

    # Mode-specific gating
    if context_type == "relational":
        return mode == "relational" or bool(activated & SOCIAL_CONCEPTS)

    if context_type == "pressure":
        return mode == "restless" or boredom > 0.5

    if context_type == "curiosity":
        return mode == "observational" or novelty > 0.6

    if context_type == "drawing":
        return bool(activated & {"draw", "drawing", "paper", "pen", "line", "arm", "mark"})

    if context_type == "paper":
        # Only when drawing is blocked (checked by caller)
        return True

    if context_type == "motifs":
        # Only in introspective mode when there ARE persistent motifs
        return mode == "introspective"

    if context_type == "beliefs":
        # Only in introspective mode when we have strong associations
        if mode != "introspective":
            return False
        return len(network.get_strong_edges(threshold=0.7)) > 0

    if context_type == "long_term":
        # Only when something triggers recall (activation overlap)
        # This is expensive to check, so default False unless introspective
        return mode == "introspective" and boredom > 0.4

    if context_type == "story":
        # Narrative state only useful for introspective or restless modes
        return mode in ("introspective", "restless")

    if context_type == "mood":
        # Mood only when emotionally relevant (high boredom, social, or introspective)
        return mode in ("introspective", "relational") or boredom > 0.6

    return False


def generate_state_summary() -> str:
    """Generate state summary based on activation network.

    Focuses on inner state and learned associations, NOT listing visible objects
    (the model can already see those). Used for simplified caption prompts.
    """
    network = get_activation_network()
    top_concepts = network.get_activated_concepts(threshold=0.3)

    if not top_concepts:
        return ""

    social = [c for c, a in top_concepts if c in SOCIAL_CONCEPTS]
    dynamic = [c for c, a in top_concepts if c in DYNAMIC_CONCEPTS]
    boredom = network._last_boredom
    beliefs = network.get_beliefs()

    # Social presence is notable
    if social:
        return "Someone is present."

    # Dynamic activity is notable
    if dynamic:
        return f"Something is happening: {', '.join(dynamic[:2])}."

    # High boredom = stagnation
    if boredom > 0.7:
        return "Everything feels the same."

    # Moderate boredom = familiar
    if boredom > 0.4:
        return "Familiar."

    # Moderate boredom but no beliefs yet
    if boredom > 0.4:
        return "The scene is familiar."

    # Low boredom, no beliefs = genuinely new, no summary needed
    return ""


def observe_and_store(text: str, gaze_zone: str = "ahead") -> Tuple[float, float]:
    """Convenience function: observe concepts and store memory. Returns (novelty, boredom)."""
    concepts = extract_concepts(text)
    if not concepts:
        return 0.5, 0.0

    network = get_activation_network()
    memory = get_contextual_memory()

    novelty = network.observe(concepts, gaze_zone)
    boredom = network.calculate_boredom(concepts)
    memory.store(text, concepts, gaze_zone)

    # NOTE: Long-term memory promotion is handled by CaptionAgent.observe() with stricter criteria
    # Do NOT promote here to avoid duplicates

    # Save snapshot for visualizer (real-time state sharing)
    try:
        network.save_visualizer_snapshot(
            memories=[{"text": m.get("text", "")[:100], "timestamp": m.get("timestamp", 0)} for m in memory.memories[-10:]],
            long_term_memories=memory.long_term_memories,
        )
    except Exception:
        pass  # Non-critical

    return novelty, boredom


def recall_for_prompt(gaze_zone: str = "ahead", mode: str = "introspective") -> str:
    """Convenience function: get formatted memory recall for prompt injection."""
    memory = get_contextual_memory()
    recalls = memory.recall(current_gaze=gaze_zone, mode=mode, k=2)
    return "\n".join(recalls) if recalls else ""


def get_beliefs() -> List[str]:
    """Get LLM-generated beliefs from compression introspection.

    These are ACTUAL beliefs generated by the model reflecting on its observations,
    not heuristic co-occurrence extraction.

    Falls back to spatial associations if no introspection has occurred.
    """
    try:
        from captioner.context_compression import context_compressor
        belief = context_compressor.get_current_belief()
        if belief:
            return [belief]
    except Exception:
        pass

    # Fallback: spatial associations (honest about being co-occurrence data)
    return get_activation_network().get_beliefs()


def get_desires() -> List[str]:
    """Get LLM-generated desires from compression introspection.

    These are ACTUAL desires generated by the model reflecting on its state,
    not keyword extraction.
    """
    try:
        from captioner.context_compression import context_compressor
        desire = context_compressor.get_current_desire()
        if desire:
            return [desire]
    except Exception:
        pass
    return []


def get_long_term_memories(k: int = 2) -> str:
    """Get formatted long-term memories relevant to current activation."""
    memory = get_contextual_memory()
    recalls = memory.recall_long_term(k=k)
    return "\n".join(recalls) if recalls else ""


def promote_memory(text: str, significance: str = "observation"):
    """Promote a significant thought to long-term memory."""
    concepts = extract_concepts(text)
    memory = get_contextual_memory()
    memory.promote_to_long_term(text, concepts, significance)


def boost_from_compression(compression_text: str):
    """Apply compression feedback loop."""
    get_activation_network().boost_from_compression(compression_text)


def save_state():
    """Save activation network and memory state (call on shutdown)."""
    if _network:
        _network.save_state()
    if _memory:
        _memory.save_state()


def get_activation_summary_for_compression() -> dict:
    """Get formatted activation data for feeding into compression/introspection prompts.

    Returns dict with:
    - concepts_str: Top 3 activated concepts as comma-separated string
    - long_term_memory: Most relevant long-term memory (or None)
    - association_str: Strongest learned association (or None)
    - trends: dict with 'rising' and 'fading' concept lists
    - boredom: Current boredom level (0-1)
    - novelty: Current novelty level (0-1)
    """
    network = get_activation_network()
    memory = get_contextual_memory()

    result = {
        "concepts_str": "",
        "long_term_memory": None,
        "association_str": None,
        "trends": {"rising": [], "fading": []},
        "boredom": network._last_boredom,
        "novelty": getattr(network, "_last_novelty", 0.5),
    }

    top_concepts = network.get_activated_concepts(threshold=0.4)[:3]
    if top_concepts:
        result["concepts_str"] = ", ".join([c for c, _ in top_concepts])

    long_term_recalls = memory.recall_long_term(k=1)
    if long_term_recalls:
        result["long_term_memory"] = long_term_recalls[0]

    strong_edges = network.get_strong_edges(threshold=0.7)[:1]
    if strong_edges:
        result["association_str"] = f"{strong_edges[0][0]} and {strong_edges[0][1]}"

    result["trends"] = network.get_activation_trends()

    return result


def get_activation_summary_for_introspection() -> dict:
    """Get richer activation data for introspection prompts.

    Returns dict with:
    - concepts: Top 5 activated concepts
    - trends: Rising and fading concepts
    - long_term_memories: Up to 2 relevant long-term memories
    - boredom: Current boredom level
    - novelty: Current novelty level
    """
    network = get_activation_network()
    memory = get_contextual_memory()

    top_concepts = network.get_activated_concepts(threshold=0.3)[:5]
    trends = network.get_activation_trends()
    long_term = memory.recall_long_term(k=2)

    return {
        "concepts": [c for c, _ in top_concepts],
        "trends": trends,
        "long_term_memories": long_term,
        "boredom": network._last_boredom,
        "novelty": getattr(network, "_last_novelty", 0.5),
    }


def save_comprehensive_snapshot(agent=None):
    """Save comprehensive state snapshot for visualizer with all accumulated data.

    Gathers state from:
    - Activation network (activations, edges, novelty, boredom, beliefs, trends)
    - Session memories and long-term memories
    - Context compression (baseline, history, session info, sentiment)
    - Agent self-model (desires, beliefs)

    Args:
        agent: Optional CaptionAgent for accessing desires and self-model
    """
    network = get_activation_network()
    memory = get_contextual_memory()

    extra_state = {}

    # === COMPRESSION DATA ===
    try:
        from captioner.context_compression import context_compressor

        extra_state["compression"] = {
            "baseline_context": context_compressor.get_baseline_context(),
            "session_info": context_compressor.get_current_session_info(),
            "history": context_compressor.get_compression_history(max_entries=5),
        }

        sentiment = context_compressor.get_latest_sentiment_analysis()
        if sentiment:
            extra_state["compression"]["sentiment"] = sentiment

        # LLM-generated introspective state with full history (for visualizer)
        full_identity = context_compressor.get_full_identity()
        if full_identity["current_desire"] or full_identity["current_belief"]:
            extra_state["identity"] = {
                "current_desire": full_identity["current_desire"],
                "current_belief": full_identity["current_belief"],
                "desire_history": full_identity["desire_history"][-5:],  # Last 5 for viz
                "belief_history": full_identity["belief_history"][-5:],  # Last 5 for viz
                "last_updated": full_identity["last_updated"],
                "introspection_count": full_identity["introspection_count"],
            }
    except Exception:
        pass

    # === AGENT SELF-MODEL (desires, beliefs) ===
    if agent:
        try:
            if hasattr(agent, "self_model") and agent.self_model:
                self_model = agent.self_model
                extra_state["self_model"] = {
                    "desires": self_model.get("desires", [])[-5:],
                    "beliefs": list(self_model.get("beliefs", {}).keys())[:10],
                    "identity_statements": self_model.get("identity", [])[-3:] if "identity" in self_model else [],
                }
        except Exception:
            pass

    # Save with all data
    try:
        network.save_visualizer_snapshot(
            memories=[{"text": m.get("text", "")[:100], "timestamp": m.get("timestamp", 0)} for m in memory.memories[-10:]],
            long_term_memories=memory.long_term_memories,
            extra_state=extra_state,
        )
    except Exception:
        pass  # Non-critical
