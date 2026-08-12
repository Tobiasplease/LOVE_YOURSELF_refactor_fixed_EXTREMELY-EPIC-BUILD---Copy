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
EDGE_PERSISTENCE_FILE = os.path.join(MOOD_SNAPSHOT_FOLDER, "activation_edges.json")
VISUALIZER_SNAPSHOT_FILE = os.path.join(MOOD_SNAPSHOT_FOLDER, "activation_snapshot.json")


def _truncate_at_sentence(text: str, max_len: int) -> str:
    """Truncate text at a sentence boundary, falling back to word boundary with ellipsis."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    for i in range(min(len(text), max_len), 15, -1):
        if text[i - 1] in ".!?":
            return text[:i]
    truncated = text[:max_len].rsplit(" ", 1)[0]
    return truncated.rstrip(",.;:") + "..."


class ActivationNetwork:
    """Spreading activation network for concept associations.

    Nodes are ChromaDB concept IDs (e.g. "concept_1713500000_3") rather than
    bare words. The concept_labels cache maps IDs to human-readable names
    for the visualizer and prompt formatting.
    """

    def __init__(self):
        self.activations: Dict[str, float] = {}
        self.edges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.spatial_tags: Dict[str, str] = {}
        self.last_seen: Dict[str, float] = {}
        self.last_decay_time: float = time.time()
        self._lock = threading.RLock()

        # Concept ID → human-readable label (e.g. "Cracked ceiling above the desk")
        self.concept_labels: Dict[str, str] = {}

        # Tracking for visualizer
        self._last_novelty: float = 0.5
        self._last_boredom: float = 0.0

        self._load_edges()

    def observe(self, concept_ids: List[str], gaze_zone: str = "ahead", concept_data: List[Dict] = None) -> float:
        """Observe concepts, boost activation, build edges, return novelty.

        Args:
            concept_ids: ChromaDB concept IDs to observe
            gaze_zone: current gaze direction
            concept_data: optional list of dicts with "id", "label", "times_seen", "is_new"
                          from SemanticMemory.match_or_create_concepts()
        """
        if not concept_ids:
            return 0.5

        # Update label cache from concept data
        if concept_data:
            for cd in concept_data:
                self.concept_labels[cd["id"]] = cd["label"]

        with self._lock:
            self._apply_decay()

            novelty_sum = 0.0
            now = time.time()

            for concept_id in concept_ids:
                old_activation = self.activations.get(concept_id, 0.0)
                novelty_sum += (1.0 - old_activation)

                new_activation = min(1.0, old_activation + ACTIVATION_BOOST)
                self.activations[concept_id] = new_activation
                self.last_seen[concept_id] = now

                if gaze_zone:
                    self.spatial_tags[concept_id] = gaze_zone

            for i, c1 in enumerate(concept_ids):
                for c2 in concept_ids[i + 1:]:
                    self.edges[c1][c2] = min(1.0, self.edges[c1][c2] + EDGE_BUILD_INCREMENT)
                    self.edges[c2][c1] = min(1.0, self.edges[c2][c1] + EDGE_BUILD_INCREMENT)

            self._spread_activation(concept_ids)

            novelty = novelty_sum / len(concept_ids)
            self._last_novelty = novelty
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
        """Get spatial associations — what concepts appear together.

        Uses concept labels for human-readable output.
        """
        strong = self.get_strong_edges(EDGE_STRENGTH_THRESHOLD)
        if not strong:
            return []

        associations = []
        for c1, c2, weight in strong[:3]:
            label1 = self.concept_labels.get(c1, c1)
            label2 = self.concept_labels.get(c2, c2)
            associations.append(f"{label1} and {label2}")

        if associations:
            return [f"Often together: {', '.join(associations)}"]
        return []

    def calculate_boredom(self, concept_data: List[Dict] = None) -> float:
        """Calculate boredom from concept metadata rather than word lists.

        Args:
            concept_data: list of dicts with "id", "label", "times_seen", "is_new"
                          from SemanticMemory.match_or_create_concepts()

        Boredom scoring:
        - High times_seen + high activation = stale scene (boring)
        - New concepts = not boring
        - Person concepts = engagement, not boring
        """
        if not concept_data:
            return 0.0

        with self._lock:
            total_weighted = 0.0
            total_weight = 0.0

            for cd in concept_data:
                concept_id = cd["id"]
                activation = self.activations.get(concept_id, 0.0)
                times_seen = cd.get("times_seen", 1)
                is_new = cd.get("is_new", False)
                label = cd.get("label", "").lower()

                # New concepts are not boring at all
                if is_new:
                    weight = 0.1
                # Person concepts = engagement
                elif any(w in label for w in ["person", "someone", "man", "woman", "people", "figure", "sitting", "standing"]):
                    weight = 0.1
                # Highly familiar = boring when repeated
                elif times_seen > 20:
                    weight = 1.0
                # Moderately familiar
                else:
                    weight = min(1.0, times_seen / 20.0)

                total_weighted += activation * weight
                total_weight += weight

            boredom = total_weighted / total_weight if total_weight > 0 else 0.0
            self._last_boredom = boredom
            return boredom

    def boost_from_compression(self, concept_ids: List[str]):
        """Boost concepts that made it into compression (feedback loop).

        Args:
            concept_ids: ChromaDB concept IDs to boost
        """
        if not concept_ids:
            return

        with self._lock:
            for concept_id in concept_ids:
                current = self.activations.get(concept_id, 0)
                self.activations[concept_id] = min(1.0, current + COMPRESSION_BOOST)

            for i, c1 in enumerate(concept_ids):
                for c2 in concept_ids[i + 1:]:
                    self.edges[c1][c2] = min(1.0, self.edges[c1][c2] + 0.08)
                    self.edges[c2][c1] = min(1.0, self.edges[c2][c1] + 0.08)

        labels = [self.concept_labels.get(cid, cid) for cid in concept_ids[:3]]
        log_json_entry(
            LogType.INFO,
            {"message": "Compression feedback applied", "concept_ids": concept_ids, "count": len(concept_ids)},
            print_message=f"[🔄] Compression boosted: {', '.join(labels)}..."
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
        """Persist edge weights and concept labels to file."""
        try:
            os.makedirs(os.path.dirname(EDGE_PERSISTENCE_FILE), exist_ok=True)
            with self._lock:
                data = {
                    "edges": {k: dict(v) for k, v in self.edges.items()},
                    "spatial_tags": self.spatial_tags,
                    "concept_labels": self.concept_labels,
                    "saved_at": time.time(),
                }
            with open(EDGE_PERSISTENCE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log_json_entry(LogType.ERROR, {"message": f"Failed to save edges: {e}"})

    def _load_edges(self):
        """Load edge weights from file. Migrates old bare-word edges to concept IDs."""
        if not os.path.exists(EDGE_PERSISTENCE_FILE):
            return

        try:
            with open(EDGE_PERSISTENCE_FILE, "r") as f:
                data = json.load(f)

            edges = data.get("edges", {})

            # Migration: if any key lacks "concept_" prefix, it's old bare-word data — discard
            if edges and not any(k.startswith("concept_") for k in list(edges.keys())[:5]):
                print("[🧠] Detected old bare-word edges — clearing for concept ID migration")
                return

            with self._lock:
                for c1, neighbors in edges.items():
                    for c2, weight in neighbors.items():
                        self.edges[c1][c2] = weight

                self.spatial_tags = data.get("spatial_tags", {})
                self.concept_labels = data.get("concept_labels", {})

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

    def save_visualizer_snapshot(self, memories: list = None, extra_state: dict = None):
        """Save complete state snapshot for visualizer (called frequently).

        Args:
            memories: Recent session memories
            extra_state: Additional state dict with compression, desires, etc.
        """
        try:
            with self._lock:
                snapshot = {
                    "activations": dict(self.activations),
                    "edges": {k: dict(v) for k, v in self.edges.items()},
                    "spatial_tags": dict(self.spatial_tags),
                    "concept_labels": dict(self.concept_labels),
                    "novelty": self._last_novelty,
                    "boredom": self._last_boredom,
                    "timestamp": time.time(),
                    "beliefs": self.get_beliefs(),
                    "trends": self.get_activation_trends(),
                }

            if memories:
                snapshot["memories"] = memories[-10:]

            if extra_state:
                snapshot.update(extra_state)

            os.makedirs(os.path.dirname(VISUALIZER_SNAPSHOT_FILE), exist_ok=True)
            with open(VISUALIZER_SNAPSHOT_FILE, "w") as f:
                json.dump(snapshot, f, indent=2)
        except Exception:
            pass


class ContextualMemory:
    """In-session memory store with activation-based recall.

    Cross-session persistence is handled by ChromaDB (SemanticMemory).
    This class only manages the current session's observations.
    """

    def __init__(self, network: ActivationNetwork):
        self.network = network
        self.memories: List[dict] = []
        self._lock = threading.RLock()

    def store(self, text: str, concept_ids: List[str], gaze_zone: str, timestamp: float = None):
        """Store a memory with concept IDs."""
        if not text or not concept_ids:
            return

        timestamp = timestamp or time.time()

        with self._lock:
            self.memories.append({
                "text": text,
                "concept_ids": concept_ids,
                "zone": gaze_zone,
                "timestamp": timestamp,
            })

            if len(self.memories) > MAX_MEMORIES:
                self.memories = self.memories[-MAX_MEMORIES:]

    def recall(self, current_gaze: str = None, mode: str = None, k: int = 2) -> List[str]:
        """Recall relevant memories based on activation overlap."""
        activated = self.network.get_activated_concepts()
        if not activated:
            return []

        activated_set = {c for c, _ in activated}
        activation_map = {c: a for c, a in activated}
        now = time.time()

        with self._lock:
            scored = []

            for mem in self.memories:
                mem_concepts = set(mem.get("concept_ids", mem.get("concepts", [])))
                overlap = mem_concepts & activated_set

                if not overlap:
                    continue

                overlap_score = sum(activation_map.get(c, 0) for c in overlap)
                spatial_boost = 1.5 if mem["zone"] == current_gaze else 1.0

                age_hours = (now - mem["timestamp"]) / 3600
                recency = 1.0 / (1.0 + age_hours * 0.1)

                score = overlap_score * spatial_boost * recency
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

    def get_drawing_context(self) -> dict:
        """Get rich context for drawing prompts using concept labels."""
        activated = self.network.get_activated_concepts(threshold=0.4)
        active_labels = [self.network.concept_labels.get(c, c) for c, _ in activated[:8]]
        associations = self.network.get_strong_edges()[:5]
        trends = self.network.get_activation_trends()

        return {
            "active_concepts": active_labels,
            "associations": [
                (self.network.concept_labels.get(c1, c1), self.network.concept_labels.get(c2, c2))
                for c1, c2, _ in associations
            ],
            "rising": [self.network.concept_labels.get(c, c) for c in trends.get("rising", [])],
            "fading": [self.network.concept_labels.get(c, c) for c in trends.get("fading", [])],
        }

    def format_drawing_context(self) -> str:
        """Format drawing context as natural language."""
        ctx = self.get_drawing_context()
        parts = []

        if ctx["active_concepts"]:
            concepts_str = ", ".join(ctx["active_concepts"][:5])
            parts.append(f"On your mind: {concepts_str}")

        if ctx["associations"]:
            assoc = ctx["associations"][0]
            parts.append(f"You know {assoc[0]} and {assoc[1]} go together")

        if ctx["fading"]:
            fading_str = ctx["fading"][0]
            parts.append(f"{fading_str} is fading from attention")

        return ". ".join(parts) + "." if parts else ""


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
    """Whether a context type belongs in the prompt for this mode.

    Only "beliefs" and "story" are requested at runtime (prompts.py). The old
    pressure/curiosity/relational/mood types — and the "restless" mode that
    never had a producer — were torn out Aug 12 2026.
    """
    network = get_activation_network()

    if context_type == "beliefs":
        if mode != "introspective":
            return False
        return len(network.get_strong_edges(threshold=0.7)) > 0

    if context_type == "story":
        return mode == "introspective"

    return False



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



def boost_from_compression(compression_text: str):
    """Apply compression feedback loop by matching compression text to known concepts."""
    try:
        from captioner.semantic_memory import get_semantic_memory
        matched = get_semantic_memory().match_or_create_concepts(compression_text)
        if matched:
            concept_ids = [c["id"] for c in matched]
            get_activation_network().boost_from_compression(concept_ids)
    except Exception:
        pass


def save_state():
    """Save activation network state (call on shutdown)."""
    if _network:
        _network.save_state()


def get_activation_summary_for_compression() -> dict:
    """Get formatted activation data for compression/introspection prompts.

    Returns concept labels (not IDs) for human-readable prompt injection.
    """
    network = get_activation_network()

    result = {
        "concepts_str": "",
        "association_str": None,
    }

    top_concepts = network.get_activated_concepts(threshold=0.4)[:3]
    if top_concepts:
        labels = [network.concept_labels.get(c, c) for c, _ in top_concepts]
        result["concepts_str"] = ", ".join(labels)

    strong_edges = network.get_strong_edges(threshold=0.7)[:1]
    if strong_edges:
        l1 = network.concept_labels.get(strong_edges[0][0], strong_edges[0][0])
        l2 = network.concept_labels.get(strong_edges[0][1], strong_edges[0][1])
        result["association_str"] = f"{l1} and {l2}"

    return result


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
            extra_state=extra_state,
        )
    except Exception:
        pass
