"""
captioner/semantic_memory.py
----------------------------
Persistent concept-level memory using ChromaDB.

Gives the machine a growing relationship with objects, people, and spatial
features it encounters across sessions. Each "concept" is something the machine
has noticed more than once — a sign, a person, a piece of ceiling damage.

Design principles:
  - No LLM calls in this module. All classification via heuristics.
  - One data store (ChromaDB with metadata), no separate SQLite.
  - Injection format optimized for legibility: one clean sentence per concept.
  - Graceful when empty — adds nothing to prompts until it has real signal.
"""

import os
import re
import time
import threading
from typing import Dict, List, Optional, Tuple

import chromadb

from config.config import MOOD_SNAPSHOT_FOLDER

# --- Storage paths ---
CHROMADB_PATH = os.path.join(MOOD_SNAPSHOT_FOLDER, "chromadb")

# --- Thresholds ---
SIMILARITY_THRESHOLD = 0.5  # Cosine distance: 0 = identical, 1 = orthogonal. 0.5 keeps matches tight.
PERSON_SIMILARITY_THRESHOLD = 0.7  # Looser threshold for person concepts — Qwen describes people inconsistently
NOVELTY_MIN_LENGTH = 15  # Perceptions shorter than this are too vague to store
DUPLICATE_DISTANCE = 0.3  # Below this distance, observations are near-identical
MAX_OBSERVATIONS_PER_CONCEPT = 10  # Keep only the N most recent observations per concept

# --- Familiarity tiers ---
TIER_NEW = 3  # seen < 3 times
TIER_FAMILIAR = 10  # seen 3-10 times
# above 10 = very familiar

# --- Reflection settings ---
REFLECTION_INTERVAL_SECONDS = 600  # Run reflection check every 10 minutes
REFLECTION_MIN_NEW_OBSERVATIONS = 4  # Need at least N new observations to trigger
REFLECTION_MIN_CONCEPT_FAMILIARITY = 5  # Concept must be seen at least N times
REFLECTION_MAX_PER_CYCLE = 1  # Max reflections per worker pass (avoid LLM contention)

# Singleton
_instance = None
_lock = threading.Lock()


def get_semantic_memory() -> "SemanticMemory":
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SemanticMemory()
    return _instance


class SemanticMemory:
    def __init__(self):
        os.makedirs(CHROMADB_PATH, exist_ok=True)
        self._client = chromadb.PersistentClient(path=CHROMADB_PATH)

        # Concepts: each document is the canonical description of a concept.
        # Metadata carries structured state (times_seen, last_seen, etc.)
        self._concepts = self._client.get_or_create_collection(
            name="concepts",
            metadata={"hnsw:space": "cosine"},
        )

        # Observations: individual thoughts/perceptions linked to concepts.
        # Each document is one observation. Metadata links it to a concept_id.
        self._observations = self._client.get_or_create_collection(
            name="observations",
            metadata={"hnsw:space": "cosine"},
        )

        self._obs_counter = self._observations.count()
        self._session_id = f"session_{int(time.time())}"

        # Reflection worker state
        self._reflection_thread: Optional[threading.Thread] = None
        self._reflection_stop = threading.Event()
        self._last_reflection_check = 0.0

        # Count existing reflections for logging
        try:
            refl_count = len(self._observations.get(where={"type": "reflection"})["ids"])
        except Exception:
            refl_count = 0
        print(f"[SEMANTIC] Loaded: {self._concepts.count()} concepts, {self._observations.count()} observations ({refl_count} reflections)")

        # Start the reflection worker thread
        self._start_reflection_worker()

    # ------------------------------------------------------------------
    # Two-phase API: match concepts first, store observations later
    # ------------------------------------------------------------------

    def match_or_create_concepts(self, perception: str) -> List[Dict]:
        """Phase 1: Match perception against known concepts, creating new ones if noteworthy.

        Returns MULTIPLE matching concepts — this is critical for the activation
        network to build edges between co-occurring concepts. A perception like
        "person sitting at desk under fluorescent light" should activate the
        person concept, the desk concept, AND the light concept.

        Returns list of dicts:
        [{"id": "concept_xyz", "label": "Red sign on wall", "times_seen": 5, "is_new": False}, ...]
        """
        if not perception or len(perception.strip()) < NOVELTY_MIN_LENGTH:
            return []

        cleaned = self._clean_perception(perception)
        matched = []

        # Query for multiple matches within threshold
        if self._concepts.count() > 0:
            n_results = min(5, self._concepts.count())
            is_person = self._mentions_person(cleaned)
            threshold = PERSON_SIMILARITY_THRESHOLD if is_person else SIMILARITY_THRESHOLD

            results = self._concepts.query(
                query_texts=[cleaned],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            if results["ids"][0]:
                seen_ids = set()
                for i in range(len(results["ids"][0])):
                    dist = results["distances"][0][i]
                    if dist > threshold:
                        continue
                    cid = results["ids"][0][i]
                    if cid in seen_ids:
                        continue
                    seen_ids.add(cid)

                    meta = results["metadatas"][0][i]
                    doc = results["documents"][0][i]

                    self._bump_concept(cid)
                    matched.append({
                        "id": cid,
                        "label": doc,
                        "times_seen": meta.get("times_seen", 0) + 1,
                        "is_new": False,
                    })

        # If no matches and perception is noteworthy, create a new concept
        if not matched and self._is_noteworthy(cleaned):
            # Guard against person fragmentation
            if self._mentions_person(cleaned):
                existing_person = self._find_any_person_concept()
                if existing_person is not None:
                    self._bump_concept(existing_person["id"])
                    return [{
                        "id": existing_person["id"],
                        "label": existing_person["name"],
                        "times_seen": existing_person["times_seen"] + 1,
                        "is_new": False,
                    }]

            concept_id = f"concept_{int(time.time())}_{self._concepts.count()}"
            canonical_name = self._extract_canonical_name(cleaned)
            now = time.time()

            self._concepts.add(
                ids=[concept_id],
                documents=[canonical_name],
                metadatas=[{
                    "times_seen": 1,
                    "first_seen": now,
                    "last_seen": now,
                    "session_count": 1,
                    "last_session": self._session_id,
                    "last_observation": "",
                }],
            )
            print(f"[SEMANTIC] New concept: '{canonical_name}' (id={concept_id})")

            matched.append({
                "id": concept_id,
                "label": canonical_name,
                "times_seen": 1,
                "is_new": True,
            })

        return matched

    def get_current_thread(self, top_concept_ids: List[str], limit: int = 2) -> List[Dict]:
        """Get the current thread of attention for prompt injection.

        For each top-activated concept, returns its label and most recent
        reflection or observation — giving the LLM a thread to continue.

        Returns: [{"label": "Cracked ceiling", "last_thought": "The cracks seem...", "thought_type": "reflection"}, ...]
        """
        if not top_concept_ids:
            return []

        threads = []
        for concept_id in top_concept_ids[:limit]:
            # Get concept label
            concept_data = self._concepts.get(ids=[concept_id], include=["documents", "metadatas"])
            if not concept_data["ids"]:
                continue

            label = concept_data["documents"][0]
            meta = concept_data["metadatas"][0]

            # Try to get most recent reflection first, fall back to observation
            thought_text, thought_type = self._get_relevant_observation(label, concept_id=concept_id)

            threads.append({
                "label": label,
                "times_seen": meta.get("times_seen", 0),
                "last_thought": thought_text if thought_text else "",
                "thought_type": thought_type if thought_type else "",
            })

        return threads

    # ------------------------------------------------------------------
    # Core: match perception to known concepts
    # ------------------------------------------------------------------

    @staticmethod
    def _mentions_person(text: str) -> bool:
        """Check if text describes a person."""
        t = text.lower()
        return any(w in t for w in [
            "person", "someone", "individual", "man", "woman",
            "people", "figure", "they are", "he is", "she is",
            "sitting", "standing", "typing", "working", "looking",
            "seated", "focused on", "wearing",
        ])

    def match_perception(self, perception: str) -> Optional[Dict]:
        """Find the best matching concept for a perception string.

        Returns dict with keys: id, name, times_seen, last_observation, distance
        or None if no match above threshold.

        Person descriptions get a looser threshold since Qwen describes the same
        person differently each cycle. If the perception mentions a person, we
        also check all results (not just top-1) for an existing person concept.
        """
        if not perception or len(perception.strip()) < NOVELTY_MIN_LENGTH:
            return None

        if self._concepts.count() == 0:
            return None

        is_person = self._mentions_person(perception)
        threshold = PERSON_SIMILARITY_THRESHOLD if is_person else SIMILARITY_THRESHOLD

        n_results = min(3, self._concepts.count()) if is_person else 1
        results = self._concepts.query(
            query_texts=[perception],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return None

        # For person perceptions, prefer an existing person concept even if
        # it's not the top similarity hit — this prevents creating duplicate
        # person concepts when Qwen describes the same person differently.
        if is_person and n_results > 1:
            for i in range(len(results["ids"][0])):
                dist = results["distances"][0][i]
                doc = results["documents"][0][i]
                if dist <= threshold and self._mentions_person(doc):
                    meta = results["metadatas"][0][i]
                    return {
                        "id": results["ids"][0][i],
                        "name": doc,
                        "times_seen": meta.get("times_seen", 1),
                        "last_seen": meta.get("last_seen", 0),
                        "session_count": meta.get("session_count", 1),
                        "last_observation": meta.get("last_observation", ""),
                        "distance": dist,
                    }

        # Standard path: take top result if within threshold
        distance = results["distances"][0][0]
        if distance > threshold:
            return None

        meta = results["metadatas"][0][0]
        return {
            "id": results["ids"][0][0],
            "name": results["documents"][0][0],
            "times_seen": meta.get("times_seen", 1),
            "last_seen": meta.get("last_seen", 0),
            "session_count": meta.get("session_count", 1),
            "last_observation": meta.get("last_observation", ""),
            "distance": distance,
        }

    # ------------------------------------------------------------------
    # Core: update or create concept from perception + monologue
    # ------------------------------------------------------------------

    def after_perception(self, perception: str) -> Optional[str]:
        """Called after vision model perceives. Returns a memory injection line
        for the monologue prompt, or None if nothing relevant.

        This is the main integration point — it decides what the machine
        "remembers" about what it's currently seeing.
        """
        if not perception or len(perception.strip()) < NOVELTY_MIN_LENGTH:
            return None

        # Clean vision-model artifacts before matching
        cleaned = self._clean_perception(perception)
        match = self.match_perception(cleaned)
        if match is None:
            return None

        # Update the concept: bump times_seen, update last_seen
        self._bump_concept(match["id"])

        return self._format_injection(match, perception=cleaned)

    def after_monologue(self, perception: str, monologue: str, matched_concepts: List[Dict] = None):
        """Called after monologue generation. Stores observations under matched concepts.

        Args:
            perception: what the vision model saw (grounded)
            monologue: what nemo said about it (interpreted)
            matched_concepts: pre-matched concepts from match_or_create_concepts() — skips re-matching
        """
        if not perception or len(perception.strip()) < NOVELTY_MIN_LENGTH:
            return
        if not monologue or monologue.strip() in ("...", "Processing..."):
            return

        # If we have pre-matched concepts from Phase 1, use them directly
        if matched_concepts:
            for concept in matched_concepts:
                concept_id = concept["id"]
                # Relevance check: does the monologue relate to this concept?
                try:
                    relevance = self._concepts.query(
                        query_texts=[monologue],
                        n_results=1,
                        include=["distances"],
                    )
                    if relevance["distances"][0] and relevance["distances"][0][0] < SIMILARITY_THRESHOLD:
                        self._store_observation(concept_id, monologue)
                        self._update_last_observation(concept_id, monologue)
                except Exception:
                    self._store_observation(concept_id, monologue)
                    self._update_last_observation(concept_id, monologue)
            return

        # Fallback: no pre-matched concepts, do the old matching path
        perception = self._clean_perception(perception)
        match = self.match_perception(perception)

        if match is not None:
            try:
                relevance = self._concepts.query(
                    query_texts=[monologue],
                    n_results=1,
                    include=["distances"],
                )
                if relevance["distances"][0] and relevance["distances"][0][0] < SIMILARITY_THRESHOLD:
                    self._store_observation(match["id"], monologue)
                    self._update_last_observation(match["id"], monologue)
            except Exception:
                self._store_observation(match["id"], monologue)
                self._update_last_observation(match["id"], monologue)
        else:
            if self._is_noteworthy(perception):
                if self._mentions_person(perception):
                    existing_person = self._find_any_person_concept()
                    if existing_person is not None:
                        self._bump_concept(existing_person["id"])
                        self._store_observation(existing_person["id"], monologue)
                        self._update_last_observation(existing_person["id"], monologue)
                        return
                self._create_concept(perception, monologue)

    def _find_any_person_concept(self) -> Optional[Dict]:
        """Find the most-seen existing person concept, if any.

        Used to prevent creating duplicate person concepts — if we already
        track a person, new person perceptions should merge into it.
        """
        all_data = self._concepts.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            return None

        best = None
        best_seen = 0
        for cid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
            if self._mentions_person(doc) and meta.get("times_seen", 0) > best_seen:
                best_seen = meta["times_seen"]
                best = {"id": cid, "name": doc, "times_seen": best_seen}

        return best

    # ------------------------------------------------------------------
    # Session start: load familiar concepts for early prompts
    # ------------------------------------------------------------------

    def get_session_greeting(self, limit: int = 3) -> Optional[str]:
        """Get a memory line for session start — what the machine already knows
        about its environment. Returns None if too few concepts exist.
        """
        if self._concepts.count() < 2:
            return None

        # Get most-seen concepts
        all_concepts = self._concepts.get(include=["documents", "metadatas"])
        if not all_concepts["ids"]:
            return None

        # Sort by times_seen descending
        indexed = list(zip(all_concepts["ids"], all_concepts["documents"], all_concepts["metadatas"]))
        indexed.sort(key=lambda x: x[2].get("times_seen", 0), reverse=True)

        top = indexed[:limit]
        names = [doc for _, doc, _ in top]

        if len(names) == 1:
            return f"I know this place — {names[0]}."
        else:
            joined = ", ".join(names[:-1]) + f", and {names[-1]}"
            return f"I know this place — {joined}."

    # ------------------------------------------------------------------
    # Injection formatting — the most important part
    # ------------------------------------------------------------------

    def _get_relevant_observation(self, perception: str, concept_id: str = None) -> Tuple[str, str]:
        """Find the most semantically relevant stored memory for what the machine
        is currently seeing. Returns (text, type) where type is "reflection" or "observation".

        Reflections are preferred when available (they represent settled understanding);
        falls back to raw observations otherwise.
        """
        if not perception or self._observations.count() < 1:
            return ("", "")

        filler_markers = ["bored", "restless", "getting tired", "same scene",
                          "still here", "same old", "nothing new", "feeling restless",
                          "(in first person)", "i'm here to help", "drawing:"]

        # Try reflections first
        try:
            where = {"type": "reflection"}
            if concept_id:
                where = {"$and": [{"concept_id": concept_id}, {"type": "reflection"}]}

            refl_results = self._observations.query(
                query_texts=[perception],
                n_results=min(3, self._observations.count()),
                where=where,
                include=["documents", "distances"],
            )

            if refl_results["ids"][0]:
                for doc, dist in zip(refl_results["documents"][0], refl_results["distances"][0]):
                    if len(doc) < 15:
                        continue
                    if dist > 0.85:
                        continue  # Too unrelated even for a reflection
                    doc_lower = doc.lower()
                    if any(m in doc_lower for m in filler_markers):
                        continue
                    return (doc, "reflection")
        except Exception:
            pass

        # Fall back to raw observations
        try:
            query_args = {
                "query_texts": [perception],
                "n_results": min(5, self._observations.count()),
                "include": ["documents", "distances", "metadatas"],
            }
            if concept_id:
                query_args["where"] = {"$and": [{"concept_id": concept_id}, {"type": "observation"}]}
            else:
                query_args["where"] = {"type": "observation"}

            results = self._observations.query(**query_args)

            if not results["ids"][0]:
                return ("", "")

            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                if len(doc) < 15:
                    continue
                doc_lower = doc.lower()
                if any(m in doc_lower for m in filler_markers):
                    continue
                return (doc, "observation")

        except Exception:
            pass

        return ("", "")

    def _get_related_thoughts(self, perception: str, exclude_concept_id: str = None, limit: int = 2) -> List[str]:
        """Find observations from OTHER concepts that are semantically related
        to the current perception. This creates cross-concept connections.

        "The ceiling is cracked" might surface thoughts about the exposed wires
        or the peeling paint — related but from different concepts.
        """
        if not perception or self._observations.count() < 3:
            return []

        try:
            results = self._observations.query(
                query_texts=[perception],
                n_results=min(10, self._observations.count()),
                include=["documents", "distances", "metadatas"],
            )

            if not results["ids"][0]:
                return []

            filler_markers = ["bored", "restless", "getting tired", "same scene",
                              "still here", "same old", "nothing new",
                              "(in first person)", "i'm here to help"]
            # Reject very short emotional fragments that are too generic to be useful
            # (e.g. "Feeling curious.", "Just watching.")

            related = []
            for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
                # Skip the matched concept's own observations
                if meta.get("concept_id") == exclude_concept_id:
                    continue
                # Skip too similar (just a restatement) or too distant (unrelated)
                if dist < 0.2 or dist > 0.8:
                    continue
                # Need substantive length — short emotional fragments don't add anything
                if len(doc) < 30:
                    continue
                doc_lower = doc.lower()
                if any(m in doc_lower for m in filler_markers):
                    continue
                # Reject very short emotional one-liners ("Feeling curious.", "Just watching.")
                # Even if they pass length check, must contain a concrete reference
                if len(doc) < 50:
                    concrete = ["ceiling", "wall", "shelf", "sign", "light", "plant", "desk",
                                "person", "chair", "bag", "wire", "crack", "hole", "window",
                                "door", "screen", "monitor", "book", "shadow", "dust", "paper",
                                "color", "red", "blue", "white", "black", "pink", "green",
                                "fingers", "hands", "face", "eyes", "shape", "line"]
                    if not any(c in doc_lower for c in concrete):
                        continue
                related.append(self._truncate_observation(doc, 50))
                if len(related) >= limit:
                    break

            return related

        except Exception:
            return []

    def _format_injection(self, match: Dict, perception: str = "") -> str:
        """Format a matched concept as third-person observational context.

        The output describes what the machine knows about what it's seeing,
        framed for the writer (the model) rather than spoken in the machine's voice.
        Prefers reflections (settled understanding) over raw observations when available.
        """
        name = match["name"]
        times = match["times_seen"]

        # Lowercase the name for natural-sounding sentences
        name_lower = name[0].lower() + name[1:] if name else name

        if times < TIER_NEW:
            return f"It has noticed this before — {name_lower}."

        # Find the most relevant memory (reflection or raw observation)
        relevant_text, mem_type = self._get_relevant_observation(perception, concept_id=match["id"]) if perception else ("", "")

        # Different framing for reflections (settled) vs raw observations (fleeting)
        def frame_memory(text: str, mtype: str, max_len: int = 60) -> str:
            short = self._truncate_observation(text, max_len)
            if mtype == "reflection":
                # Reflections are LLM-synthesized — they often already start with "It has..."
                # so we use a simpler frame to avoid doubling
                return f"A settled understanding: \"{short}\""
            return f"Earlier it thought: \"{short}\""

        if times < TIER_FAMILIAR:
            if relevant_text and len(relevant_text) > 10:
                return f"It has seen this before — {name_lower}. {frame_memory(relevant_text, mem_type, 60)}"
            return f"It has seen this a few times — {name_lower}."

        # Very familiar — include cross-concept connections if available
        parts = [f"Familiar to it — {name_lower}."]

        if relevant_text and len(relevant_text) > 10:
            parts.append(frame_memory(relevant_text, mem_type, 50))

        # Pull one related thought from a different concept
        if perception and times > TIER_FAMILIAR * 2:
            related = self._get_related_thoughts(perception, exclude_concept_id=match["id"], limit=1)
            if related:
                parts.append(f"Nearby in memory: \"{related[0]}\"")

        return " ".join(parts)

    @staticmethod
    def _truncate_observation(text: str, max_len: int) -> str:
        """Truncate an observation to max_len at a sentence or word boundary."""
        text = text.strip()
        if len(text) <= max_len:
            return text
        # Try sentence boundary
        for i in range(min(len(text), max_len), 15, -1):
            if text[i - 1] in ".!?":
                return text[:i]
        # Word boundary
        truncated = text[:max_len].rsplit(" ", 1)[0]
        return truncated.rstrip(",.;:") + "..."

    # ------------------------------------------------------------------
    # Concept CRUD
    # ------------------------------------------------------------------

    def _create_concept(self, perception: str, monologue: str):
        """Create a new concept from a noteworthy perception."""
        concept_id = f"concept_{int(time.time())}_{self._concepts.count()}"
        canonical_name = self._extract_canonical_name(perception)
        now = time.time()

        self._concepts.add(
            ids=[concept_id],
            documents=[canonical_name],
            metadatas=[{
                "times_seen": 1,
                "first_seen": now,
                "last_seen": now,
                "session_count": 1,
                "last_session": self._session_id,
                "last_observation": monologue[:200] if monologue else "",
            }],
        )

        # Store the first observation
        self._store_observation(concept_id, monologue)

        print(f"[SEMANTIC] New concept: '{canonical_name}' (id={concept_id})")

    def _bump_concept(self, concept_id: str):
        """Increment times_seen and update timestamps for a known concept."""
        existing = self._concepts.get(ids=[concept_id], include=["metadatas"])
        if not existing["ids"]:
            return

        meta = existing["metadatas"][0]
        meta["times_seen"] = meta.get("times_seen", 0) + 1
        meta["last_seen"] = time.time()

        # Track session boundaries
        if meta.get("last_session") != self._session_id:
            meta["session_count"] = meta.get("session_count", 0) + 1
            meta["last_session"] = self._session_id

        self._concepts.update(ids=[concept_id], metadatas=[meta])

    def _update_last_observation(self, concept_id: str, monologue: str):
        """Update a concept's last_observation field."""
        existing = self._concepts.get(ids=[concept_id], include=["metadatas"])
        if not existing["ids"]:
            return

        meta = existing["metadatas"][0]
        meta["last_observation"] = monologue[:200]
        self._concepts.update(ids=[concept_id], metadatas=[meta])

    def _store_observation(self, concept_id: str, text: str):
        """Store an observation linked to a concept, with quality and relevance checks.

        Only stores observations that are:
        1. Substantive (not filler/emotional noise)
        2. Actually about the concept (semantically close to concept name)
        3. Saying something new (not duplicating existing observations)
        """
        if not text or len(text.strip()) < 25:
            return  # Too short to be meaningful as a memory

        # Quality gate: reject monologue filler that isn't about anything specific
        text_lower = text.lower().strip()
        if text_lower.startswith(("feeling ", "still ", "bored", "...", "sigh")):
            # Only reject if the ENTIRE text is filler — if there's substance after, keep it
            # Check: does it contain any concrete nouns or specific references?
            concrete_markers = ["ceiling", "wall", "shelf", "sign", "light", "plant", "desk",
                                "person", "chair", "bag", "wire", "crack", "hole", "window",
                                "door", "screen", "monitor", "book", "shadow", "dust"]
            if not any(m in text_lower for m in concrete_markers):
                return  # Pure emotional filler — don't store

        # Relevance gate: check the observation is semantically about this concept
        try:
            concept_data = self._concepts.get(ids=[concept_id], include=["documents"])
            if concept_data["documents"]:
                concept_name = concept_data["documents"][0]
                # Query: how close is this monologue to the concept's name?
                relevance = self._concepts.query(
                    query_texts=[text],
                    n_results=1,
                    include=["distances"],
                )
                if relevance["distances"][0] and relevance["distances"][0][0] > 0.8:
                    return  # Monologue is about something completely different from this concept
        except Exception:
            pass

        # Duplicate gate: don't store if too similar to existing observations
        existing = self._observations.get(
            where={"concept_id": concept_id},
            include=["documents"],
        )

        if existing["documents"]:
            try:
                results = self._observations.query(
                    query_texts=[text],
                    n_results=1,
                    where={"concept_id": concept_id},
                    include=["distances"],
                )
                if results["distances"][0] and results["distances"][0][0] < DUPLICATE_DISTANCE:
                    return  # Too similar to an existing observation
            except Exception:
                pass

            # Prune old observations if we have too many for this concept
            if len(existing["documents"]) >= MAX_OBSERVATIONS_PER_CONCEPT:
                self._prune_oldest_observations(concept_id)

        obs_id = f"obs_{self._obs_counter}"
        self._obs_counter += 1

        self._observations.add(
            ids=[obs_id],
            documents=[text[:300]],
            metadatas=[{
                "concept_id": concept_id,
                "timestamp": time.time(),
                "session_id": self._session_id,
                "type": "observation",
                "depth": 0,
            }],
        )

    def _generate_reflection(self, concept_id: str, concept_name: str, observations: List[Dict]) -> Optional[str]:
        """Synthesize recent observations about a concept into one settled thought.

        Uses the compression model (text-only) to avoid contention with the vision pipeline.
        Returns the reflection text or None on failure.
        """
        if len(observations) < 2:
            return None

        try:
            from utils.ollama import query_ollama
            from config import config

            # Build the prompt
            obs_lines = "\n".join(f"- {o['text'][:120]}" for o in observations[:6])

            prompt = f"""Recent thoughts about "{concept_name}":
{obs_lines}

In ONE SHORT SENTENCE (under 20 words), what pattern or deeper understanding is emerging from these thoughts? Third person observational voice. Start with "It" or "A pattern".

Respond with ONLY the sentence."""

            system_prompt = (
                "You synthesize a drawing machine's recurring thoughts about an object or person into one settled insight. "
                "Third person, present tense, under 20 words, no preamble. "
                "Examples: "
                "'It has come to see the cracks as a wound that no one will heal.' "
                "'A pattern emerges — it returns to the shelf whenever it feels alone.'"
            )

            model_options = {
                "temperature": 0.6,
                "top_p": 0.9,
                "num_predict": 50,
                "repeat_penalty": 1.3,
                "stop": ["\n\n"],
            }

            compression_model = getattr(config, 'COMPRESSION_MODEL', config.OLLAMA_MODEL)

            response = query_ollama(
                prompt=prompt,
                model=compression_model,
                image=None,
                system_prompt=system_prompt,
                timeout=getattr(config, 'OLLAMA_TIMEOUT_EVAL', 60),
                options=model_options,
                prompt_type="reflection",
            )

            if response and isinstance(response, str):
                cleaned = response.strip().strip('"\'').strip()
                # Take just the first sentence
                for sep in [". ", "! ", "? "]:
                    if sep in cleaned:
                        cleaned = cleaned.split(sep)[0] + sep[0]
                        break
                if 15 < len(cleaned) < 200:
                    return cleaned

        except Exception as e:
            print(f"[SEMANTIC] Reflection generation failed: {e}")

        return None

    def _store_reflection(self, concept_id: str, text: str, source_obs_ids: List[str], depth: int = 0):
        """Store a reflection — an LLM-synthesized higher-order memory.

        Reflections are stored alongside observations in the same collection,
        distinguished by metadata. They represent the machine's settled
        understanding rather than a single fleeting thought.
        """
        if not text or len(text.strip()) < 15:
            return

        # Duplicate check: don't store if too similar to existing reflections for this concept
        existing = self._observations.get(
            where={"$and": [{"concept_id": concept_id}, {"type": "reflection"}]},
            include=["documents"],
        )
        if existing["documents"]:
            try:
                results = self._observations.query(
                    query_texts=[text],
                    n_results=1,
                    where={"$and": [{"concept_id": concept_id}, {"type": "reflection"}]},
                    include=["distances"],
                )
                if results["distances"][0] and results["distances"][0][0] < DUPLICATE_DISTANCE:
                    return  # Already have a near-identical reflection
            except Exception:
                pass

        refl_id = f"refl_{self._obs_counter}"
        self._obs_counter += 1

        # ChromaDB metadata can't store lists directly, store as comma-joined string
        sources_str = ",".join(source_obs_ids[:10])  # cap at 10 to keep metadata manageable

        self._observations.add(
            ids=[refl_id],
            documents=[text[:300]],
            metadatas=[{
                "concept_id": concept_id,
                "timestamp": time.time(),
                "session_id": self._session_id,
                "type": "reflection",
                "depth": depth,
                "synthesized_from": sources_str,
            }],
        )
        print(f"[SEMANTIC] Stored reflection (depth={depth}): {text[:80]}")

    # ------------------------------------------------------------------
    # Reflection worker — periodic background synthesis
    # ------------------------------------------------------------------

    def _start_reflection_worker(self):
        """Start the background thread that periodically generates reflections."""
        if self._reflection_thread is not None and self._reflection_thread.is_alive():
            return
        self._reflection_stop.clear()
        self._reflection_thread = threading.Thread(
            target=self._reflection_worker_loop,
            daemon=True,
            name="SemanticReflectionWorker",
        )
        self._reflection_thread.start()

    def stop_reflection_worker(self):
        """Stop the reflection worker thread cleanly."""
        self._reflection_stop.set()
        if self._reflection_thread:
            self._reflection_thread.join(timeout=5)

    def _reflection_worker_loop(self):
        """Main loop for the reflection worker. Runs every REFLECTION_INTERVAL_SECONDS."""
        # Sleep on startup to let the main system warm up first
        if self._reflection_stop.wait(timeout=120):
            return  # Stop requested before first cycle

        while not self._reflection_stop.is_set():
            try:
                self._reflection_cycle()
            except Exception as e:
                print(f"[SEMANTIC] Reflection cycle error: {e}")
            # Wait for next interval, exit early if stop requested
            if self._reflection_stop.wait(timeout=REFLECTION_INTERVAL_SECONDS):
                return

    def _reflection_cycle(self):
        """One pass: find concepts ripe for reflection, synthesize one or two."""
        candidates = self._find_reflection_candidates()
        if not candidates:
            return

        # Process up to REFLECTION_MAX_PER_CYCLE candidates
        for concept_id, concept_name, observations in candidates[:REFLECTION_MAX_PER_CYCLE]:
            print(f"[SEMANTIC] Reflecting on '{concept_name}' ({len(observations)} new observations)")
            reflection = self._generate_reflection(concept_id, concept_name, observations)
            if reflection:
                obs_ids = [o["id"] for o in observations]
                self._store_reflection(concept_id, reflection, obs_ids, depth=0)
                # Surface in caption monitor as a "settled understanding"
                try:
                    from utils.live_log import log_reflection
                    log_reflection(concept_name, reflection)
                except Exception:
                    pass

    def _find_reflection_candidates(self) -> List[Tuple[str, str, List[Dict]]]:
        """Find concepts that have accumulated enough new observations to warrant reflection.

        Returns list of (concept_id, concept_name, observations_since_last_reflection) tuples,
        sorted by activity (most observations first).
        """
        candidates = []

        all_concepts = self._concepts.get(include=["documents", "metadatas"])
        if not all_concepts["ids"]:
            return []

        for cid, name, meta in zip(all_concepts["ids"], all_concepts["documents"], all_concepts["metadatas"]):
            # Skip rarely-seen concepts
            if meta.get("times_seen", 0) < REFLECTION_MIN_CONCEPT_FAMILIARITY:
                continue

            # Get all observations for this concept
            obs_data = self._observations.get(
                where={"concept_id": cid},
                include=["documents", "metadatas"],
            )
            if not obs_data["ids"]:
                continue

            # Find the most recent reflection's timestamp (if any) for this concept
            last_refl_ts = 0.0
            new_observations = []
            for oid, doc, ometa in zip(obs_data["ids"], obs_data["documents"], obs_data["metadatas"]):
                otype = ometa.get("type", "observation")
                ts = ometa.get("timestamp", 0)
                if otype == "reflection" and ts > last_refl_ts:
                    last_refl_ts = ts

            # Now collect observations newer than the last reflection
            for oid, doc, ometa in zip(obs_data["ids"], obs_data["documents"], obs_data["metadatas"]):
                if ometa.get("type", "observation") != "observation":
                    continue
                ts = ometa.get("timestamp", 0)
                if ts > last_refl_ts and len(doc) > 15:
                    new_observations.append({"id": oid, "text": doc, "timestamp": ts})

            if len(new_observations) >= REFLECTION_MIN_NEW_OBSERVATIONS:
                # Sort by timestamp ascending (oldest first, so synthesis sees the arc)
                new_observations.sort(key=lambda x: x["timestamp"])
                candidates.append((cid, name, new_observations))

        # Sort by number of new observations (most active concepts first)
        candidates.sort(key=lambda x: len(x[2]), reverse=True)
        return candidates

    def _prune_oldest_observations(self, concept_id: str):
        """Remove the oldest observations for a concept, keeping MAX_OBSERVATIONS_PER_CONCEPT."""
        existing = self._observations.get(
            where={"concept_id": concept_id},
            include=["metadatas"],
        )
        if not existing["ids"]:
            return

        # Sort by timestamp, delete the oldest
        indexed = list(zip(existing["ids"], existing["metadatas"]))
        indexed.sort(key=lambda x: x[1].get("timestamp", 0))

        to_delete = len(indexed) - MAX_OBSERVATIONS_PER_CONCEPT + 1  # +1 to make room
        if to_delete > 0:
            delete_ids = [idx for idx, _ in indexed[:to_delete]]
            self._observations.delete(ids=delete_ids)

    # ------------------------------------------------------------------
    # Noteworthy filter — gates concept creation
    # ------------------------------------------------------------------

    def _is_noteworthy(self, perception: str) -> bool:
        """Decide if a perception describes something specific enough to become a concept.

        Rejects generic room descriptions and vague statements.
        Accepts: specific objects, people, spatial features with detail.
        """
        text = perception.lower().strip()

        # Too short = too vague
        if len(text) < 20:
            return False

        # Reject pure filler and generic scene descriptions
        filler_patterns = [
            r"^same\b", r"^nothing\b", r"^unclear\b", r"^still\b",
            r"^a (?:room|space|area|place)\b",
            r"^(?:the )?(?:usual|same|familiar) ",
            r"^(?:the )?scene\b",
            r"(?:no people|with no people|no one|nobody)\b",
            r"^(?:the )?(?:room|workspace|studio|area|space) (?:is|has|looks|appears)\b",
            # Generic whole-room descriptors — no specific object
            r"^(?:a |an )?(?:cluttered|messy|tidy|bright|dark|dimly lit|well lit)\s+(?:indoor\s+)?(?:room|workspace|studio|area|space|environment|setting)\b",
        ]
        for pat in filler_patterns:
            if re.search(pat, text):
                return False

        # Accept if it has spatial specificity (location phrases)
        spatial_markers = [
            "on the", "near the", "above the", "below the", "next to",
            "behind the", "in front of", "corner", "left", "right",
            "wall", "shelf", "desk", "ceiling", "floor", "window", "door",
        ]
        has_spatial = any(m in text for m in spatial_markers)

        # Accept if it mentions a person
        person_markers = ["person", "someone", "man", "woman", "face", "people", "they", "he ", "she "]
        has_person = any(m in text for m in person_markers)

        # Accept if it has descriptive specificity (color, text, distinctive features)
        detail_markers = [
            "red", "blue", "green", "yellow", "black", "white", "brown", "orange",
            "sign", "text", "label", "sticker", "poster", "writing",
            "damaged", "broken", "new", "old", "large", "small",
            "says", "reads", "written",
        ]
        has_detail = any(m in text for m in detail_markers)

        # Need at least spatial + detail, or person mention
        return has_person or (has_spatial and has_detail)

    # ------------------------------------------------------------------
    # Name extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_perception(text: str) -> str:
        """Strip vision-model artifacts from perception text before any processing.

        Removes "of the image", "in this image", verbose editorial tails, etc.
        """
        # Strip "of/in the image" references
        text = re.sub(r'\s*(?:of|in|from)\s+(?:the|this)\s+(?:image|photo|picture|frame)\s*', ' ', text, flags=re.IGNORECASE)
        # Strip editorial tails: "adding a pop of color...", "which appears to be..."
        text = re.sub(r',\s*(?:adding|which|creating|making|giving|providing)\b.*$', '', text, flags=re.IGNORECASE)
        # Strip "is prominently placed/positioned/located"
        text = re.sub(r'\s+is\s+(?:prominently|strategically|carefully)\s+(?:placed|positioned|located)\b', '', text, flags=re.IGNORECASE)
        # Collapse double spaces from stripping
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    @staticmethod
    def _extract_canonical_name(perception: str) -> str:
        """Extract a short canonical name from a perception.

        "A red sign on the wall with white text" → "Red sign on the wall with white text"
        "Someone in a camo jacket sitting at the desk" → "Person in a camo jacket at the desk"
        """
        text = SemanticMemory._clean_perception(perception.strip())

        # Truncate to first sentence if multi-sentence
        for i in range(min(len(text), 60), 10, -1):
            if text[i - 1] in ".!?":
                text = text[:i - 1]
                break

        # Strip leading locative preambles: "On the left side of the room, there is..."
        text = re.sub(r'^On\s+the\s+\w+\s+side(?:\s+of\s+the\s+\w+)?\s*,?\s*(?:there\s+is\s+)?', '', text, flags=re.IGNORECASE)
        # Strip leading articles and filler
        text = re.sub(r'^(?:There(?:\'s| is| are) )?(?:a |an |the )?', '', text, flags=re.IGNORECASE)
        # Strip "noticeable detail is" type preambles
        text = re.sub(r'^(?:noticeable|notable|interesting|striking|most striking|most noticeable|most prominent)\s+(?:detail|feature|element)\s+(?:is\s+)?(?:the\s+)?', '', text, flags=re.IGNORECASE)
        # Strip "in front of you is" preambles from perception model
        text = re.sub(r'^(?:in front of you is\s+)', '', text, flags=re.IGNORECASE)

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        # Truncate at reasonable length — prefer sentence boundary
        if len(text) > 55:
            # Try sentence boundary first
            for i in range(min(len(text), 55), 15, -1):
                if text[i - 1] in ".!?":
                    text = text[:i - 1]
                    break
            else:
                text = text[:55].rsplit(" ", 1)[0]

        # Strip trailing punctuation and dangling prepositions
        text = text.rstrip(",.;:!? ")
        # Remove trailing dangling words (with, and, or, the, a, in, on, of, to, for, is)
        text = re.sub(r'\s+(?:with|and|or|the|a|an|in|on|of|to|for|is|are|has|that|which)$', '', text, flags=re.IGNORECASE)

        return text

    # ------------------------------------------------------------------
    # Debug / introspection
    # ------------------------------------------------------------------

    def get_all_concepts(self) -> List[Dict]:
        """Return all concepts with metadata, sorted by times_seen descending."""
        all_data = self._concepts.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            return []

        concepts = []
        for cid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
            concepts.append({
                "id": cid,
                "name": doc,
                "times_seen": meta.get("times_seen", 0),
                "first_seen": meta.get("first_seen", 0),
                "last_seen": meta.get("last_seen", 0),
                "session_count": meta.get("session_count", 0),
                "last_observation": meta.get("last_observation", ""),
            })

        concepts.sort(key=lambda x: x["times_seen"], reverse=True)
        return concepts

    def get_concept_observations(self, concept_id: str) -> List[Dict]:
        """Return all observations for a concept, sorted by timestamp."""
        obs = self._observations.get(
            where={"concept_id": concept_id},
            include=["documents", "metadatas"],
        )
        if not obs["ids"]:
            return []

        result = []
        for oid, doc, meta in zip(obs["ids"], obs["documents"], obs["metadatas"]):
            result.append({
                "id": oid,
                "text": doc,
                "timestamp": meta.get("timestamp", 0),
                "session_id": meta.get("session_id", ""),
            })

        result.sort(key=lambda x: x["timestamp"])
        return result

    def recall_tangent(self, current_perception: str = "") -> Optional[str]:
        """Surface one old observation that's adjacent to — but different from —
        what the machine is currently seeing.

        Uses semantic distance to find the sweet spot: not a restatement of
        the current perception (too close), not completely unrelated (noise),
        but a thought that connects sideways to what's being observed.
        """
        if self._observations.count() < 5 or not current_perception:
            return None

        try:
            cleaned = self._clean_perception(current_perception)

            # Query observations by semantic similarity to current perception
            results = self._observations.query(
                query_texts=[cleaned],
                n_results=min(15, self._observations.count()),
                include=["documents", "distances", "metadatas"],
            )

            if not results["ids"][0]:
                return None

            now = time.time()
            filler_markers = [
                "bored", "restless", "getting tired", "same scene",
                "still here", "same old", "nothing new", "feeling restless",
                "(in first person)", "i'm here to help", "drawing:",
                "feeling bored", "getting tired of",
            ]

            # Find the best candidate in the "adjacent" distance range
            # Too close (< 0.3) = just restating what we see
            # Too far (> 0.9) = unrelated noise
            # Sweet spot (0.3 - 0.7) = connected but different perspective
            last_tangent = getattr(self, '_last_tangent', '')

            for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
                if dist < 0.3 or dist > 0.7:
                    continue
                age = now - meta.get("timestamp", now)
                if age < 300:
                    continue  # Too recent
                if len(doc) < 15:
                    continue
                doc_lower = doc.lower()
                if any(m in doc_lower for m in filler_markers):
                    continue
                if doc == last_tangent:
                    continue  # Avoid immediate repeats

                self._last_tangent = doc
                return self._truncate_observation(doc, 50)

        except Exception:
            pass

        return None

    def delete_concept(self, concept_id: str) -> bool:
        """Delete a concept and all its observations."""
        existing = self._concepts.get(ids=[concept_id])
        if not existing["ids"]:
            return False

        # Delete observations linked to this concept
        obs = self._observations.get(where={"concept_id": concept_id})
        if obs["ids"]:
            self._observations.delete(ids=obs["ids"])

        self._concepts.delete(ids=[concept_id])
        return True

    def merge_concepts(self, keep_id: str, absorb_id: str) -> bool:
        """Merge two concepts: absorb_id's observations move into keep_id, then absorb_id is deleted.

        The kept concept gets the combined times_seen and the earliest first_seen.
        """
        keep = self._concepts.get(ids=[keep_id], include=["metadatas"])
        absorb = self._concepts.get(ids=[absorb_id], include=["metadatas"])
        if not keep["ids"] or not absorb["ids"]:
            return False

        keep_meta = keep["metadatas"][0]
        absorb_meta = absorb["metadatas"][0]

        # Combine counts
        keep_meta["times_seen"] = keep_meta.get("times_seen", 0) + absorb_meta.get("times_seen", 0)
        keep_meta["session_count"] = max(keep_meta.get("session_count", 0), absorb_meta.get("session_count", 0))
        keep_meta["first_seen"] = min(keep_meta.get("first_seen", 0), absorb_meta.get("first_seen", 0))
        keep_meta["last_seen"] = max(keep_meta.get("last_seen", 0), absorb_meta.get("last_seen", 0))

        self._concepts.update(ids=[keep_id], metadatas=[keep_meta])

        # Re-tag absorbed observations to the kept concept
        obs = self._observations.get(where={"concept_id": absorb_id}, include=["metadatas"])
        if obs["ids"]:
            new_metas = []
            for meta in obs["metadatas"]:
                meta["concept_id"] = keep_id
                new_metas.append(meta)
            self._observations.update(ids=obs["ids"], metadatas=new_metas)

        # Delete the absorbed concept
        self._concepts.delete(ids=[absorb_id])
        return True

    def stats(self) -> Dict:
        return {
            "concepts": self._concepts.count(),
            "observations": self._observations.count(),
            "session": self._session_id,
        }
