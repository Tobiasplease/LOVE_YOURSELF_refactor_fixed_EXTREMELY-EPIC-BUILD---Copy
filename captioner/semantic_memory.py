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
import threading
import time
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

# --- Spatial extraction patterns ---
# Maps phrases from Qwen's output to rough spatial zones (pan/tilt)
_SPATIAL_PAN_PATTERNS = [
    (r"\bto the left\b|\bon the left\b|\bleft side\b", "left"),
    (r"\bto the right\b|\bon the right\b|\bright side\b", "right"),
    (r"\bin the center\b|\bin the middle\b|\bin front\b|\bdirectly ahead\b", "ahead"),
    (r"\bin the background\b|\bbehind\b|\bback wall\b|\bagainst the .* wall\b", "ahead"),
]
_SPATIAL_TILT_PATTERNS = [
    (r"\babove\b|\bceiling\b|\bhung from\b|\bmounted.*ceiling\b|\boverhead\b", "up"),
    (r"\bbelow\b|\bfloor\b|\bground\b|\bon the (?:desk|table|surface)\b", "down"),
    (r"\bsuspended from\b|\bhanging from\b", "up"),
]

# Singleton
_instance = None
_lock = threading.Lock()


# Reject sentence-fragment labels — concepts are short noun phrases, not caption
# scraps. Earlier runs minted "Chaos on the floor is", "Light from their desk is"
# (a regex mangler, now removed); this also screens legacy garbage at surfacing.
_LABEL_TAIL_STOPWORDS = {
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "of",
    "and",
    "but",
    "with",
    "to",
    "in",
    "on",
    "at",
    "it",
    "as",
    "that",
    "this",
    "their",
    "its",
    "from",
}


def _looks_like_noun_phrase(label: str) -> bool:
    words = (label or "").strip().split()
    if not (1 <= len(words) <= 4):
        return False
    return words[-1].lower().strip(".,;:") not in _LABEL_TAIL_STOPWORDS


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

        # Reflections: long-form thoughts from the reflection loop
        # (captioner/reflection.py). First-class memories — each document is
        # one full reflection, retrieved by relevance into quiet captions.
        self._reflections = self._client.get_or_create_collection(
            name="reflections",
            metadata={"hnsw:space": "cosine"},
        )

        self._obs_counter = self._observations.count()
        self._session_id = f"session_{int(time.time())}"

        print(
            f"[SEMANTIC] Loaded: {self._concepts.count()} concepts, "
            f"{self._observations.count()} observations, {self._reflections.count()} reflections"
        )

    # ------------------------------------------------------------------
    # Reflections — storage and retrieval for the reflection loop
    # ------------------------------------------------------------------

    def store_reflection_entry(self, text: str, subject: str) -> Optional[str]:
        """Store one long-form reflection. Returns its id, or None if rejected."""
        text = (text or "").strip()
        if len(text) < 80:
            return None
        refl_id = f"refl_{int(time.time() * 1000)}"
        self._reflections.add(
            ids=[refl_id],
            documents=[text[:4000]],
            metadatas=[{"subject": subject, "timestamp": time.time(), "session_id": self._session_id}],
        )
        return refl_id

    def set_reflection_kernel(self, refl_id: str, kernel: str) -> None:
        """Attach the distilled kernel (the reflection's one load-bearing
        sentence) to an already-stored entry. Merge-update: Chroma's update
        replaces the metadata dict, so re-read and merge. Old entries without
        a kernel keep surfacing as bare subjects (purple-era containment)."""
        kernel = (kernel or "").strip()
        if not refl_id or not kernel:
            return
        try:
            got = self._reflections.get(ids=[refl_id], include=["metadatas"])
            metas = got.get("metadatas") or []
            meta = dict(metas[0]) if metas else {}
            meta["kernel"] = kernel[:200]
            self._reflections.update(ids=[refl_id], metadatas=[meta])
        except Exception:
            pass

    def get_recent_reflections(self, limit: int = 3, subject: str = "") -> List[Dict]:
        """Most recent reflections, oldest first — the thread of self-thought
        each new reflection gets to see (across sessions).

        subject: restrict to one subject's thread. Each reflection organ reads
        its OWN past (July 31). Quoting the last three reflections of ANY
        subject was a direct copy-the-theme channel: three counting-themed
        reflections in a row were pasted verbatim into the top of the fourth,
        whatever its subject — one of the homogenisers behind the
        five-lenses-one-thought collapse.
        """
        try:
            got = self._reflections.get(include=["documents", "metadatas"])
        except Exception:
            return []
        rows = sorted(
            zip(got["ids"], got["documents"], got["metadatas"]),
            key=lambda r: r[2].get("timestamp", 0),
        )
        if subject:
            rows = [r for r in rows if (r[2].get("subject") or "") == subject]
        return [
            {"id": rid, "text": doc, "subject": meta.get("subject", ""), "timestamp": meta.get("timestamp", 0)} for rid, doc, meta in rows[-limit:]
        ]

    def query_reflections(self, query_text: str, n_results: int = 2, max_distance: float = 0.6) -> List[Dict]:
        """Reflections relevant to the current thought, nearest first."""
        if not query_text or self._reflections.count() == 0:
            return []
        try:
            res = self._reflections.query(
                query_texts=[query_text],
                n_results=min(n_results, self._reflections.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []
        out = []
        for rid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if dist <= max_distance:
                out.append(
                    {
                        "id": rid,
                        "text": doc,
                        "subject": meta.get("subject", ""),
                        "kernel": meta.get("kernel", ""),
                        "timestamp": meta.get("timestamp", 0),
                        "distance": dist,
                    }
                )
        return out

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

                    self._bump_concept(cid, perception=cleaned)
                    matched.append(
                        {
                            "id": cid,
                            "label": doc,
                            "times_seen": meta.get("times_seen", 0) + 1,
                            "is_new": False,
                            "session_count": meta.get("session_count", 1),
                            "last_observation": meta.get("last_observation", ""),
                            "first_seen": meta.get("first_seen", 0),
                            "last_seen": meta.get("last_seen", 0),
                            "spatial_pan": meta.get("spatial_pan"),
                            "spatial_tilt": meta.get("spatial_tilt"),
                        }
                    )

        # If no matches and it mentions a person, bump the existing person concept
        # New concept creation is now handled by compression-time LLM extraction
        # (register_concepts_from_compression), not per-caption regex.
        if not matched and self._is_noteworthy(cleaned):
            if self._mentions_person(cleaned):
                existing_person = self._find_any_person_concept()
                if existing_person is not None:
                    self._bump_concept(existing_person["id"], perception=cleaned)
                    return [
                        {
                            "id": existing_person["id"],
                            "label": existing_person["name"],
                            "times_seen": existing_person["times_seen"] + 1,
                            "is_new": False,
                            "last_observation": existing_person.get("last_observation", ""),
                            "spatial_pan": existing_person.get("spatial_pan"),
                            "spatial_tilt": existing_person.get("spatial_tilt"),
                        }
                    ]

        return matched

    @staticmethod
    def _mentions_person(text: str) -> bool:
        """Check if text describes a person."""
        t = text.lower()
        return any(
            w in t
            for w in [
                "person",
                "someone",
                "individual",
                "man",
                "woman",
                "people",
                "figure",
                "they are",
                "he is",
                "she is",
                "sitting",
                "standing",
                "typing",
                "working",
                "looking",
                "seated",
                "focused on",
                "wearing",
            ]
        )

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
                # NO per-caption concept creation: _extract_canonical_name is a
                # regex mangler that minted sentence-fragment labels from raw
                # captions ("Chaos on the floor is", "Light from their desk is").
                # New concepts come ONLY from the clean compression-time LLM
                # extraction (register_concepts_from_compression). See
                # docs/memory-redesign-plan.md (concepts/objects ledger).

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
    # Concept CRUD
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_concept_label(label: str) -> bool:
        """Storage gate for concept names. Captions sometimes arrive with
        markdown scaffolding or situational fragments, and those were being
        stored verbatim ('* **Senses:** The hum of', 'Awake 2 minutes', '',
        'none' — June 12). A concept is a plain name for a thing."""
        label = (label or "").strip()
        if not 3 <= len(label) <= 40:
            return False
        if any(ch in label for ch in "*#`_:[]{}|"):
            return False
        if not any(ch.isalpha() for ch in label):
            return False
        lowered = label.lower()
        if lowered in ("none", "nothing", "unknown", "n/a"):
            return False
        if lowered.startswith(("awake ", "looking ", "just woke", "someone ")):
            return False
        return True

    @staticmethod
    def _extract_spatial_zone(perception: str) -> tuple:
        """Extract rough spatial location (pan_zone, tilt_zone) from Qwen's perception text.

        Returns ("left"/"right"/"ahead"/None, "up"/"down"/None).
        Only returns a direction when there's a clear spatial phrase.
        """
        text = perception.lower()
        pan = None
        tilt = None
        for pattern, zone in _SPATIAL_PAN_PATTERNS:
            if re.search(pattern, text):
                pan = zone
                break
        for pattern, zone in _SPATIAL_TILT_PATTERNS:
            if re.search(pattern, text):
                tilt = zone
                break
        return (pan, tilt)

    def _bump_concept(self, concept_id: str, perception: str = ""):
        """Increment times_seen, update timestamps, and refine spatial location."""
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

        # Update spatial location if perception provides it
        if perception:
            pan, tilt = self._extract_spatial_zone(perception)
            if pan:
                meta["spatial_pan"] = pan
            if tilt:
                meta["spatial_tilt"] = tilt

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
            concrete_markers = [
                "ceiling",
                "wall",
                "shelf",
                "sign",
                "light",
                "plant",
                "desk",
                "person",
                "chair",
                "bag",
                "wire",
                "crack",
                "hole",
                "window",
                "door",
                "screen",
                "monitor",
                "book",
                "shadow",
                "dust",
            ]
            if not any(m in text_lower for m in concrete_markers):
                return  # Pure emotional filler — don't store

        # Relevance gate: check the observation is semantically about this concept
        try:
            concept_data = self._concepts.get(ids=[concept_id], include=["documents"])
            if concept_data["documents"]:
                # NOTE: this queries the whole concepts collection with the monologue
                # text — it does NOT compare against this concept's name, despite the
                # old comment claiming so (census-aug30 §2.5, latent behavior gap).
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
            metadatas=[
                {
                    "concept_id": concept_id,
                    "timestamp": time.time(),
                    "session_id": self._session_id,
                    "type": "observation",
                    "depth": 0,
                }
            ],
        )

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
            r"^same\b",
            r"^nothing\b",
            r"^unclear\b",
            r"^still\b",
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
            "on the",
            "near the",
            "above the",
            "below the",
            "next to",
            "behind the",
            "in front of",
            "corner",
            "left",
            "right",
            "wall",
            "shelf",
            "desk",
            "ceiling",
            "floor",
            "window",
            "door",
        ]
        has_spatial = any(m in text for m in spatial_markers)

        # Accept if it mentions a person
        person_markers = ["person", "someone", "man", "woman", "face", "people", "they", "he ", "she "]
        has_person = any(m in text for m in person_markers)

        # Accept if it has descriptive specificity (color, text, distinctive features)
        detail_markers = [
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "brown",
            "orange",
            "sign",
            "text",
            "label",
            "sticker",
            "poster",
            "writing",
            "damaged",
            "broken",
            "new",
            "old",
            "large",
            "small",
            "says",
            "reads",
            "written",
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
        text = re.sub(r"\s*(?:of|in|from)\s+(?:the|this)\s+(?:image|photo|picture|frame)\s*", " ", text, flags=re.IGNORECASE)
        # Strip editorial tails: "adding a pop of color...", "which appears to be..."
        text = re.sub(r",\s*(?:adding|which|creating|making|giving|providing)\b.*$", "", text, flags=re.IGNORECASE)
        # Strip "is prominently placed/positioned/located"
        text = re.sub(r"\s+is\s+(?:prominently|strategically|carefully)\s+(?:placed|positioned|located)\b", "", text, flags=re.IGNORECASE)
        # Collapse double spaces from stripping
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def register_concepts_from_compression(self, labels: list[str]) -> None:
        """Register pre-cleaned concept labels extracted by LLM during compression.

        For each label: if a similar concept exists (by embedding similarity), bump it.
        Otherwise create a new concept. This bypasses _extract_canonical_name entirely.
        """
        try:
            from captioner.context_compression import _is_abstract_label
        except Exception:
            _is_abstract_label = lambda l: False

        for label in labels:
            label = label.strip()
            if not self._valid_concept_label(label):
                continue
            # Reject sentence fragments (must be a short noun phrase, not a
            # caption scrap ending in "is"/"the"/etc.)
            if "." in label or "?" in label or "!" in label:
                continue
            if not _looks_like_noun_phrase(label):
                continue
            # Reject affect/abstraction labels ("unseen presence", "glitching nightmare")
            if _is_abstract_label(label):
                continue

            # Check for existing similar concept
            if self._concepts.count() > 0:
                results = self._concepts.query(
                    query_texts=[label],
                    n_results=1,
                    include=["documents", "metadatas", "distances"],
                )
                if results["ids"][0] and results["distances"][0][0] < SIMILARITY_THRESHOLD:
                    self._bump_concept(results["ids"][0][0])
                    continue

            # Create new concept
            concept_id = f"concept_{int(time.time())}_{self._concepts.count()}"
            now = time.time()
            self._concepts.add(
                ids=[concept_id],
                documents=[label],
                metadatas=[
                    {
                        "times_seen": 1,
                        "first_seen": now,
                        "last_seen": now,
                        "session_count": 1,
                        "last_session": self._session_id,
                        "last_observation": "",
                        "source": "compression",
                    }
                ],
            )
            print(f"[SEMANTIC] New concept (from compression): '{label}'")

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
            concepts.append(
                {
                    "id": cid,
                    "name": doc,
                    "times_seen": meta.get("times_seen", 0),
                    "first_seen": meta.get("first_seen", 0),
                    "last_seen": meta.get("last_seen", 0),
                    "session_count": meta.get("session_count", 0),
                    "last_observation": meta.get("last_observation", ""),
                }
            )

        concepts.sort(key=lambda x: x["times_seen"], reverse=True)
        return concepts

    def get_memorable_concept(self, min_times_seen: int = 3) -> Optional[Dict]:
        """One recurring concept worth remembering, as a NEUTRAL record (no
        stored prose). For memory mode: the ledger surfaces WHAT the machine has
        come to know (a recurring object, how often, across how many visits) and
        lets it re-voice the remembering — instead of replaying an old caption.
        Prefers cross-session, well-established concepts; random among the top
        for variety (memories surface unprompted).
        """
        import random as _random

        concepts = [
            c
            for c in self.get_all_concepts()
            if c.get("times_seen", 0) >= min_times_seen and len((c.get("name") or "").strip()) >= 3 and _looks_like_noun_phrase(c.get("name"))
        ]
        if not concepts:
            return None
        cross_session = [c for c in concepts if c.get("session_count", 0) > 1]
        pool = cross_session if cross_session else concepts
        return _random.choice(pool[: min(8, len(pool))])

    def get_place_inventory(self, max_items: int = 6, min_times_seen: int = 3) -> str:
        """Neutral inventory of the place — the recurring objects the machine has
        come to know, straight from the concepts ledger (the real 'what's in this
        room'), NOT an LLM prose sentence. e.g. "pink shelves, the desk, hanging
        wires". This is the ledger replacement for core_facts['place'] prose.
        """
        labels = []
        for c in self.get_all_concepts():  # already sorted by times_seen desc
            name = (c.get("name") or "").strip()
            if c.get("times_seen", 0) >= min_times_seen and _looks_like_noun_phrase(name):
                labels.append(name[0].lower() + name[1:])
                if len(labels) >= max_items:
                    break
        return ", ".join(labels)

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
            result.append(
                {
                    "id": oid,
                    "text": doc,
                    "timestamp": meta.get("timestamp", 0),
                    "session_id": meta.get("session_id", ""),
                }
            )

        result.sort(key=lambda x: x["timestamp"])
        return result

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
