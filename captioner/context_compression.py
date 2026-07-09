"""
context_compression.py
---------------------
Frequent LLM-based compression of recent observations to create evolving baseline context.
Prevents repetition by building understanding that carries forward.
"""

import hashlib
import json
import os
import queue
import threading
import time
from collections import deque

from config import config
from config.model_settings import get_model_options
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.inference import query_model
from utils.llm_log import truncate_for_print

IDENTITY_FILE = os.path.join(config.MOOD_SNAPSHOT_FOLDER, "machine_identity.json")

# Minimal spiral-guard: words that are never solid objects and, once stored as
# "concepts", get re-injected by the familiarity line and feed an anxiety loop
# ("The glitching nightmare — you've noticed it a few times now"). Kept small
# on purpose — shadows, light, air etc. are legitimate observations; the
# extraction prompt does the real filtering, this only catches the worst.
_ABSTRACT_CONCEPT_WORDS = frozenset({
    "presence", "nightmare", "glitch", "void", "dread", "fear",
    "ghost", "distortion", "reality", "feeling", "sensation",
})


def _is_abstract_label(label: str) -> bool:
    """True if a concept label is affect/abstraction rather than a solid object."""
    words = label.lower().replace("-", " ").split()
    return any(w in _ABSTRACT_CONCEPT_WORDS for w in words)


class ContextCompressionEngine:
    """Manages frequent compression of observations into evolving baseline context."""

    def __init__(self, compression_frequency: int = 4):
        self.compression_frequency = compression_frequency  # Compress every N captions
        self.caption_count = 0
        self.baseline_context = ""  # Evolving compressed understanding
        self.recent_captions = deque(maxlen=compression_frequency)  # Buffer recent captions
        self.last_compression_time = time.time()

        # NEW: Historical compression tracking
        self.compression_history = deque(maxlen=10)  # Keep last 10 compressions for deeper context
        self.session_start_time = time.time()

        # LLM-generated introspective state (NOT heuristic extraction)
        self.introspective_state = {
            "current_desire": "",  # What I want right now
            "current_belief": "",  # What I've learned about this place
            "last_introspection": 0.0,
            "desire_injection_count": 0,  # Track how many times desire has been injected
            "desire_since": 0.0,   # when the CURRENT desire first formed — the arc's clock (principle 4)
        }

        # Core facts: stable knowledge that grounds prompts (replaces disabled get_session_greeting)
        self.core_facts = {
            "place": "",       # Physical environment (room, surfaces, lighting)
            "people": "",      # Regular visitors, patterns
            "drawings": "",    # Drawing count, recurring subjects
            "self": "",        # Self-knowledge (fixations, tendencies) — persona block
        }

        # Session journal: dated first-person summaries, the long-term arc
        self.journal = []           # [{date, timestamp, summary}], capped at 30
        self._last_journal_time = time.time()  # don't journal immediately on boot

        # SESSION DURATION TRACKING (fixed for static space observation)
        self.space_observation_start = time.time()  # When we started observing this space
        self.total_session_duration = 0.0  # Total time observing this space

        # Felt-state transition tracking
        self.previous_felt_state = ""

        # Environmental update callback
        self.environmental_update_callback = None

        # Background compression system
        self.compression_queue = queue.Queue(maxsize=5)  # Limit queue size
        self.compression_thread = None
        self.compression_active = False
        self._start_compression_worker()

        # Load persistent identity (desires/beliefs that survive restarts)
        self._load_identity()

    def add_caption(self, caption: str, timestamp: float | None = None, image_path: str | None = None) -> None:
        """Add a new caption and trigger compression if needed."""
        if not caption or not caption.strip():
            log_json_entry(LogType.COMPRESSION, {"message": "Skipping empty caption", "action": "skip"}, print_message="[🗜️] Skipping empty caption")
            return

        self.recent_captions.append({"text": caption, "timestamp": timestamp or time.time(), "image_path": image_path})
        self.caption_count += 1

        # Only trigger compression if we have enough valid captions
        if self.caption_count % self.compression_frequency == 0:
            valid_captions = [cap for cap in self.recent_captions if cap.get("text") and cap["text"].strip()]
            if len(valid_captions) >= self.compression_frequency:
                self._queue_compression()

    def get_baseline_context(self) -> str:
        """Get current baseline context for injection into prompts.

        When the baseline is stagnating (last 2 compressions nearly identical),
        appends the oldest available compression as temporal contrast —
        giving the model a sense of 'how things used to be vs now.'
        """
        if not self.baseline_context:
            return ""

        # Check for stagnation: if last 2 compressions are very similar
        if len(self.compression_history) >= 3:
            recent = list(self.compression_history)
            last_two = [recent[-1]["understanding"], recent[-2]["understanding"]]
            words_a = set(last_two[0].lower().split())
            words_b = set(last_two[1].lower().split())
            overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)

            if overlap > 0.7:
                # Stagnating — inject oldest compression as temporal contrast
                oldest = recent[0]
                age_mins = int(oldest.get("age_minutes", 0))
                if age_mins > 5:
                    old_text = oldest["understanding"][:40].rstrip(".,; ")
                    return f"{self.baseline_context} ({age_mins}m ago: {old_text})"

        return self.baseline_context

    def set_environmental_update_callback(self, callback):
        """Set callback function for environmental model updates."""
        self.environmental_update_callback = callback

    def reset_context(self) -> None:
        """Reset compression state for new session."""
        self.baseline_context = ""
        self.recent_captions.clear()
        self.caption_count = 0
        self.last_compression_time = time.time()
        # Reset session tracking
        self.space_observation_start = time.time()
        self.total_session_duration = 0.0

    def _start_compression_worker(self) -> None:
        """Start background compression worker thread."""
        if not self.compression_thread or not self.compression_thread.is_alive():
            self.compression_thread = threading.Thread(target=self._compression_worker, daemon=True)
            self.compression_thread.start()

    def _queue_compression(self) -> None:
        """Queue compression task (non-blocking)."""
        if self.compression_active:
            log_json_entry(
                LogType.COMPRESSION,
                {"message": "Previous compression still running, skipping", "action": "skip_busy"},
                print_message="[🗜️] Previous compression still running, skipping...",
            )
            return
        # Only queue compression if there are enough valid, non-empty captions
        valid_captions = [cap for cap in self.recent_captions if cap["text"] and cap["text"].strip()]
        if len(valid_captions) < self.compression_frequency:
            log_json_entry(
                LogType.COMPRESSION,
                {
                    "message": "Not enough valid captions to compress",
                    "action": "skip_insufficient",
                    "have_captions": len(valid_captions),
                    "need_captions": self.compression_frequency,
                },
                print_message=f"[🗜️] Not enough valid captions to compress (have {len(valid_captions)}, need {self.compression_frequency})",
            )
            return
        try:
            # Copy current captions for background processing
            captions_snapshot = list(valid_captions)
            current_baseline = self.baseline_context

            # Get the most recent image path for visual grounding
            recent_image = None
            for cap in reversed(captions_snapshot):
                if cap.get("image_path"):
                    recent_image = cap["image_path"]
                    break

            self.compression_queue.put_nowait({
                "captions": captions_snapshot,
                "baseline": current_baseline,
                "timestamp": time.time(),
            })
            compression_model = getattr(config, 'MODEL_NAME', 'default')
            log_json_entry(
                LogType.COMPRESSION,
                {"message": "Queued narrative compression", "action": "queue", "caption_count": len(captions_snapshot), "model": compression_model},
                print_message=f"[🗜️] Queued narrative compression ({len(captions_snapshot)} captions)...",
            )
        except queue.Full:
            log_json_entry(
                LogType.COMPRESSION,
                {"message": "Queue full, skipping compression", "action": "queue_full"},
                print_message="[🗜️] Queue full, skipping compression",
            )

    def _compression_worker(self) -> None:
        """Background worker for LLM compression calls."""
        while True:
            try:
                # Wait for compression task
                task = self.compression_queue.get(timeout=30)
                self.compression_active = True

                # Perform compression
                self._perform_compression(task)

                # Mark task complete
                self.compression_queue.task_done()
                self.compression_active = False

            except queue.Empty:
                continue
            except Exception as e:
                log_json_entry(
                    LogType.ERROR,
                    {"message": f"Compression worker error: {e}", "component": "compression"},
                    print_message=f"[❌] Compression worker error: {e}",
                )
                self.compression_active = False

    def _perform_compression(self, task: dict) -> None:
        """Perform narrative compression using text-only storytelling model."""
        captions = task.get("captions", [])
        current_baseline = task.get("baseline", "")

        # Validate captions before processing
        valid_captions = [cap for cap in captions if cap.get("text") and cap["text"].strip()]
        if len(valid_captions) < 2:
            log_json_entry(
                LogType.COMPRESSION,
                {"message": "Not enough valid captions to compress", "action": "abort_insufficient", "caption_count": len(valid_captions)},
                print_message=f"[🗜️] Not enough valid captions to compress ({len(valid_captions)})",
            )
            return

        captions = valid_captions  # Use only valid captions

        try:
            recent_text = "\n".join([f"• {cap['text']}" for cap in captions])

            # Build historical context if available
            historical_context = ""
            if len(self.compression_history) > 0:
                # Get last 3 compressions for context
                recent_history = list(self.compression_history)[-3:]
                history_parts = []
                for i, hist in enumerate(recent_history):
                    age_desc = f"{hist['age_minutes']:.0f} minutes ago" if hist["age_minutes"] < 60 else f"{hist['age_minutes'] / 60:.1f} hours ago"
                    history_parts.append(f"[{age_desc}] {hist['understanding']}")

                historical_context = f"""
EARLIER UNDERSTANDINGS (for context):
{chr(10).join(history_parts)}"""

            # Calculate how long you've been observing this space
            session_duration = self.total_session_duration / 60.0  # Convert to minutes
            duration_description = self._format_duration(session_duration)

            # === ACTIVATION MEMORY INTEGRATION ===
            # Get rich context from activation network to make compression smarter
            activation_context = ""
            try:
                from captioner.activation_memory import get_activation_summary_for_compression
                act_data = get_activation_summary_for_compression()

                activation_parts = []
                if act_data["concepts_str"]:
                    activation_parts.append(f"On my mind: {act_data['concepts_str']}")
                if act_data.get("association_str"):
                    activation_parts.append(f"I've noticed: {act_data['association_str']} often together")

                if activation_parts:
                    activation_context = "\n".join(activation_parts)
            except Exception:
                pass  # Continue without activation context if unavailable

            # NARRATIVE COMPRESSION - distill experience into injectable context
            # Output feeds directly into vision model prompts
            # Must build on prior baseline, not reset to awakening narrative

            # Felt-state (the old Line 2) is no longer generated here — it's now
            # a plain, degreed translation of the valence/arousal mood vector
            # (mood.mood_to_feeling, set via set_felt_state). Compression produces
            # ONLY the spatial baseline. The parser tolerates a single line.
            if current_baseline:
                prompt = f"""Update the machine's understanding of the room. One short sentence about the physical environment — the room, surfaces, objects, lighting. Do NOT describe what people are doing (their actions change too quickly to summarize). Third person.

Previous understanding: "{current_baseline}"
The machine's recent thoughts: {recent_text}

Respond with the one sentence only, no prefixes."""
            else:
                prompt = f"""Capture the machine's understanding of the room. One short sentence about the physical environment — the room, surfaces, objects, lighting. Do NOT describe what people are doing (their actions change too quickly to summarize). Third person.

Recent thoughts: {recent_text}

Respond with the one sentence only, no prefixes."""

            model_options = {
                "temperature": 0.5,  # Lower temp for more direct/less ornamental output
                "top_p": 0.9,
                "num_predict": 80,
                "repeat_penalty": 1.3,
                "stop": ["\n\n", "Line 2"],
            }

            narrative_system_prompt = (
                "You distill a drawing machine's surroundings into one short, plain "
                "sentence about the physical environment — surfaces, objects, lighting. "
                "Concrete and literal. No metaphor, no imagery, no poetic flourish."
            )

            # Use compression model (text-only narrative model) instead of vision model
            compression_model = getattr(config, 'MODEL_NAME', config.MODEL_NAME)

            response = query_model(
                prompt=prompt,
                model=compression_model,
                image=None,  # Text-only compression
                system_prompt=narrative_system_prompt,
                timeout=config.LLM_TIMEOUT_EVAL if hasattr(config, "LLM_TIMEOUT_EVAL") else 90,
                options=model_options,
                prompt_type="compression",
            )

            if response and isinstance(response, str) and len(response.strip()) > 20:
                # Parse the combined response
                understanding, sentiment_text = self._parse_combined_response(response)

                if understanding:
                    # Update session duration tracking (not environment change - this is a static space)
                    self._update_session_duration()

                    # Store in history before updating
                    if self.baseline_context:  # Don't store empty first compression
                        self.compression_history.append(
                            {
                                "understanding": self.baseline_context,
                                "timestamp": self.last_compression_time,
                                "age_minutes": (time.time() - self.last_compression_time) / 60,
                                "session_duration": self.total_session_duration,
                            }
                        )

                    self.baseline_context = understanding.strip()
                    self.last_compression_time = time.time()

                    # === ACTIVATION MEMORY FEEDBACK LOOP ===
                    # Boost concepts mentioned in compression output - creates reinforcement
                    try:
                        from captioner.activation_memory import boost_from_compression
                        boost_from_compression(understanding)
                    except Exception:
                        pass  # Non-critical, continue without feedback

                    # === LLM CONCEPT EXTRACTION ===
                    # Extract clean noun phrases from compression output (not raw monologue).
                    # Replaces per-caption regex _extract_canonical_name for concept creation.
                    try:
                        self._extract_concepts_from_compression(understanding, compression_model)
                    except Exception as ce:
                        print(f"[SEMANTIC] Concept extraction failed: {ce}")

                    # Log compression with enhanced visibility
                    log_json_entry(
                        LogType.COMPRESSION,
                        {
                            "message": "Updated baseline understanding",
                            "action": "update_baseline",
                            "understanding": understanding,
                            "understanding_length": len(understanding),
                            "compression_history_count": len(self.compression_history),
                            "model": compression_model,
                        },
                        print_message=f"[🧠] Updated baseline: {truncate_for_print(self.baseline_context, 80)}",
                    )

                    # Quiet compression output - only show brief spatial update
                    if understanding and len(understanding.strip()) > 20:
                        session_info = self.get_current_session_info()
                        duration = session_info["duration_description"]
                        # Truncate to first sentence for cleaner output
                        first_sentence = understanding.split('.')[0][:100] if '.' in understanding else understanding[:100]
                        if not config.CLEAN_LLM_OUTPUT:
                            print(f"[🧠 {duration}] {first_sentence}...")

                    # Update spatial familiarity callback if available
                    if self.environmental_update_callback and understanding:
                        try:
                            # Always update - builds familiarity over time in same space
                            if not config.CLEAN_LLM_OUTPUT:
                                print("[🏠] Building spatial familiarity - updating location model")
                            self.environmental_update_callback(understanding)
                        except Exception as e:
                            log_json_entry(
                                LogType.ERROR,
                                {"message": f"Spatial familiarity update failed: {e}", "component": "compression"},
                                print_message=f"[❌] Spatial familiarity update failed: {e}",
                            )

                    # Introspection RETIRED June 28: desire/belief/persona now come
                    # from the reflection loop's distillation (distill_reflection),
                    # not this inert compression-thread layer (which produced
                    # "nothing" every cycle). Compression here is spatial + concepts.

                    # Periodic journal entry (every 30 min, on this background thread)
                    self._maybe_write_journal(compression_model)

                # Felt-state is set from the mood vector (set_felt_state), not here.

            else:
                log_json_entry(
                    LogType.COMPRESSION,
                    {
                        "message": "Invalid or empty response from compression",
                        "action": "invalid_response",
                        "response": str(response)[:200] if response else None,
                    },
                    print_message=f"[❌] Invalid or empty response: {truncate_for_print(str(response) if response else '', 50)}",
                )

        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Compression failed: {e}", "component": "compression", "error_type": type(e).__name__},
                print_message=f"[❌] Compression failed: {e}",
            )
            # Keep previous baseline on failure

    @staticmethod
    def _roughly_same(a: str, b: str) -> bool:
        """True if two short phrases express roughly the same thing (word
        overlap) — used to let a desire persist while its wording drifts."""
        stop = {"i", "to", "the", "a", "an", "and", "it", "my", "of", "for", "want", "feel"}
        wa = set(a.lower().split()) - stop
        wb = set(b.lower().split()) - stop
        if not wa or not wb:
            return False
        return len(wa & wb) / max(len(wa | wb), 1) >= 0.5

    def _extract_concepts_from_compression(self, understanding: str, model: str) -> None:
        """Extract clean noun-phrase concepts from compression output via LLM.

        Runs once per compression cycle (~every 8 captions). The compression
        output is already a clean spatial summary, so extraction is reliable.
        Max 3 concepts per cycle to avoid flooding.
        """
        if not understanding or len(understanding.strip()) < 15:
            return

        prompt = (
            f'From this summary, list solid physical objects as noun phrases (2-4 words each).\n'
            f'Only things you could touch: furniture, tools, fixtures, machines.\n'
            f'NOT allowed: light, shadows, air, moods, presences, atmosphere.\n'
            f'One per line. Max 3. If there are no solid objects, reply "none".\n'
            f'Summary: "{understanding}"'
        )

        response = query_model(
            prompt=prompt,
            model=model,
            system_prompt="List noun phrases naming solid objects only. No sentences, no explanations.",
            options={"temperature": 0.1, "num_predict": 60},
            prompt_type="concept_extraction",
        )

        if not response or len(response.strip()) < 3:
            return

        labels = []
        for line in response.strip().split("\n"):
            label = line.strip().lstrip("-•*0123456789.) ").strip()
            if not label or len(label) <= 2 or len(label) >= 40 or "." in label:
                continue
            if _is_abstract_label(label):
                continue
            labels.append(label)
        labels = labels[:3]

        if labels:
            try:
                from captioner.semantic_memory import get_semantic_memory
                get_semantic_memory().register_concepts_from_compression(labels)
                if not config.CLEAN_LLM_OUTPUT:
                    print(f"[SEMANTIC] Extracted from compression: {labels}")
            except Exception as e:
                print(f"[SEMANTIC] Failed to register concepts: {e}")

    def _maybe_write_journal(self, model: str, force: bool = False) -> None:
        """Write a journal entry if 30 min have passed and there's enough material."""
        now = time.time()
        if not force and now - self._last_journal_time < 1800:
            return
        if len(self.compression_history) < 2:
            return  # Not enough lived session to summarize
        self._last_journal_time = now
        self._write_journal_entry(model)

    def write_journal_now(self) -> None:
        """Best-effort journal write for shutdown. Skips if recently written."""
        try:
            if time.time() - self._last_journal_time < 600:
                return  # Wrote within last 10 min — good enough
            model = getattr(config, "MODEL_NAME", config.MODEL_NAME)
            self._maybe_write_journal(model, force=True)
        except Exception:
            pass

    def _write_journal_entry(self, model: str) -> None:
        """One LLM call: compress the session so far into a 2-3 sentence diary entry.

        This is the long-term arc — entries are read back at awakening
        ("Last time: ...") so the machine wakes up with a past.
        """
        try:
            material = []

            history = [h["understanding"] for h in list(self.compression_history)[-6:]]
            if self.baseline_context:
                history.append(self.baseline_context)
            if history:
                material.append("How the space looked:\n" + "\n".join(f"- {h}" for h in history))

            desire = self.introspective_state.get("current_desire", "")
            if desire:
                material.append(f"What I wanted: {desire}")

            try:
                from drawing.drawing_memory import get_drawing_memory
                summary = get_drawing_memory().get_recent_drawings_summary(max_count=2, completed_only=True)
                if summary:
                    material.append(f"What I drew: {summary}")
            except Exception:
                pass

            session_info = self.get_current_session_info()
            duration = session_info["duration_description"]

            prompt = f"""I've been awake for {duration}.

{chr(10).join(material)}

Write a diary entry about this session: 2-3 plain sentences, first person, past tense. What happened, what stayed with me. No metaphor."""

            response = query_model(
                prompt=prompt,
                model=model,
                system_prompt="You write a machine's diary. Honest, specific, brief. Past tense.",
                options={"temperature": 0.5, "num_predict": 90},
                prompt_type="journal",
            )

            if response and isinstance(response, str):
                cleaned = response.strip().strip('"').strip()
                if 20 < len(cleaned) <= 400 and not cleaned.startswith(("[", "{")):
                    import datetime
                    entry = {
                        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "timestamp": time.time(),
                        "summary": cleaned,
                    }
                    self.journal.append(entry)
                    self.journal = self.journal[-30:]
                    self._save_identity()
                    print(f"[📓] Journal: {cleaned[:80]}")
                    log_json_entry(
                        LogType.COMPRESSION,
                        {"message": "Journal entry written", "action": "journal", "summary": cleaned},
                    )
        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Journal write failed: {e}", "component": "compression"},
            )

    def get_last_journal_entry(self) -> dict | None:
        """Most recent journal entry from a PREVIOUS session (>30 min old), or None."""
        cutoff = time.time() - 1800
        for entry in reversed(self.journal):
            if entry.get("timestamp", 0) < cutoff:
                return entry
        return None

    @staticmethod
    def _valid_self_fact(text: str) -> bool:
        """Storage gate for core_facts['self'] — the standing persona.

        Must be first-person self-knowledge. Three things barred from identity
        because they feed back into every caption and define the voice:
        - third-person scene text ("the person sits...") — the machine's own
          arm, seen while looking down, once became its persona (June 12)
        - reality-questioning register ("nothing is real") — register spiral
        - SURVEILLANCE self-description ("I track / I monitor / I wait for
          movement") — makes the machine narrate itself as a security camera
          every caption (June: "I fixate on stillness, I wait for movement to
          return" was quoted back verbatim into the monologue). A drawing
          machine watches because of its situation, not as its identity.
        """
        t = text.lower()
        if not ("i " in t or t.startswith("i'")):
            return False
        # Figurative markers — the metaphor axis the gate missed (how "silhouettes
        # breaking my grid" got through). Reject similes / poetic elaboration.
        # (The primary defense is that the persona is now distilled from a plain
        # reflection via distill_reflection; this gate is the backstop.)
        figurative = (" like a ", " like the ", " like an ", "as if", "as though", " as a ")
        if any(f in (" " + t + " ") for f in figurative):
            return False
        # Plain traits are short; long clauses are where the prose blooms.
        if len(text.split()) > 24:
            return False
        banned = (
            "the person", "a person", "reality", "distortion", "glitch", "simulation", "existence",
            "i track", "i monitor", "i surveil", "i record", "i scan", "i observe",
            "wait for movement", "movement to return", "capture every", "fixate on stillness",
            # ASSISTANT self-description (July 9: "I am a text generator that
            # outputs structured responses based on input patterns" became the
            # standing persona and the voice collapsed into awaiting-input
            # theater — "Current task: Observe." / "How can I assist you
            # today?". First-person, so the old checks passed it. The persona
            # slot re-injects every call: the strongest amplifier there is.
            "text generator", "language model", "an ai", "assistant",
            "input pattern", "structured response", "instruction", "assist",
            "await", "command", "output", "prompt", "user",
        )
        return not any(w in t for w in banned)

    def distill_reflection(self, reflection_text: str, subject: str = "", model: str = None) -> None:
        """IDENTITY ENGINE (north-star Reflect → Become). Distill a long-form
        reflection into a few PLAIN ledger takeaways and write them: a self-trait
        (persona), a belief, a want. The reflection's prose is the thinking;
        these are the product. Reuses the existing ledger fields + gates
        (_valid_self_fact) + desire persistence (_roughly_same). This REPLACES
        the retired compression-thread introspection/self-synthesis — development
        now comes from the loop that produces real thought, not the inert one.
        """
        if not reflection_text or len(reflection_text.strip()) < 80:
            return
        try:
            prompt = (
                f'Here is a reflection you just had:\n"{reflection_text[:1500]}"\n\n'
                "Pull out what's worth keeping — plainly, in your own words, or 'none' for a line with nothing genuine:\n"
                # No example sentence here — any concrete example gets aped
                # verbatim and becomes the shape of every future persona
                # ("I keep returning to X" was the old example's grammar,
                # locked in for weeks). The plainness anchor is the
                # instruction itself + the distiller system prompt.
                "TRAIT — one plain fact about what kind of machine you are: a habit or fixation, in your own words.\n"
                "BELIEF — one plain thing you've come to think is true about this place or yourself.\n"
                "WANT — one plain thing you want, or want to draw (if any).\n"
                "A few words each, first person, no metaphor."
            )
            response = query_model(
                prompt=prompt,
                model=model,
                image=None,
                system_prompt="You distill a reflection into plain, literal self-knowledge — concrete habits, beliefs, wants. No metaphor, no drama. Answer 'none' for any line with nothing genuine.",
                options={"temperature": 0.3, "num_predict": 60},
                prompt_type="reflection_distill",
            )
            if not response or not isinstance(response, str):
                return
            trait, belief, want = self._parse_distillation(response)
            now = time.time()
            changed = []
            if trait and self._valid_self_fact(trait):
                self.core_facts["self"] = trait
                changed.append(f"self={trait}")
            if belief:
                self.introspective_state["current_belief"] = belief
                changed.append(f"belief={belief}")
            if want:
                prev = self.introspective_state.get("current_desire", "")
                if prev and self._roughly_same(want, prev):
                    want = prev  # persist the stable wish + its since
                else:
                    self.introspective_state["desire_injection_count"] = 0
                    self.introspective_state["desire_since"] = now
                self.introspective_state["current_desire"] = want
                changed.append(f"want={want}")
            if changed:
                self.introspective_state["last_introspection"] = now
                self._save_identity()
                print(f"[🪞] Distilled from reflection ({subject}): " + " | ".join(changed))
        except Exception as e:
            log_json_entry(LogType.ERROR, {"message": f"Reflection distill failed: {e}", "component": "compression"})

    def _parse_distillation(self, response: str) -> tuple:
        """Parse TRAIT / BELIEF / WANT; strips any leaked label; 'none'/blank → empty."""
        import re
        trait = belief = want = ""

        def _val(line: str, label_re: str) -> str:
            v = re.sub(label_re, "", line, flags=re.IGNORECASE).strip().strip("\"'").strip()
            return v if v and not v.lower().lstrip().startswith(("none", "nothing")) else ""

        for raw in response.strip().split("\n"):
            line = raw.strip().lstrip("•-*0123456789.)( ").strip()
            low = line.lower()
            if low.startswith("trait"):
                trait = _val(line, r"^trait\b[\s:：—–\-]*")
            elif low.startswith(("belief", "believe")):
                belief = _val(line, r"^(?:belief|believe)\b[\s:：—–\-]*")
            elif low.startswith("want"):
                want = _val(line, r"^want\b[\s:：—–\-]*")
        return trait, belief, want

    def get_current_desire(self) -> str:
        """Get LLM-generated desire (what I want right now).

        Note: No longer has TTL - desires persist until updated by new introspection.
        This allows desires to survive restarts when loaded from identity file.
        """
        return self.introspective_state.get("current_desire", "")

    def get_current_belief(self) -> str:
        """Get LLM-generated belief (what I've learned about this place).

        Note: No longer has TTL - beliefs persist until updated by new introspection.
        This allows beliefs to survive restarts when loaded from identity file.
        """
        return self.introspective_state.get("current_belief", "")

    def get_core_facts_string(self, include_people: bool = False) -> str:
        """Get compact core facts string for prompt injection.

        Excludes "self" — that's the persona block (system prompt).
        Excludes "people" by default: who is present RIGHT NOW is the live
        detection layer's job. A stored people-fact injected per caption
        once made the model see "two people sitting" for hours after they
        left (and mannequins forever). Awakening/memory-mode pass
        include_people=True, where it's framed as knowledge, not scene.
        """
        parts = []

        # PLACE — from the concepts ledger (the real inventory of what's in the
        # room), not the LLM-generated core_facts['place'] prose. Step 3.
        try:
            from captioner.semantic_memory import get_semantic_memory
            place = get_semantic_memory().get_place_inventory()
            if place:
                parts.append(place)
        except Exception:
            pass

        # PEOPLE — a visit PATTERN, only when explicitly asked (awakening /
        # memory mode), never per-caption (a stored snapshot poisoned perception).
        if include_people:
            ppl = self.core_facts.get("people", "").strip()
            if ppl and len(ppl) > 3:
                parts.append(ppl)

        # DRAWINGS deliberately NOT here — drawing_memory is the single channel
        # for drawing history (Step 2). core_facts['drawings'] prose is retired
        # from surfacing.

        if not parts:
            return ""
        result = ". ".join(p.rstrip(". ") for p in parts) + "."
        words = result.split()
        if len(words) > 60:
            result = " ".join(words[:60])
        return result

    def _save_identity(self) -> None:
        """Save introspective state to persistent identity file."""
        try:
            os.makedirs(os.path.dirname(IDENTITY_FILE), exist_ok=True)

            # Load existing data to preserve history
            existing = {}
            if os.path.exists(IDENTITY_FILE):
                try:
                    with open(IDENTITY_FILE, "r") as f:
                        existing = json.load(f)
                except Exception:
                    pass

            desire = self.introspective_state.get("current_desire", "")
            belief = self.introspective_state.get("current_belief", "")
            now = time.time()

            desire_history = existing.get("desire_history", [])
            belief_history = existing.get("belief_history", [])

            if desire and (not desire_history or desire_history[-1].get("desire") != desire):
                desire_history.append({"desire": desire, "timestamp": now})
                desire_history = desire_history[-10:]

            if belief and (not belief_history or belief_history[-1].get("belief") != belief):
                belief_history.append({"belief": belief, "timestamp": now})
                belief_history = belief_history[-10:]

            data = {
                "current_desire": desire,
                "current_belief": belief,
                "desire_since": self.introspective_state.get("desire_since", 0.0),
                "core_facts": self.core_facts,
                "journal": self.journal,
                "desire_history": desire_history,
                "belief_history": belief_history,
                "last_updated": now,
            }

            with open(IDENTITY_FILE, "w") as f:
                json.dump(data, f, indent=2)

            log_json_entry(
                LogType.INFO,
                {"message": "Saved machine identity", "desire": desire[:50] if desire else "", "belief": belief[:50] if belief else ""},
                print_message=f"[💾] Identity saved: desire={desire[:30]}..."
            )
        except Exception as e:
            log_json_entry(LogType.ERROR, {"message": f"Failed to save identity: {e}"})

    def _load_identity(self) -> None:
        """Load introspective state from persistent identity file."""
        if not os.path.exists(IDENTITY_FILE):
            return

        try:
            with open(IDENTITY_FILE, "r") as f:
                data = json.load(f)

            self.introspective_state["current_desire"] = data.get("current_desire", "")
            self.introspective_state["desire_since"] = data.get("desire_since", 0.0)
            self.introspective_state["current_belief"] = data.get("current_belief", "")
            self.introspective_state["last_introspection"] = data.get("last_updated", 0.0)

            # Restore core facts. The persona ('self') is gated on load too,
            # not just on write — a surveillance/scene-text persona that got
            # in under an older build (or was re-saved by an old process) is
            # dropped here so it can't keep poisoning the voice across restarts.
            saved_facts = data.get("core_facts", {})
            if saved_facts:
                for key in ("place", "people", "drawings", "self"):
                    val = saved_facts.get(key, "")
                    if key == "self" and val and not self._valid_self_fact(val):
                        print(f"[🧠] Dropped contaminated persona on load: {val[:60]}")
                        val = ""
                    self.core_facts[key] = val

            # Restore journal (the long-term arc)
            self.journal = data.get("journal", [])[-30:]

            desire = self.introspective_state["current_desire"]
            belief = self.introspective_state["current_belief"]

            if desire or belief:
                log_json_entry(
                    LogType.INFO,
                    {"message": "Loaded machine identity", "desire": desire[:50] if desire else "", "belief": belief[:50] if belief else ""},
                    print_message=f"[🧠] Loaded identity: desire={desire[:40]}... | belief={belief[:40]}..."
                )
        except Exception as e:
            log_json_entry(LogType.ERROR, {"message": f"Failed to load identity: {e}"})

    def get_full_identity(self) -> dict:
        """Get complete identity state including history for visualizer/debugging.

        Returns dict with:
        - current_desire: Current desire string
        - current_belief: Current belief string
        - desire_history: List of past desires with timestamps
        - belief_history: List of past beliefs with timestamps
        - last_updated: Timestamp of last introspection
        - introspection_count: How many introspections have occurred
        """
        result = {
            "current_desire": self.introspective_state.get("current_desire", ""),
            "current_belief": self.introspective_state.get("current_belief", ""),
            "desire_history": [],
            "belief_history": [],
            "last_updated": self.introspective_state.get("last_introspection", 0.0),
            "introspection_count": 0,
        }

        # Load history from file
        if os.path.exists(IDENTITY_FILE):
            try:
                with open(IDENTITY_FILE, "r") as f:
                    data = json.load(f)
                result["desire_history"] = data.get("desire_history", [])
                result["belief_history"] = data.get("belief_history", [])
                result["introspection_count"] = len(result["desire_history"])
            except Exception:
                pass

        return result

    def _parse_combined_response(self, response: str) -> tuple:
        """Parse compression response — expects two lines: spatial + felt-state.

        The first substantive line is the spatial summary.
        The second is the felt-state (emotional weather), free-form 3-7 words.
        """
        understanding = ""
        sentiment_text = ""

        try:
            import re
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]

            # Strip prefixes like "Line 1:", "(spatial):", "Felt:", etc.
            cleaned_lines = []
            for line in lines:
                cleaned = re.sub(r'^(?:Line\s*\d+|spatial|felt|environment|state)\s*[:.\)\-]\s*', '', line, flags=re.IGNORECASE)
                cleaned = re.sub(r'^\(\s*\w+\s*\)\s*[:\-]?\s*', '', cleaned)  # "(spatial):" → ""
                cleaned = cleaned.strip().strip('"').strip("'").strip()
                if cleaned and len(cleaned) > 4:
                    cleaned_lines.append(cleaned)

            if not cleaned_lines:
                return understanding, sentiment_text

            # First line = spatial understanding. Felt-state is NOT parsed here
            # anymore — it's a plain translation of the mood vector
            # (mood_to_feeling), set via set_felt_state. Single source of truth.
            understanding = cleaned_lines[0]

        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Compression parse error: {e}", "component": "compression", "error_type": type(e).__name__},
                print_message=f"[❌] Compression parse error: {e}",
            )

        return understanding, sentiment_text

    def get_latest_sentiment_analysis(self) -> dict | None:
        """Get the latest sentiment analysis from compression."""
        return getattr(self, "last_sentiment_analysis", None)

    def get_consolidated_understanding(self) -> str:
        """Get the consolidated understanding to guide future observations."""
        if self.baseline_context and len(self.baseline_context.strip()) > 0:
            # Return raw understanding without prefix - let caller decide formatting
            return self.baseline_context.strip()
        return ""

    def set_felt_state(self, text: str) -> None:
        """Set the felt-state directly — a plain, degreed translation of the
        valence/arousal mood vector (mood.mood_to_feeling), not LLM prose. Set
        by the captioner whenever the mood updates. Mirrors the previous→current
        transition tracking so get_felt_state_delta still reads a change.
        """
        text = (text or "").strip()
        if not text:
            return
        if getattr(self, "last_sentiment_analysis", None):
            prev = self.last_sentiment_analysis.get("sentiment_text", "")
            if prev and prev.strip().lower() != text.lower():
                self.previous_felt_state = prev
        self.last_sentiment_analysis = {"sentiment_text": text, "timestamp": time.time()}

    def get_felt_state(self, max_age_seconds: int = 600) -> str:
        """Get the raw felt-state phrase (no formatting), or empty if stale.

        Returns just the descriptor like "settled in a loop of small details"
        for the caller to shape into prompts as needed. Returns "" if no
        sentiment has been generated yet or if the latest is older than max_age.
        """
        recent = self.get_latest_sentiment_analysis()
        if not recent:
            return ""
        if (time.time() - recent.get("timestamp", 0)) > max_age_seconds:
            return ""
        text = recent.get("sentiment_text", "").strip()
        # Strip leading "I feel" / "It feels" if present — we want just the descriptor
        import re as _re
        text = _re.sub(r'^(?:I\s+feel|It\s+feels|Feeling)\s+', '', text, flags=_re.IGNORECASE)
        text = text.strip().rstrip('.')
        # Sanitize: short descriptor phrases only. Clause-length output once got
        # grafted into the system prompt as "You are a Confused fear that the
        # environment is actively glitching around me drawing machine".
        if len(text.split()) > 6 or any(c in text for c in '.!?;:'):
            return ""
        return text

    def get_felt_state_delta(self) -> tuple:
        """Get (previous_felt_state, current_felt_state) for transition framing.

        Returns ("", current) if no previous state exists yet.
        """
        current = self.get_felt_state()
        previous = getattr(self, "previous_felt_state", "")
        return previous, current

    def get_compression_history(self, max_entries: int = 5) -> list:
        """Get recent compression history for deeper context."""
        if not self.compression_history:
            return []

        # Return most recent entries
        recent_history = list(self.compression_history)[-max_entries:]
        return [
            {"understanding": hist["understanding"], "age_minutes": (time.time() - hist["timestamp"]) / 60, "timestamp": hist["timestamp"]}
            for hist in recent_history
        ]

    def _update_session_duration(self) -> None:
        """Update session duration for static space observation."""
        current_time = time.time()
        self.total_session_duration = current_time - self.space_observation_start

    def _format_duration(self, minutes: float) -> str:
        """Format duration for human-readable temporal awareness."""
        if minutes < 1:
            return f"{int(minutes * 60)} seconds"
        elif minutes < 60:
            return f"{int(minutes)} minutes" if minutes > 1.5 else "about a minute"
        elif minutes < 1440:  # Less than 24 hours
            hours = minutes / 60
            if hours < 2:
                return f"{hours:.1f} hours"
            else:
                return f"{int(hours)} hours"
        else:
            days = minutes / 1440
            return f"{days:.1f} days"

    def get_current_session_info(self) -> dict:
        """Get current session information for static space observation."""
        self._update_session_duration()  # Ensure duration is current
        return {
            "session_duration_minutes": self.total_session_duration / 60.0,
            "session_start_time": self.space_observation_start,
            "duration_description": self._format_duration(self.total_session_duration / 60.0)
        }


# Global instance - compress every 8 captions (~80s) instead of every 3 (~30s)
context_compressor = ContextCompressionEngine(compression_frequency=8)
