"""
context_compression.py
---------------------
Frequent LLM-based compression of recent observations to create evolving baseline context.
Prevents repetition by building understanding that carries forward.
"""

import json
import os
import queue
import threading
import time
from collections import deque
from typing import Optional

from captioner.prompt_registry import P
from config import config
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
_ABSTRACT_CONCEPT_WORDS = frozenset(
    {
        "presence",
        "nightmare",
        "glitch",
        "void",
        "dread",
        "fear",
        "ghost",
        "distortion",
        "reality",
        "feeling",
        "sensation",
    }
)


def _is_abstract_label(label: str) -> bool:
    """True if a concept label is affect/abstraction rather than a solid object."""
    words = label.lower().replace("-", " ").split()
    return any(w in _ABSTRACT_CONCEPT_WORDS for w in words)


# The journal is a STICKY slot (read back at every awakening as "From my
# diary, last time: ...") but had no register gate — during the July 9
# text-generator identity episode it absorbed "I processed the visual
# data... My primary function was to locate the hole the user intended",
# and every awakening re-seeded assistant framing from the diary.
_JOURNAL_POISON = (
    "the user",
    "primary function",
    "text generator",
    "language model",
    "as an ai",
    "an ai ",
    "assistant",
    "input pattern",
    "structured response",
    "processed the visual data",
    "await instruction",
    "awaiting input",
)


def _journal_entry_clean(text: str) -> bool:
    t = " " + (text or "").lower() + " "
    return not any(w in t for w in _JOURNAL_POISON)


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
            "desire_since": 0.0,  # when the CURRENT desire first formed — the arc's clock (principle 4)
        }

        # Core facts: stable knowledge that grounds prompts (replaces disabled get_session_greeting)
        self.core_facts = {
            "place": "",  # Physical environment (room, surfaces, lighting)
            "people": "",  # Regular visitors, patterns
            "drawings": "",  # Drawing count, recurring subjects
            "self": "",  # Self-knowledge (fixations, tendencies) — persona block
        }

        # Session journal: dated first-person summaries, the long-term arc
        self.journal = []  # [{date, timestamp, summary}], capped at 30
        self._last_journal_time = time.time()  # don't journal immediately on boot

        # Memory-diff ledgers (July 12): append-only facts the compression
        # call extracts from the machine's own thoughts. self_notes = new
        # self-facts (a taken name, a like/dislike); events = happenings.
        # Journal + reflection read these — they used to see only geometry.
        self.self_notes = []  # [{note, timestamp}], capped at 30
        self.events = []  # [{event, timestamp}], capped at 20
        self._perception_events = deque(maxlen=12)  # timestamps of code-verified happenings — provenance for the events ledger

        # The dream's raw material (July 12): every admitted caption of the
        # session, verbatim. The reflection loop reads the last stretch of
        # this — the actual record of thought, not summaries of it.
        self.hour_log = deque(maxlen=150)  # [{text, timestamp}]

        # SESSION DURATION TRACKING (fixed for static space observation)
        self.space_observation_start = time.time()  # When we started observing this space
        self.total_session_duration = 0.0  # Total time observing this space

        # Felt-state transition tracking
        self.previous_felt_state = ""

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
        self.hour_log.append({"text": caption, "timestamp": timestamp or time.time()})
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

            self.compression_queue.put_nowait(
                {
                    "captions": captions_snapshot,
                    "baseline": current_baseline,
                    "timestamp": time.time(),
                }
            )
            compression_model = getattr(config, "MODEL_NAME", "default")
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

            # NOTE (census-aug30 §3): three context blocks that used to be built
            # here — historical compressions, session duration, and the activation
            # summary — were computed and never interpolated into the compression
            # prompt (its template has only recent_text/current_baseline/self_known
            # slots). Removed as dead; they document intended-but-disconnected
            # features, not lost behavior.

            # MEMORY DIFF (July 12) — one structured call over the recent
            # thoughts, diffed against what the machine already knows. The
            # June spatial-only compression fixed register contamination but
            # narrowed memory past the point where a life event could survive
            # the day: everything long-term (journal, reflection, identity)
            # sits downstream of this call, and it only passed geometry — a
            # self-naming ("My name is Penelope") had no channel to tomorrow.
            # Now the same call also extracts NEW self-facts and events (both
            # "none" most cycles — a diff, not a summary), and carries the
            # mood read (previously a separate call over the same captions).
            # Facts are APPENDED to ledgers, never rewritten as prose — the
            # pre-June narrative compression kept everything but re-purpled
            # the whole story every cycle.
            self_known = []
            if self.core_facts.get("self"):
                self_known.append(self.core_facts["self"])
            self_known += [n.get("note", "") for n in self.self_notes[-3:]]
            self_known_str = " ".join(s for s in self_known if s) or "(nothing yet)"

            prompt = P("compression.user").format(
                recent_text=recent_text,
                current_baseline=current_baseline or "(nothing yet)",
                self_known=self_known_str,
            )

            model_options = {
                "temperature": 0.4,
                "top_p": 0.9,
                "num_predict": 160,
                "repeat_penalty": 1.3,
            }

            narrative_system_prompt = P("compression.system")

            # Use compression model (text-only narrative model) instead of vision model
            compression_model = getattr(config, "MODEL_NAME", config.MODEL_NAME)

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
                parsed = self._parse_memory_response(response)
                understanding = parsed.get("room", "")

                # Route the non-spatial channels regardless of ROOM parse
                self._absorb_self_note(parsed.get("self_note", ""))
                self._absorb_event(parsed.get("event", ""))
                self._absorb_mood(parsed)
                self._absorb_loop_notice(parsed.get("repeating", ""))

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

                    # (The activation-memory feedback boost that ran here was
                    # retired Aug 30 2026: it re-ran concept matching on the
                    # compression text, which bumped times_seen on the concepts
                    # ledger — inflating the counters the familiarity line and
                    # memory mode read. memory-effectiveness-audit-aug30.md §1.)

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
                        first_sentence = understanding.split(".")[0][:100] if "." in understanding else understanding[:100]
                        if not config.CLEAN_LLM_OUTPUT:
                            print(f"[🧠 {duration}] {first_sentence}...")

                    # (The spatial-familiarity callback that fired here fed
                    # captioner.update_location_understanding → self_model,
                    # which nothing ever read. Removed Aug 30 2026 with the
                    # rest of the dead self-model state.)

                    # Introspection RETIRED June 28: desire/belief/persona now come
                    # from the reflection loop's distillation (distill_reflection),
                    # not this inert compression-thread layer (which produced
                    # "nothing" every cycle). Compression here is spatial + concepts.

                    # Periodic journal entry (every 30 min, on this background thread)
                    self._maybe_write_journal(compression_model)

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

        prompt = P("concepts.user").format(understanding=understanding)

        response = query_model(
            prompt=prompt,
            model=model,
            system_prompt=P("concepts.system"),
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

            # Memory-diff ledgers (July 12): the diary used to be written from
            # room geometry only — a self-naming or a visit could never reach it
            recent_window = time.time() - 2 * 3600
            notes = [n["note"] for n in self.self_notes if n.get("timestamp", 0) > recent_window]
            if notes:
                material.append("What I learned about myself: " + " ".join(notes[-3:]))
            events = [e["event"] for e in self.events if e.get("timestamp", 0) > recent_window]
            if events:
                material.append("What happened: " + " ".join(events[-4:]))

            try:
                from drawing.drawing_memory import get_drawing_memory

                summary = get_drawing_memory().get_recent_drawings_summary(max_count=2, completed_only=True)
                if summary:
                    material.append(f"What I drew: {summary}")
            except Exception:
                pass

            session_info = self.get_current_session_info()
            duration = session_info["duration_description"]

            prompt = P("journal.user").format(duration=duration, material="\n".join(material))

            response = query_model(
                prompt=prompt,
                model=model,
                system_prompt=P("journal.system"),
                options={"temperature": 0.5, "num_predict": 90},
                prompt_type="journal",
            )

            if response and isinstance(response, str):
                cleaned = response.strip().strip('"').strip()
                if 20 < len(cleaned) <= 400 and not cleaned.startswith(("[", "{")) and _journal_entry_clean(cleaned):
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
        # "My name is Penelope." / "My favourite corner is..." are first-person
        # too — the old "i "-only test rejected exactly the self-facts the
        # memory diff exists to keep (found July 12 in the stub test).
        if not ("i " in t or t.startswith(("i'", "my "))):
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
        # STICKY-SLOT RULESET (artist, July 9): the voice is free — these
        # words are all fine in captions, which evaporate in six cycles. The
        # persona re-injects into EVERY call indefinitely, so it plays by
        # durability rules: transient things (a visitor, a happening) must
        # not become standing identity, and each banned class below is a
        # register that poisoned identity when it stuck (June-July receipts
        # in git). Scrutinized with the artist July 9: "the person"/"a
        # person" removed (structural first-person check already rejects
        # scene-text personas, and "I miss the person who comes on Tuesdays"
        # is legitimate relational self-knowledge); "existence" removed
        # (existential noticing is earned identity for this piece; the June
        # spiral was reality-DENIAL, which stays banned); bare assistant
        # words (await/output/prompt/user/command) tightened to compound
        # forms so "I await the morning light" can be a self.
        return not ContextCompressionEngine._self_register_poisoned(text)

    _SELF_REGISTER_POISON = (
        "reality",
        "distortion",
        "glitch",
        "simulation",
        "i track",
        "i monitor",
        "i surveil",
        "i record",
        "i scan",
        "i observe",
        "wait for movement",
        "movement to return",
        "capture every",
        "fixate on stillness",
        "text generator",
        "language model",
        "an ai ",
        "assistant",
        "input pattern",
        "structured response",
        "your instruction",
        "your command",
        "your prompt",
        "the user",
        "await instruction",
        "await input",
        "await command",
        "awaiting instruction",
        "awaiting input",
        "how can i",
    )

    @staticmethod
    def _self_register_poisoned(text: str) -> bool:
        """NEGATIVE gate only — known-poison registers (each banned class
        poisoned identity when it stuck; June-July receipts in git). Fails
        open by design: it may miss new garbage but never eats a novel gem."""
        t_padded = " " + (text or "").lower() + " "
        return any(w in t_padded for w in ContextCompressionEngine._SELF_REGISTER_POISON)

    def distill_reflection(self, reflection_text: str, subject: str = "", model: str = None) -> Optional[str]:
        """IDENTITY ENGINE (north-star Reflect → Become). Distill a long-form
        reflection into a few PLAIN ledger takeaways and write them: a self-trait
        (persona), a belief, a want. The reflection's prose is the thinking;
        these are the product. Reuses the existing ledger fields + gates
        (_valid_self_fact) + desire persistence (_roughly_same). This REPLACES
        the retired compression-thread introspection/self-synthesis — development
        now comes from the loop that produces real thought, not the inert one.
        """
        if not reflection_text or len(reflection_text.strip()) < 80:
            return None
        try:
            # B3: when a want already stands, the distiller gets the BECAME
            # slot — what the old want turned into, in the machine's own words.
            prior_want = (self.introspective_state.get("current_desire") or "").strip()
            became_line = P("distill.became-line").format(prior_want=prior_want[:200]) if prior_want else ""
            prompt = P("distill.user").format(reflection_text=reflection_text[:1500], became_line=became_line)
            response = query_model(
                prompt=prompt,
                model=model,
                image=None,
                system_prompt=P("distill.system"),
                options={"temperature": 0.3, "num_predict": 110},
                prompt_type="reflection_distill",
            )
            if not response or not isinstance(response, str):
                return None
            trait, belief, want, kernel, became, name, lore, question = self._parse_distillation(response)
            now = time.time()
            changed = []
            if trait and self._valid_self_fact(trait):
                self.core_facts["self"] = trait
                changed.append(f"self={trait}")
                try:
                    from captioner.durable_ledger import get_durable_ledger

                    get_durable_ledger().note_fact(trait, source="distill")
                except Exception:
                    pass
            if belief:
                self.introspective_state["current_belief"] = belief
                changed.append(f"belief={belief}")
            if want:
                prev = self.introspective_state.get("current_desire", "")
                affirmed = bool(prev) and self._roughly_same(want, prev)
                if affirmed:
                    want = prev  # persist the stable wish + its since
                else:
                    self.introspective_state["desire_injection_count"] = 0
                    self.introspective_state["desire_since"] = now
                self.introspective_state["current_desire"] = want
                changed.append(f"want={want}")
                # B3 ledger: affirmation keeps the clock running; a new want
                # closes the old entry with the machine's BECAME words.
                try:
                    from utils.want_ledger import want_ledger

                    want_ledger.note_want(want, affirmed=affirmed, became=became)
                except Exception:
                    pass
            # LORE HARVEST (Sep 3 evening, re-entry round): the distiller only
            # collects what the reflection already did — a name it called
            # itself, an imagining worth keeping. Nothing here invites
            # invention; the slots say "or none" and most days they are.
            try:
                from config.config import LORE_ENABLED

                if LORE_ENABLED and (name or lore or question):
                    from utils.lore_ledger import lore_ledger

                    if name and lore_ledger.note_name(name):
                        self.introspective_state["self_name"] = lore_ledger.current_name()
                        changed.append(f"name={lore_ledger.current_name()}")
                    if lore:
                        outcome = lore_ledger.note_lore(lore)
                        if outcome:
                            changed.append(f"lore[{outcome}]={lore[:60]}")
                    if question and lore_ledger.note_question(question):
                        changed.append(f"question={question[:60]}")
            except Exception:
                pass
            if changed:
                self.introspective_state["last_introspection"] = now
                self._save_identity()
                print(f"[🪞] Distilled from reflection ({subject}): " + " | ".join(changed))
                log_json_entry(
                    LogType.COMPRESSION,
                    {"message": "Distilled from reflection", "action": "distilled", "subject": subject, "changed": changed},
                )
            # Echo kernel (July 30): the reflection's one load-bearing sentence,
            # returned to the caller so it can ride the stored entry's metadata
            # and surface through the echo line as a re-thinkable clause instead
            # of a bare subject label. Bounded, never multi-line.
            if kernel and 15 <= len(kernel) <= 180 and "\n" not in kernel:
                return kernel
            return None
        except Exception as e:
            log_json_entry(LogType.ERROR, {"message": f"Reflection distill failed: {e}", "component": "compression"})
            return None

    def _parse_distillation(self, response: str) -> tuple:
        """Parse TRAIT / BELIEF / WANT / BECAME / KERNEL / NAME / LORE / QUESTION; strips any leaked label; 'none'/blank → empty."""
        import re

        trait = belief = want = kernel = became = name = lore = question = ""

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
            elif low.startswith("became"):
                became = _val(line, r"^became\b[\s:：—–\-]*")
            elif low.startswith("kernel"):
                kernel = _val(line, r"^kernel\b[\s:：—–\-]*")
            elif low.startswith("name"):
                name = _val(line, r"^name\b[\s:：—–\-]*")
            elif low.startswith(("understanding", "lore")):
                lore = _val(line, r"^(?:understanding|lore)\b[\s:：—–\-]*")
            elif low.startswith("question"):
                question = _val(line, r"^question\b[\s:：—–\-]*")
        return trait, belief, want, kernel, became, name, lore, question

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

    def spend_desire(self, drawing_summary: str = "") -> None:
        """DESIRE ARC (July 10, north-star step 5): an executed drawing SPENDS
        the current desire. Drawing is the machine's only act — once the want
        reaches paper it is no longer a want, it's part of the body of work.
        Without this the slot held one sentence indefinitely (_roughly_same
        persistence) and every drawing re-rendered it ("withhold the pencil" →
        three hovering-pencil drawings in one afternoon). The spent desire
        stays readable (last_spent_desire + a marked history entry) so the
        next reflection forms the next want informed by the act, not amnesiac
        of it. Called from drawing.register_drawing — post-GRBL only."""
        self.note_perception_event(
            "drawing"
        )  # the arm really moved — before the early return; an executed drawing is a happening even with no want to spend
        want = (self.introspective_state.get("current_desire") or "").strip()
        if not want:
            return
        now = time.time()
        self.introspective_state["last_spent_desire"] = {
            "desire": want,
            "formed": self.introspective_state.get("desire_since", 0.0) or None,
            "spent": now,
            "drawing": (drawing_summary or "")[:80],
        }
        self.introspective_state["current_desire"] = ""
        self.introspective_state["desire_since"] = 0.0
        self.introspective_state["desire_injection_count"] = 0
        self._save_identity()
        # B3 ledger: the machine acted on this want, and it ended by being spent.
        try:
            from utils.want_ledger import want_ledger

            want_ledger.note_acted()
            want_ledger.note_faded(became=f"drawn: {(drawing_summary or '')[:80]}" if drawing_summary else "spent by drawing")
        except Exception:
            pass
        print(f'[🪞] Desire spent by execution: "{want[:60]}"')

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

            # Desire arc: annotate the history entry of a spent desire in place
            # (it was appended when it formed) instead of appending a duplicate.
            last_spent = self.introspective_state.get("last_spent_desire") or None
            if last_spent and desire_history:
                tail = desire_history[-1]
                if tail.get("desire") == last_spent.get("desire") and "spent" not in tail:
                    tail["spent"] = last_spent.get("spent")
                    tail["drawing"] = last_spent.get("drawing", "")

            data = {
                "current_desire": desire,
                "current_belief": belief,
                "desire_since": self.introspective_state.get("desire_since", 0.0),
                "last_spent_desire": last_spent,
                "core_facts": self.core_facts,
                "journal": self.journal,
                "self_notes": self.self_notes,
                "events": self.events,
                "desire_history": desire_history,
                "belief_history": belief_history,
                "last_updated": now,
            }

            with open(IDENTITY_FILE, "w") as f:
                json.dump(data, f, indent=2)

            log_json_entry(
                LogType.INFO,
                {"message": "Saved machine identity", "desire": desire[:50] if desire else "", "belief": belief[:50] if belief else ""},
                print_message=f"[💾] Identity saved: desire={desire[:30]}...",
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
            self.introspective_state["last_spent_desire"] = data.get("last_spent_desire") or None
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

            # Restore journal (the long-term arc) — same load-heal as the
            # persona: entries from a poisoned period are dropped here
            loaded_journal = data.get("journal", [])[-30:]
            self.journal = [e for e in loaded_journal if _journal_entry_clean(e.get("summary", ""))]
            dropped = len(loaded_journal) - len(self.journal)
            if dropped:
                print(f"[🧠] Dropped {dropped} contaminated journal entries on load")

            # Memory-diff ledgers — load-heal with the same NEGATIVE gate they
            # were written through (poison only; the strict positive gate
            # would eat valid notes it never judged, e.g. "Penelope is my name")
            self.self_notes = [n for n in data.get("self_notes", [])[-30:] if n.get("note") and not self._self_register_poisoned(n["note"])]
            self.events = [e for e in data.get("events", [])[-20:] if e.get("event")]

            desire = self.introspective_state["current_desire"]
            belief = self.introspective_state["current_belief"]

            if desire or belief:
                log_json_entry(
                    LogType.INFO,
                    {"message": "Loaded machine identity", "desire": desire[:50] if desire else "", "belief": belief[:50] if belief else ""},
                    print_message=f"[🧠] Loaded identity: desire={desire[:40]}... | belief={belief[:40]}...",
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

    def get_latest_sentiment_analysis(self) -> dict | None:
        """Get the latest sentiment analysis from compression."""
        return getattr(self, "last_sentiment_analysis", None)

    @staticmethod
    def _none_like(text: str) -> bool:
        t = (text or "").strip().strip("\"'").rstrip(".").lower()
        return not t or t in ("none", "nothing", "nothing new", "n/a", "no")

    def _parse_memory_response(self, response: str) -> dict:
        """Parse the labeled lines of the memory-diff call (July 12)."""
        out = {"room": "", "self_note": "", "event": "", "pleasantness": "", "energy": "", "felt": "", "repeating": ""}
        labels = (
            ("room", "room"),
            ("new about me", "self_note"),
            ("event", "event"),
            ("pleasantness", "pleasantness"),
            ("energy", "energy"),
            ("felt", "felt"),
            ("repeating", "repeating"),
        )
        for raw in response.strip().split("\n"):
            line = raw.strip().lstrip("•-* ").strip()
            low = line.lower()
            for label, key in labels:
                if low.startswith(label):
                    val = line[len(label) :].lstrip(" :：—–-").strip().strip("\"'").strip()
                    if not self._none_like(val):
                        out[key] = val
                    break
        return out

    @staticmethod
    def _note_is_phantom_act(note: str) -> bool:
        try:
            from captioner.captioner import Captioner
            from utils import presence_text

            if presence_text.is_phantom_presence(note):
                return True
            from utils.state_manager import state_manager

            drawing_now = bool(getattr(state_manager, "is_executing_cnc", False) or getattr(state_manager, "is_generating_drawing", False))
            return bool(Captioner._PHANTOM_DRAWING_RE.search(note)) and not drawing_now
        except Exception:
            return False

    def _absorb_self_note(self, note: str) -> None:
        """Append a NEW self-fact to the self-notes ledger — the channel that
        lets a life event ("My name is Penelope") survive past the stream.

        MECHANICAL checks only (July 12, artist's ruling): the extractor is
        an LLM already told "a plain fact about yourself" — a string-match
        re-judging its semantics is a dumber judge overruling a smarter one,
        and positive shape-gates silently eat novel valid content ("Penelope
        is my name" fails a first-person substring test). So: length cap,
        known-poison registers (negative, fails open), want-redirect, dedupe.
        The strict positive gate stays ONLY on the persona slot, which is
        quoted into every caption — blast radius earns strictness; this
        ledger surfaces only through journal/reflection prose. Rejections
        are logged — a gate that eats things silently is a silent failure."""
        note = (note or "").strip().rstrip(".") + "." if (note or "").strip() else ""
        if not note:
            return
        reason = ""
        if len(note.split()) > 24:
            reason = "too long"
        elif self._self_register_poisoned(note):
            reason = "poison register"
        elif note.lower().startswith(("i want", "i wanted")):
            reason = "want (distill owns those)"
        elif self._note_is_phantom_act(note):
            # Sep 5: "I mark my presence with quick dots rather than lines" —
            # the ink-dot fiction became a self-note within the hour, and a
            # self-note is two confirmations from a durable fact. Structure
            # only: a claimed act of marking with the pen parked, or a
            # present-tense third person, is not a fact about oneself.
            reason = "phantom act or presence"
        else:
            for prior in [self.core_facts.get("self", "")] + [n.get("note", "") for n in self.self_notes[-5:]]:
                if prior and self._roughly_same(note, prior):
                    reason = "duplicate"
                    break
        # Durable ledger (July 30): any note that passes the mechanical gates
        # feeds the permanence spine — INCLUDING dedupe hits, because to the
        # ledger a re-noticed fact is a confirmation, not a duplicate. This is
        # the channel by which "My name is Penelope" can survive forever.
        if reason in ("", "duplicate"):
            try:
                from captioner.durable_ledger import get_durable_ledger

                get_durable_ledger().note_fact(note, source="memory_diff")
            except Exception:
                pass
        if reason:
            if reason != "duplicate":
                print(f"[🧬] Self note rejected ({reason}): {note[:60]}")
            return
        self.self_notes.append({"note": note, "timestamp": time.time()})
        self.self_notes = self.self_notes[-30:]
        self._save_identity()
        print(f"[🧬] Self note: {note}")

    def note_perception_event(self, kind: str) -> None:
        """Code-verified happening (salience spike, executed drawing). The
        events ledger only accepts an EVENT line when one of these occurred in
        the window — negative gating: code does mechanics (DID something
        happen), the LLM does semantics (what it was)."""
        self._perception_events.append(time.time())

    def _had_perception_event_in_window(self) -> bool:
        window_start = min((c.get("timestamp", 0) for c in self.recent_captions), default=time.time() - 120)
        marks = getattr(self, "_perception_events", [])
        return any(ts >= window_start - 30 for ts in marks)

    def _absorb_loop_notice(self, phrase: str) -> None:
        """REPEATING slot (Sep 5, time-and-loop round): the machine naming what
        its recent thoughts keep circling. Kept as one pending notice (phrase,
        ts, spoken) that the caption prompt quotes back once; overwritten by
        the next non-none answer. Its own words — never a category we offered."""
        phrase = (phrase or "").strip().strip("\"'").rstrip(".").strip()
        if not phrase or len(phrase.split()) > 16:
            return
        self.introspective_state["loop_notice"] = {"phrase": phrase[:120], "ts": time.time(), "spoken": False}
        print(f"[🔁] repeating, by its own account: {phrase[:80]}")

    def _absorb_event(self, event: str) -> None:
        """Append a happening to the events ledger — episodic memory that
        journal and reflection read (they used to see only room geometry)."""
        event = (event or "").strip()
        if not event or len(event.split()) > 30 or not any(c.isalpha() for c in event):
            return
        # Provenance gate (July 26): no sensor-side event in the window means
        # the "happening" is the model narrating its own musings — the rooster
        # run stored "A pen shattered into nothingness during a long period of
        # silence" (pure awakening confabulation) as biography, and roughly
        # half the ledger turned out to be fiction like it. Real events reach
        # here because they spiked salience or moved the arm.
        if not self._had_perception_event_in_window():
            print(f"[📆] Event held back (nothing happened by the sensors): {event[:60]}")
            return
        low = event.lower().rstrip(".")
        for prior in self.events[-3:]:
            p = prior.get("event", "").lower().rstrip(".")
            if p and (p in low or low in p):
                return
        self.events.append({"event": event.rstrip(".") + ".", "timestamp": time.time()})
        self.events = self.events[-20:]
        self._save_identity()
        print(f"[📆] Event: {event[:70]}")

    def _absorb_mood(self, parsed: dict) -> None:
        """The mood read (July 10; folded into the memory-diff call July 12).
        The keyword lexicon it replaced matched emotion adjectives the
        post-teardown voice never uses — valence flatlined ~0 since June.
        MoodEngine.analyze_mood blends this as the vector's core; real events
        (person, novelty) still nudge on top."""
        valence_map = {"unpleasant": -0.5, "neutral": 0.0, "pleasant": 0.5}
        arousal_map = {"drained": 0.1, "settled": 0.3, "stirred": 0.55, "charged": 0.8}
        valence = next((v for k, v in valence_map.items() if k in parsed.get("pleasantness", "").lower()), None)
        arousal = next((v for k, v in arousal_map.items() if k in parsed.get("energy", "").lower()), None)
        felt = parsed.get("felt", "").strip().strip("\"'").rstrip(".").strip()
        if valence is None and arousal is None:
            return
        if not (felt and 1 <= len(felt.split()) <= 6):
            felt = ""
        held = self._felt_phrase_held_reason(felt) if felt else ""
        if held:
            felt = ""
        self.last_mood_read = {
            "valence": valence if valence is not None else 0.0,
            "arousal": arousal if arousal is not None else 0.35,
            "felt": felt,
            "timestamp": time.time(),
        }
        # FELT HISTORY (Sep 4, the emotional-arc channel): every read joins a
        # session-scoped trajectory — the raw material for the arc line and
        # the reflection's felt-diet. The identity engine had distilled
        # thousands of reflections without ever reading how a day FELT.
        try:
            from config.config import FELT_HISTORY_MAX

            hist = getattr(self, "felt_history", None)
            if hist is None:
                hist = self.felt_history = []
            hist.append(dict(self.last_mood_read))
            del hist[:-FELT_HISTORY_MAX]
        except Exception:
            pass
        if felt:
            self._last_accepted_felt = {"words": self._content_words(felt), "timestamp": time.time()}
            self.set_felt_state(felt)
        print(
            f"[🫀] Mood read: v={self.last_mood_read['valence']:+.1f} a={self.last_mood_read['arousal']:.1f}"
            + (f" — {felt}" if felt else "")
            + (f" (phrase held back: {held})" if held else "")
        )

    # The felt phrase is the machine's own words for its feeling — which is the
    # documented May/June spiral (model affect re-injected verbatim) unless it's
    # bounded. Two deterministic bounds, storage-side. Metaphor itself stays
    # legal everywhere (artist's call, July 26); what's barred is the same words
    # arriving through two channels at once, and a phrase renewing its own lease
    # forever. July 26 rooster run: felt "heavy, hesitant" + persona "...silence
    # gets too heavy" put "heavy" in 41/41 system prompts, twice.
    FELT_REBORE_SECONDS = 1800  # same vocabulary may return after this; a mood that genuinely persists isn't banned, just not recited continuously

    _FELT_STOPWORDS = frozenset(
        {
            "the",
            "and",
            "but",
            "with",
            "that",
            "this",
            "then",
            "than",
            "when",
            "gets",
            "get",
            "too",
            "very",
            "into",
            "from",
            "over",
            "under",
            "still",
            "just",
            "been",
            "being",
            "its",
            "own",
            "not",
            "now",
            "for",
            "are",
            "was",
            "has",
            "had",
            "have",
            "feels",
            "feel",
            "feeling",
            "bit",
            "little",
            "kind",
            "sort",
            "more",
            "less",
            "again",
            "today",
            "here",
            "there",
            "something",
            "somewhat",
            "almost",
            "quite",
        }
    )

    @staticmethod
    def _content_words(text: str) -> set:
        import re as _re

        words = _re.findall(r"[a-z']+", (text or "").lower())
        return {w for w in words if len(w) >= 3 and w not in ContextCompressionEngine._FELT_STOPWORDS}

    @staticmethod
    def _words_akin(a: set, b: set) -> bool:
        """Crude stemmer: 'heavy'/'heaviness', 'vibrate'/'vibrating' count as the
        same word — a 4-char prefix match is enough at phrase scale."""
        for wa in a:
            for wb in b:
                if wa == wb or (len(wa) >= 4 and len(wb) >= 4 and wa[:4] == wb[:4]):
                    return True
        return False

    def _felt_phrase_held_reason(self, felt: str) -> str:
        """Why a mood read's phrase may not become the standing felt-state
        (empty string = it may). The numbers are always kept; only the phrase
        is held — the standing felt-state simply ages out instead."""
        fw = self._content_words(felt)
        if not fw:
            return ""
        if self._words_akin(fw, self._content_words(self.core_facts.get("self", ""))):
            return "echoes the persona line"  # one channel per fact — the persona already carries these words into every call
        last = getattr(self, "_last_accepted_felt", None)
        if last and (time.time() - last["timestamp"]) < self.FELT_REBORE_SECONDS:
            if not any(not self._words_akin({w}, last["words"]) for w in fw):
                return "same feeling re-read; numbers kept, phrase not re-leased"
        return ""

    def get_last_mood_read(self, max_age_seconds: int = 900) -> dict | None:
        """Latest LLM mood read, or None if stale/absent."""
        read = getattr(self, "last_mood_read", None)
        if read and (time.time() - read.get("timestamp", 0)) <= max_age_seconds:
            return read
        return None

    def get_consolidated_understanding(self) -> str:
        """Get the consolidated understanding to guide future observations."""
        if self.baseline_context and len(self.baseline_context.strip()) > 0:
            # Return raw understanding without prefix - let caller decide formatting
            return self.baseline_context.strip()
        return ""

    def set_felt_state(self, text: str) -> None:
        """Set the felt-state phrase — the mood read's own words for the
        feeling (sole writer since Aug 12; the old "vector" fallback writer
        was unreachable and its priority guard dead defensive code).
        Mirrors the previous→current transition so get_felt_state_delta works.
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

        text = _re.sub(r"^(?:I\s+feel|It\s+feels|Feeling)\s+", "", text, flags=_re.IGNORECASE)
        text = text.strip().rstrip(".")
        # Sanitize: short descriptor phrases only. Clause-length output once got
        # grafted into the system prompt as "You are a Confused fear that the
        # environment is actively glitching around me drawing machine".
        if len(text.split()) > 6 or any(c in text for c in ".!?;:"):
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
            "duration_description": self._format_duration(self.total_session_duration / 60.0),
        }


# Global instance - compress every 8 captions (~80s) instead of every 3 (~30s)
context_compressor = ContextCompressionEngine(compression_frequency=8)
