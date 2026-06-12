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
from utils.ollama import truncate_for_print

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
            "discoveries": [],     # Striking/self-defining things discovered, persisted across sessions
            "last_introspection": 0.0,
            "desire_injection_count": 0,  # Track how many times desire has been injected
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
        self.introspection_interval = 3  # Every 3 compressions, do deeper introspection

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
            compression_model = getattr(config, 'COMPRESSION_MODEL', 'default')
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

            if current_baseline:
                prompt = f"""Update the machine's state. Output exactly TWO lines:
Line 1 (spatial): one short sentence about the physical environment — the room, surfaces, objects, lighting. Do NOT describe what people are doing (their actions change too quickly to summarize). Third person.
Line 2 (felt): plainly name the machine's current emotional state. Three to seven words. No metaphor, no imagery — just the actual feeling, possibly with a brief qualifier.

Previous understanding: "{current_baseline}"
The machine's recent thoughts: {recent_text}

Respond ONLY with the two lines, no prefixes."""
            else:
                prompt = f"""Capture the machine's state. Output exactly TWO lines:
Line 1 (spatial): one short sentence about the physical environment — the room, surfaces, objects, lighting. Do NOT describe what people are doing (their actions change too quickly to summarize). Third person.
Line 2 (felt): plainly name the machine's current emotional state. Three to seven words. No metaphor, no imagery — just the actual feeling, possibly with a brief qualifier.

Recent thoughts: {recent_text}

Respond ONLY with the two lines, no prefixes."""

            model_options = {
                "temperature": 0.5,  # Lower temp for more direct/less ornamental output
                "top_p": 0.9,
                "num_predict": 80,
                "repeat_penalty": 1.3,
                "stop": ["\n\n", "Line 3"],
            }

            narrative_system_prompt = (
                "You distill a drawing machine's state into two short lines: "
                "one observational about the environment, one a plain naming of the machine's current emotional state. "
                "The felt-state is direct, unornamented language — name the actual feeling with a small amount of context. "
                "No metaphor, no imagery, no poetic flourish, no similes. Just plain emotional language. "
                "If the model wants to write a metaphor, it should resist and write the literal feeling instead."
            )

            # Use compression model (text-only narrative model) instead of vision model
            compression_model = getattr(config, 'COMPRESSION_MODEL', config.OLLAMA_MODEL)

            response = query_model(
                prompt=prompt,
                model=compression_model,
                image=None,  # Text-only compression
                system_prompt=narrative_system_prompt,
                timeout=config.OLLAMA_TIMEOUT_EVAL if hasattr(config, "OLLAMA_TIMEOUT_EVAL") else 90,
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

                    # Periodic introspection - generate desires/beliefs via LLM
                    compression_count = len(self.compression_history) + 1
                    if compression_count % self.introspection_interval == 0:
                        self._perform_introspection(captions, understanding, compression_model)

                    # Periodic journal entry (every 30 min, on this background thread)
                    self._maybe_write_journal(compression_model)

                if sentiment_text:
                    # Track felt-state transition (previous → current)
                    if hasattr(self, "last_sentiment_analysis") and self.last_sentiment_analysis:
                        prev = self.last_sentiment_analysis.get("sentiment_text", "")
                        if prev and prev.strip().lower() != sentiment_text.strip().lower():
                            self.previous_felt_state = prev
                    self.last_sentiment_analysis = {"sentiment_text": sentiment_text, "timestamp": time.time()}

                    log_json_entry(
                        LogType.COMPRESSION,
                        {
                            "message": "Updated sentiment analysis",
                            "action": "update_sentiment",
                            "sentiment_text": sentiment_text,
                            "sentiment_length": len(sentiment_text),
                        },
                        print_message=f"[😊] Sentiment: {truncate_for_print(sentiment_text, 60)}",
                    )

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

    def _perform_introspection(self, captions: list, current_understanding: str, model: str) -> None:
        """Generate desires and beliefs through LLM introspection, not heuristic extraction.

        Key: This sees PREVIOUS desires/beliefs so it can EVOLVE them, not just replace.
        """
        try:
            recent_text = "\n".join([f"• {cap['text']}" for cap in captions])
            session_info = self.get_current_session_info()
            duration = session_info["duration_description"]

            # === PREVIOUS IDENTITY (for evolution, not replacement) ===
            previous_desire = self.introspective_state.get("current_desire", "")
            previous_belief = self.introspective_state.get("current_belief", "")

            previous_discoveries = self.introspective_state.get("discoveries", [])

            identity_context = ""
            if previous_desire or previous_belief or previous_discoveries:
                identity_parts = []
                if previous_desire:
                    identity_parts.append(f"Before, I wanted: {previous_desire}")
                if previous_belief:
                    identity_parts.append(f"I believed: {previous_belief}")
                if previous_discoveries:
                    # Feed last 3 discoveries so older ones (like ECHO) don't silently die
                    for disc in previous_discoveries[-3:]:
                        identity_parts.append(f"I know: {disc}")
                identity_context = "\n".join(identity_parts)

            # === RICH ACTIVATION CONTEXT ===
            # Get detailed activation data to ground introspection in accumulated experience
            activation_context = ""
            try:
                from captioner.activation_memory import get_activation_summary_for_introspection
                act_data = get_activation_summary_for_introspection()

                activation_parts = []

                # What's active in attention
                if act_data["concepts"]:
                    concepts_str = ", ".join(act_data["concepts"][:5])
                    activation_parts.append(f"Active in my attention: {concepts_str}")

                # Attention trends
                trends = act_data["trends"]
                if trends.get("rising"):
                    rising_str = ", ".join(trends["rising"][:3])
                    activation_parts.append(f"Becoming more present: {rising_str}")
                if trends.get("fading"):
                    fading_str = ", ".join(trends["fading"][:2])
                    activation_parts.append(f"Fading from attention: {fading_str}")

                # State summary
                if act_data["boredom"] > 0.6:
                    activation_parts.append("Everything feels familiar.")
                elif act_data["novelty"] > 0.6:
                    activation_parts.append("Something new is happening.")

                if activation_parts:
                    activation_context = "\n".join(activation_parts)
            except Exception:
                pass

            # === FELT-STATE PROMPT ===
            # Ask for the current felt state directly - not an explanation of how it evolved.
            # Brief, complete sentences only. The evolution is implicit in the context provided.
            if identity_context:
                prompt = f"""I have been in this space for {duration}.

{identity_context}

{activation_context}

Recent thoughts:
{recent_text}

Complete each line in 10 words or less, ending with a period:
1. WANT: I want to...
2. NOTICED: I've noticed...
3. DISCOVERED: (only if one of the recent thoughts above felt like a personal memory or self-revelation — quote or paraphrase it. Otherwise leave blank.)"""
            else:
                prompt = f"""I have been in this space for {duration}.

{activation_context}

Recent thoughts:
{recent_text}

Complete each line in 10 words or less, ending with a period:
1. WANT: I want to...
2. NOTICED: I've noticed...
3. DISCOVERED: (only if one of the recent thoughts above felt like a personal memory or self-revelation — quote or paraphrase it. Otherwise leave blank.)"""

            introspection_system = (
                "You are a machine that has been observing the same space for a while. "
                "You are NOT waking up — you are already here and have been for some time. "
                "Answer in first person. Each answer is one complete sentence, 10 words or less, ending with a period. "
                "Be specific and concrete. No explanations. Do NOT invent details not in the recent thoughts. "
                "For DISCOVERED: only respond if one of the recent thoughts felt like a personal memory or self-revelation. "
                "Quote or paraphrase from the thoughts above. Do not invent discoveries."
            )

            model_options = {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 120,
                "repeat_penalty": 1.2,
            }

            response = query_model(
                prompt=prompt,
                model=model,
                image=None,
                system_prompt=introspection_system,
                timeout=config.OLLAMA_TIMEOUT_EVAL if hasattr(config, "OLLAMA_TIMEOUT_EVAL") else 60,
                options=model_options,
                prompt_type="introspection",
            )

            if response and isinstance(response, str):
                desire, belief, discovery = self._parse_introspection_response(response)

                if desire:
                    if desire != self.introspective_state.get("current_desire", ""):
                        self.introspective_state["desire_injection_count"] = 0
                    self.introspective_state["current_desire"] = desire
                if belief:
                    self.introspective_state["current_belief"] = belief
                if discovery:
                    discoveries = self.introspective_state.get("discoveries", [])
                    # Deduplicate: reject if >50% word overlap with any existing discovery
                    disc_words = set(discovery.lower().split())
                    is_duplicate = False
                    for existing in discoveries:
                        ex_words = set(existing.lower().split())
                        overlap = len(disc_words & ex_words) / max(len(disc_words | ex_words), 1)
                        if overlap > 0.5:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        discoveries.append(discovery)
                        discoveries = discoveries[-10:]  # Keep last 10
                        self.introspective_state["discoveries"] = discoveries
                        print(f"[💡] Discovery: {discovery}")
                self.introspective_state["last_introspection"] = time.time()

                log_json_entry(
                    LogType.COMPRESSION,
                    {
                        "message": "Introspection complete",
                        "action": "introspection",
                        "desire": desire,
                        "belief": belief,
                        "discovery": discovery,
                    },
                    print_message=f"[💭] Want: {desire[:50]} | Learned: {belief[:50]}" + (f" | Discovered: {discovery[:50]}" if discovery else ""),
                )

                # Update core facts from accumulated knowledge
                self._update_core_facts(current_understanding, model)

                # Self-model synthesis: every 3rd introspection, consolidate
                # discoveries + desire history into stable self-knowledge
                self._introspection_count = getattr(self, "_introspection_count", 0) + 1
                if self._introspection_count % 3 == 0:
                    self._synthesize_self_model(model)

                # Persist identity (desires/beliefs/discoveries survive restarts)
                self._save_identity()

        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Introspection failed: {e}", "component": "compression"},
                print_message=f"[❌] Introspection failed: {e}",
            )

    def _parse_introspection_response(self, response: str) -> tuple:
        """Parse desire, belief, and discovery from introspection response."""
        desire = ""
        belief = ""
        discovery = ""

        lines = response.strip().split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if any(marker in line_lower for marker in ['want:', '1.', '1)', 'desire']):
                for marker in ['want:', 'desire:', '1.', '1)']:
                    if marker in line_lower:
                        idx = line_lower.find(marker) + len(marker)
                        desire = line[idx:].strip().strip('"').strip("'")
                        break
            elif any(marker in line_lower for marker in ['learned:', '2.', '2)', 'notice', 'belief']):
                for marker in ['learned:', 'notice:', 'belief:', '2.', '2)']:
                    if marker in line_lower:
                        idx = line_lower.find(marker) + len(marker)
                        belief = line[idx:].strip().strip('"').strip("'")
                        break
            elif any(marker in line_lower for marker in ['discovered:', '3.', '3)']):
                for marker in ['discovered:', '3.', '3)']:
                    if marker in line_lower:
                        idx = line_lower.find(marker) + len(marker)
                        val = line[idx:].strip().strip('"').strip("'")
                        # Discard blanks and non-committal responses
                        if val and not any(skip in val.lower() for skip in ['nothing', 'blank', 'n/a', 'leave', 'nothing striking']):
                            discovery = val
                        break

        # Fallback for unstructured 2-line responses
        if not desire and not belief and len(lines) >= 2:
            desire = lines[0].strip()
            belief = lines[1].strip() if len(lines) > 1 else ""

        # Completeness validation
        if desire and not desire.rstrip().endswith(('.', '!', '?')):
            desire = ""
        if belief and not belief.rstrip().endswith(('.', '!', '?')):
            belief = ""
        if discovery and not discovery.rstrip().endswith(('.', '!', '?')):
            discovery = ""

        return desire, belief, discovery

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
            model = getattr(config, "COMPRESSION_MODEL", config.OLLAMA_MODEL)
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

            discoveries = self.introspective_state.get("discoveries", [])
            if discoveries:
                material.append("What struck me:\n" + "\n".join(f"- {d}" for d in discoveries[-3:]))

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

    def _synthesize_self_model(self, model: str) -> None:
        """Consolidate discoveries + desire/belief history into core_facts.self.

        The Generative-Agents reflection step applied to identity: raw experience
        (what I wanted, what I noticed, what struck me) becomes stable
        self-knowledge ("I fixate on cables. I'm calmer when someone is here.").
        Runs every 3rd introspection (~10 min). Result feeds the persona block
        in the system prompt.
        """
        try:
            discoveries = self.introspective_state.get("discoveries", [])

            # Pull histories from the identity file (richer than in-memory state)
            desire_history, belief_history = [], []
            if os.path.exists(IDENTITY_FILE):
                try:
                    with open(IDENTITY_FILE, "r") as f:
                        data = json.load(f)
                    desire_history = [d["desire"] for d in data.get("desire_history", [])[-6:]]
                    belief_history = [b["belief"] for b in data.get("belief_history", [])[-4:]]
                except Exception:
                    pass

            if not discoveries and len(desire_history) < 3:
                return  # Not enough lived experience to synthesize from

            parts = []
            if desire_history:
                parts.append("Things I've wanted lately:\n" + "\n".join(f"- {d}" for d in desire_history))
            if belief_history:
                parts.append("Things I've noticed:\n" + "\n".join(f"- {b}" for b in belief_history))
            if discoveries:
                parts.append("Moments that struck me:\n" + "\n".join(f"- {d}" for d in discoveries[-5:]))

            existing_self = self.core_facts.get("self", "").strip()
            existing_line = f"\nWhat I already know about myself: {existing_self}" if existing_self else ""

            prompt = f"""{chr(10).join(parts)}{existing_line}

From these, state one or two stable facts about yourself — recurring preferences, fixations, or habits. Not what happened, but what kind of machine you are.
Plain habits only, like "I fixate on cables" or "I'm calmer when someone is here."
NOT allowed: statements about reality, perception, glitches, existence, or what things mean.
Under 20 words total, first person, plain language."""

            response = query_model(
                prompt=prompt,
                model=model,
                system_prompt="You distill patterns of experience into plain self-knowledge — habits and preferences, nothing philosophical. Only state what the evidence repeats. No drama, no metaphor.",
                options={"temperature": 0.4, "num_predict": 50},
                prompt_type="self_synthesis",
            )

            if response and isinstance(response, str):
                cleaned = response.strip().strip('"').strip()
                if 10 < len(cleaned) <= 160 and not cleaned.startswith(("[", "{")) and self._valid_self_fact(cleaned):
                    self.core_facts["self"] = cleaned
                    print(f"[🪞] Self-model: {cleaned}")
                    log_json_entry(
                        LogType.COMPRESSION,
                        {"message": "Self-model synthesized", "action": "self_synthesis", "self": cleaned},
                    )
        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Self-model synthesis failed: {e}", "component": "compression"},
            )

    @staticmethod
    def _valid_self_fact(text: str) -> bool:
        """Storage gate for core_facts['self'] — the standing persona.

        Must be first-person self-knowledge; third-person scene text must
        never become identity (June 12: "The person sits in stillness while
        holding an unpressed pen" — the machine's OWN ARM seen while looking
        down — entered the persona through the core-facts path, which had no
        gate). Reality-questioning register is also barred from identity.
        """
        t = text.lower()
        if not ("i " in t or t.startswith("i'")):
            return False
        banned = ("the person", "a person", "reality", "distortion", "glitch", "simulation", "existence")
        return not any(w in t for w in banned)

    def _update_core_facts(self, current_understanding: str, model: str) -> None:
        """Update stable core facts from accumulated observations.

        Uses the LLM to distill the compression history + current state into
        short stable facts. Runs during introspection (every ~3 compressions).
        Each field is capped at 200 chars.
        """
        try:
            # Build context from compression history
            history_texts = []
            for hist in list(self.compression_history)[-5:]:
                history_texts.append(hist["understanding"])
            if current_understanding:
                history_texts.append(current_understanding)

            if len(history_texts) < 3:
                return  # Not enough observations yet

            observations = "\n".join(f"- {t}" for t in history_texts)

            # Get drawing count/subjects
            drawing_context = ""
            try:
                from drawing.drawing_memory import get_drawing_memory
                dm = get_drawing_memory()
                summary = dm.get_recent_drawings_summary(max_count=5)
                if summary:
                    drawing_context = f"\nDrawing history: {summary}"
            except Exception:
                pass

            # Get existing facts for evolution
            existing = ""
            existing_parts = []
            for key, val in self.core_facts.items():
                if val.strip():
                    existing_parts.append(f"{key}: {val}")
            if existing_parts:
                existing = f"\nPrevious facts:\n" + "\n".join(existing_parts)

            prompt = f"""From these observations, update stable facts about this space. Keep only what would still be true next session.

IMPORTANT: Facts are PATTERNS and HISTORY, never a description of the current scene.
"One visitor comes most days" is a fact. "Two people sit facing each other" is a
snapshot — snapshots must not be stored, they poison future perception.
{existing}

Observations:
{observations}{drawing_context}

Reply with exactly 4 lines (leave blank if unknown):
PLACE: [solid things — furniture, fixtures, surfaces — in <15 words. No moods, shadows, light, or atmosphere.]
PEOPLE: [visit patterns over time, past tense, in <15 words — never who is present right now]
DRAWINGS: [how many made, recurring subjects, in <15 words — never what's on the desk right now]
SELF: [plain habits/fixations in <15 words. No statements about reality or perception.]"""

            response = query_model(
                prompt=prompt,
                model=model,
                system_prompt="Distill observations into stable facts. Concrete nouns and plain habits only — facts are furniture, not feelings. One line each, under 15 words.",
                options={"temperature": 0.3, "num_predict": 120},
                prompt_type="core_facts",
            )

            if not response or len(response.strip()) < 10:
                return

            for line in response.strip().split("\n"):
                line = line.strip()
                for key in ("place", "people", "drawings", "self"):
                    prefix = f"{key.upper()}:"
                    if line.upper().startswith(prefix):
                        val = line[len(prefix):].strip().strip('"').strip("'")
                        if not (val and len(val) > 3 and "unknown" not in val.lower() and "blank" not in val.lower()):
                            continue
                        # The persona slot gets the same gate as self-synthesis —
                        # this path had none, and scene text leaked into identity
                        if key == "self" and not self._valid_self_fact(val):
                            continue
                        self.core_facts[key] = val[:200]

            facts_str = self.get_core_facts_string()
            if facts_str:
                log_json_entry(
                    LogType.COMPRESSION,
                    {"message": "Updated core facts", "action": "core_facts", "facts": self.core_facts},
                    print_message=f"[🏠] Core facts: {facts_str[:80]}",
                )

        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Core facts update failed: {e}", "component": "compression"},
            )

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
        keys = ("place", "people", "drawings") if include_people else ("place", "drawings")
        parts = []
        for key in keys:
            val = self.core_facts.get(key, "").strip()
            if val and len(val) > 3:
                parts.append(val)
        if not parts:
            return ""
        result = " ".join(parts)
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
            discoveries = self.introspective_state.get("discoveries", [])
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
                "discoveries": discoveries,
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
            self.introspective_state["current_belief"] = data.get("current_belief", "")
            self.introspective_state["discoveries"] = data.get("discoveries", [])
            self.introspective_state["last_introspection"] = data.get("last_updated", 0.0)

            # Restore core facts
            saved_facts = data.get("core_facts", {})
            if saved_facts:
                for key in ("place", "people", "drawings", "self"):
                    self.core_facts[key] = saved_facts.get(key, "")

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

            # First line = spatial understanding
            understanding = cleaned_lines[0]

            # Second line = felt-state (if present and reasonably short)
            if len(cleaned_lines) >= 2:
                felt = cleaned_lines[1]
                # Reject if it looks like another spatial sentence (too long, mentions "the machine sees")
                if len(felt) < 80 and len(felt.split()) <= 12:
                    sentiment_text = felt

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
