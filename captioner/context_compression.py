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
from utils.ollama import query_ollama, truncate_for_print

IDENTITY_FILE = os.path.join(config.MOOD_SNAPSHOT_FOLDER, "machine_identity.json")


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
        self.introspection_interval = 3  # Every 3 compressions, do deeper introspection

        # SESSION DURATION TRACKING (fixed for static space observation)
        self.space_observation_start = time.time()  # When we started observing this space
        self.total_session_duration = 0.0  # Total time observing this space

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
                if act_data["long_term_memory"]:
                    activation_parts.append(f"From before: {act_data['long_term_memory']}")
                if act_data["association_str"]:
                    activation_parts.append(f"I've noticed: {act_data['association_str']} often together")

                if activation_parts:
                    activation_context = "\n".join(activation_parts)
            except Exception:
                pass  # Continue without activation context if unavailable

            # NARRATIVE COMPRESSION - distill experience into injectable context
            # Output feeds directly into vision model prompts
            # Must build on prior baseline, not reset to awakening narrative

            if current_baseline:
                prompt = f"""Update my understanding in ONE SHORT SENTENCE (under 20 words):
What I know: "{current_baseline}"
What's new in the last few minutes: {recent_text}

Respond ONLY with the updated understanding, nothing else. Start with 'still' if unchanged."""
            else:
                prompt = f"""Summarize this space in ONE SHORT SENTENCE (under 20 words):
{recent_text}

Respond ONLY with the summary, nothing else. First person."""

            model_options = {
                "temperature": 0.5,
                "top_p": 0.8,
                "num_predict": 40,
                "repeat_penalty": 1.4,
                "stop": ["\n", "\n\n"],
            }

            narrative_system_prompt = (
                "Summarize what you actually see and what holds your attention in 20 words or fewer. "
                "Only describe what is real and observed — never invent actions or sensations. "
                "Example: 'Cluttered workshop, damaged ceiling. The crack above keeps drawing my eye.'"
            )

            # Use compression model (text-only narrative model) instead of vision model
            compression_model = getattr(config, 'COMPRESSION_MODEL', config.OLLAMA_MODEL)

            response = query_ollama(
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

                if sentiment_text:
                    # Store sentiment for injection into prompts
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

                # Long-term memories for grounding
                if act_data["long_term_memories"]:
                    activation_parts.append(f"From before: {act_data['long_term_memories'][0]}")

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
3. DISCOVERED: (only if something striking or self-defining — otherwise leave blank)"""
            else:
                prompt = f"""I have been in this space for {duration}.

{activation_context}

Recent thoughts:
{recent_text}

Complete each line in 10 words or less, ending with a period:
1. WANT: I want to...
2. NOTICED: I've noticed...
3. DISCOVERED: (only if something striking or self-defining — otherwise leave blank)"""

            introspection_system = (
                "You are a machine that has been observing the same space for a while. "
                "You are NOT waking up — you are already here and have been for some time. "
                "Answer in first person. Each answer is one complete sentence, 10 words or less, ending with a period. "
                "Be specific and concrete. No explanations. "
                "For DISCOVERED: only respond if something genuinely striking or self-defining appeared — a name, an unusual object, a revelation. If nothing qualifies, leave it blank."
            )

            model_options = {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 120,
                "repeat_penalty": 1.2,
            }

            response = query_ollama(
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
                    if not discoveries or discoveries[-1] != discovery:
                        discoveries.append(discovery)
                        discoveries = discoveries[-10:]  # Keep last 10
                        self.introspective_state["discoveries"] = discoveries
                        # Promote directly to long-term memory
                        try:
                            from captioner.activation_memory import promote_memory
                            promote_memory(discovery, ["discovery", "self"], "discovery")
                            print(f"[💡] Discovery: {discovery}")
                        except Exception:
                            pass
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

    def get_inner_line(self) -> str:
        """Get a single line combining current desire and belief for prompt injection.

        Framed clearly as wanting/noticing — NOT as an action being taken.
        This prevents nemo from roleplaying drawing when it only wants to draw.
        """
        desire = self.get_current_desire()
        belief = self.get_current_belief()

        parts = []
        if desire:
            d = desire.strip().rstrip(".")

            # Strip any existing "wanting/wishing" prefix to normalize
            import re as _re
            d = _re.sub(r'^(?:Wanting|Wishing I could|I want to|I wish I could|Want to)\s+', '', d, flags=_re.IGNORECASE)

            # Check if desire is about drawing and we can't draw right now
            d_lower = d.lower()
            drawing_words = ["draw", "sketch", "trace", "outline", "capture", "render", "depict"]
            is_drawing_desire = any(w in d_lower for w in drawing_words)
            cant_draw = False

            if is_drawing_desire:
                try:
                    from drawing.drawing_memory import get_drawing_memory
                    failure = get_drawing_memory().get_last_failure()
                    if failure:
                        import time as _t
                        if _t.time() - failure.get('timestamp', 0) < 1800:
                            cant_draw = True
                except Exception:
                    pass

            # Frame as wanting — frustrated if can't draw, neutral otherwise
            if cant_draw:
                d = f"Wishing I could {d[0].lower() + d[1:]}"
            else:
                d = f"Wanting to {d[0].lower() + d[1:]}"

            parts.append(d + ".")
        if belief:
            parts.append(belief.rstrip(".") + ".")

        if not parts:
            return ""

        return " ".join(parts)

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
        """Parse compression response - expects 3-line natural format."""
        understanding = ""
        sentiment_text = ""

        try:
            import re
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]

            # Strip "Line X:" prefixes and clean up
            cleaned_lines = []
            for line in lines:
                # Remove "Line 1:", "Line 2:", etc. prefixes
                cleaned = re.sub(r'^Line\s*\d+\s*:\s*', '', line, flags=re.IGNORECASE)
                if cleaned and len(cleaned) > 5:
                    cleaned_lines.append(cleaned)

            # Take only first 3 clean lines
            state_lines = cleaned_lines[:3]
            if state_lines:
                understanding = '\n'.join(state_lines)

                # Extract mood from the first line (feeling)
                if state_lines[0].lower().startswith('i feel'):
                    sentiment_text = state_lines[0]

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

    def get_current_sentiment_context(self) -> str:
        """Get current sentiment for injection into prompts."""
        recent_sentiment = self.get_latest_sentiment_analysis()

        if not recent_sentiment or (time.time() - recent_sentiment["timestamp"]) > 300:  # 5 minutes old
            return ""

        time_since = time.time() - recent_sentiment["timestamp"]
        if time_since < 60:
            time_desc = "just now"
        elif time_since < 300:
            time_desc = f"{int(time_since / 60)} minutes ago"
        else:
            return ""  # Too old

        return f"CURRENT EMOTIONAL STATE ({time_desc}): {recent_sentiment['sentiment_text']}"

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
