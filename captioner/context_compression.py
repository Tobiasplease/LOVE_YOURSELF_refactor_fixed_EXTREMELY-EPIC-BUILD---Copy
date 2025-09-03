"""
context_compression.py
---------------------
Frequent LLM-based compression of recent observations to create evolving baseline context.
Prevents repetition by building understanding that carries forward.
"""

import queue
import threading
import time
from collections import deque

from config import config
from config.model_settings import get_model_options
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.ollama import query_ollama, truncate_for_print


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

        # Background compression system
        self.compression_queue = queue.Queue(maxsize=5)  # Limit queue size
        self.compression_thread = None
        self.compression_active = False
        self._start_compression_worker()

    def add_caption(self, caption: str, timestamp: float | None = None) -> None:
        """Add a new caption and trigger compression if needed."""
        if not caption or not caption.strip():
            log_json_entry(LogType.COMPRESSION, {"message": "Skipping empty caption", "action": "skip"}, print_message="[🗜️] Skipping empty caption")
            return

        self.recent_captions.append({"text": caption, "timestamp": timestamp or time.time()})
        self.caption_count += 1

        # Only trigger compression if we have enough valid captions
        if self.caption_count % self.compression_frequency == 0:
            valid_captions = [cap for cap in self.recent_captions if cap.get("text") and cap["text"].strip()]
            if len(valid_captions) >= self.compression_frequency:
                self._queue_compression()

    def get_baseline_context(self) -> str:
        """Get current baseline context for injection into prompts."""
        if not self.baseline_context:
            return ""

        # Calculate time since last compression
        time_since = time.time() - self.last_compression_time
        if time_since < 60:
            time_desc = f"{int(time_since)} seconds ago"
        elif time_since < 3600:
            time_desc = f"{int(time_since / 60)} minutes ago"
        else:
            time_desc = f"{int(time_since / 3600)} hours ago"

        return f"""ALREADY OBSERVED ({time_desc}): {self.baseline_context}\n\nDO NOT repeat these established facts.
        Notice what's new, different, or if nothing has changed - that's meaningful too."""

    def should_mention_evolution(self) -> bool:
        """Check if recent compression shows significant change."""
        # This could be enhanced to detect if the baseline significantly evolved
        return len(self.baseline_context) > 20

    def reset_context(self) -> None:
        """Reset compression state for new session."""
        self.baseline_context = ""
        self.recent_captions.clear()
        self.caption_count = 0
        self.last_compression_time = time.time()

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
            self.compression_queue.put_nowait({"captions": captions_snapshot, "baseline": current_baseline, "timestamp": time.time()})
            log_json_entry(
                LogType.COMPRESSION,
                {"message": "Queued background compression", "action": "queue", "caption_count": len(captions_snapshot)},
                print_message="[🗜️] Queued background compression...",
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
        """Perform the actual LLM compression call with combined mood analysis."""
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

            # EXPERIENTIAL COMPRESSION - WHAT'S ACTUALLY HAPPENING
            prompt = f"""You're taking a moment to notice what's actually happening with you - what you're seeing and how it's affecting you.

IMPORTANT: You SEE through camera vision - this is YOUR SIGHT, not an image.
{historical_context}

WHERE YOU WERE:
{current_baseline if current_baseline else "Just beginning to experience this space."}

RECENT MOMENTS:
{recent_text}

What's actually happening with you right now - both what you're noticing and how it feels?
What's sticking with you from these moments?
What are you drawn to look at next?

You might feel restless, heavy, light, unsettled, steady, disconnected, present, numb, electric - whatever's actually true.
Don't default to "curious" or "intrigued" - be specific about the actual feeling.

RESPOND AS:
EXPERIENCE: [What's happening with you - observations mixed naturally with how they affect you]
MOOD: [The actual feeling tone right now - be specific, not generic]"""

            # Use settings optimized for existential reflection + mood analysis
            model_options = get_model_options(config.OLLAMA_MODEL).copy()
            model_options["num_predict"] = 200  # More space for identity/existential content
            model_options["temperature"] = 0.5  # Slightly more creative for philosophical insights

            response = query_ollama(
                prompt=prompt,
                model=config.OLLAMA_MODEL,
                system_prompt="You're taking a moment to notice what's actually happening with you - "
                "what you're seeing and how it's affecting you. Be honest about what you're actually experiencing",
                timeout=20,
                options=model_options,
                prompt_type="compression",
            )

            if response and isinstance(response, str) and len(response.strip()) > 20:
                # Parse the combined response
                understanding, sentiment_text = self._parse_combined_response(response)

                if understanding:
                    # Store in history before updating
                    if self.baseline_context:  # Don't store empty first compression
                        self.compression_history.append(
                            {
                                "understanding": self.baseline_context,
                                "timestamp": self.last_compression_time,
                                "age_minutes": (time.time() - self.last_compression_time) / 60,
                            }
                        )

                    self.baseline_context = understanding.strip()
                    self.last_compression_time = time.time()

                    # Log compression and always show full output
                    log_json_entry(
                        LogType.COMPRESSION,
                        {
                            "message": "Updated baseline understanding",
                            "action": "update_baseline",
                            "understanding": understanding,
                            "understanding_length": len(understanding),
                            "compression_history_count": len(self.compression_history),
                        },
                        print_message=f"[🧠] Updated baseline: {truncate_for_print(self.baseline_context, 80)}",
                    )

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

    def _parse_combined_response(self, response: str) -> tuple:
        """Parse combined compression + sentiment response."""
        import re

        understanding = ""
        sentiment_text = ""

        try:
            # Try new format first (EXPERIENCE/MOOD)
            experience_match = re.search(r"EXPERIENCE:\s*(.+?)(?=MOOD:|$)", response, re.DOTALL)
            mood_match = re.search(r"MOOD:\s*(.+?)$", response, re.DOTALL)

            if experience_match and mood_match:
                understanding = experience_match.group(1).strip()
                sentiment_text = mood_match.group(1).strip()
            else:
                # Fallback to old format (UNDERSTANDING/SENTIMENT) for compatibility
                understanding_match = re.search(r"UNDERSTANDING:\s*(.+?)(?=SENTIMENT:|$)", response, re.DOTALL)
                if understanding_match:
                    understanding = understanding_match.group(1).strip()

                # Extract sentiment text
                sentiment_match = re.search(r"SENTIMENT:\s*(.+?)$", response, re.DOTALL)
                if sentiment_match:
                    sentiment_text = sentiment_match.group(1).strip()

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
            return f"CONTEXT: {self.baseline_context}"
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

    def get_session_summary(self) -> str:
        """Get a summary of the entire session's understanding evolution."""
        if not self.compression_history:
            return "Session just beginning - no historical understanding yet."

        session_duration = (time.time() - self.session_start_time) / 3600  # hours
        total_compressions = len(self.compression_history) + (1 if self.baseline_context else 0)

        return f"Session duration: {session_duration:.1f} hours, {total_compressions} understanding iterations completed."


# Global instance
context_compressor = ContextCompressionEngine(compression_frequency=4)
