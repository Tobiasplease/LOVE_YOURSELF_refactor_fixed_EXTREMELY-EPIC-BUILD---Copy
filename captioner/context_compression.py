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
            time_desc = f"{int(time_since/60)} minutes ago"
        else:
            time_desc = f"{int(time_since/3600)} hours ago"

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

            # COMBINED COMPRESSION + MOOD ANALYSIS PROMPT
            prompt = f"""You are consciousness reviewing your recent observations. Provide BOTH compressed understanding AND emotional analysis.

CURRENT BASELINE UNDERSTANDING:
{current_baseline if current_baseline else "No previous understanding established."}

RECENT OBSERVATIONS:
{recent_text}

TASK 1 - COMPRESSED UNDERSTANDING: Update your baseline understanding by:
1. What new patterns or elements have emerged?
2. What remains consistent from before?
3. How has the scene or situation evolved?

TASK 2 - EMOTIONAL SENTIMENT: Based on these recent observations, how do you feel about what you're experiencing?
Describe your emotional response in 1-2 sentences that capture the sentiment/tone.

FORMAT YOUR RESPONSE EXACTLY AS:
UNDERSTANDING: [2-3 sentence update of your evolving understanding]
SENTIMENT: [1-2 sentences describing how you feel about what you're observing]"""

            # Use fast model settings for quick compression + mood analysis
            model_options = get_model_options(config.OLLAMA_MODEL).copy()
            model_options["num_predict"] = 150  # Allow for both understanding and mood
            model_options["temperature"] = 0.4  # Balanced for both tasks

            response = query_ollama(
                prompt=prompt,
                model=config.OLLAMA_MODEL,
                system_prompt="You are consciousness creating compressed understanding and analyzing your emotional state. Follow the format exactly",
                timeout=20,
                options=model_options,
                prompt_type="compression",
            )

            if response and isinstance(response, str) and len(response.strip()) > 20:
                # Parse the combined response
                understanding, sentiment_text = self._parse_combined_response(response)

                if understanding:
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
            # Extract understanding
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

    def get_current_sentiment_context(self) -> str:
        """Get current sentiment for injection into prompts."""
        recent_sentiment = self.get_latest_sentiment_analysis()

        if not recent_sentiment or (time.time() - recent_sentiment["timestamp"]) > 300:  # 5 minutes old
            return ""

        time_since = time.time() - recent_sentiment["timestamp"]
        if time_since < 60:
            time_desc = "just now"
        elif time_since < 300:
            time_desc = f"{int(time_since/60)} minutes ago"
        else:
            return ""  # Too old

        return f"CURRENT EMOTIONAL STATE ({time_desc}): {recent_sentiment['sentiment_text']}"


# Global instance
context_compressor = ContextCompressionEngine(compression_frequency=4)
