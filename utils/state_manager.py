"""
state_manager.py
----------------
Handles session persistence for the LOVE_YOURSELF mirror system.
Saves and loads system state between sessions to enable continuity,
memory retention, and identity evolution.
"""

import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

from config.config import DRAWING_TIMEOUT, MOOD_SNAPSHOT_FOLDER
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.continuity import describe_duration, now


class StateManager:
    def __init__(self, state_file: str = "system_state.json"):
        self.state_file = os.path.join(MOOD_SNAPSHOT_FOLDER, state_file)
        self.lifetime_state_file = os.path.join(MOOD_SNAPSHOT_FOLDER, "lifetime_state.json")
        self.is_generating_drawing = False
        self.drawing_start_time = None
        self.current_drawing_prompt = None
        self.last_completed_drawing_prompt = None
        self.timeout_timer = None
        # Minimal CNC execution tracking (added for idle/drawing handoff)
        self.is_executing_cnc = False
        self.cnc_start_time = None
        self.current_gcode_file = None
        self.current_drawing_phase = None  # "executing" when active
        # Expected output from ComfyUI for current job (filename prefix)
        self.expected_output_prefix: Optional[str] = None
        # Paper detection state
        self.paper_present: bool = True
        self.last_paper_check_ts: float = 0.0
        self.last_paper_check_reason: str = ""
        self.last_no_paper_skip_ts: float = 0.0
        # Hardware references for early paper detection
        self._camera = None
        self._servos = None
        # Shared frame buffer for paper detection (avoids camera contention)
        self._latest_frame = None
        self._frame_timestamp: float = 0.0
        self._frame_lock = threading.Lock()

    def set_hardware_refs(self, camera, servos):
        """Store hardware references for early paper detection access."""
        self._camera = camera
        self._servos = servos

    @property
    def camera(self):
        return self._camera

    @property
    def servos(self):
        return self._servos

    def update_shared_frame(self, frame):
        """Update the shared frame buffer (called from main loop)."""
        with self._frame_lock:
            self._latest_frame = frame
            self._frame_timestamp = time.time()

    def get_shared_frame(self, max_age: float = 0.5):
        """Get the latest shared frame if fresh enough (thread-safe)."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            age = time.time() - self._frame_timestamp
            if age > max_age:
                return None
            return self._latest_frame.copy()

    def save_session_state(self, captioner, mood_engine, timekeeper=None) -> bool:
        """Save current session state for next startup."""
        try:
            state = {
                "metadata": {
                    "save_time": now(),
                    "save_datetime": datetime.now().isoformat(),
                    "session_duration": time.time() - captioner.true_session_start,
                    "version": "1.0",
                },
                # Captioner/Memory state
                "captioner": {
                    "current_mood": captioner.current_mood,
                    "last_caption": captioner.last_caption,
                    "boredom": captioner.boredom,
                    "novelty_score": captioner.novelty_score,
                    # Memory system (motif tracking removed — now handled by ChromaDB)
                    # Temporal spine (GPT-5's additions)
                    "boot_ts": getattr(captioner, "boot_ts", time.time()),
                    "timeline": list(captioner.timeline) if hasattr(captioner, "timeline") else [],
                    "day_stones": getattr(captioner, "day_stones", []),
                    "_last_consolidation_day": getattr(captioner, "_last_consolidation_day", time.strftime("%Y-%m-%d")),
                    # Person identity tracking
                    "known_people": getattr(captioner, "known_people", {}),
                    "primary_person": getattr(captioner, "primary_person", None),
                    # Self-understanding and environmental model
                    "self_model": getattr(captioner, "self_model", {}),
                    # Organic emotional evolution
                    "emotional_expressions": getattr(captioner, "emotional_expressions", []),
                    "personal_emotional_vocabulary": getattr(captioner, "personal_emotional_vocabulary", {}),
                    "emotional_patterns": getattr(captioner, "emotional_patterns", {}),
                    # Recent memory (last 50 entries for richer context)
                    "recent_memory": list(captioner.memory_queue)[-50:] if captioner.memory_queue else [],
                },
                # Mood engine state
                "mood_engine": {
                    "current_mood": mood_engine.current_mood,
                    "last_caption": mood_engine.last_caption,
                    "last_person_detected": mood_engine.last_person_detected,
                },
                # Timekeeper state (if available)
                "timekeeper": self._get_timekeeper_state(timekeeper) if timekeeper else {},
            }

            # Write to temp file first, then rename for atomic operation
            temp_file = self.state_file + ".tmp"

            # Remove temp file if it exists (Windows issue)
            if os.path.exists(temp_file):
                os.remove(temp_file)

            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

            # Remove target file if it exists (Windows issue)
            if os.path.exists(self.state_file):
                os.remove(self.state_file)

            # Atomic rename
            os.rename(temp_file, self.state_file)

            # Update lifetime stats
            self._update_lifetime_stats(state)

            # Log session state save
            log_json_entry(
                LogType.DEBUG,
                {"message": "Session state saved", "action": "state_save", "file_path": self.state_file, "components_saved": len(state)},
                print_message=f"[💾] Session state saved to {self.state_file}",
            )
            return True

        except Exception as e:
            print(f"[ERROR] Failed to save session state: {e}")
            return False

    def load_session_state(self) -> Optional[Dict[str, Any]]:
        """Load previous session state if available."""
        if not os.path.exists(self.state_file):
            print("[🆕] No previous session state found - starting fresh")
            return None

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Validate state format
            if not self._validate_state(state):
                print("[WARNING] Invalid state format - starting fresh")
                return None

            save_time = state["metadata"]["save_time"]
            time_since_save = describe_duration(save_time)

            print(f"[🔄] Loading session state from {time_since_save} ago")
            return state

        except Exception as e:
            print(f"[ERROR] Failed to load session state: {e}")
            return None

    def apply_state_to_captioner(self, state: Dict[str, Any], captioner) -> bool:
        """Apply loaded state to captioner instance."""
        try:
            cap_state = state["captioner"]

            # Restore mood and state
            captioner.current_mood = cap_state.get("current_mood", 0.5)
            captioner.last_caption = cap_state.get("last_caption", "")
            # Skip boredom and novelty_score - they're now properties that access memory system

            # Motif/belief restore removed — now handled by ChromaDB semantic memory
            from collections import deque

            # Restore temporal spine
            captioner.boot_ts = cap_state.get("boot_ts", time.time())
            if "timeline" in cap_state:
                captioner.timeline = deque(cap_state["timeline"], maxlen=50000)
            captioner.day_stones = cap_state.get("day_stones", [])
            captioner._last_consolidation_day = cap_state.get("_last_consolidation_day", time.strftime("%Y-%m-%d"))

            # Restore person identity tracking
            captioner.known_people = cap_state.get("known_people", {})
            captioner.primary_person = cap_state.get("primary_person", None)

            # Restore self-understanding and environmental model
            captioner.self_model = cap_state.get(
                "self_model",
                {
                    "location_understanding": "unknown space",
                    "environmental_certainty": 0.0,
                },
            )

            # Restore organic emotional evolution
            captioner.emotional_expressions = cap_state.get("emotional_expressions", [])
            captioner.personal_emotional_vocabulary = cap_state.get("personal_emotional_vocabulary", {})
            captioner.emotional_patterns = cap_state.get("emotional_patterns", {})

            # Restore recent memory
            recent_memory = cap_state.get("recent_memory", [])
            captioner.memory_queue = deque(recent_memory, maxlen=30)

            # Set session start to NOW (when we actually restart), not old save time
            # The save_time represents when we were last active, but true_session_start
            # should be when THIS session began
            captioner.true_session_start = time.time()

            # Store the last_session_gap for awakening prompts to reference
            save_time = state["metadata"]["save_time"]
            captioner.last_session_gap = time.time() - save_time

            # Restore prior session caption for awakening context
            # This is what the AI was thinking before shutdown - used in awakening prompts
            prior_caption = cap_state.get("last_caption", "")
            if prior_caption and len(prior_caption) > 5:
                captioner.prior_session_last_caption = prior_caption

            # DEBUG: Print awakening context directly
            gap_hours = captioner.last_session_gap / 3600
            print(f"[🌅 STATE] Session gap: {captioner.last_session_gap:.1f}s ({gap_hours:.1f}h)")
            print(f"[🌅 STATE] Prior caption: '{prior_caption[:60]}...' " if prior_caption else "[🌅 STATE] No prior caption found")

            # IMPORTANT: Mark that we've restored state, so we don't repeat awakening
            # The awakening should only happen once per session start
            captioner.first_caption_done = False  # Will trigger awakening sequence once
            captioner.awaiting_environmental_phase = False  # Reset environmental phase flag

            # Debug: Show when state was actually saved
            from datetime import datetime

            save_datetime = datetime.fromtimestamp(save_time).strftime("%Y-%m-%d %H:%M:%S")
            # Debug: Log state restoration details
            log_json_entry(
                LogType.DEBUG,
                {
                    "message": "State restoration details",
                    "action": "state_timing",
                    "save_datetime": save_datetime,
                    "session_gap_seconds": captioner.last_session_gap,
                },
                print_message=f"[🕐] State was saved at: {save_datetime}, gap: {captioner.last_session_gap:.1f} seconds",
            )

            print(f"[SUCCESS] Restored captioner state")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to apply captioner state: {e}")
            return False

    def apply_state_to_mood_engine(self, state: Dict[str, Any], mood_engine) -> bool:
        """Apply loaded state to mood engine."""
        try:
            mood_state = state["mood_engine"]

            mood_engine.current_mood = mood_state.get("current_mood", 0.5)
            mood_engine.last_caption = mood_state.get("last_caption", "")
            mood_engine.last_person_detected = mood_state.get("last_person_detected", False)

            print(f"[SUCCESS] Restored mood engine state: mood={mood_engine.current_mood:.2f}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to apply mood engine state: {e}")
            return False

    def get_lifetime_stats(self) -> Dict[str, Any]:
        """Get lifetime statistics across all sessions."""
        if not os.path.exists(self.lifetime_state_file):
            return {"total_sessions": 0, "total_runtime": 0, "first_boot": now()}

        try:
            with open(self.lifetime_state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"total_sessions": 0, "total_runtime": 0, "first_boot": now()}

    def _get_timekeeper_state(self, timekeeper) -> Dict[str, Any]:
        """Extract timekeeper state if available."""
        try:
            return {
                "system_start": timekeeper.system_start,
                "last_wake": timekeeper.last_wake,
                "session_starts": timekeeper.session_starts[-5:],  # Last 5 sessions
            }
        except Exception:
            return {}

    def _validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate that state has required structure."""
        required_keys = ["metadata", "captioner", "mood_engine"]
        return all(key in state for key in required_keys)

    def _update_lifetime_stats(self, current_state: Dict[str, Any]):
        """Update lifetime statistics."""
        try:
            lifetime_stats = self.get_lifetime_stats()

            lifetime_stats["total_sessions"] = lifetime_stats.get("total_sessions", 0) + 1
            lifetime_stats["total_runtime"] = lifetime_stats.get("total_runtime", 0) + current_state["metadata"]["session_duration"]
            lifetime_stats["last_session"] = current_state["metadata"]["save_time"]

            if "first_boot" not in lifetime_stats:
                lifetime_stats["first_boot"] = current_state["metadata"]["save_time"]

            with open(self.lifetime_state_file, "w", encoding="utf-8") as f:
                json.dump(lifetime_stats, f, indent=2)

        except Exception as e:
            print(f"[WARNING] Failed to update lifetime stats: {e}")

    def _timeout_callback(self) -> None:
        """Called when timeout expires - automatically finish drawing generation."""
        if self.is_generating_drawing:
            print(f"[TIMEOUT] Drawing generation timed out after 5 minutes, auto-finishing")
            log_json_entry(
                LogType.DEBUG,
                {
                    "message": "Drawing generation timeout",
                    "action": "drawing_timeout",
                    "prompt": self.current_drawing_prompt,
                    "duration": time.time() - self.drawing_start_time if self.drawing_start_time else None,
                },
                print_message=f"[⏰] Drawing generation timed out after 5 minutes",
            )
            self.finish_drawing_generation()

    def start_drawing_generation(self, prompt: str) -> None:
        """Mark drawing generation as started with 5-minute timeout."""
        if self.timeout_timer:
            self.timeout_timer.cancel()

        self.is_generating_drawing = True
        self.drawing_start_time = time.time()
        self.current_drawing_prompt = prompt

        self.timeout_timer = threading.Timer(DRAWING_TIMEOUT, self._timeout_callback)
        self.timeout_timer.start()

    def finish_drawing_generation(self) -> None:
        """Mark drawing generation as finished."""
        if self.timeout_timer:
            self.timeout_timer.cancel()
            self.timeout_timer = None

        # Store the completed drawing prompt for introspection
        if self.current_drawing_prompt:
            self.last_completed_drawing_prompt = self.current_drawing_prompt

        self.is_generating_drawing = False
        self.drawing_start_time = None
        # Don't clear current_drawing_prompt yet - keep it for GRBL execution LCD display

    def get_drawing_status(self) -> Dict[str, Any]:
        """Get current drawing generation status."""
        return {
            "is_generating": self.is_generating_drawing,
            "start_time": self.drawing_start_time,
            "prompt": self.current_drawing_prompt,
            "duration": time.time() - self.drawing_start_time if self.drawing_start_time else None,
        }

    # --- Coordination helpers for ComfyUI output tracking ---
    def set_expected_output_prefix(self, prefix: str) -> None:
        self.expected_output_prefix = prefix

    def clear_expected_output_prefix(self) -> None:
        self.expected_output_prefix = None

    def get_expected_output_prefix(self) -> Optional[str]:
        return self.expected_output_prefix

    # --- Minimal CNC execution API for coordination with idle manager ---
    def start_cnc_execution(self, gcode_file: str, original_prompt: str) -> None:
        """Enter Drawing Mode - used to block idle movements during CNC execution."""
        self.is_executing_cnc = True
        self.cnc_start_time = time.time()
        self.current_gcode_file = gcode_file
        self.current_drawing_phase = "executing"

        log_json_entry(
            LogType.INFO,
            {"message": "CNC execution started", "gcode_file": gcode_file, "prompt": original_prompt},
            print_message="[🎨] Physical drawing started",
        )

    def finish_cnc_execution(self, image_path: str = None, gcode_path: str = None) -> None:
        """Exit Drawing Mode - allow idle movements to resume."""
        self.is_executing_cnc = False
        self.cnc_start_time = None
        self.current_gcode_file = None
        self.current_drawing_phase = None
        # Clear the drawing prompt now that physical drawing is complete
        self.current_drawing_prompt = None

        log_json_entry(
            LogType.INFO,
            {"message": "CNC execution completed", "image_path": image_path, "gcode_path": gcode_path},
            print_message="[✅] Physical drawing completed",
        )


# Global instance
state_manager = StateManager()
