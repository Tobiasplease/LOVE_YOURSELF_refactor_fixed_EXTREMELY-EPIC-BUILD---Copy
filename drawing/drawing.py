from __future__ import annotations

# from utils.ollama import query_ollama

"""drawing.py – final version

Captioner now hands us a *ready-made* drawing prompt that already contains
scene caption + reflection. This version does not query the LLM again,
and passes the prompt directly to ComfyUI.
"""

import base64
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from config.config import (
    COMFY_CNET_STRENGTH,
    COMFY_FLUX_GUIDANCE,
    COMFY_LATENT_HEIGHT,
    COMFY_LATENT_WIDTH,
    COMFY_LORA_STRENGTH,
    COMFY_STEPS,
    DRAWING_COOLDOWN,
    MOOD_SNAPSHOT_FOLDER,
    TRIGGER_PROMPT,
)
from captioner.prompts import SELF_CRITIQUE_PROMPT, SELF_CRITIQUE_SYSTEM_PROMPT
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from event_logging.run_manager import get_run_image_path
from grbl.idle_movement_manager import pause_for_drawing
from utils.ollama import query_ollama, truncate_for_print
from utils.state_manager import state_manager

from .comfy import create_impostor_controller

if TYPE_CHECKING:
    from captioner.captioner import Captioner


class DrawingController:
    """Decides when to draw and queues ComfyUI jobs."""

    def __init__(self) -> None:
        self.last_drawing_time: float = time.time() - DRAWING_COOLDOWN - 10  # Allow immediate first drawing
        self.cooldown: float = DRAWING_COOLDOWN  # seconds between drawings
        self.last_prompt: Optional[str] = None
        self.last_drawing_prompt: str = ""
        self.last_reflection: Optional[str] = None
        self.quota_manager = None  # No quota system - use timer-based drawing

    # ------------------------------------------------------------------
    # decision helpers
    # ------------------------------------------------------------------
    def ready_to_draw(self) -> bool:
        # Check if cooldown period has passed
        cooldown_ready = time.time() - self.last_drawing_time > self.cooldown

        # Check if currently generating or executing a drawing
        try:
            from utils.drawing_state import DrawingState
            from utils.state_manager import state_manager

            currently_executing = DrawingState.is_drawing()
            currently_generating = getattr(state_manager, "is_generating_drawing", False)

            if currently_generating:
                print(f"[🎨] Drawing blocked: ComfyUI generation currently in progress")
                return False
            if currently_executing:
                print(f"[🎨] Drawing blocked: GRBL execution currently in progress")
                return False
        except Exception as e:
            print(f"[⚠️] Could not check drawing state: {e}")

        return cooldown_ready

    def should_draw(self, *, mood: float, novelty: float, boredom: float, reflection: Optional[str] = None) -> bool:
        # Use state-motivated drawing logic when enabled, otherwise timer-based
        try:
            from config.config import DRAWING_USE_STATE_MOTIVATION
            if DRAWING_USE_STATE_MOTIVATION:
                return self._should_draw_state_motivated(mood=mood, novelty=novelty, boredom=boredom, reflection=reflection)
            else:
                return self._should_draw_original(mood=mood, novelty=novelty, boredom=boredom, reflection=reflection)
        except ImportError:
            return self._should_draw_original(mood=mood, novelty=novelty, boredom=boredom, reflection=reflection)

    def _should_draw_original(self, *, mood: float, novelty: float, boredom: float, reflection: Optional[str] = None) -> bool:
        """Pure timer-based drawing decision logic for debugging."""
        if not self.ready_to_draw():
            cooldown_remaining = max(0, self.cooldown - (time.time() - self.last_drawing_time))
            print(f"[🎨] Timer drawing check: BLOCKED by cooldown ({cooldown_remaining:.0f}s remaining)")
            return False

        # Pure timer-based: if cooldown passed, always draw
        print(f"[🎨] ✨ TIMER DRAWING TRIGGERED (debug mode - ignoring mood/novelty/boredom)")
        return True

    def _should_draw_state_motivated(self, *, mood: float, novelty: float, boredom: float, reflection: Optional[str] = None) -> bool:
        """Sophisticated state-motivated drawing decision logic."""
        import random
        from config.config import (
            DRAWING_MIN_INTERVAL, DRAWING_MAX_INTERVAL, DRAWING_BASE_THRESHOLD,
            DRAWING_NOVELTY_WEIGHT, DRAWING_BOREDOM_WEIGHT, DRAWING_MOOD_WEIGHT
        )

        current_time = time.time()
        time_since_last = current_time - self.last_drawing_time

        # Absolute minimum interval - safety check
        if time_since_last < DRAWING_MIN_INTERVAL:
            remaining = DRAWING_MIN_INTERVAL - time_since_last
            print(f"[🎨] State drawing check: BLOCKED by minimum interval ({remaining:.0f}s remaining)")
            return False

        # Force drawing if maximum interval exceeded (ensure some activity)
        if time_since_last >= DRAWING_MAX_INTERVAL:
            print(f"[🎨] ✨ STATE DRAWING TRIGGERED: Maximum interval exceeded ({time_since_last:.0f}s)")
            return True

        # Calculate state-based drawing motivation score
        # Normalize inputs to 0-1 range
        normalized_mood = max(0, min(1, (mood + 1) / 2))  # mood is -1 to 1, normalize to 0-1
        normalized_novelty = max(0, min(1, novelty))      # novelty should be 0-1
        normalized_boredom = max(0, min(1, boredom))       # boredom should be 0-1

        # Calculate weighted motivation score
        motivation_score = (
            normalized_novelty * DRAWING_NOVELTY_WEIGHT +
            normalized_boredom * DRAWING_BOREDOM_WEIGHT +
            normalized_mood * DRAWING_MOOD_WEIGHT
        )

        # Add time pressure - gradually increase motivation over time
        time_factor = min(1.0, time_since_last / DRAWING_MAX_INTERVAL)
        time_pressure = time_factor * 0.3  # Up to 0.3 additional motivation

        total_motivation = motivation_score + time_pressure

        # Add small random factor for unpredictability (±0.1)
        randomness = (random.random() - 0.5) * 0.2
        final_score = total_motivation + randomness

        # Decision threshold
        will_draw = final_score >= DRAWING_BASE_THRESHOLD

        print(f"[🎨] State drawing evaluation:")
        print(f"  Time since last: {time_since_last:.0f}s (min: {DRAWING_MIN_INTERVAL}s, max: {DRAWING_MAX_INTERVAL}s)")
        print(f"  Mood: {normalized_mood:.3f}, Novelty: {normalized_novelty:.3f}, Boredom: {normalized_boredom:.3f}")
        print(f"  Base motivation: {motivation_score:.3f}, Time pressure: {time_pressure:.3f}")
        print(f"  Final score: {final_score:.3f} (threshold: {DRAWING_BASE_THRESHOLD})")
        print(f"  Decision: {'DRAW' if will_draw else 'WAIT'}")

        if will_draw:
            print(f"[🎨] ✨ STATE DRAWING TRIGGERED: Internal motivation reached threshold")

        return will_draw

    def register_drawing(self, prompt: str) -> None:
        self.last_drawing_time = time.time()
        self.last_prompt = prompt
        self.last_drawing_prompt = prompt

        # Send drawing prompt to LCD display
        try:
            from utils.caption_display import send_caption_to_display
            send_caption_to_display(f"Drawing: {prompt}")
            print(f"[LCD] Sent drawing prompt: {prompt[:40]}...")
        except Exception as e:
            print(f"[LCD] Failed to send drawing prompt: {e}")

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def critique_drawing(self, image_path: str) -> None:
        """Critique a completed drawing using Ollama."""
        try:
            if not self.last_drawing_prompt or not self.last_reflection:
                return

            critique_prompt = SELF_CRITIQUE_PROMPT.format(
                original_prompt=self.last_drawing_prompt, reflection=self.last_reflection or "No specific reflection recorded"
            )

            critique_response = query_ollama(
                prompt=critique_prompt,
                image=image_path,
                log_dir=MOOD_SNAPSHOT_FOLDER,
                system_prompt=SELF_CRITIQUE_SYSTEM_PROMPT,
                prompt_type="reflection",
            )

            log_json_entry(
                LogType.REFLECTION,
                {
                    "event": "drawing_self_critique",
                    "image_path": image_path,
                    "original_prompt": self.last_drawing_prompt,
                    "critique": critique_response,
                    "timestamp": time.time(),
                },
                print_message=f"[🎯] Self-critique: {truncate_for_print(critique_response, 100)}",
            )

            # Store concise reflection for drawing memory
            try:
                from config.config import INCLUDE_DRAWING_HISTORY

                if INCLUDE_DRAWING_HISTORY:
                    one_liner = critique_response.strip().split("\n")[0]
                    if len(one_liner) > 160:
                        one_liner = one_liner[:157] + "..."
                    # We don't have direct agent reference here; try to import a global captioner if available
                    # Fallback: log as an event in state manager timeline if exposed later

            except Exception:
                pass

        except Exception as exc:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Error in drawing critique: {exc}", "component": "drawing_critique"},
                print_message=f"[❌] Error critiquing drawing: {exc}",
            )

    def handle_drawing_flow(
        self,
        agent: "Captioner",
        drawing_prompt: str,
        latest_image: str,
        *,
        reflection: Optional[str] = None,
    ) -> None:
        """Captioner passes the prompt already built – we just queue it."""
        self.last_reflection = reflection
        try:
            novelty = getattr(agent, "novelty_score", 0.0)
            boredom = getattr(agent, "boredom", 0.0)
            if not self.should_draw(
                mood=agent.current_mood,
                novelty=novelty,
                boredom=boredom,
                reflection=reflection,
            ):
                print_message = f"[🎨] Not inspired (novelty:{novelty},boredom:{boredom},mood:{agent.current_mood:.2f})"
                log_json_entry(
                    LogType.DECISION,
                    {
                        "decision": "skip_drawing",
                        "reason": "not_inspired",
                        "mood": agent.current_mood,
                        "novelty": novelty,
                        "boredom": boredom,
                        "ready_to_draw": self.ready_to_draw(),
                        "cooldown_remaining": max(0, self.cooldown - (time.time() - self.last_drawing_time)),
                    },
                    print_message=print_message,
                )
                return

            # NOTE: register_drawing() is now called AFTER GRBL execution completes, not here
            # This ensures cooldown starts after physical drawing, not after prompt generation

            # Record a concise drawing intent into memory for future reference
            try:
                from config.config import INCLUDE_DRAWING_HISTORY

                if INCLUDE_DRAWING_HISTORY and hasattr(agent, "observe"):
                    # Extract drawing summary from the model's own response
                    drawing_summary = "drawing based on current observations"  # fallback
                    try:
                        # Ask the model to summarize what it's drawing
                        from utils.ollama import query_ollama
                        from config.config import MOOD_SNAPSHOT_FOLDER

                        summary_prompt = f"Summarize what this drawing shows in 2-4 words:\n\n{drawing_prompt}\n\nSummary:"

                        drawing_summary = query_ollama(
                            prompt=summary_prompt,
                            log_dir=MOOD_SNAPSHOT_FOLDER,
                            system_prompt="You are summarizing visual content. Give only a brief 2-4 word description of the subject matter.",
                            prompt_type="drawing_summary",
                            options={"temperature": 0.3, "num_predict": 20}
                        ).strip()

                        print(f"[📝] Model-generated drawing summary: {drawing_summary}")

                        # Update state_manager's current_drawing_prompt with the concise summary
                        # so introspection references the summary instead of the full ComfyUI prompt
                        state_manager.current_drawing_prompt = drawing_summary

                    except Exception as e:
                        print(f"[⚠️] Summary generation failed: {e}")
                        drawing_summary = "drawing based on current observations"

                    agent.observe(f"Drawing intent: {drawing_summary}", agent.current_mood, latest_image or "", memory_type="drawing_intent")
                    print(f"[📝] Stored drawing intent in memory: {drawing_summary}")
                else:
                    print(f"[⚠️] Drawing intent not stored: INCLUDE_DRAWING_HISTORY={INCLUDE_DRAWING_HISTORY}, agent.observe exists={hasattr(agent, 'observe')}")
            except Exception as e:
                print(f"[❌] Failed to store drawing intent: {e}")
                import traceback
                traceback.print_exc()

            log_json_entry(
                LogType.DECISION,
                {
                    "decision": "trigger_drawing",
                    "reason": "inspired",
                    "mood": agent.current_mood,
                    "novelty": getattr(agent, "novelty_score", 0.0),
                    "boredom": getattr(agent, "boredom", 0.0),
                    "drawing_prompt": drawing_prompt,
                    "reflection": (reflection or "").strip(),
                },
                print_message=f"[🎨] Drawing prompt:\n{drawing_prompt}",
            )
            # Always echo full drawing prompt to console regardless of log filters
            try:
                print("[🖼️ Drawing Prompt]\n" + drawing_prompt)
            except Exception:
                pass

            if latest_image and os.path.exists(latest_image):
                self._invoke_comfyui_drawing(drawing_prompt, latest_image)
            else:
                log_json_entry(
                    LogType.ERROR,
                    {"message": "Cannot invoke ComfyUI - no valid image available", "component": "drawing", "image_path": latest_image},
                    print_message="[❌] Cannot invoke ComfyUI – no valid image available",
                )

        except Exception as exc:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Error in drawing flow: {exc}", "component": "drawing"},
                print_message=f"[❌] Error in drawing flow: {exc}",
            )

    # ------------------------------------------------------------------
    # ComfyUI invocation helper
    # ------------------------------------------------------------------
    def _invoke_comfyui_drawing(self, drawing_prompt: str, latest_image: str) -> None:
        try:
            # Don't pause idle movements yet - let them continue with "generating" pattern
            # We'll only pause when actual G-code execution starts
            try:
                from grbl.idle_movement_manager import update_emotion
                # Switch to "generating" pattern - continuous circular movements
                update_emotion("generating")
            except Exception as e:
                print(f"[⚠️] Could not switch to generating pattern: {e}")

            if os.path.exists(latest_image):
                image_path = latest_image
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"draw_input_{timestamp}.jpg")
                image_data = base64.b64decode(latest_image)
                with open(image_path, "wb") as f:
                    f.write(image_data)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"impostor-{timestamp}"
            controller = create_impostor_controller(
                load_image_path=image_path,
                override_prompt=drawing_prompt,
                primitive_string=TRIGGER_PROMPT,
                filename_prefix=filename_prefix,
                latent_width=COMFY_LATENT_WIDTH,
                latent_height=COMFY_LATENT_HEIGHT,
                cnet_strength=COMFY_CNET_STRENGTH,
                flux_guidance=COMFY_FLUX_GUIDANCE,
                steps=COMFY_STEPS,
                lora_strength=COMFY_LORA_STRENGTH,
            )
            if controller.queue_prompt():
                # Track expected output so the monitor only processes this job's image
                try:
                    state_manager.set_expected_output_prefix(filename_prefix)
                except Exception:
                    pass
                state_manager.start_drawing_generation(drawing_prompt)
                log_json_entry(
                    LogType.COMFY_PROMPT,
                    {"message": "ComfyUI drawing queued successfully", "drawing_prompt": drawing_prompt},
                    print_message=f"[🎨] Queued to ComfyUI with prompt:\n{drawing_prompt}",
                )
                # Always echo queued prompt to console as well
                try:
                    print("[🖼️ Queued Prompt]\n" + drawing_prompt)
                except Exception:
                    pass
                # Store an immediate post-queue note for drawing history
                try:
                    from config.config import INCLUDE_DRAWING_HISTORY

                    if INCLUDE_DRAWING_HISTORY:
                        note = drawing_prompt.strip().split("\n")[-1]
                        if len(note) > 160:
                            note = note[:157] + "..."
                        # We don't have the agent here; rely on state manager after image completes
                except Exception:
                    pass
            else:
                log_json_entry(
                    LogType.ERROR,
                    {"message": "Failed to queue ComfyUI drawing", "component": "comfy", "drawing_prompt": drawing_prompt},
                    print_message="[❌] Failed to queue ComfyUI drawing",
                )
        except Exception as exc:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Error invoking ComfyUI: {exc}", "component": "comfy"},
                print_message=f"[❌] Error invoking ComfyUI: {exc}",
            )
