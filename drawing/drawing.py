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
from config.prompt_templates import SELF_CRITIQUE_PROMPT
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
        self.last_drawing_time: float = time.time()  # Initialize to current time to prevent immediate trigger
        self.cooldown: float = DRAWING_COOLDOWN  # seconds between drawings
        self.last_prompt: Optional[str] = None
        self.last_drawing_prompt: str = ""
        self.last_reflection: Optional[str] = None

    # ------------------------------------------------------------------
    # decision helpers
    # ------------------------------------------------------------------
    def ready_to_draw(self) -> bool:
        return time.time() - self.last_drawing_time > self.cooldown

    def should_draw(self, *, mood: float, novelty: float, boredom: float, reflection: Optional[str] = None) -> bool:
        if not self.ready_to_draw():
            return False
        # More reasonable thresholds for drawing triggers
        if novelty > 0.4 or boredom > 0.5 or mood < 0.4:
            return True
        reflections = ("i feel stuck", "i need to express", "nothing is changing", "want to draw", "create something")
        if reflection and any(key in reflection.lower() for key in reflections):
            return True
        return False

    def register_drawing(self, prompt: str) -> None:
        self.last_drawing_time = time.time()
        self.last_prompt = prompt
        self.last_drawing_prompt = prompt

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
                system_prompt="You are critiquing your own artwork. Be honest and constructive.",
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

            self.register_drawing(drawing_prompt)

            # Record a concise drawing intent into memory for future reference
            try:
                from config.config import INCLUDE_DRAWING_HISTORY

                if INCLUDE_DRAWING_HISTORY and hasattr(agent, "observe"):
                    intent = drawing_prompt.strip().split("\n")[0]
                    # Trim to a short one-liner
                    if len(intent) > 160:
                        intent = intent[:157] + "..."
                    agent.observe(f"Drawing intent: {intent}", agent.current_mood, latest_image or "", memory_type="drawing_intent")
            except Exception:
                pass

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
