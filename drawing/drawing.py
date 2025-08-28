from __future__ import annotations

# from utils.ollama import query_ollama

"""drawing.py – final version

Captioner now hands us a *ready-made* drawing prompt that already contains
scene caption + reflection. This version does not query the LLM again,
and passes the prompt directly to ComfyUI.
"""

import os
import time
import base64
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from event_logging.run_manager import get_run_image_path

from config.config import DRAWING_COOLDOWN, MOOD_SNAPSHOT_FOLDER, TRIGGER_PROMPT
from config.prompt_templates import SELF_CRITIQUE_PROMPT
from utils.ollama import query_ollama
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
                print_message=f"[🎯] Self-critique: {critique_response[:100]}...",
            )

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
            if not self.should_draw(
                mood=agent.current_mood,
                novelty=getattr(agent, "novelty_score", 0.0),
                boredom=getattr(agent, "boredom", 0.0),
                reflection=reflection,
            ):
                log_json_entry(
                    LogType.DECISION,
                    {
                        "decision": "skip_drawing",
                        "reason": "not_inspired",
                        "mood": agent.current_mood,
                        "novelty": getattr(agent, "novelty_score", 0.0),
                        "boredom": getattr(agent, "boredom", 0.0),
                        "ready_to_draw": self.ready_to_draw(),
                        "cooldown_remaining": max(0, self.cooldown - (time.time() - self.last_drawing_time)),
                    },
                    print_message=f"""[🎨] Not inspired (novelty:{getattr(agent, 'novelty_score', 0.0):.2f},
                    boredom:{getattr(agent, 'boredom', 0.0):.2f},
                    mood:{agent.current_mood:.2f})""",
                )
                return

            self.register_drawing(drawing_prompt)

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
                print_message=f"[✨] Inspired! Creating artwork...",
            )

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
            if os.path.exists(latest_image):
                image_path = latest_image
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"draw_input_{timestamp}.jpg")
                image_data = base64.b64decode(latest_image)
                with open(image_path, "wb") as f:
                    f.write(image_data)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            controller = create_impostor_controller(
                load_image_path=image_path,
                override_prompt=drawing_prompt,
                primitive_string=TRIGGER_PROMPT,
                filename_prefix=f"impostor-{timestamp}",
                flux_guidance=4.0,  # @todo: mood controlled?
                cnet_strength=0.3,
                steps=25,
            )
            if controller.queue_prompt():
                state_manager.start_drawing_generation(drawing_prompt)
                log_json_entry(
                    LogType.COMFY_PROMPT,
                    {"message": "ComfyUI drawing queued successfully", "drawing_prompt": drawing_prompt},
                    print_message="[🎨] ComfyUI drawing queued successfully",
                )
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
