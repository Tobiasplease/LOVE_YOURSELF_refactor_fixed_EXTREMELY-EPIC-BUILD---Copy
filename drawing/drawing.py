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

from event_logging.event_logger import log_json_entry, LogType
from event_logging.run_manager import get_run_image_path

from config.config import DRAWING_COOLDOWN, MOOD_SNAPSHOT_FOLDER
from .comfy import create_impostor_controller

if TYPE_CHECKING:
    from captioner.captioner import Captioner


class DrawingController:
    """Decides when to draw and queues ComfyUI jobs."""

    def __init__(self) -> None:
        self.last_drawing_time: float = 0.0
        self.cooldown: float = DRAWING_COOLDOWN  # seconds between drawings
        self.last_prompt: Optional[str] = None
        self.last_drawing_prompt: str = ""

    # ------------------------------------------------------------------
    # decision helpers
    # ------------------------------------------------------------------
    def ready_to_draw(self) -> bool:
        return time.time() - self.last_drawing_time > self.cooldown

    def should_draw(self, *, mood: float, novelty: float, boredom: float, reflection: Optional[str] = None) -> bool:
        if not self.ready_to_draw():
            return False
        if novelty > 0.65 or boredom > 0.7 or mood < 0.3:
            return True
        if reflection and any(key in reflection.lower() for key in ("i feel stuck", "i need to express", "nothing is changing")):
            return True
        return False

    def register_drawing(self, prompt: str) -> None:
        self.last_drawing_time = time.time()
        self.last_prompt = prompt
        self.last_drawing_prompt = prompt

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def handle_drawing_flow(
        self,
        agent: "Captioner",
        drawing_prompt: str,
        latest_image: str,
        *,
        reflection: Optional[str] = None,
    ) -> None:
        """Captioner passes the prompt already built – we just queue it."""
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
                    MOOD_SNAPSHOT_FOLDER,
                    auto_print=True,
                    print_message="❌ Not inspired to draw",
                )
                return

            self.register_drawing(drawing_prompt)

            # log_json_entry(
            #     LogType.DRAWING_PROMPT,
            #     {
            #         "prompt": drawing_prompt,
            #         "reflection": (reflection or "").strip(),
            #         "mood": agent.current_mood,
            #         "boredom": getattr(agent, "boredom", 0.0),
            #         "novelty_score": getattr(agent, "novelty_score", 0.0),
            #         "last_drawing_prompt": self.last_drawing_prompt,
            #     },
            #     MOOD_SNAPSHOT_FOLDER,
            # )

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
                MOOD_SNAPSHOT_FOLDER,
                auto_print=True,
                print_message="🎨 Drawing triggered",
            )

            if latest_image and os.path.exists(latest_image):
                self._invoke_comfyui_drawing(drawing_prompt, latest_image)
            else:
                log_json_entry(
                    LogType.ERROR,
                    {"message": "Cannot invoke ComfyUI - no valid image available", "component": "drawing", "image_path": latest_image},
                    MOOD_SNAPSHOT_FOLDER,
                    auto_print=True,
                    print_message="⚠️ Cannot invoke ComfyUI – no valid image available",
                )

        except Exception as exc:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Error in drawing flow: {exc}", "component": "drawing"},
                MOOD_SNAPSHOT_FOLDER,
                auto_print=True,
                print_message=f"⚠️ Error in drawing flow: {exc}",
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
                primitive_string="impostor black and white sketch line art ",
                filename_prefix=f"impostor-{timestamp}",
                flux_guidance=4.0,  # @todo: mood controlled?
                cnet_strength=0.3,
                steps=25,
            )
            if controller.queue_prompt():
                log_json_entry(
                    LogType.COMFY_PROMPT,
                    {"message": "ComfyUI drawing queued successfully", "drawing_prompt": drawing_prompt},
                    MOOD_SNAPSHOT_FOLDER,
                    auto_print=True,
                    print_message="🎨 ComfyUI drawing queued successfully",
                )
            else:
                log_json_entry(
                    LogType.ERROR,
                    {"message": "Failed to queue ComfyUI drawing", "component": "comfy", "drawing_prompt": drawing_prompt},
                    MOOD_SNAPSHOT_FOLDER,
                    auto_print=True,
                    print_message="❌ Failed to queue ComfyUI drawing",
                )
        except Exception as exc:
            log_json_entry(
                LogType.ERROR,
                {"message": f"Error invoking ComfyUI: {exc}", "component": "comfy"},
                MOOD_SNAPSHOT_FOLDER,
                auto_print=True,
                print_message=f"⚠️ Error invoking ComfyUI: {exc}",
            )
