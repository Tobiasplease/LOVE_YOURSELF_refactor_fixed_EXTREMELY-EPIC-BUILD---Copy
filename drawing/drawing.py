from __future__ import annotations

import time
import os
import base64
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from captioner.captioner import Captioner

from .comfy import create_impostor_controller
from event_logging.json_logger import log_json_entry
from event_logging.run_manager import get_run_image_path
from ollama import query_ollama
from config.config import MOOD_SNAPSHOT_FOLDER, LLAVA_TIMEOUT_SUMMARY


class DrawingController:
    """
    Determines whether Lint JR should draw, based on mood, novelty,
    self-evaluation, and elapsed time.
    """

    def __init__(self) -> None:
        self.last_drawing_time: float = 0.0
        self.cooldown: float = 600.0  # seconds between possible drawings
        self.last_prompt: Optional[str] = None
        self.last_drawing_prompt: str = ""

    def ready_to_draw(self) -> bool:
        """Checks whether enough time has passed to consider drawing."""
        return time.time() - self.last_drawing_time > self.cooldown

    def should_draw(
        self,
        mood: float,
        novelty: float,
        boredom: float,
        evaluation: Optional[str] = None,
    ) -> bool:
        """
        Uses a rough heuristic to decide whether to draw.
        Returns True only if all criteria suggest sufficient emotional or symbolic content.
        """
        if not self.ready_to_draw():
            return False

        if novelty > 0.65 or boredom > 0.7:
            return True

        if evaluation:
            lowered = evaluation.lower()
            if any(phrase in lowered for phrase in ["i feel stuck", "i need to express", "nothing is changing", "this might be important"]):
                return True

        if mood < 0.3:
            return True

        return False

    def register_drawing(self, prompt: str) -> None:
        """
        Call this after a drawing decision has been made.
        Stores time and prompt.
        """
        self.last_drawing_time = time.time()
        self.last_prompt = prompt
        self.last_drawing_prompt = prompt

    def handle_drawing_flow(self, agent: "Captioner", evaluation: str, latest_image: str) -> None:
        """
        Handle the complete drawing flow including prompt generation and ComfyUI invocation.

        Args:
            agent: The Captioner instance with current state
            evaluation: The self-evaluation text
            latest_image: Path to the latest image to use for drawing
        """
        try:
            if not self.should_draw(
                mood=agent.current_mood,
                novelty=agent.novelty_score,
                boredom=agent.boredom,
                evaluation=evaluation,
            ):
                print("[❌] Not inspired to draw.")
                # @todo event log?
                return

            # Generate drawing prompt
            from captioner.prompts import build_drawing_prompt

            drawing_prompt = build_drawing_prompt(
                evaluation=evaluation,
                agent=agent,
                last_drawing_prompt=self.last_drawing_prompt,
            )

            self.register_drawing(drawing_prompt)

            log_json_entry(
                "drawing_prompt",
                {
                    "prompt": drawing_prompt,
                    "evaluation": evaluation.strip(),
                    "mood": agent.current_mood,
                    "boredom": agent.boredom,
                    "novelty_score": agent.novelty_score,
                    "last_drawing_prompt": self.last_drawing_prompt,
                },
                MOOD_SNAPSHOT_FOLDER,
            )

            print("[🎨] Drawing triggered.")

            # Generate ComfyUI prompt
            comfy_prompt_text = query_ollama(
                prompt=drawing_prompt, model="llava", image=None, timeout=LLAVA_TIMEOUT_SUMMARY, log_dir=MOOD_SNAPSHOT_FOLDER
            )
            log_json_entry("comfy_prompt", {"prompt": comfy_prompt_text, "latest_image": latest_image}, MOOD_SNAPSHOT_FOLDER)

            print(f"[🎨] ComfyUI prompt generated: {comfy_prompt_text}")

            # Invoke ComfyUI drawing
            if latest_image and os.path.exists(latest_image):
                self._invoke_comfyui_drawing(comfy_prompt_text, latest_image, agent)
            else:
                print("[⚠️] Cannot invoke ComfyUI - no valid image available for drawing")

        except Exception as e:
            print(f"[⚠️] Error in drawing flow: {e}")

    def _invoke_comfyui_drawing(self, drawing_prompt: str, latest_image: str, agent: "Captioner") -> None:
        """
        Invoke ComfyUI with the drawing prompt and latest mood image.

        Args:
            drawing_prompt: The generated drawing prompt
            latest_image: The encoded image from memory_queue (base64 or filepath)
            agent: The Captioner instance with current emotional state
        """
        try:
            # Read from state instead and avoid this mess? or just always deal with images on disk.
            # Determine if latest_image is an encoded image or a file path
            if os.path.exists(latest_image):
                # It's a file path, use it directly
                image_path = latest_image
                print(f"[🎨] Using existing image file: {image_path}")
            else:
                # It's likely a base64 encoded image, decode and save to disk
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = get_run_image_path(MOOD_SNAPSHOT_FOLDER, f"draw_input_{timestamp}.jpg")

                try:
                    # Decode base64 image and write to disk
                    image_data = base64.b64decode(latest_image)
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    print(f"[🎨] Decoded and saved image to: {image_path}")
                except Exception as decode_error:
                    print(f"[❌] Failed to decode image: {decode_error}")
                    return

            # Create ComfyUI controller with custom configuration
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            controller = create_impostor_controller(
                load_image_path=image_path,
                override_prompt=drawing_prompt,
                primitive_string="impostor black and white sketch line art ",
                filename_prefix=f"impostor-{timestamp}",
                flux_guidance=4.0,  # Example value, adjust as needed
                cnet_strength=0.3,  # Example value, adjust as needed
                # Use current emotional state to influence generation?
                # flux_guidance=max(2.0, min(6.0, 4.0 + (agent.current_mood - 0.5) * 2)),
                # cnet_strength=max(0.2, min(0.8, 0.5 + agent.novelty_score * 0.3)),
                # steps=max(15, min(35, int(25 + agent.boredom * 10))),
                steps=25,
            )

            # Queue the prompt to ComfyUI
            success = controller.queue_prompt()

            # time.sleeep here? thinking state?

            if success:
                print("[🎨] ComfyUI drawing queued successfully")
                print(f"[🎨] Input image: {image_path}")
                print(f"[🎨] Drawing prompt: {drawing_prompt}")
                print(f"[🎨] Emotional influence - Mood: {agent.current_mood:.2f}, Novelty: {agent.novelty_score:.2f}, Boredom: {agent.boredom:.2f}")
            else:
                print("[❌] Failed to queue ComfyUI drawing")

        except Exception as e:
            print(f"[⚠️] Error invoking ComfyUI: {e}")
