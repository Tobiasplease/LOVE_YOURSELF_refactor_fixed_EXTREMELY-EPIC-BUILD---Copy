"""
Clean model wrapper - pure API handler only.
All prompt logic moved to prompt_interface.py for centralization.
"""

import re
from typing import Optional

from config.config import (
    MOOD_SNAPSHOT_FOLDER,
    MOTIF_MODEL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_REFLECTION,
    TINYLLAMA_NUM_PREDICT,
    TINYLLAMA_TEMPERATURE,
    TINYLLAMA_TIMEOUT,
    TINYLLAMA_TOP_P,
)
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.ollama import query_ollama, truncate_for_print

from .prompt_interface import PromptInterface
from .prompts import NUMBER_GENERATOR_SYSTEM_PROMPT


class MultimodalModel:
    """Simplified model wrapper - pure API handler."""

    def __init__(self, memory_ref: Optional[any] = None) -> None:  # type: ignore
        self.memory_ref = memory_ref
        self.model_name = OLLAMA_MODEL
        self.prompt_interface = PromptInterface(self.model_name)

    def caption_image(self, image_path: str, *, flowing: bool = True, first_time: bool = False, drawing_introspection_mode: bool = False, person_present: bool = False) -> tuple:
        """Generate image caption using centralized prompt interface.

        Returns:
            tuple: (caption_text, prompt_mode) where prompt_mode is 'introspective', 'observational', 'relational', etc.
        """
        # Get prompt and options from centralized interface
        prompt, model_options, system_prompt, prompt_mode = self.prompt_interface.build_caption_prompt_with_options(
            self.memory_ref, image_path, flowing=flowing, first_time=first_time, drawing_introspection_mode=drawing_introspection_mode, person_present=person_present
        )

        if prompt is None:
            return ("Vision initializing... camera systems coming online...", "awakening")

        log_json_entry(
            LogType.DEBUG,
            {
                "message": "Caption prompt generated",
                "action": "prompt_hash",
                "prompt_hash": hash(prompt),
                "prompt_preview": prompt[:200],
                "flowing": flowing,
                "first_time": first_time,
                "prompt_mode": prompt_mode,
            },
            print_message=f"[🐞] Prompt hash: {hash(prompt)}, mode: {prompt_mode}, preview: {truncate_for_print(prompt, 200)}",
        )

        # Use Natsumura for introspective mode (text-only, narrative continuation)
        if prompt_mode == "introspective":
            result = self._call_natsumura_introspective(prompt, system_prompt, model_options)
        else:
            # Use LLaVA for visual modes (observational, relational, etc.)
            result = self._call_ollama(prompt, image_path=image_path, system_prompt=system_prompt, model_options=model_options, prompt_type="caption")

        log_json_entry(
            LogType.DEBUG,
            {
                "message": "Caption response received",
                "action": "response_hash",
                "response_hash": hash(result),
                "response_preview": result[:50],
                "response_length": len(result),
            },
            print_message=f"[🐞] Response hash: {hash(result)}, preview: {truncate_for_print(result, 50)}",
        )
        return (result, prompt_mode)

    def reason_about_caption(
        self, caption: str, *, agent: Optional[any] = None, mood_text: Optional[str] = None, extra: Optional[str] = None  # type: ignore
    ) -> str:  # type: ignore
        """Generate reflection using centralized prompt interface."""
        try:
            prompt, model_options, system_prompt = self.prompt_interface.build_reflection_prompt_with_options(caption, agent=agent, extra=extra)

            log_json_entry(
                LogType.REFLECTION,
                {
                    "message": "Starting reflection",
                    "action": "reflection_start",
                    "timeout": OLLAMA_TIMEOUT_REFLECTION,
                    "caption_preview": caption[:50],
                },
                print_message=f"[🤔] Starting reflection with timeout={OLLAMA_TIMEOUT_REFLECTION}s",
            )

            response = self._call_ollama(
                prompt, system_prompt=system_prompt, model_options=model_options, timeout=OLLAMA_TIMEOUT_REFLECTION, prompt_type="reflection"
            )

            log_json_entry(
                LogType.REFLECTION,
                {
                    "message": "Reflection completed",
                    "action": "reflection_success",
                    "response_length": len(response),
                    "response_preview": response[:100],
                },
                print_message=f"[✅] Reflection completed: {len(response)} chars",
            )
            return response
        except Exception as e:
            print(f"[ERROR] Reflection failed: {e}")
            import traceback

            traceback.print_exc()
            return "[WARNING] Reflection generation failed"

    def generate_drawing_prompt(self, *, extra: Optional[str] = None, image_path: Optional[str] = None) -> str:
        """Generate drawing prompt using centralized prompt interface with VISUAL GROUNDING."""
        prompt, model_options, system_prompt = self.prompt_interface.build_drawing_prompt_with_options(
            self.memory_ref, extra=extra, image_path=image_path
        )

        if prompt is None:
            return "[WARNING] No memory available for drawing prompt"

        # Log the exact input we send to the LLM for drawing prompt generation
        try:
            log_json_entry(
                LogType.DEBUG,
                {
                    "message": "Visual drawing LLM input prepared",
                    "action": "llm_input",
                    "prompt_preview": truncate_for_print(prompt, 400),
                    "prompt_length": len(prompt),
                    "image_provided": image_path is not None,
                    "image_path": image_path,
                    "options": {k: model_options.get(k) for k in ("temperature", "top_p", "repeat_penalty", "top_k", "num_predict", "seed")},
                },
                print_message=f"[🎨] Visual drawing prompt generation {'WITH IMAGE' if image_path else 'TEXT ONLY'}: {truncate_for_print(prompt, 220)}",
            )
        except Exception:
            pass

        # If using natsumura or multi-step analysis, the prompt IS the final result, don't call LLM again
        try:
            from config.config import DRAWING_ANALYSIS_MODE
            if DRAWING_ANALYSIS_MODE in ("natsumura", "multi_step"):
                return prompt  # These modes already return the final drawing prompt
        except ImportError:
            pass

        # For single-prompt approach, call LLM
        return self._call_ollama(prompt, image_path=image_path, system_prompt=system_prompt, model_options=model_options, prompt_type="drawing")

    def query_tinyllama(self, prompt: str) -> str:
        """Query TinyLlama model for motif scoring and emotional analysis."""
        tinyllama_options = {
            "temperature": TINYLLAMA_TEMPERATURE,
            "top_p": TINYLLAMA_TOP_P,
            "num_predict": TINYLLAMA_NUM_PREDICT,
        }

        try:
            response = query_ollama(
                prompt=prompt,
                model=MOTIF_MODEL,
                timeout=TINYLLAMA_TIMEOUT,
                log_dir=MOOD_SNAPSHOT_FOLDER,
                system_prompt=NUMBER_GENERATOR_SYSTEM_PROMPT,
                options=tinyllama_options,
                prompt_type="motif_scoring",
            )
            return response.strip()
        except Exception:
            return "0.5"

    def _call_natsumura_introspective(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_options: dict | None = None,
    ) -> str:
        """Call Natsumura model for introspective/narrative captions (no image needed)."""
        try:
            from config.config import COMPRESSION_MODEL
            natsumura_model = COMPRESSION_MODEL
        except ImportError:
            natsumura_model = "natsumura-storytelling-rp:latest"

        # Use provided options or get defaults, but adjust for narrative mode
        if model_options is None:
            model_options = self.prompt_interface._get_base_model_options()

        # Adjust options for narrative introspection - SHORT inner thoughts, not prose
        model_options.update({
            "temperature": 0.9,
            "top_p": 0.7,
            "repeat_penalty": 1.5,
            "num_predict": 60,  # Force brevity - let num_predict handle length
            # Removed aggressive stop sequences - they cause mid-thought truncation
        })

        log_json_entry(
            LogType.DEBUG,
            {
                "message": "Natsumura introspective call",
                "action": "natsumura_caption",
                "model": natsumura_model,
                "prompt_preview": prompt[:150],
            },
            print_message=f"[🌸] Natsumura introspective: {truncate_for_print(prompt, 100)}",
        )

        response = query_ollama(
            prompt=prompt,
            model=natsumura_model,
            image=None,  # No image for introspective mode
            timeout=60,
            log_dir=MOOD_SNAPSHOT_FOLDER,
            system_prompt=system_prompt,
            options=model_options,
            prompt_type="introspective_caption",
        )

        if not response:
            response = ""

        return self._clean_response(response)

    def _call_ollama(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        timeout: int = 90,
        model_options: dict | None = None,
        prompt_type: str = "general",
    ) -> str:
        """Pure API call handler - no prompt logic here."""

        # Use provided options or get defaults
        if model_options is None:
            model_options = self.prompt_interface._get_base_model_options()

        response = query_ollama(
            prompt=prompt,
            model=self.model_name,
            image=image_path,
            timeout=timeout,
            log_dir=MOOD_SNAPSHOT_FOLDER,
            system_prompt=system_prompt,
            options=model_options,
            prompt_type=prompt_type,
        )

        if not response:
            response = ""

        return self._clean_response(response)

    def _clean_response(self, response: str) -> str:
        """Remove unwanted AI-generated prompt leakage from responses."""
        unwanted_patterns = [
            r"\\n\\nFeelings:.*?\\?",
            r"\\n\\nReflection:.*?\\?",
            r"\\n\\nWhat do you feel\\?",
            r"\\n\\nHow does.*?feel\\?",
            r"Feelings: What do you feel\\?",
            r"Reflection: How does.*?\\?",
        ]

        cleaned = response
        for pattern in unwanted_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

        return cleaned.strip()
