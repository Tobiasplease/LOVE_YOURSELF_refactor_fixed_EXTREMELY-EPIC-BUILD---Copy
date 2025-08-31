"""
Centralized prompt interface - all prompt logic consolidated here.
This replaces the scattered prompt logic across model_wrapper.py and prompts.py.
"""

import os
import random
from typing import Optional

from config import config

# from config.config import OLLAMA_TIMEOUT_REFLECTION
from config.model_settings import get_model_options

from .prompts import build_drawing_prompt, build_environmental_caption_prompt, build_reflection_prompt, build_simple_caption_prompt


class PromptInterface:
    """Centralized interface for all prompt building and preparation."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.OLLAMA_MODEL

    def build_caption_prompt_with_options(self, memory_ref, image_path: str, *, flowing: bool = True, first_time: bool = False):
        """Build caption prompt and prepare all options for API call."""
        if not os.path.exists(image_path):
            return None, None, None

        # Build the prompt based on context
        if first_time:
            if memory_ref:
                session_gap = getattr(memory_ref, "last_session_gap", None)
                prompt = build_environmental_caption_prompt(
                    memory_ref,
                    mood=getattr(memory_ref, "current_mood", 0.5),
                    boredom=getattr(memory_ref, "boredom", 0.0),
                    novelty=getattr(memory_ref, "novelty_score", 0.5),
                    last_session_gap=session_gap,
                )
            else:
                prompt = "What do I perceive as I awaken to consciousness for the first time?"
        elif flowing and memory_ref:
            # Inject contexts from compression system
            emotional_context = self._get_emotional_context()
            baseline_context = self._get_baseline_context()

            prompt = build_simple_caption_prompt(
                memory_ref, getattr(memory_ref, "current_mood_vector", (0.5, 0.0, 0.0)), getattr(memory_ref, "last_caption", None)
            )

            # Add contexts if available
            context_parts = []
            if baseline_context:
                context_parts.append(baseline_context)
            if emotional_context:
                context_parts.append(emotional_context)

            if context_parts:
                context_string = "\n\n".join(context_parts)
                prompt = f"{context_string}\n\n{prompt}"
        else:
            prompt = "Describe this image."

        # Prepare model options with variation settings
        model_options = self._get_base_model_options()
        model_options["seed"] = random.randint(1, 1000000)
        model_options.update({"temperature": 1.5, "top_p": 0.7, "repeat_penalty": 1.5, "top_k": 20})

        # Return prompt, options, and system prompt
        return prompt, model_options, config.SYSTEM_PROMPT

    def build_reflection_prompt_with_options(self, caption: str, agent=None, extra: Optional[str] = None):
        """Build reflection prompt and prepare options."""
        prompt = build_reflection_prompt(caption, extra=extra, agent=agent)
        model_options = self._get_base_model_options()
        model_options["seed"] = random.randint(1, 1000000)

        return prompt, model_options, config.SYSTEM_PROMPT

    def build_drawing_prompt_with_options(self, memory_ref, extra: Optional[str] = None):
        """Build drawing prompt and prepare options."""
        if not memory_ref:
            return None, None, None

        prompt = build_drawing_prompt(memory_ref, extra=extra)
        model_options = self._get_base_model_options()
        model_options["seed"] = random.randint(1, 1000000)

        return prompt, model_options, config.SYSTEM_PROMPT

    def _get_emotional_context(self) -> str:
        """Get current emotional context from compression system for prompt injection."""
        try:
            from captioner.context_compression import context_compressor

            return context_compressor.get_current_sentiment_context()
        except Exception as e:
            print(f"[PROMPT] Could not get emotional context: {e}")
            return ""

    def _get_baseline_context(self) -> str:
        """Get baseline understanding context from compression system."""
        try:
            from captioner.context_compression import context_compressor

            return context_compressor.get_baseline_context()
        except Exception as e:
            print(f"[PROMPT] Could not get baseline context: {e}")
            return ""

    def _get_base_model_options(self):
        """Get base model options for the current model."""
        return get_model_options(self.model_name)
