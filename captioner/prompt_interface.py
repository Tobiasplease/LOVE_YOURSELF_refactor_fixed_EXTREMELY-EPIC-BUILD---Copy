"""
Centralized prompt interface - all prompt logic consolidated here.
This replaces the scattered prompt logic across model_wrapper.py and prompts.py.
"""

import os
import random
from typing import Optional

from config import config
from config.model_settings import get_model_options

from .prompts import (
    DRAWING_SYSTEM_PROMPT,
    STATIC_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_drawing_prompt,
    build_environmental_caption_prompt,
    build_ongoing_caption_prompt,
    build_reflection_prompt,
)
from utils.view_orientation import describe_view_orientation


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
            # Only inject non-drawing contexts - drawing context is handled directly in prompts.py
            emotional_context = self._get_emotional_context()
            baseline_context = self._get_baseline_context()
            # REMOVED: drawing_context injection - handled in prompts.py build_ongoing_caption_prompt()

            prompt = build_ongoing_caption_prompt(memory_ref, getattr(memory_ref, "last_caption", None))

            # Add non-drawing contexts only
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

        # Format dynamic SYSTEM_PROMPT with temporal/motif context if available
        system_prompt = SYSTEM_PROMPT

        # Try to format with dynamic context if memory_ref has the method
        if memory_ref and hasattr(memory_ref, "get_dynamic_system_context"):
            try:
                dynamic_context = memory_ref.get_dynamic_system_context()
                if isinstance(dynamic_context, dict):
                    system_prompt = SYSTEM_PROMPT.format(
                        emotional_state=dynamic_context.get("emotional_state", "contemplative"),
                        temporal_context=dynamic_context.get("temporal_context", ""),
                        accumulated_understanding=dynamic_context.get("accumulated_understanding", ""),
                    )
            except Exception as e:
                # Fall back to static prompt if formatting fails
                print(f"[PROMPT] Dynamic context formatting failed: {e}")
                system_prompt = STATIC_SYSTEM_PROMPT
        else:
            # No dynamic context available, use static fallback
            system_prompt = STATIC_SYSTEM_PROMPT

        # Subtle egocentric head orientation guidance (implicit context only)
        try:
            vp = getattr(memory_ref, "view_pan", None)
            vt = getattr(memory_ref, "view_tilt", None)
            if isinstance(vp, (int, float)) and isinstance(vt, (int, float)):
                orientation = describe_view_orientation(vp, vt)
                if orientation:
                    system_prompt = f"{system_prompt}\n\nHEAD ORIENTATION (implicit): {orientation}"
        except Exception:
            pass

        # Return prompt, options, and formatted system prompt
        return prompt, model_options, system_prompt

    def build_reflection_prompt_with_options(self, caption: str, agent=None, extra: Optional[str] = None):
        """Build reflection prompt and prepare options."""
        prompt = build_reflection_prompt(caption, extra=extra, agent=agent)
        model_options = self._get_base_model_options()
        model_options["seed"] = random.randint(1, 1000000)

        return prompt, model_options, SYSTEM_PROMPT

    def build_drawing_prompt_with_options(self, memory_ref, extra: Optional[str] = None):
        """Build drawing prompt and prepare options."""
        if not memory_ref:
            return None, None, None

        prompt = build_drawing_prompt(memory_ref, extra=extra)
        model_options = self._get_base_model_options()
        model_options["seed"] = random.randint(1, 1000000)

        # Make drawing prompts more concrete and directive
        model_options.update(
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.3,
                "top_k": 40,
                "num_predict": max(model_options.get("num_predict", 160), 140),
            }
        )

        drawing_system_prompt = DRAWING_SYSTEM_PROMPT

        return prompt, model_options, drawing_system_prompt

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

# REMOVED: _get_drawing_context() - drawing context now handled directly in prompts.py

    def _get_base_model_options(self):
        """Get base model options for the current model."""
        return get_model_options(self.model_name)
