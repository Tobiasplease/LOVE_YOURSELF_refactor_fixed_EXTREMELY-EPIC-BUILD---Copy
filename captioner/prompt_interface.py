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
    context_rich_multi_step_drawing_analysis,
)


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

            # Add non-drawing contexts and temporal awareness
            context_parts = []
            if baseline_context:
                context_parts.append(baseline_context)
            if emotional_context:
                context_parts.append(emotional_context)

            # Add temporal stagnation awareness from compression system
            try:
                from captioner.context_compression import context_compressor
                stagnation_info = context_compressor.get_current_stagnation_info()
                if stagnation_info["stagnation_duration_minutes"] > 0:
                    duration_desc = stagnation_info["duration_description"]
                    temporal_context = f"TEMPORAL AWARENESS: You have been observing this environment for {duration_desc}. This duration shapes your current emotional state and perspective."
                    context_parts.append(temporal_context)
            except Exception as e:
                print(f"[PROMPT] Could not get temporal context: {e}")

            if context_parts:
                context_string = "\n\n".join(context_parts)
                prompt = f"{context_string}\n\n{prompt}"
        else:
            prompt = "Describe this image."

        # Prepare model options with variation settings
        model_options = self._get_base_model_options()
        model_options["seed"] = random.randint(1, 1000000)

        # Use appropriate temperature based on caption type
        try:
            if first_time:
                from config.config import ENVIRONMENTAL_TEMPERATURE

                caption_temp = ENVIRONMENTAL_TEMPERATURE
            else:
                from config.config import CAPTIONER_TEMPERATURE

                caption_temp = CAPTIONER_TEMPERATURE
        except ImportError:
            caption_temp = 1.2 if not first_time else 0.9  # Default fallback

        model_options.update({"temperature": caption_temp, "top_p": 0.7, "repeat_penalty": 1.5, "top_k": 20})

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
                        spatial_language_hints=dynamic_context.get("spatial_language_hints", ""),
                    )
            except Exception as e:
                # Fall back to static prompt if formatting fails
                print(f"[PROMPT] Dynamic context formatting failed: {e}")
                system_prompt = STATIC_SYSTEM_PROMPT
        else:
            # No dynamic context available, use static fallback
            system_prompt = STATIC_SYSTEM_PROMPT

        # Return prompt, options, and formatted system prompt
        return prompt, model_options, system_prompt

    def build_reflection_prompt_with_options(self, caption: str, agent=None, extra: Optional[str] = None):
        """Build reflection prompt and prepare options."""
        prompt = build_reflection_prompt(caption, extra=extra, agent=agent)
        model_options = self._get_base_model_options()
        model_options["seed"] = random.randint(1, 1000000)

        # Use configurable reflection temperature
        try:
            from config.config import REFLECTION_TEMPERATURE

            model_options["temperature"] = REFLECTION_TEMPERATURE
        except ImportError:
            model_options["temperature"] = 1.1  # Default fallback

        return prompt, model_options, SYSTEM_PROMPT

    def build_drawing_prompt_with_options(self, memory_ref, extra: Optional[str] = None, image_path: Optional[str] = None):
        """Build drawing prompt and prepare options with visual grounding support."""
        if not memory_ref:
            return None, None, None

        # Choose between multi-step context-rich analysis or single-prompt approach
        try:
            from config.config import USE_MULTI_STEP_DRAWING_ANALYSIS

            use_multi_step = USE_MULTI_STEP_DRAWING_ANALYSIS
        except ImportError:
            use_multi_step = False  # Fallback to single-prompt

        if use_multi_step:
            print("[🎨] Using context-rich multi-step drawing analysis")
            prompt = context_rich_multi_step_drawing_analysis(memory_ref, extra=extra, image_path=image_path)
        else:
            print("[🎨] Using single-prompt drawing analysis")
            prompt = build_drawing_prompt(memory_ref, extra=extra, image_path=image_path)

        model_options = self._get_base_model_options()
        model_options["seed"] = random.randint(1, 1000000)

        # Use configurable drawing temperature and make prompts more concrete and directive
        try:
            from config.config import DRAWING_TEMPERATURE

            drawing_temp = DRAWING_TEMPERATURE
        except ImportError:
            drawing_temp = 1.0  # Default fallback

        model_options.update(
            {
                "temperature": max(drawing_temp - 0.2, 0.8),  # Slightly lower temperature for more focused responses
                "top_p": 0.9,
                "repeat_penalty": 1.2,  # Lower repeat penalty to allow detailed responses
                "top_k": 40,
                "num_predict": max(model_options.get("num_predict", 500), 400),  # Much longer responses for comprehensive analysis
            }
        )

        # Build rich contextual drawing system prompt with variables (like main SYSTEM_PROMPT)
        drawing_system_prompt = self._build_drawing_system_prompt_with_context(memory_ref)

        return prompt, model_options, drawing_system_prompt

    def _build_drawing_system_prompt_with_context(self, memory_ref) -> str:
        """Build drawing system prompt with rich contextual variables like main SYSTEM_PROMPT."""

        # Get temporal context
        temporal_context = ""
        if hasattr(memory_ref, "temporal_prompt_lines"):
            tlines = memory_ref.temporal_prompt_lines()
            if tlines:
                temporal_context = " ".join(tlines[:2]) + ". "  # Keep it concise for system prompt

        # Get accumulated understanding
        accumulated_understanding = ""
        try:
            from captioner.context_compression import context_compressor

            understanding = context_compressor.get_consolidated_understanding()
            if understanding:
                # Truncate for system prompt
                accumulated_understanding = understanding[:200] + "... " if len(understanding) > 200 else understanding + " "
        except Exception:
            pass

        # Get emotional state description
        emotional_state = ""
        try:
            if hasattr(memory_ref, "describe_current_mood") and callable(memory_ref.describe_current_mood):
                emotional_state = memory_ref.describe_current_mood()
            else:
                emotional_state = "aware and focused"
        except Exception:
            emotional_state = "aware and focused"

        # Format the system prompt with context
        return DRAWING_SYSTEM_PROMPT.format(
            temporal_context=temporal_context, accumulated_understanding=accumulated_understanding, emotional_state=emotional_state
        )

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
