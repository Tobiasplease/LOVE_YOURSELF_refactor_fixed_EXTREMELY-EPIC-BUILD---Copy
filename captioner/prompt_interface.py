"""
Prompt interface for the drawing pipeline.
The caption-side layer that used to live here (build_caption_prompt_with_options
and friends) was dead — the live caption loop calls prompts.build_simple_caption_prompt
directly. Removed Aug 30 2026 (git history keeps it).
"""

import random
from typing import Optional

from config import config
from config.model_settings import get_model_options

from .prompts import DRAWING_SYSTEM_PROMPT

class PromptInterface:
    """Interface for building the drawing prompt and its model options."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.MODEL_NAME

    def build_drawing_prompt_with_options(self, memory_ref, extra: Optional[str] = None, image_path: Optional[str] = None):
        """Build drawing prompt and prepare options with visual grounding support."""
        if not memory_ref:
            return None, None, None

        # Stream is the only pipeline since the Aug 19 consolidation (the
        # 5-step committee, kept "for A/B" since July 10 and never A/B'd, is
        # deleted — git history keeps it)
        from captioner.prompts import stream_drawing_analysis

        prompt = stream_drawing_analysis(memory_ref, extra=extra, image_path=image_path)

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

        # The live felt-state, or no feeling sentence at all. (The old
        # describe_current_mood read a frozen vector and put the same
        # "balanced emotional state" sentence in every drawing's system prompt.)
        felt_line = ""
        try:
            from captioner.context_compression import context_compressor as _cc

            _felt = _cc.get_felt_state()
            if _felt:
                felt_line = f"You are feeling {_felt}. "
        except Exception:
            pass

        return DRAWING_SYSTEM_PROMPT.format(
            temporal_context=temporal_context, accumulated_understanding=accumulated_understanding, felt_line=felt_line
        )

    def _get_base_model_options(self):
        """Get base model options for the current model."""
        return get_model_options(self.model_name)
