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

        # Prepare model options with mood-aware variation settings
        model_options = self._get_base_model_options()
        model_options["seed"] = random.randint(1, 1000000)
        # Apply light mood modulation (no language steering)
        try:
            v, a, _ = self._get_mood_vector(memory_ref)
            model_options = self._apply_mood_modulation(model_options, v, a)
        except Exception:
            # Fallback to baseline
            pass

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

        # Minimal embodied identity prefix (roleplay seed) – single concise line only
        try:
            v, a, _ = self._get_mood_vector(memory_ref)
            label, intensity = self._label_from_mood(v, a, memory_ref)
            role_line = f"You are a {intensity}{label} drawing machine.\n"
            system_prompt = role_line + system_prompt
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

    # --- Mood helpers (no style scripting) ---
    def _get_mood_vector(self, memory_ref):
        v, a, c = (0.0, 0.0, 0.0)
        if memory_ref is not None and hasattr(memory_ref, "current_mood_vector"):
            mv = getattr(memory_ref, "current_mood_vector", (0.0, 0.0, 0.0))
            if isinstance(mv, (tuple, list)) and len(mv) >= 3:
                v, a, c = float(mv[0]), float(mv[1]), float(mv[2])
        return v, a, c

    def _label_from_mood(self, v: float, a: float, memory_ref) -> tuple[str, str]:
        """Map valence/arousal to a compact label and intensity adverb, with hysteresis via memory_ref."""
        import time
        label = "calm"
        # Primary label by quadrants
        if v > 0.4 and a > 0.5:
            label = "ecstatic"
        elif v > 0.2 and a > 0.2:
            label = "eager"
        elif v > 0.2 and a < -0.2:
            label = "serene"
        elif v < -0.4 and a > 0.4:
            label = "agitated"
        elif v < -0.5 and a < -0.2:
            label = "withdrawn"
        elif v < -0.2:
            label = "uneasy"
        elif a > 0.5:
            label = "restless"
        else:
            label = "calm"

        magnitude = max(abs(v), abs(a))
        intensity = ""
        if magnitude > 0.7:
            intensity = "deeply "
        elif magnitude > 0.45:
            intensity = "very "
        elif magnitude > 0.25:
            intensity = "slightly "

        # Hysteresis: soften label flips using memory_ref
        try:
            now = time.time()
            last_label = getattr(memory_ref, "_last_mood_label", None)
            last_ts = getattr(memory_ref, "_last_mood_label_ts", 0.0)
            if last_label and last_label != label and (now - last_ts) < 20.0:
                # Keep previous label if change is too soon and magnitude not extreme
                if magnitude < 0.75:
                    label = last_label
            # Persist
            setattr(memory_ref, "_last_mood_label", label)
            setattr(memory_ref, "_last_mood_label_ts", now)
        except Exception:
            pass

        return label, intensity

    def _apply_mood_modulation(self, opts: dict, v: float, a: float) -> dict:
        """Adjust decoding options from valence/arousal (no language rules)."""
        # Baseline from model settings
        temp = float(opts.get("temperature", 0.8))
        top_p = float(opts.get("top_p", 0.9))
        rep = float(opts.get("repeat_penalty", 1.1))
        top_k = int(opts.get("top_k", 40))
        npred = int(opts.get("num_predict", 200))

        # Modulate by arousal/intensity
        intensity = max(abs(v), abs(a))
        if a > 0.4:
            temp += 0.15 * intensity
            top_p = min(0.95, top_p + 0.02)
            rep = max(1.05, rep - 0.02)
            npred = min(240, npred + int(20 * intensity))
        elif a < -0.2:
            temp -= 0.1 * (0.3 - max(-0.3, a))
            top_p = max(0.82, top_p - 0.03)
            rep = min(1.2, rep + 0.03)
            npred = max(140, npred - 10)

        # Clamp reasonable bounds
        temp = max(0.5, min(temp, 1.2))
        top_p = max(0.8, min(top_p, 0.95))
        rep = max(1.03, min(rep, 1.25))

        opts.update({
            "temperature": temp,
            "top_p": top_p,
            "repeat_penalty": rep,
            "top_k": top_k,
            "num_predict": npred,
        })
        return opts
