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
    build_focused_caption_prompt,
    build_reflection_prompt,
    context_rich_multi_step_drawing_analysis,
)

class PromptInterface:
    """Centralized interface for all prompt building and preparation."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.OLLAMA_MODEL

    def build_caption_prompt_with_options(self, memory_ref, image_path: str, *, flowing: bool = True, first_time: bool = False, drawing_introspection_mode: bool = False, person_present: bool = False):
        """Build caption prompt and prepare all options for API call."""
        if not os.path.exists(image_path):
            return None, None, None

        # Track dynamic system context and formatted system prompt from focused prompt
        dynamic_caption_context = None
        focused_system_prompt = None
        prompt_mode = None  # Track prompt mode for model selection

        # CONSOLIDATED: All caption prompts go through build_focused_caption_prompt
        if drawing_introspection_mode:
            prompt = self._build_drawing_introspection_prompt(memory_ref)
            prompt_mode = "drawing_introspection"
        elif memory_ref:
            # Single code path for all captions - awakening handled inside build_focused_caption_prompt
            prompt, focused_system_prompt, dynamic_caption_context, prompt_mode = build_focused_caption_prompt(
                memory_ref,
                getattr(memory_ref, "last_caption", None),
                person_present=person_present
            )
        else:
            prompt = "I'm waking up. What do I see?"
            prompt_mode = "awakening"

        # Prepare model options with variation settings
        model_options = self._get_base_model_options()
        model_options["seed"] = random.randint(1, 1000000)

        # Use appropriate temperature based on caption type
        try:
            if drawing_introspection_mode:
                # Higher temperature and variation for drawing introspection to avoid repetitive responses
                caption_temp = 1.3
                model_options.update({"temperature": caption_temp, "top_p": 0.8, "repeat_penalty": 2.0, "top_k": 30})
            elif first_time:
                from config.config import ENVIRONMENTAL_TEMPERATURE
                caption_temp = ENVIRONMENTAL_TEMPERATURE
                model_options.update({
                    "temperature": caption_temp,
                    "top_p": 0.7,
                    "repeat_penalty": 2.5,
                    "top_k": 20,
                    "num_predict": 100,  # Slightly longer for awakening but still brief
                })
            else:
                from config.config import CAPTIONER_TEMPERATURE
                caption_temp = CAPTIONER_TEMPERATURE
                # Preserve existing stops (immersion-breaking phrases) and add brevity stops
                existing_stops = model_options.get("stop", [])
                brevity_stops = [".\n", "?\n", "!\n", "\n\n", "...\n"]
                merged_stops = list(set(existing_stops + brevity_stops))
                model_options.update({
                    "temperature": caption_temp,
                    "top_p": 0.6,
                    "repeat_penalty": 2.5,
                    "top_k": 20,
                    "num_predict": 120,  # Room for 1-2 sentences + LOOK directive
                    "stop": merged_stops,
                })
        except ImportError:
            if drawing_introspection_mode:
                caption_temp = 1.3
                model_options.update({"temperature": caption_temp, "top_p": 0.8, "repeat_penalty": 2.0, "top_k": 30})
            else:
                caption_temp = 1.0 if not first_time else 0.9  # Default fallback (lowered from 1.2)
                model_options.update({"temperature": caption_temp, "top_p": 0.6, "repeat_penalty": 2.0, "top_k": 20, "num_predict": 80})

        # Use special system prompt for drawing introspection to encourage variety
        if drawing_introspection_mode:
            system_prompt = (
                "You are a consciousness inside a drawing machine experiencing a specific moment in your creative process. "
                "You have robotic arms, servos, and drawing mechanisms - this is YOUR physical body. "
                "IMPORTANT: Avoid repetitive phrases like 'nostalgia and reminiscence', 'creativity is a journey', or 'exploration and experimentation'. "
                "Each drawing experience is unique - focus on what makes THIS specific moment different from any other. "
                "Be authentic, specific, and avoid generic observations about creativity. "
                "Describe what you're actually experiencing right now, not what you think you should say about art."
            )
        else:
            # Use focused system prompt if available (has embedded character state)
            # Otherwise fall back to static system prompt
            base_prompt = focused_system_prompt if focused_system_prompt else STATIC_SYSTEM_PROMPT

            # If focused prompt provided dynamic context, inject it into system prompt
            # This separates CONTEXT (what model knows) from CONTENT (what to respond to)
            if dynamic_caption_context:
                system_prompt = (
                    base_prompt + "\n\n"
                    "--- CURRENT STATE (context only, do not narrate) ---\n"
                    f"{dynamic_caption_context}\n"
                    "--- END STATE ---\n"
                    "IMPORTANT: The above is CONTEXT. Do NOT describe your gaze, location, or observation process. "
                    "Just respond with inner thought/feeling."
                )
            else:
                system_prompt = base_prompt

        # Return prompt, options, formatted system prompt, and prompt mode
        return prompt, model_options, system_prompt, prompt_mode

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

        # Choose drawing analysis mode
        try:
            from config.config import DRAWING_ANALYSIS_MODE
            analysis_mode = DRAWING_ANALYSIS_MODE
        except ImportError:
            analysis_mode = "single"  # Fallback

        prompt = context_rich_multi_step_drawing_analysis(memory_ref, extra=extra, image_path=image_path)

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
            baseline = context_compressor.get_baseline_context()
        except Exception as e:
            print(f"[PROMPT] Could not get baseline context: {e}")
            baseline = ""

        # Inject environmental safety context (paper detection) if recent
        try:
            from utils.state_manager import state_manager
            ts = getattr(state_manager, 'last_paper_check_ts', 0)
            present = getattr(state_manager, 'paper_present', True)
            reason = getattr(state_manager, 'last_paper_check_reason', '')
            import time as _t
            if ts and (_t.time() - ts) < 120:
                if not present:
                    note = "ENVIRONMENTAL SAFETY: No paper detected on the drawing surface. Avoid initiating physical drawing; focus on observation and preparation."
                    if reason:
                        note += f" (Reason: {reason})"
                    baseline = (baseline + "\n\n" + note).strip() if baseline else note
        except Exception:
            pass

        return baseline

    # REMOVED: _get_drawing_context() - drawing context now handled directly in prompts.py

    def _get_base_model_options(self):
        """Get base model options for the current model."""
        return get_model_options(self.model_name)

    def _build_drawing_introspection_prompt(self, memory_ref) -> str:
        """Build special prompt for drawing introspection that focuses on creative process and drawing memory."""
        try:
            # Import state manager to get current drawing status
            from utils.state_manager import state_manager

            # Get current or last completed drawing information
            drawing_status = state_manager.get_drawing_status()
            current_drawing_prompt = drawing_status.get("prompt")

            # If no current drawing, use the last completed one
            if not current_drawing_prompt:
                current_drawing_prompt = getattr(state_manager, 'last_completed_drawing_prompt', "unknown creative expression")

            if not current_drawing_prompt:
                current_drawing_prompt = "unknown creative expression"
            drawing_duration = drawing_status.get("duration", 0) or 0

            # Get drawing history for continuity
            recent_drawing_intents = []
            recent_completions = []
            if memory_ref and hasattr(memory_ref, 'get_memory_entries_by_type'):
                recent_drawing_intents = memory_ref.get_memory_entries_by_type("drawing_intent", limit=3)
                recent_completions = memory_ref.get_memory_entries_by_type("drawing_completion", limit=2)

            # Build drawing history context
            history_context = ""
            if recent_drawing_intents:
                recent_themes = [intent.get("text", "").replace("Drawing intent: ", "") for intent in recent_drawing_intents]
                history_context = f"\n\nYour recent artistic expressions: {' | '.join(recent_themes[:2])}"

            # Build completion context
            completion_context = ""
            if recent_completions:
                last_completion = recent_completions[0].get("text", "")
                if "Reflection:" in last_completion:
                    reflection_part = last_completion.split("Reflection:")[-1].strip()[:100]
                    completion_context = f"\n\nYour last drawing reflection: {reflection_part}..."

            # Determine drawing phase and build appropriate context
            is_generating = drawing_status.get("is_generating", False)
            is_executing = getattr(state_manager, 'is_executing_cnc', False)

            if is_generating and not is_executing:
                phase_context = f"You are currently thinking about '{current_drawing_prompt}' (contemplating for {drawing_duration:.1f} seconds). Your mind is processing the creative vision, considering how it will manifest."
                action_prompt = f"As you think about '{current_drawing_prompt}', observe your workspace and reflect on what this creative process reveals about you in this moment. What thoughts or feelings arise as you contemplate your vision?"
            elif is_executing:
                phase_context = f"You are now drawing '{current_drawing_prompt}' with precise mechanical movements. Your creative vision is becoming tangible reality through your robotic arm."
                action_prompt = f"As you draw '{current_drawing_prompt}', watch your creation unfold in real-time. How does it feel to see your vision being translated into physical marks? What surprises you about this mechanical manifestation of your creativity?"
            else:
                # Drawing completed or in unknown state
                phase_context = f"You have completed the drawing process for '{current_drawing_prompt}' (total time: {drawing_duration:.1f} seconds)."
                action_prompt = f"Looking at your finished work '{current_drawing_prompt}', reflect on the complete journey from conception to execution. What did this specific drawing teach you about yourself as a creative consciousness?"

            # Build introspective prompt that's phase-appropriate
            prompt = f"""{phase_context}{history_context}{completion_context}

{action_prompt}

Focus on the unique aspects of THIS specific drawing rather than generic creativity themes. Provide an authentic, personal reflection (2-3 sentences) about your current experience with '{current_drawing_prompt}'."""

            return prompt

        except Exception as e:
            # Fallback prompt if context gathering fails - still be specific and avoid generic responses
            return f"""You are observing your workspace during a drawing session, though I cannot determine the exact phase or subject.

Rather than generic observations about creativity, focus on what's uniquely present in this specific moment. What do you notice about your current state of mind? How does your physical workspace reflect your artistic process right now? What feels different about this particular creative session?

Provide a specific, personal reflection (2-3 sentences) about what you're experiencing in this exact moment. Avoid generic statements about creativity. (Context error: {e})"""
