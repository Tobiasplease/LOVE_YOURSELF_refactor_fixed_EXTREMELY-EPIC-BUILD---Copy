"""
Clean model wrapper - pure API handler only.
All prompt logic moved to prompt_interface.py for centralization.
"""

import re
from typing import Optional

from config.config import (
    MOOD_SNAPSHOT_FOLDER,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_REFLECTION,
)
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.ollama import query_ollama, truncate_for_print

from .prompt_interface import PromptInterface

def _is_plantable_prior(text: str) -> bool:
    """Return True if the caption is safe to plant in voice thread.

    PHILOSOPHY: An imperfect thread entry is vastly better than an empty thread.
    Only reject things that are UNAMBIGUOUSLY not first-person inner monologue:
    - Literal image analysis ("The image shows...", "can be seen")
    - AI identity breaks ("As an AI", "artificial intelligence")
    - Second-person / roleplay bleed ("You:", "your")
    - Garbage tokens
    Everything else passes. The sentence extractor handles truncation.
    """
    t = text.strip()
    if not t:
        return False
    # Strip leading thread-echo dashes before all checks
    if t.startswith("—"):
        t = t[1:].strip()
    if t.startswith("- "):
        t = t[2:].strip()
    if not t:
        return False
    # Reject known garbage tokens
    if t.startswith("addCriterion") or t.startswith("[WARNING]") or t.startswith("Vision initializing"):
        return False

    t_lower = t.lower()

    # Reject literal image-analysis register (the model talking ABOUT an image/photograph)
    image_words = ["the image", "this image", "in this image", "a photograph", "a photo of",
                   "the photograph", "the photo", "an image of", "this photograph"]
    if any(t_lower.startswith(w) for w in image_words):
        return False
    # Reject VQA language anywhere in the text
    if "can be seen" in t_lower or "is depicted" in t_lower or "in the image" in t_lower:
        return False
    if "appears to be" in t_lower and ("photograph" in t_lower or "photo" in t_lower or "image" in t_lower):
        return False

    # AI identity breaks
    if "artificial intelligence" in t_lower or "as an ai" in t_lower or "i am an ai" in t_lower:
        return False

    # Second-person / roleplay bleed
    if t_lower.startswith("you:") or t_lower.startswith("you "):
        return False
    if "*you " in t_lower or "*your " in t_lower:
        return False

    # Chatbot mode
    if "i'm sorry" in t_lower or "i apologize" in t_lower:
        return False

    # Drawing title/idea/catalogue mode — reject any output that's about
    # generating drawings rather than inner monologue
    drawing_prefixes = ["drawing title:", "drawing idea:", "new drawing:", "drawing complete",
                        "drawing description:", "drawing note:", "new observation:",
                        "drawing in progress", "new subject detected", "drawing:",
                        "draw:", "draws:", "i draw", "let's draw"]
    if any(t_lower.startswith(p) for p in drawing_prefixes):
        return False
    # Also catch "Draw:" or "Drawing:" mid-text
    if "draw:" in t_lower or "drawing:" in t_lower or "draws:" in t_lower:
        return False

    # Near-empty
    if len(t.split()) < 3:
        return False

    return True


def _extract_first_sentence(text: str, min_chars: int = 10, max_chars: int = 120) -> str:
    """Extract first complete sentence from text.

    Handles LLaVA's preferred punctuation: ... ; — as well as standard . ! ?
    """
    text = text.strip()
    # Strip leading dash if model echoed the thread format
    if text.startswith("—"):
        text = text[1:].strip()
    if text.startswith("- "):
        text = text[2:].strip()

    if len(text) < min_chars:
        return ""

    for end_idx in range(min_chars, min(len(text), max_chars)):
        char = text[end_idx]

        if char in ".!?":
            # "..." — treat as sentence end (strip trailing dots)
            if char == "." and end_idx + 2 < len(text) and text[end_idx + 1 : end_idx + 3] == "..":
                return text[:end_idx].strip()
            return text[: end_idx + 1].strip()

        if char == ";":
            return text[:end_idx].strip() + "..."
        if char == "—" and end_idx > min_chars:
            return text[:end_idx].strip() + "..."
        if char == "-" and end_idx > 0 and text[end_idx - 1] == "-":
            return text[: end_idx - 1].strip() + "..."

    # No boundary found — truncate at word boundary with ellipsis to imply continuity
    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    if len(truncated) >= min_chars:
        return truncated + "..."
    return ""


def _get_valid_captions(agent, max_captions: int = 3, include_perception: bool = False):
    """Extract valid, filtered caption sentences from agent's recent captions.

    If include_perception=True, returns list of (thought, perception) tuples.
    Otherwise returns list of thought strings (backwards compatible).
    """
    if not hasattr(agent, "recent_captions") or not agent.recent_captions:
        return []

    available = list(agent.recent_captions)[-max_captions:]
    valid = []

    for caption_entry in available:
        if isinstance(caption_entry, dict):
            caption_text = caption_entry.get("text", "")
            perception_text = caption_entry.get("perception", "")
        elif isinstance(caption_entry, (list, tuple)):
            caption_text = caption_entry[0] if len(caption_entry) > 0 else ""
            # New format: (caption, timestamp, mode, perception)
            perception_text = caption_entry[3] if len(caption_entry) > 3 else ""
        else:
            caption_text = str(caption_entry)
            perception_text = ""

        if not caption_text:
            continue
        if not _is_plantable_prior(caption_text):
            continue

        sentence = _extract_first_sentence(caption_text)
        if not sentence:
            continue

        if include_perception:
            valid.append((sentence, perception_text.strip() if perception_text else ""))
        else:
            valid.append(sentence)

    return valid


def build_caption_thread(agent, max_captions: int = 3) -> str:
    """Build dashed caption thread (legacy format, used by single-pass path)."""
    valid_captions = _get_valid_captions(agent, max_captions)
    if not valid_captions:
        return ""

    thread_lines = ["My thoughts:"]
    for caption in valid_captions:
        thread_lines.append(f"— {caption}")
    thread_lines.append("—")
    return "\n".join(thread_lines)


def _compress_perception(perception: str, max_len: int = 60) -> str:
    """Compress a perception string to a short phrase for thread display."""
    if not perception:
        return ""
    p = perception.strip()
    # Remove common preambles
    for preamble in ["In front of me is ", "In front of you is ", "Right now ", "The scene in front of you is "]:
        if p.lower().startswith(preamble.lower()):
            p = p[len(preamble):]
            break
    # Truncate at word boundary
    if len(p) > max_len:
        p = p[:max_len].rsplit(" ", 1)[0]
    # Lowercase first char
    if p and p[0].isupper():
        p = p[0].lower() + p[1:]
    return p.rstrip(".,;")


def build_flowing_thread(agent, max_captions: int = 3) -> str:
    """Build flowing thought stream from recent captions.

    Returns: "...thought 1. Thought 2. Thought 3." or empty string.
    Simple flowing format — perception context is handled by the
    'Already noticed:' line in the perceive() method instead.
    """
    valid_captions = _get_valid_captions(agent, max_captions)
    if not valid_captions:
        return ""

    cleaned = []
    for cap in valid_captions:
        c = cap.rstrip(".")
        if c.endswith("..."):
            c = c[:-3].rstrip()
        cleaned.append(c)

    return "..." + ". ".join(cleaned) + "."


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

        # Introspective mode uses text-only Natsumura for LLaVA (no image needed).
        # Qwen is a vision model — route ALL modes through the visual API path.
        is_qwen = "qwen" in self.model_name.lower()
        if prompt_mode == "introspective" and not is_qwen:
            result = self._call_natsumura_introspective(prompt, system_prompt, model_options)
        else:
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

        if model_options is None:
            model_options = self.prompt_interface._get_base_model_options()

        # For Qwen captions: plant prior caption as owned assistant turn so Qwen
        # continues its voice rather than describing the prior caption as external context.
        # Only plant if the prior caption is genuine first-person inner monologue —
        # VQA register (starts with "The scene/room/image") or garbage poisons the voice.
        prior_turn = None
        if prompt_type == "caption" and "qwen" in self.model_name.lower() and self.memory_ref:
            last = getattr(self.memory_ref, "last_caption", None)
            if last and _is_plantable_prior(last):
                prior_turn = last

        # Debug: print full prompt structure for review
        if prompt_type in ("caption", "reflection", "drawing"):
            print(f"\n{'='*80}\n[PROMPT] {prompt_type.upper()}\n{'='*80}")
            if system_prompt:
                print(f"SYSTEM:\n{system_prompt}\n")
            print(f"USER:\n{prompt}\n")
            if prior_turn:
                print(f"[PLANTED PRIOR] (injected as assistant turn for voice continuity):\n{prior_turn}\n")
            print(f"{'='*80}\n")

        response = query_ollama(
            prompt=prompt,
            model=self.model_name,
            image=image_path,
            timeout=timeout,
            log_dir=MOOD_SNAPSHOT_FOLDER,
            system_prompt=system_prompt,
            options=model_options,
            prompt_type=prompt_type,
            prior_assistant_turn=prior_turn,
        )

        if not response:
            response = ""

        # Debug: show raw response for perception calls to diagnose empty results
        if prompt_type == "perception":
            print(f"[PERCEPTION RAW] len={len(response)}, response={repr(response[:200])}")

        cleaned = self._clean_response(response)

        # Print the caption immediately after the prompt for continuity review
        if prompt_type == "caption" and cleaned:
            print(f"[RESPONSE] CAPTION:\n{cleaned}\n")

        return cleaned

    def perceive(self, image_path: str, *, perception_prompt: str, mode: str = "observational") -> str:
        """Pass 1: Vision model sees the image and answers a directed perception question.

        Returns a short grounded sentence about what it notices.
        This output is NOT stored as a caption — it feeds into the monologue pass.
        """
        import random as _random
        from captioner.prompts import get_perception_system_prompt

        model_options = {
            "temperature": 0.7,
            "top_p": 0.8,
            "repeat_penalty": 1.3,
            "num_predict": 120,
            "num_ctx": 4096,
            "seed": _random.randint(1, 1000000),
            "stop": ["\n\nUser:", "\n\nHuman:", "\n\nAssistant:"],
        }

        perception_system_prompt = get_perception_system_prompt(mode)

        # Add prior perception context so the vision model can go deeper
        # instead of repeating the same baseline description each cycle.
        full_prompt = perception_prompt
        if self.memory_ref:
            last_perc = getattr(self.memory_ref, "_last_perception", "")
            if last_perc and len(last_perc) > 10:
                compressed = _compress_perception(last_perc, max_len=80)
                if compressed:
                    full_prompt = f"Already noticed: {compressed}.\n{perception_prompt}"

        print(f"\n{'='*80}\n[PERCEPTION] {self.model_name} ({mode})\n{'='*80}")
        print(f"PROMPT: {full_prompt}")
        print(f"{'='*80}\n")

        result = self._call_ollama(
            full_prompt,
            image_path=image_path,
            system_prompt=perception_system_prompt,
            model_options=model_options,
            prompt_type="perception",
        )

        # Clean up perception: strip image-analysis preamble, then truncate
        cleaned = (result or "").strip()

        # Detect refusals first — if LLaVA refuses, return empty immediately
        refusal_phrases = [
            "does not contain", "do not see", "cannot provide", "cannot confidently",
            "not enough information", "no human figures", "no people visible",
            "privacy concerns", "lacks any human", "no individual",
            "don't have enough", "not clearly visible", "cannot determine",
            "no person", "no one is", "unable to identify",
        ]
        if any(phrase in cleaned.lower() for phrase in refusal_phrases):
            # Only refuse if the refusal phrase is the MAIN content, not just mentioned
            # in passing. If there's substantial descriptive content before the refusal,
            # keep the good part.
            first_refusal_pos = min(
                (cleaned.lower().find(p) for p in refusal_phrases if p in cleaned.lower()),
                default=0
            )
            if first_refusal_pos > 40:
                # Good content before the refusal — keep just the good part
                cleaned = cleaned[:first_refusal_pos].rstrip(" ,;.")
                print(f"[PERCEPTION] Trimmed refusal tail, keeping: {cleaned[:60]}")
            else:
                print(f"[PERCEPTION] {self.model_name} refused/uncertain, skipping: {cleaned[:60]}")
                return ""

        # Strip any leading clause that references "image"/"photo"/"picture"/etc.
        # This catches all variants: "In this/the image...", "The image provided...",
        # "The photo shows...", "The photograph captures...", etc.
        # NOTE: photograph MUST come before photo in alternation to avoid partial match leaving "graph"
        cleaned = re.sub(
            r'^(?:In\s+(?:this|the)\s+)?(?:The\s+)?(?:image|photograph|photo|picture)(?:\s+provided)?\s*'
            r'(?:,\s*I\s+can\s+\w+\s+)?'
            r'(?:appears?\s+to\s+(?:show|be|depict|have\s+been\s+taken)\s*'
            r'|shows?\s+|depicts?\s+|contains?\s+|presents?\s+|captures?\s+)?'
            r'(?:the\s+following\s+(?:elements?|objects?|items?)\s*(?:present|visible)?\s*(?:in)?\s*)?'
            r'[,.:;]?\s*',
            '', cleaned, count=1, flags=re.IGNORECASE
        )
        # Strip "in the/this image" and "in the provided photo" wherever they appear
        # e.g. "The ceiling in the image is peeling" → "The ceiling is peeling"
        # e.g. "In the provided photo of what appears..." → "of what appears..."
        cleaned = re.sub(r'\s*[Ii]n\s+(?:the|this)\s+(?:provided\s+)?(?:image|photograph|photo|picture)\s*(?:of\s+)?', ' ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'^of\s+', '', cleaned.strip())  # Clean up leftover "of" at start
        # Also strip "I can see/observe/describe" preamble
        cleaned = re.sub(
            r'^I\s+(?:can\s+)?(?:see|observe|describe|notice)\s+(?:that\s+)?',
            '', cleaned, count=1, flags=re.IGNORECASE
        )
        # Strip "Compared to a previous version" or similar cross-reference
        cleaned = re.sub(
            r'^(?:Compared|According)\s+to\s+.*?[,]\s*',
            '', cleaned, count=1, flags=re.IGNORECASE
        )
        # Strip numbered list markers and list preamble
        cleaned = re.sub(r'^\d+\.\s*(?:\*\*\w+\*\*:?\s*)?', '', cleaned)
        cleaned = re.sub(r'^Here\s+are\s+(?:the\s+)?(?:differences?|details?|observations?|changes?)\s*[:;]?\s*', '', cleaned, flags=re.IGNORECASE)
        # Strip "Taken from" remnant after photo preamble removal
        cleaned = re.sub(r'^[Tt]aken\s+from\s+', 'From ', cleaned)

        # Nuclear option: if "image"/"photo"/"photograph" still appears anywhere
        # after all stripping, try to excise the contaminated sentence
        cleaned = cleaned.strip()
        if cleaned:
            cl = cleaned.lower()
            if any(w in cl for w in ["image", "photo", "photograph", "picture"]):
                # Try dropping the first sentence
                first_end = -1
                for i in range(10, min(len(cleaned), 150)):
                    if cleaned[i] in ".!?":
                        first_end = i + 1
                        break
                if first_end > 0:
                    rest = cleaned[first_end:].strip()
                    if len(rest) > 15:
                        cleaned = rest

        # Capitalize after stripping
        cleaned = cleaned.strip()
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        # Truncate at sentence boundary within 200 chars, or word boundary
        if len(cleaned) > 200:
            # Try to find sentence end
            for i in range(min(len(cleaned), 200), 40, -1):
                if cleaned[i - 1] in ".!?":
                    cleaned = cleaned[:i]
                    break
            else:
                cleaned = cleaned[:200].rsplit(" ", 1)[0].strip()

        print(f"[PERCEPTION] Result: {cleaned}\n")
        return cleaned

    def generate_monologue(self, perception: str, *, monologue_prompt: str, mode: str, agent=None) -> tuple:
        """Pass 2: Text model generates inner monologue from perception + thread.

        Returns:
            tuple: (monologue_text, mode)
        """
        import random as _random
        from config.config import MONOLOGUE_MODEL
        from captioner.prompts import get_monologue_system_prompt

        emotion_state = getattr(agent, "current_emotion_state", "calm") if agent else "calm"
        system_prompt = get_monologue_system_prompt(mode, emotion_state)

        model_options = {
            "temperature": 0.4,
            "top_p": 0.8,
            "repeat_penalty": 1.25,
            "num_predict": 80,
            "num_ctx": 4096,
            "seed": _random.randint(1, 1000000),
            "stop": [
                "\n\n",
                "[Image", "[image",
                "[What I see", "[what i see",
                "\n\nUser:", "\n\nHuman:", "\n\nAssistant:",
                "[INST]", "[/INST]",
            ],
        }

        print(f"\n{'='*80}\n[MONOLOGUE] {MONOLOGUE_MODEL}\n{'='*80}")
        print(f"SYSTEM: {system_prompt}\n")
        print(f"USER:\n{monologue_prompt}\n")
        print(f"{'='*80}\n")

        result = query_ollama(
            prompt=monologue_prompt,
            model=MONOLOGUE_MODEL,
            image=None,
            system_prompt=system_prompt,
            timeout=60,
            log_dir=MOOD_SNAPSHOT_FOLDER,
            options=model_options,
            prompt_type="monologue",
        )

        cleaned = self._clean_response(result) if result else ""

        # Post-process monologue output
        if cleaned:
            # Strip markdown formatting that corrupts the thread
            cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)  # **bold** → bold
            cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)      # *italic* → italic
            # Strip leading/trailing quotes
            cleaned = cleaned.strip('"').strip()
            # Strip structured prefixes (model treating output as a form)
            cleaned = re.sub(r'^(?:Feeling|Emotion|Mood|Status|State)\s*:\s*\w+\.?\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^(?:Thoughts?|Observation|Note)\s*:\s*["\']?', '', cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip('"\'').strip()
            # Strip bracket format leak — nemo sometimes echoes [What I see right now: ...]
            cleaned = re.sub(r'\[What I see[^]]*?\]', '', cleaned, flags=re.IGNORECASE).strip()
            # Fix leading "You" → "I"
            if cleaned.startswith("You "):
                cleaned = "I " + cleaned[4:]
            elif cleaned.startswith("Your "):
                cleaned = "My " + cleaned[5:]
            # Fix mid-sentence "you"
            cleaned = re.sub(r'\byou\b(?! ["\'])', 'I', cleaned, count=2)
            cleaned = re.sub(r'\byour\b(?! ["\'])', 'my', cleaned, count=2)

            # Replace instruct-mode non-responses with silence
            # These produce no inner monologue and corrupt the thread
            filler = cleaned.lower().rstrip(".!? ")
            if filler in ("understood", "noted", "acknowledged", "ok", "okay",
                          "not drawing yet", "not drawing right now",
                          "currently observing", "observing", "pauses"):
                cleaned = "..."

        print(f"[MONOLOGUE] Result: {cleaned}\n")

        return (cleaned, mode)

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
