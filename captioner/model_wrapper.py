"""
Clean model wrapper - pure API handler only.
All prompt logic moved to prompt_interface.py for centralization.
"""

import re
from typing import Optional

from config.config import (
    MOOD_SNAPSHOT_FOLDER,
    MODEL_NAME,
)
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType
from utils.inference import query_model
from utils.llm_log import truncate_for_print

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

    # Reject mid-sentence fragments (opens with a lowercase letter). A
    # truncation-cascade tail ("by those who came before me long ago when...")
    # got saved as the prior thought and seeded the July 26 awakening with
    # free-association it then dutifully continued. First-char only: a
    # digit-led thought ("135 hours is too long...") is a real sentence.
    if t[0].isalpha() and t[0].islower():
        return False

    t_lower = t.lower()

    # Reject literal image-analysis register (the model talking ABOUT an image/photograph)
    image_words = [
        "the image",
        "this image",
        "in this image",
        "a photograph",
        "a photo of",
        "the photograph",
        "the photo",
        "an image of",
        "this photograph",
    ]
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
    # Outward register by density (July 28): the previous session ended in
    # "What do you think?" escalation, got saved as the prior thought, and
    # seeded the next session's awakening straight into assistant mode. Two
    # second-person tokens = the text has a reader; it must not seed a
    # monologue. One stays plantable (self-address, talking to objects).
    if len(re.findall(r"\b(?:you|your|yours|yourself)\b", t_lower)) >= 2:
        return False

    # Chatbot mode
    if "i'm sorry" in t_lower or "i apologize" in t_lower:
        return False

    # Drawing title/idea/catalogue mode — reject any output that's about
    # generating drawings rather than inner monologue
    drawing_prefixes = [
        "drawing title:",
        "drawing idea:",
        "new drawing:",
        "drawing complete",
        "drawing description:",
        "drawing note:",
        "new observation:",
        "drawing in progress",
        "new subject detected",
        "drawing:",
        "draw:",
        "draws:",
        "i draw",
        "let's draw",
    ]
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

        # Filter captions that compound badly in the thread.
        s_lower = sentence.lower()

        # Drawing-intent musings create false memories ("I drew X" from "I want to draw X")
        if any(
            kw in s_lower
            for kw in [
                "should draw",
                "want to draw",
                "could draw",
                "would draw",
                "next drawing",
                "my next piece",
                "i'll draw",
                "i will draw",
                "draw it",
                "draw them",
                "draw this",
                "draw that",
                "sketch it",
                "sketch the",
                "capture it in",
                "put it on paper",
                "[drawing idea",
                "drawing idea:",
            ]
        ):
            continue

        # Time/status statements compound into fictional timelines when they
        # re-enter the thread. "about 14 minutes" + "about 20 minutes" + "about 45 minutes"
        # all in the same prompt creates a fake time progression.
        if re.search(r"^about \d+ (?:minutes?|hours?) (?:awake|now|in|active)", s_lower):
            continue
        if re.search(r"^\d+ (?:minutes?|hours?) (?:awake|since|passed)", s_lower):
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




class MultimodalModel:
    """Simplified model wrapper - pure API handler."""

    def __init__(self, memory_ref: Optional[any] = None) -> None:  # type: ignore
        self.memory_ref = memory_ref
        self.model_name = MODEL_NAME
        self.prompt_interface = PromptInterface(self.model_name)


    def generate_drawing_prompt(self, *, extra: Optional[str] = None, image_path: Optional[str] = None, drawing_intentions: list = None) -> str:
        """Generate drawing prompt using centralized prompt interface with VISUAL GROUNDING."""
        prompt, model_options, system_prompt = self.prompt_interface.build_drawing_prompt_with_options(
            self.memory_ref, extra=extra, image_path=image_path, drawing_intentions=drawing_intentions
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

        # If using stream/natsumura/multi-step analysis, the prompt IS the final result, don't call LLM again
        try:
            from config.config import DRAWING_ANALYSIS_MODE

            if DRAWING_ANALYSIS_MODE in ("stream", "natsumura", "multi_step"):
                return prompt  # These modes already return the final drawing prompt
        except ImportError:
            pass

        # Unreachable: every DRAWING_ANALYSIS_MODE returns above. Kept as an
        # explicit failure rather than a silent fall-through to a retired path.
        raise RuntimeError(f"unknown DRAWING_ANALYSIS_MODE — no builder produced a prompt")





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
