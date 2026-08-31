from enum import Enum


class LogType(Enum):
    """Enumeration of all valid log types for event logging."""

    # Core system events
    SESSION_START = "session_start"
    INFO = "info"
    ERROR = "error"
    DEBUG = "debug"
    RUN_METADATA = "run_metadata"

    # AI/ML processing events
    CAPTION = "caption"
    REFLECTION = "reflection"
    VOCAB_PROMOTION = "vocab_promotion"
    # SENTIMENT = "sentiment_analysis"
    COMPRESSION = "compression"
    LLM_API_CALL = "llm_api_call"  # was OLLAMA_API_CALL/"ollama_api_call" pre-July 9 (old logs keep the old string)

    # Drawing and creative process events
    DECISION = "decision"
    COMFY_PROMPT = "comfy_prompt"
    NEW_DRAWING = "new_drawing"
    GRBL = "grbl"
