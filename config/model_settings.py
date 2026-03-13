"""
Model-specific settings for the active model (LLaVA Mistral).
All legacy Qwen and alternative model configurations removed to simplify the stack.
"""

# Model-specific generation options
MODEL_GENERATION_OPTIONS = {
    "llava:7b-v1.6-mistral-q5_1": {
        "temperature": 0.8,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_predict": 200,
        "num_ctx": 4096,
        "stop": [
            "\n\nUser:",
            "\n\nHuman:",
            "\n\nAssistant:",
            # Block AI/image-analysis language that breaks immersion
            "This is an image of",
            "I'm looking at an image of",
            "the image depicts",
            "This is a photograph of",
            "The photograph shows",
            "This image shows",
            "The image shows",
            "In this image",
            "The image contains",
            "I can see an image",
            "The image appears",
            "appears to be a",
            "which suggests that",
            "indicating that",
            # Block AI identity breaks
            "As an AI",
            "as an AI",
            "I am an AI",
            "I'm an AI",
            "language model",
            "text-based AI",
            "without visual capabilities",
            "I do not have",
            "I cannot perceive",
            "I am unable to",
            # Block self-description (narrating identity instead of embodying it)
            "As a drawing machine",
            "As a quiet drawing",
            "As the quiet drawing",
            "drawing machine that",
            # Block hedging/programming language
            "as per my programming",
            "not programmed",
            "my programming",
            "programmed to",
            "I was programmed",
            "I'm programmed",
            # Block prompt leakage
            "EXperience",
            "EXERCISE",
            "EXPLANATION",
            # NOTE: Removed ". The ", ". This ", ". As I " - these caused truncation feedback loops
            # NOTE: Removed "\n\n" - was causing mid-thought truncation ("The room" etc)
            # Let num_predict handle length instead of aggressive stop sequences
        ],
    }
}

# Note: System prompts are centrally defined in captioner/prompts.py.


def get_model_options(model_name: str) -> dict:
    """Get generation options for a specific model. Returns a COPY to prevent mutation."""
    import copy
    base = MODEL_GENERATION_OPTIONS.get(model_name, MODEL_GENERATION_OPTIONS["llava:7b-v1.6-mistral-q5_1"])
    return copy.deepcopy(base)


def get_model_system_prompt(model_name: str) -> dict:
    """Deprecated. System prompts live in captioner/prompts.py."""
    return {"style": "detailed", "base_prompt": "See captioner/prompts.SYSTEM_PROMPT"}
