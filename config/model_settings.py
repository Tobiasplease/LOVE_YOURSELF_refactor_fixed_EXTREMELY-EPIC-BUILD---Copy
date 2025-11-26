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
            # Block image analysis language that breaks immersion
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
            # Encourage natural sentence completion
            "\n\n",
        ],
    }
}

# Note: System prompts are centrally defined in captioner/prompts.py.


def get_model_options(model_name: str) -> dict:
    """Get generation options for a specific model."""
    return MODEL_GENERATION_OPTIONS.get(model_name, MODEL_GENERATION_OPTIONS["llava:7b-v1.6-mistral-q5_1"])


def get_model_system_prompt(model_name: str) -> dict:
    """Deprecated. System prompts live in captioner/prompts.py."""
    return {"style": "detailed", "base_prompt": "See captioner/prompts.SYSTEM_PROMPT"}
