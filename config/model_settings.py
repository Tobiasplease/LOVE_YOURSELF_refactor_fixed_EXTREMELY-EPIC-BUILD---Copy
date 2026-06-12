"""
Model-specific settings.
"""

# Model-specific generation options
MODEL_GENERATION_OPTIONS = {
    "qwen2.5vl:7b": {
        # Critical: Qwen default temp=0.0001 (near-deterministic VQA). Must always override.
        # repeat_penalty 1.55: 2.5 destroys vocabulary and forces VQA phrasing.
        # No "\n" standalone stop — gaze tokens in user prompt could trigger premature stop.
        "temperature": 0.85,
        "top_p": 0.7,
        "top_k": 20,
        "repeat_penalty": 1.55,
        "num_predict": 40,
        "num_ctx": 4096,
        "penalize_newline": True,
        "stop": [
            "\n\n",
            ". I ", ". My ", ". The ",
            "<|im_start|>", "<|im_end|>",
            "\n\nUser:", "\n\nHuman:", "\n\nAssistant:",
            "The image", "This image", "In this image",
            "This is an image", "I'm looking at an image",
            "the image depicts", "This is a photograph",
            "The photograph shows", "I can see an image",
            "I can see", "I observe", "I see that",
            "As an AI", "as an AI", "I am an AI", "I'm an AI",
            "language model", "text-based AI", "without visual capabilities",
            "I do not have", "I cannot perceive", "I am unable to",
            "As a drawing machine", "As a quiet drawing", "drawing machine that",
            "as per my programming", "my programming", "programmed to",
            "I was programmed", "I'm programmed",
        ],
    },
    "llava-llama3:latest": {
        # Llama 3 base: stronger instruction following than Mistral but RLHF causes
        # assistant-mode relapses ("I am unsure of the purpose..."). Block those explicitly.
        "temperature": 0.9,
        "top_p": 0.9,
        "repeat_penalty": 1.15,
        "num_predict": 120,
        "num_ctx": 4096,
        "stop": [
            "\n\nUser:", "\n\nHuman:", "\n\nAssistant:",
            # Block Llama 3 RLHF assistant relapses
            "I am unsure", "I'm unsure",
            "I cannot determine", "I'm not able to",
            "as there are no clear instructions",
            "no clear instructions",
            "I don't have enough", "I don't have the",
            "without more context", "without additional context",
            "please provide", "Please provide",
            "could you clarify", "Could you clarify",
            # Block image-analysis language
            "This is an image of", "I'm looking at an image",
            "the image depicts", "This is a photograph of",
            "The photograph shows", "This image shows",
            "The image shows", "In this image", "The image contains",
            "I can see an image", "The image appears",
            "appears to be a", "which suggests that", "indicating that",
            # Block AI identity breaks
            "As an AI", "as an AI", "I am an AI", "I'm an AI",
            "language model", "text-based AI", "without visual capabilities",
            "I do not have", "I cannot perceive", "I am unable to",
            # Block self-description
            "As a drawing machine", "As a quiet drawing",
            "drawing machine that", "as per my programming",
            "my programming", "programmed to",
            "I was programmed", "I'm programmed",
        ],
    },
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
            "The image captures",
            "The image presents",
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


