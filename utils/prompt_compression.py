"""
Drawing Prompt Compression
Intelligent compression of drawing prompts into concise descriptions using TinyLlama
"""

import re
from typing import Optional
from utils.ollama import query_ollama
from config.config import MOTIF_MODEL, TINYLLAMA_TEMPERATURE, TINYLLAMA_TOP_P, TINYLLAMA_NUM_PREDICT, TINYLLAMA_TIMEOUT


def compress_drawing_prompt(drawing_prompt: str) -> str:
    """
    Compress a drawing prompt into a concise, natural description using TinyLlama.
    
    Args:
        drawing_prompt: Full drawing prompt to compress
        
    Returns:
        Compressed description (e.g., "a line sketch of an empty room")
    """
    
    if not drawing_prompt or not drawing_prompt.strip():
        return "a drawing based on recent observations"
    
    # Try intelligent compression first
    try:
        compressed = _compress_with_tinyllama(drawing_prompt)
        if compressed and len(compressed.strip()) > 5:
            return compressed.strip()
    except Exception as e:
        print(f"[⚠️] TinyLlama compression failed: {e}")
    
    # Fallback to pattern-based extraction
    try:
        compressed = _extract_key_concepts(drawing_prompt)
        if compressed:
            return compressed
    except Exception as e:
        print(f"[⚠️] Pattern extraction failed: {e}")
    
    # Final fallback to simple truncation (better than nothing)
    first_line = drawing_prompt.strip().split("\n")[0]
    if len(first_line) > 60:
        first_line = first_line[:57] + "..."
    
    return first_line


def _compress_with_tinyllama(drawing_prompt: str) -> Optional[str]:
    """Use TinyLlama to compress the drawing prompt intelligently."""
    
    compression_prompt = f"""What is being drawn? Answer in 3-6 words only.

PROMPT: "{drawing_prompt}"

WHAT'S BEING DRAWN:"""
    
    model_options = {
        "temperature": TINYLLAMA_TEMPERATURE,
        "top_p": TINYLLAMA_TOP_P,
        "num_predict": max(50, TINYLLAMA_NUM_PREDICT * 10),  # Allow enough tokens for meaningful compression (50+ tokens)
    }
    
    try:
        # Import consolidated system prompt for compression
        from captioner.prompts import NUMBER_GENERATOR_SYSTEM_PROMPT
        compression_system_prompt = "You compress drawing prompts into short, natural descriptions. Be concise and specific."
        
        # Use main model instead of TinyLlama for better text comprehension
        from config.config import OLLAMA_MODEL
        response = query_ollama(
            prompt=compression_prompt,
            model=OLLAMA_MODEL,  # Use main model instead of MOTIF_MODEL (TinyLlama)
            timeout=30,  # Longer timeout for main model
            system_prompt=compression_system_prompt,
            options={
                "temperature": 0.3,  # Low temperature for consistent compression
                "top_p": 0.8,
                "num_predict": 30,  # Enough for 8-15 words
            },
            prompt_type="compression"
        )
        
        if response:
            # Clean up the response
            cleaned = response.strip().lower()
            
            # Remove common prefixes/artifacts
            cleaned = re.sub(r'^(compressed description:|description:|result:|answer:)\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^["\']|["\']$', '', cleaned)  # Remove quotes
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
            
            # Ensure it starts with "a" or "an" for natural flow
            if cleaned and not cleaned.startswith(('a ', 'an ', 'the ')):
                if cleaned[0] in 'aeiou':
                    cleaned = f"an {cleaned}"
                else:
                    cleaned = f"a {cleaned}"
            
            return cleaned if len(cleaned) <= 80 else None
            
    except Exception as e:
        print(f"[⚠️] TinyLlama API call failed: {e}")
        return None


def _extract_key_concepts(drawing_prompt: str) -> Optional[str]:
    """Extract key concepts using pattern matching as fallback."""
    
    text = drawing_prompt.lower()
    
    # Common drawing subjects and styles
    subjects = []
    styles = []
    
    # Extract drawing styles
    if "line sketch" in text or "line drawing" in text:
        styles.append("line sketch")
    elif "sketch" in text:
        styles.append("sketch")
    elif "drawing" in text:
        styles.append("drawing")
    
    # Extract subjects (simple pattern matching)
    subject_patterns = [
        r'\b(room|bedroom|kitchen|office|space)\b',
        r'\b(person|figure|human|face|portrait)\b',
        r'\b(building|house|structure|architecture)\b',
        r'\b(landscape|scene|view|environment)\b',
        r'\b(object|item|tool|device)\b',
        r'\b(plant|tree|flower|nature)\b',
        r'\b(geometric|abstract|pattern)\b'
    ]
    
    for pattern in subject_patterns:
        matches = re.findall(pattern, text)
        if matches:
            subjects.extend(matches)
            break  # Take first category found
    
    # Build description
    description_parts = []
    if styles:
        description_parts.append(styles[0])
    
    if subjects:
        if len(subjects) == 1:
            description_parts.append(f"of {subjects[0]}")
        else:
            description_parts.append(f"of {subjects[0]}")
    
    if description_parts:
        result = " ".join(description_parts)
        # Ensure it starts with "a" or "an"
        if not result.startswith(('a ', 'an ', 'the ')):
            if result[0] in 'aeiou':
                result = f"an {result}"
            else:
                result = f"a {result}"
        return result
    
    return None