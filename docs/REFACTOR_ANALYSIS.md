# Architectural Refactoring Analysis

## Phase 1: Actual Execution Flow

### Functions ACTUALLY Called (From Debug Output):
1. `captioner.py` → `self.model.caption_image()`
2. `model_wrapper.py` → `build_caption_prompt()` 
3. `prompts.py` → `build_caption_prompt()` (dispatcher)
4. `prompts.py` → `build_simple_caption_prompt()` **← ONLY ACTIVE PROMPT FUNCTION**
5. `model_wrapper.py` → `_call_ollama()`

### Functions NEVER Called (Dead Code):
- `build_environmental_caption_prompt()` - Only used for first_time=True (rare)
- All QWEN-specific logic in model_wrapper.py
- `get_model_prompt_style()` - Always returns same result
- `is_qwen_model()` - Never true since we only use LLaVA
- Multiple duplicate functions in prompts.py

### Model Usage Reality:
- **Only Model:** `llava:7b-v1.6-mistral-q5_1`
- **QWEN Support:** Completely unused dead code
- **Model-Agnostic Logic:** Unnecessary - only one model used

### Current Problematic Architecture:
```
captioner.py 
    ↓
model_wrapper.py (has prompt logic + API calls)
    ↓  
prompts.py (actual prompt building)
    ↓
model_wrapper.py (API execution)
```

### Target Clean Architecture:
```
captioner.py 
    ↓
prompts.py (all prompt logic centralized)
    ↓
model_wrapper.py (pure API handler)
```

## Next Steps:
1. Consolidate all prompt logic in prompts.py
2. Remove dead QWEN code
3. Remove duplicate functions
4. Simplify model_wrapper.py to pure API handler