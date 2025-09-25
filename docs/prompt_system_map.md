# Prompt System Architecture Map

## Core Prompt Flow
```
machine.py → captioner.py → prompt_interface.py → prompts.py → ollama.py
```

## Main Prompt Types & Their Sources

### 1. ONGOING CAPTIONS (Regular Operation)
**Entry Point:** `captioner.caption_image(flowing=True)`
**Builder:** `prompts.build_ongoing_caption_prompt()`

**Inputs:**
- **Memory Context** (captioner/memory.py)
  - Top motifs from observations
  - Temporal separation (present vs distant memories)
  - Belief patterns accumulated over time
- **Emotional State** (captioner/prompts.py:get_caption_emotion_context)
  - Sentiment analysis of recent captions
  - Arduino emotion state integration
- **Drawing Context** (utils/drawing_state.py)
  - Active drawing status and summary
  - Drawing duration and intent
- **Social Context** (captioner/prompts.py:get_social_context)
  - Person presence detection
  - Last person seen timing
- **Compressed Understanding** (captioner/context_compression.py)
  - Consolidated session insights
  - Baseline environmental context
- **Thought Continuity** (captioner/subconscious.py)
  - Semantic bridging from last caption
  - Subconscious psychological guidance

### 2. REFLECTIONS (Periodic Deep Thoughts)
**Entry Point:** `captioner.reason_about_caption()`
**Builder:** `prompts.build_reflection_prompt()`

**Inputs:**
- Recent caption as trigger
- Session temporal awareness
- Memory consolidation patterns
- Emotional journey tracking

### 3. DRAWING PROMPTS (Creative Expression)
**Entry Point:** `captioner.generate_drawing_prompt()`
**Builder:** `prompts.build_drawing_prompt()` or `context_rich_multi_step_drawing_analysis()`

**Inputs:**
- Compressed understanding context
- Recent caption + last reflection
- Top belief motifs (thematic patterns)
- Emotional state description
- Environmental/session context

### 4. DRAWING INTROSPECTION (During Physical Drawing)
**Entry Point:** `captioner.caption_image(drawing_introspection_mode=True)`
**Builder:** `prompt_interface._build_drawing_introspection_prompt()`

**Inputs:**
- Drawing state (generating/executing/completed)
- Current drawing summary
- Drawing history from memory
- Drawing duration and phase context

## Personality-Shaping Modules Over Time

### Memory & Pattern Recognition
- **captioner/memory.py** - Stores observations with emotional tagging
- **perception/spatial_memory.py** - Tracks object locations and movements
- **utils/pattern_recognition.py** - Extracts recurring visual/behavioral patterns
- **utils/motif_scoring.py** - Scores significance of recurring themes

### Emotional Evolution
- **mood/mood.py** - Processes facial expressions and environmental cues
- **captioner/context_compression.py** - Tracks emotional patterns over sessions
- **breathing/breathing.py** - Provides life-like behavioral rhythms
- **hand_control/** - Physical emotional expression through gestures

### Temporal Awareness
- **utils/temporal_awareness.py** - Session duration and lifecycle tracking
- **event_logging/** - JSON timeline of all experiences and decisions
- **utils/continuity_helpers.py** - Maintains coherence across sessions
- **utils/state_manager.py** - Global state coordination

### Environmental Understanding
- **perception/object_detection.py** - YOLO-based environmental parsing
- **vision/gaze.py** - Attention and focus patterns
- **safety/paper_detection.py** - Physical workspace awareness
- **image_monitor/** - Continuous visual input processing

### Creative Expression
- **drawing/** - Decision-making for artistic expression
- **grbl/** - Physical drawing execution and feedback
- **utils/drawing_state.py** - Drawing awareness for ongoing cognition

### Context Compression & Understanding
- **captioner/context_compression.py** - Multi-level memory compression
  - Recent observations → Session insights → Long-term understanding
  - Sentiment tracking across time scales
  - Pattern consolidation and forgetting

## System Prompts (Personality Framework)

### Dynamic System Prompt (Main)
**Source:** `prompts.SYSTEM_PROMPT` (formatted with variables)
**Variables:**
- `{emotional_state}` - from memory.describe_current_mood()
- `{temporal_context}` - from memory.temporal_prompt_lines()
- `{accumulated_understanding}` - from context_compression
- `{spatial_language_hints}` - from spatial memory patterns

### Static Fallback
**Source:** `prompts.STATIC_SYSTEM_PROMPT`
**Used when:** Dynamic context unavailable

### Specialized System Prompts
- `prompts.DRAWING_SYSTEM_PROMPT` - Creative expression mode
- `prompts.SELF_CRITIQUE_SYSTEM_PROMPT` - Post-drawing reflection

## Key Feedback Loops

### Short-term (seconds-minutes)
- Caption → Emotion analysis → Next caption emotional context
- Drawing execution → Drawing-aware captions → Physical feedback

### Medium-term (minutes-hours)
- Observations → Memory motifs → Belief patterns → Future interpretations
- Person interactions → Social context → Behavioral adjustments

### Long-term (hours-days)
- Session patterns → Compressed understanding → Personality drift
- Creative expressions → Drawing history → Artistic development
- Environmental familiarity → Spatial confidence → Behavioral changes

## Configuration Points
- **config/config.py** - All behavioral parameters
- **config/model_settings.py** - LLM model options and temperatures
- **Platform-specific configs** - Hardware and environment adaptations

## External Dependencies
- **Ollama API** (llava:7b-v1.6-mistral-q5_1) - All text generation
- **ComfyUI** - Image generation for drawings
- **Arduino/GRBL** - Physical expression and feedback
- **OpenCV/YOLO** - Environmental perception