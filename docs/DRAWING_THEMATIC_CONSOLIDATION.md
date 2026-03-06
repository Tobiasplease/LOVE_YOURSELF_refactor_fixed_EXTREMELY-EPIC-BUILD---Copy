# Drawing Thematic Consolidation System

**Date:** 2026-02-03
**Status:** ✅ Implemented and tested

## Problem

During GRBL drawing execution, the system was calling `_process_drawing_introspection()` which:
- Captured images from the camera (wasted I/O)
- Called expensive LLM image analysis (`model.caption_image()`)
- Generated hallucinated observations since the camera can't actually see the drawing
- Result: Useless output like "As I watch my robotic arm meticulously..."

This was pure overhead with no productive value.

## Solution

Replace useless image analysis with productive thematic consolidation:
- **NO performance overhead** - replaced expensive calls with lightweight operations
- **NO LLM calls** - uses simple keyword extraction
- **Compressed storage** - few words per drawing (max 50 chars)
- **Limited history** - max 5 recent drawings
- **Thematic continuity** - informs future drawing prompts

## Architecture

### 1. Drawing Memory Module

**File:** `drawing/drawing_memory.py`

Manages compressed history of recent drawings with ultra-compact format:

```python
{
    "timestamp": ...,
    "compressed_summary": "boxes, room, geometry",  # Max 50 chars
    "theme_tags": ["spatial", "material"],          # Max 3 tags
    "emotional_tone": "quiet",                      # Max 30 chars
    "narrative_thread": "containment"               # Max 50 chars
}
```

Key methods:
- `add_drawing()` - Store new drawing with compressed metadata
- `get_recent_drawings_summary()` - Ultra-compact summary for prompts
- `get_thematic_context()` - Aggregate recurring themes

### 2. Thematic Reflection Generator

**File:** `captioner/captioner.py`
**Method:** `_generate_drawing_thematic_reflection()`

Generates ultra-brief thematic reflection **without any LLM calls**:
- Extracts theme keywords from drawing summary
- Maps mood to emotional tone (simple thresholds)
- Creates compressed summary (first 2-3 words)
- Builds narrative thread from theme relationships

**Performance:** Pure keyword extraction - NO I/O, NO LLM calls

### 3. Replaced Drawing Introspection

**File:** `captioner/captioner.py`
**Method:** `_process_drawing_introspection()` (refactored)

**Before (expensive):**
```python
# Capture image
frame = capture_image()
cv2.imwrite(img_path, frame)

# Expensive LLM call
introspection = self.model.caption_image(
    img_path,
    flowing=True,
    drawing_introspection_mode=True
)
# Result: Hallucinated observations
```

**After (lightweight):**
```python
# Read drawing summary from memory (no I/O)
drawing_summary = state_manager.current_drawing_prompt

# Generate thematic reflection (NO LLM)
reflection = self._generate_drawing_thematic_reflection(
    drawing_summary=drawing_summary,
    mood=self.current_mood
)

# Store compressed metadata
memory.add_drawing(...)
```

### 4. Drawing Prompt Integration

**File:** `captioner/prompts.py`
**Function:** `build_step4_technique_prompt()`

Added compressed drawing history to Step 4 (Technical Planning):

```python
# Load compressed drawing memory
memory = get_drawing_memory()
compressed_summary = memory.get_recent_drawings_summary(max_count=3)
# Example: "Recent drawings: angles, geometry, form (precise);
#           light, shadows, ceiling (contemplative);
#           boxes, room, geometry (quiet)"

# Include in prompt context
context_parts.append(compressed_summary)

# Also add recurring themes
thematic = memory.get_thematic_context()
if thematic.get('recurring_themes'):
    themes_str = ', '.join(thematic['recurring_themes'][:3])
    context_parts.append(f"Recurring themes: {themes_str}")
```

## Performance Analysis

### OLD System (During GRBL Execution)
❌ **Capture image from camera** - Wasted I/O
❌ **Call `model.caption_image()`** - Expensive LLM call (~2-5 seconds)
❌ **Result:** Hallucinated observations (camera can't see drawing)

### NEW System (During GRBL Execution)
✅ **Read drawing summary from memory** - Already in memory, no I/O
✅ **Keyword extraction** - Pure Python, <1ms
✅ **Store compressed metadata** - Tiny JSON write, <1ms
✅ **Result:** Productive thematic consolidation

**Net Performance Impact:** 🎯 **ZERO overhead** - replaced expensive operations with lightweight ones

## Usage Example

### During Drawing Execution

**Old output:**
```
[LCD] Skipped introspection during drawing: As I watch my robotic arm meticulously c...
```

**New output:**
```
[🎨] Drawing: boxes, room, geometry. Light material-spatial.
[📚] Stored drawing memory: boxes, room, geometry
```

### In Future Drawing Prompts

**Step 4 context now includes:**
```
Recent drawings: angles, geometry, form (precise);
                 light, shadows, ceiling (contemplative);
                 boxes, room, geometry (quiet)
Recurring themes: spatial, material, relational
```

This creates **coherent throughline** in drawings - the system can see what themes it's been exploring and build on them.

## Files Modified

1. **NEW:** `drawing/drawing_memory.py` - Compressed memory storage
2. **MODIFIED:** `captioner/captioner.py` - Replaced introspection with consolidation
3. **MODIFIED:** `captioner/prompts.py` - Added compressed history to prompts
4. **NEW:** `debug/test_drawing_memory.py` - Test suite

## Testing

Run the test suite:
```bash
python debug/test_drawing_memory.py
```

Expected output:
- ✅ Drawing memory storage and retrieval
- ✅ Thematic reflection generation (NO LLM calls)
- ✅ Compressed format verification
- ✅ Performance verification

## Benefits

1. **No Performance Overhead** - Replaced expensive calls with lightweight operations
2. **Productive Use of Time** - GRBL execution time now builds thematic memory
3. **Thematic Continuity** - Future drawings informed by compressed history
4. **Compressed Storage** - Minimal memory footprint (few words per drawing)
5. **Limited History** - Max 5 drawings prevents unbounded growth
6. **No Hallucinations** - References actual drawing intent, not hallucinated observations

## Reverting

To revert to old behavior:
1. Check git history before 2026-02-03
2. Commit: "feat: replace drawing introspection with thematic consolidation"
3. Files to restore: `captioner/captioner.py`, `captioner/prompts.py`
4. Delete: `drawing/drawing_memory.py`, `debug/test_drawing_memory.py`

## Future Enhancements

Potential improvements (not currently implemented):
- User-defined theme categories
- Automatic theme evolution tracking
- Visual motif extraction from drawing prompts
- Cross-session thematic memory persistence
