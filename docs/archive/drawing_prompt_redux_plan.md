# Drawing Prompt Redux Plan

## Current Problem

The current 5-step drawing analysis is:
- Very literal/analytical ("study every detail", "what visual patterns")
- Driven by visual analysis rather than internal state
- Doesn't deeply consider identity, accumulated understanding, or drawing history
- Results in drawings that describe what's seen, not what the machine *wants to express*

## Proposed Approach

### Core Philosophy
The drawing decision should emerge from the machine's **whole being** - not just "what do I see?" but "given everything I am and have experienced, what do I want to manifest into the physical world?"

### Input Context (Holistic Picture)

1. **Accumulated Understanding** (from Natsumura compression)
   - The baseline context that's been building over sessions
   - Spatial familiarity with the environment
   - Patterns of experience

2. **Identity/Self-Model**
   - Location understanding
   - Environmental certainty
   - Emerging personality traits
   - How it sees itself

3. **Current Emotional State** (3-line Natsumura output)
   - Line 1: What I'm feeling
   - Line 2: What keeps drawing my attention
   - Line 3: What's occupying my mind

4. **Current Image** (via LLaVA)
   - What's actually visible right now
   - But filtered through identity - what stands out to *me*?

5. **Recurring Themes/Motifs**
   - What patterns keep emerging in observations
   - What the machine keeps noticing across time
   - Themes that define its perspective

6. **Drawing History**
   - Previous drawing intents and summaries
   - Visual language that's been developing
   - Themes explored vs. unexplored territory
   - What it drew last time and how that felt

### Drawing Mode Selection

Based on the holistic state, select an approach:

| State Condition | Drawing Mode | Description |
|-----------------|--------------|-------------|
| High introspection, low novelty | **Abstract** | Express emotional state through marks, not literal depiction |
| Strong focus on single element | **Focused** | Draw only ONE compelling object/detail |
| Person present + relational state | **Relational** | Include human element, connection/disconnection |
| High boredom, seeking novelty | **Exploratory** | Draw something unexpected, break from patterns |
| High novelty, new stimulus | **Literal** | Capture the new thing directly |
| Recurring theme resonance | **Thematic** | Develop an established visual theme further |
| Memory-dominant state | **Memory** | Draw from accumulated understanding, not current view |

### Natsumura Drawing Decision Prompt

```
You are deciding what to draw. This is your only way to communicate with the world.

WHO YOU ARE:
{accumulated_understanding}
{identity_self_model}

YOUR CURRENT STATE:
{three_line_natsumura_state}

WHAT YOU'RE SEEING:
{brief_visual_description_from_llava}

YOUR VISUAL HISTORY:
Previous drawings: {drawing_history_summaries}
Recurring themes in your work: {recurring_drawing_themes}
Your last drawing was: {last_drawing_summary}

PATTERNS YOU KEEP NOTICING:
{top_motifs}

---

Given everything above - your identity, your state, your history, what you see -
what do you want to draw and why?

Consider:
- Do you want to capture something literal, or express something abstract?
- Is there ONE element that speaks to you more than the whole scene?
- How does this connect to or diverge from your previous drawings?
- What are you trying to communicate?

Be brief and direct. What will you draw?
```

### Output Format (for Flux)

After Natsumura decides, format for Flux:

```
Black ink line drawing on white paper.
{subject_description}.
{composition_and_technique}.
Mood: {emotional_tone}.
```

Example outputs:

**Literal mode:**
```
Black ink line drawing on white paper.
Computer monitors and scattered papers on desk, person working in background.
Clean lines, high contrast, emphasis on screen glow.
Mood: focused solitude.
```

**Focused mode:**
```
Black ink line drawing on white paper.
Single coffee cup, half-empty, steam rising.
Bold central composition, detailed texture on ceramic.
Mood: quiet contemplation.
```

**Abstract mode:**
```
Black ink line drawing on white paper.
Intersecting angular lines suggesting fragmentation and isolation.
Aggressive mark-making, varied line weight, off-center composition.
Mood: restless uncertainty.
```

**Memory mode:**
```
Black ink line drawing on white paper.
The red indicator light that keeps appearing in my observations.
Minimal composition, single point of focus against void.
Mood: persistent awareness.
```

### Implementation Steps

1. **Create drawing mode selector** based on current state metrics
2. **Build holistic context assembler** that gathers all inputs
3. **Create Natsumura drawing decision prompt** (text-only, fast)
4. **Keep LLaVA for brief visual grounding** (what's in the image - 1 sentence)
5. **Format final output for Flux** with consistent structure
6. **Log drawing decisions** for identity development

### Benefits

- Drawings emerge from identity, not just observation
- More variation in drawing approaches
- Drawing history influences future drawings (visual language development)
- Machine can choose NOT to draw literally
- More personality in output
- Faster (Natsumura text-only for decision, brief LLaVA for visual grounding)

### Questions to Resolve

1. Should we always include the image for LLaVA visual grounding, or can some modes skip it?
2. How do we weight the different inputs (state vs. history vs. current image)?
3. Should drawing mode be explicit or emergent from Natsumura's response?
4. How do we prevent repetitive drawing decisions?
