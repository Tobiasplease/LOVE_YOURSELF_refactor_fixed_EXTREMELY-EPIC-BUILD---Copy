# Reasoning Model Integration Plan

## The Core Insight

The CoT (chain-of-thought) thinking block in reasoning models is mechanically
identical to what we want: a continuous stream of tokens where each new thought
is conditioned on all prior thoughts via self-attention. Within a single
generation, this is genuine chained reasoning — not simulated continuity.

The current system builds an external chain-of-thought in Python because the
models can't do it internally. Activation memory, context compression,
felt-state synthesis, concept familiarity scoring — all of this is reasoning
about experience that happens in code, then gets handed to a model that has
80 tokens to phrase the conclusion. The model never thinks. It recites.

A thinking model could do the reasoning itself — associative leaps, emotional
threading, attention decisions — within its think block. The scaffolding
becomes raw input (facts, concept data, temporal markers) rather than
pre-digested conclusions.

## Current Architecture (Reference)

### Caption Cycle (~4s)
```
Camera frame
  |
  v
PASS 1: Qwen2.5-VL (vision, ~120 tokens)
  - Pure perception, no memory, no identity, stateless
  - "A person in a red shirt sits at a table with papers."
  |
  v
ChromaDB concept matching
  - "person" seen 12 times, "red shirt" first time
  |
  v
SCAFFOLDING (Python constructs inner state)
  - Identity line (awake time, drawing state)
  - Concept context (familiar/new, last thought about each)
  - Felt-state from compression engine
  - Desire/focus direction
  - Baseline context (spatial summary)
  - Last thought (140 chars of prior caption)
  - Mode selection (relational/observational/introspective/restless)
  |
  v
PASS 2: Mistral-Nemo (text-only, 80 tokens max)
  - Phrases what Python decided the machine should think about
  - Heavy stop sequences, short leash
```

### Memory/Continuity Systems
- **Activation Memory**: concept familiarity, boredom/novelty scores, edge graph
- **Semantic Memory (ChromaDB)**: persistent concepts + observations
- **Context Compression**: every 4 captions -> baseline_context + felt_state + desire
- **MemoryMixin**: session state, drawing history, emotional journey
- **Mood Engine**: sentiment -> 3D mood vector -> emotion label (drives servos)

### Side Processes
- Reflection: every 5 min, longer generation stored as memory
- Memory Mode: every 4 min, recall from long-term instead of perceiving
- Drawing Trigger: every 15s check -> 5-step analysis -> ComfyUI -> GRBL
- Gaze Parsing: extract direction from caption -> servo movement

### What carries continuity between cycles
- One sentence: "Its last thought was: '...'" (140 chars max)
- baseline_context string (compressed every 4 captions)
- Concept familiarity scores in ChromaDB
- Felt-state string from compression

## Proposed Architecture (Single-Pass Reasoning)

### Caption Cycle
```
Camera frame
  |
  v
SINGLE PASS: Qwen3-VL 8B thinking (vision + reasoning)
  System: machine identity prompt (tested, proven framing)
  Context:
    - Prior thought stream (~1500 chars of model's own output)
    - Raw facts: awake time, drawing state, last drawing
    - Raw concept data: [person: seen 12x] [red shirt: new]
    - Felt-state transition: quiet -> alert
    - [new image from camera]
  |
  v
  <think> block (200-400 tokens)
    - Model reasons about what it sees IN CONTEXT of its prior thoughts
    - Associative leaps happen via self-attention, not Python scoring
    - Emotional threading emerges from the model attending to its own
      developing thought, not from injected felt-state labels
  </think>
  |
  v
  Response (50-100 tokens)
    - Distilled monologue for display / caption stream
  |
  v
Think block appended to thought stream (rolling window)
Context compression trims oldest parts when stream > threshold
```

### What changes in each subsystem

| System | Current role | New role |
|--------|-------------|----------|
| Activation Memory | Decides boredom/novelty, constructs prompt | Provides raw concept data as facts. Model decides what's interesting. |
| Semantic Memory | Decides familiar/new, builds attention block | Provides concept history. Model decides what to attend to. |
| Context Compression | Synthesizes felt-state/desire/baseline | Compresses the THOUGHT STREAM when it grows too long. Becomes memory consolidation, not caption summarization. |
| Mode Selection | Chooses prompt mode, changes system prompt | Possibly unnecessary. Model's reasoning determines its own posture. |
| Mood Engine | Caption sentiment -> emotion -> servos | Still needed for physical outputs. Could parse think block for richer emotional signal. |
| Gaze Parsing | Extract direction from caption | Same — parse from model output. |

### What carries continuity between cycles
- The thought stream itself (~1500 chars of the model's prior reasoning)
- Raw concept data from ChromaDB (facts, not conclusions)
- Compressed older thoughts from the compression engine

## The Key Risk: VQA Gravity

Vision-language models are trained on image description and visual QA. Their
think blocks may reason about "what objects are present and their spatial
relationships" rather than "what this means to me as an entity with
continuity."

The system prompt can push against this. The prior thought stream (containing
experiential language from prior cycles) creates in-context examples of what
thinking should look like. But if the model's training bias is too strong,
the think block will just be verbose image analysis.

This is the single most important thing the test must evaluate:
does the think block produce experiential reasoning or analytical description?

If the system prompt can't overcome VQA gravity, a LoRA fine-tune on
50-100 examples of "image + thought stream -> experiential continuation"
would be the next step. The mechanism (CoT) is right; the training
distribution might need adjustment.

## Test Plan

### Prerequisites
- Update Ollama to >= 0.12.7
- Pull qwen3-vl:8b-thinking

### Test Script
`debug/test_continuous_reasoning.py` — already written, needs update to:
1. Add Qwen3-VL as the proposed model
2. Parse <think> and response separately
3. Show think block content (this IS the monologue we're evaluating)
4. Run 5-10 cycles with live camera to test continuity across real scene changes

### Evaluation Criteria
1. Does the think block read as inner monologue or image analysis?
2. Does thought develop WITHIN a single think block (associative leaps)?
3. Does the model reference its prior thoughts naturally?
4. Does it notice change across frames without being told to?
5. Latency: does think + respond fit in a 4-10s cycle on RTX 3090?

### If it works
- Refactor caption pipeline to single-pass
- Repurpose compression engine for thought stream consolidation
- Simplify scaffolding to raw fact injection
- Keep all physical output systems (mood, gaze, servo, drawing)

### If it partially works
- Use thinking model for reflection (5min) and compression passes only
- Keep Nemo for fast caption cycle
- Feed richer reflection output back into scaffolding

### If it doesn't work
- Current architecture stays
- Consider LoRA fine-tune as next investment
- Or wait for better local reasoning-vision models (12B+)

## Notes

- The current two-pass pipeline is well-optimized for non-reasoning models.
  The scaffolding, short leash, and stop sequences are correct compensation
  for models that can't chain thought. This is months of empirical work.
- Do not refactor the production pipeline until the test shows clear
  improvement. The test script exists to evaluate before committing.
- The thinking model's value is WITHIN-CALL reasoning (self-attention over
  200+ tokens of thought development), not cross-call continuity (which is
  simulated either way by feeding prior output back).
- Date: 2026-05-06
