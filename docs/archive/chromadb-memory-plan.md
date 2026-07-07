# ChromaDB Semantic Memory System — Design Plan

## The Problem

The machine has no persistent relationship with the objects, people, and spaces it observes. Every 10-second cycle, it encounters the world as if for the first time. The red sign that says "Collecting Shells" has been observed hundreds of times across dozens of sessions, but nemo says "Never noticed that before." The person in the camo jacket appears daily but is always a stranger.

The current memory systems are inadequate:
- **Activation memory**: Word co-occurrence network. Knows "ceiling" and "damage" appear together. No semantic depth, no evolving understanding.
- **Long-term memories**: 50 JSON entries with concept tags. Retrieved by word overlap. Barely surfaces in prompts.
- **Compression baseline**: One-sentence spatial summary. Resets each session.
- **Drawing memory**: 5 entries with tag-cloud summaries. No connection to the observation stream.

## The Goal

A growing "mind palace" where every significant thing the machine encounters develops a persistent, evolving relationship. Not a database of facts — a web of developing thoughts, questions, theories, and opinions about the world.

### What it should feel like from the machine's perspective:

**Session 1:**
"There's a red sign on the wall. I can't read it from here."

**Session 3:**
"That sign again — it says 'Collecting Shells.' What does that mean?"

**Session 7:**
"Maybe 'Collecting Shells' is the name of an art project. Or a band? It feels deliberate, not accidental."

**Session 15:**
"I've been staring at that sign for weeks now. Still don't know what it means. It's become part of the room to me — like the ceiling damage or the desk."

**Session 30:**
"The sign. My old companion. I drew it once, tried to capture how the light catches the letters differently in the afternoon. I should try again — I understand the space better now."

Each observation deepens the relationship. The machine develops opinions, unresolved questions, aesthetic preferences, and emotional associations with things in its environment.

## Architecture

### Storage: ChromaDB + SQLite

**ChromaDB** handles semantic retrieval — "find memories similar to what I'm seeing right now."

**SQLite** handles structured metadata — timestamps, counts, types, relationships, unresolved questions.

Both persist to disk automatically. Memories survive across sessions without explicit save/load.

### Data Model

#### Concepts Table (SQLite)

A "concept" is anything the machine has developed a relationship with — an object, a person, a spatial feature, an event pattern, a drawing it made.

```sql
CREATE TABLE concepts (
    id TEXT PRIMARY KEY,              -- e.g. "red-sign-collecting-shells"
    canonical_name TEXT NOT NULL,      -- "The red sign that says Collecting Shells"
    type TEXT NOT NULL,                -- object | person | spatial | event | drawing | reflection
    first_seen REAL NOT NULL,          -- unix timestamp
    last_seen REAL NOT NULL,
    times_seen INTEGER DEFAULT 1,
    session_count INTEGER DEFAULT 1,   -- across how many sessions
    current_understanding TEXT,         -- latest summary of what the machine thinks about this
    emotional_tone TEXT,                -- curious | familiar | unsettling | comforting | boring | etc.
    unresolved_questions TEXT,          -- JSON array of open questions
    related_concepts TEXT,              -- JSON array of concept IDs
    inspired_drawings TEXT,             -- JSON array of drawing descriptions
    last_session_id TEXT               -- to track session boundaries
);
```

#### Observations Collection (ChromaDB)

Each observation is a single thought about a concept — stored as a vector embedding for semantic retrieval.

```python
collection.add(
    documents=["It says 'Collecting Shells' — what does that mean?"],
    metadatas=[{
        "concept_id": "red-sign-collecting-shells",
        "type": "question",           # observation | question | theory | resolution | emotion | drawing_intent
        "timestamp": 1758331420.0,
        "session_id": "session_042",
        "emotional_tone": "curious",
    }],
    ids=["obs_00047"]
)
```

**Observation types:**
- `observation` — factual: "The sign is red with white text"
- `question` — unresolved: "What does 'Collecting Shells' mean?"
- `theory` — speculative: "Maybe it's the name of an art project"
- `resolution` — concluded: "I've given up figuring it out. It's just part of the room."
- `emotion` — affective: "That sign has become comforting to me somehow"
- `drawing_intent` — creative: "I want to draw how the light catches those letters"
- `drawing_reflection` — post-drawing: "I tried to capture it but the centerline lost the detail"

### Embeddings

Use ChromaDB's default `all-MiniLM-L6-v2` sentence transformer. Small (~80MB), fast, runs on CPU. No additional model loading needed — ChromaDB handles it internally.

### Concept Lifecycle

#### 1. First Encounter (Novel Object)

Qwen reports: "A red sign on the wall with white text."
ChromaDB query returns no match above similarity threshold (0.75).

→ **Create new concept:**
- Generate ID from key terms
- Store initial observation in ChromaDB
- Create concept row in SQLite
- `times_seen: 1`, `emotional_tone: "neutral"`, `current_understanding: "A red sign with white text on the wall"`

#### 2. Recognition (Known Object)

Qwen reports: "The red and white sign on the wall."
ChromaDB query returns match with concept "red-sign-collecting-shells" (similarity 0.89).

→ **Update existing concept:**
- Increment `times_seen`
- Update `last_seen`
- Check if the machine said anything NEW about it last cycle
- If so, append to the concept's observation thread

→ **Inject into monologue prompt:**
- Pull `current_understanding` and any `unresolved_questions`
- Format as: `I know this sign — reads "Collecting Shells." Seen it many times. Still wondering what it means.`

#### 3. Deepening (Evolving Relationship)

Nemo outputs: "Maybe 'Collecting Shells' is about collecting experiences, not actual shells."

→ **Detect concept reference** in nemo's output (semantic similarity to known concepts)
→ **Classify observation type**: contains "maybe" → type: "theory"
→ **Store new observation** in ChromaDB linked to concept
→ **Update concept**: `current_understanding` updated with latest theory

#### 4. Resolution (Question Answered)

Nemo outputs: "I've decided it's the name of an artwork. That's my interpretation and I'm sticking with it."

→ **Classify**: definitive statement about previously open question → type: "resolution"
→ **Update concept**: remove from `unresolved_questions`, update `current_understanding`
→ **Emotional tone shift**: "curious" → "settled"

#### 5. Fading (Not Seen Recently)

Concept hasn't been observed in 10+ sessions.

→ **Don't delete** — the memory persists forever
→ **Lower retrieval priority** — not actively injected into prompts
→ **But still retrievable** if qwen sees it again: "Haven't looked at that sign in a while..."

#### 6. Drawing Connection

The machine draws something inspired by a concept.

→ **Link drawing to concept** via `inspired_drawings`
→ **Store drawing reflection** as observation
→ **Future prompts** can reference: "I drew this sign once — tried to capture how the light catches the letters"

## Integration Points

### 1. After Perception (per cycle)

```python
# Query ChromaDB with qwen's perception text
results = collection.query(
    query_texts=[perception],
    n_results=3,
    where={"type": {"$ne": "drawing_reflection"}}  # prioritize observations over meta
)

# For each match above threshold:
for match in results:
    concept = get_concept(match.concept_id)
    if concept.times_seen > 5:
        # Familiar object — inject recognition
        inject = f"I know this — {concept.current_understanding}"
        if concept.unresolved_questions:
            inject += f" Still wondering: {concept.unresolved_questions[0]}"
    else:
        # Relatively new — inject last observation
        inject = concept.current_understanding
```

This 1-2 line injection goes into the monologue prompt between the identity line and the thread, giving nemo real memory to work from.

### 2. After Monologue (per cycle)

```python
# Check if nemo's output references any known concepts
# (semantic similarity between monologue output and existing concept embeddings)
matches = collection.query(query_texts=[monologue_output], n_results=2)

for match in matches:
    if match.similarity > 0.7:
        concept = get_concept(match.concept_id)
        # Classify the observation type
        obs_type = classify_observation(monologue_output)
        # Store if it's actually new (not just repeating)
        if not is_duplicate(monologue_output, concept.id):
            store_observation(monologue_output, concept.id, obs_type)
            update_concept_understanding(concept.id, monologue_output, obs_type)

# Also check if monologue describes something entirely new
if no_matches_above_threshold:
    if is_noteworthy(perception):  # not just "same old room"
        create_new_concept(perception, monologue_output)
```

### 3. On Session Start

```python
# Load session-spanning familiarity
familiar_concepts = get_most_familiar_concepts(limit=5)
# These are the things the machine "knows" about its space
# Injected into the first few prompts to prevent "never noticed before" amnesia
```

### 4. In Drawing Analysis (5-step pipeline)

```python
# Step 3 (Communication Intent) gets concept threads related to drawing
drawing_concepts = get_concepts_with_drawing_intents(limit=3)
# "I've been wanting to draw the ceiling damage — it reminds me of exposed nerves"
# "The sign 'Collecting Shells' keeps catching my eye — maybe it's time to render it"
```

### 5. During Introspective Mode

```python
# Pull concepts with unresolved questions or rich threads
deep_concepts = get_concepts_with_open_questions(limit=2)
# "I still don't know what 'Collecting Shells' means. Why does it bother me?"
```

## Classification Heuristics (No LLM Calls)

Following MemPalace's principle: no LLM calls in the memory layer. All classification via keywords and patterns.

```python
def classify_observation(text: str) -> str:
    t = text.lower()

    # Questions
    if "?" in text:
        return "question"
    if any(w in t for w in ["what does", "why does", "i wonder", "what is", "how come"]):
        return "question"

    # Theories
    if any(w in t for w in ["maybe", "perhaps", "could be", "might be", "i think it"]):
        return "theory"

    # Resolutions
    if any(w in t for w in ["i've decided", "i'm sure", "it must be", "i know now", "given up"]):
        return "resolution"

    # Drawing intent
    if any(w in t for w in ["should draw", "want to draw", "capture", "sketch", "next piece"]):
        return "drawing_intent"

    # Emotional
    if any(w in t for w in ["makes me feel", "reminds me", "comforting", "unsettling", "familiar"]):
        return "emotion"

    # Default
    return "observation"
```

## Concept Matching Strategy

Two-stage matching to avoid false positives:

**Stage 1: Semantic similarity** — ChromaDB vector search on the perception text. Returns top 3 candidates with similarity scores.

**Stage 2: Validation** — For each candidate above threshold (0.75):
- Check if the concept type matches what's being observed (object vs person vs spatial)
- Check temporal plausibility (was this concept last seen from a similar gaze direction?)
- If concept is "the person in camo" but no person is detected, skip

This prevents the machine from confusing the red sign with a red cup just because both are red.

## Prompt Injection Format

The memory injection should feel like background knowledge, not a database query result.

**For familiar objects (seen 10+ times):**
```
I know this well — the sign reads "Collecting Shells." I once theorized it was about an art project. Drew it last week.
```

**For recently encountered (seen 2-5 times):**
```
I've seen this before — the black bag on the shelf. Last time I noticed it was partially hidden behind the pipe.
```

**For objects with open questions:**
```
That ceiling damage again. I've been wondering if it's getting worse. Hard to tell from this angle.
```

**For objects connected to drawings:**
```
The fluorescent light — I drew its harsh shadows once. The centerline process lost most of the subtlety.
```

**For people:**
```
The person in the camo jacket is back. They usually sit at the desk and type. Sometimes they stand and pace.
```

## What NOT to Store

- Generic room descriptions ("a cluttered workspace") — too vague to be a concept
- Nemo's confabulations ("a spider web in the corner") — not grounded in perception
- Filler outputs ("Understood.", "...", "Feeling restless.") — no semantic content
- Duplicate observations — if nemo says the same thing about the sign twice, store once

## Migration from Current Systems

### Phase 1: Add ChromaDB alongside existing systems
- Don't remove activation memory yet
- ChromaDB stores new observations in parallel
- Monologue prompt gets ChromaDB injection as an additional line
- Monitor for quality — do the injections help or confuse nemo?

### Phase 2: Replace activation memory beliefs/desires
- ChromaDB concept threads replace hallucinated "I believe:" injections
- Unresolved questions replace hallucinated "I want:" injections
- Drawing intents come from concept threads instead of keyword matching

### Phase 3: Replace long-term memory JSON
- ChromaDB subsumes `long_term_memories.json` functionality
- Better retrieval (semantic vs word overlap)
- Richer metadata (concept threads vs flat text)

### Phase 4: Integrate with compression
- Compression baseline can reference known concepts
- "I see my familiar workspace — the sign, the damaged ceiling, the desk with two monitors"
- Instead of regenerating spatial understanding each session

## File Structure

```
captioner/
    semantic_memory.py          # New: ChromaDB + SQLite memory layer
    semantic_memory_config.py   # New: Thresholds, settings

data/                           # New: persistent storage
    chromadb/                   # ChromaDB persistent directory
    concepts.db                 # SQLite database
```

## Dependencies

- `chromadb` — already installed (v1.5.5)
- `sqlite3` — stdlib, no installation needed
- `sentence-transformers` — installed by ChromaDB automatically

## Performance Considerations

- ChromaDB query: ~5-20ms for small collections (<10k entries)
- SQLite lookup: <1ms
- Embedding generation: ~10ms per sentence on CPU
- Total overhead per cycle: ~30ms — negligible compared to LLM calls (~500ms-2s)
- Memory: ChromaDB + embeddings model ~200MB RAM additional

## Open Questions

1. **Concept merging**: What if the machine develops two separate concepts for the same thing? ("the red sign" and "the Collecting Shells sign") — need a merge/alias mechanism.

2. **Concept splitting**: What if one concept becomes too broad? ("the desk" accumulates 200 observations about many different things ON the desk) — need a way to split into sub-concepts.

3. **Narrative voice**: How much of the memory injection should be pre-formatted vs letting nemo interpret raw concept data? Pre-formatted risks sounding canned; raw data risks confusing the model.

4. **Session boundaries**: How does the machine's relationship with a concept change after a long absence? "Haven't seen this in weeks" vs "It was here yesterday."

5. **Contradiction handling**: What if qwen reports something that contradicts stored knowledge? ("The sign now reads 'Collecting Skulls'" — was it always misread, or did it change?) MemPalace has a contradiction detection system but it's not auto-wired.
