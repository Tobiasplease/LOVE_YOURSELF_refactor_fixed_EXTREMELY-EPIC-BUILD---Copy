# Activation Memory System Plan

## Overview

Replace the shallow motif extraction system with a cognitive-inspired activation-spreading memory network. This system provides intelligent, contextual memory recall without LLM calls.

---

## Problems with Current System

| Component | Current Behavior | Problem |
|-----------|-----------------|---------|
| `motif_counter` | Counts word frequency | "desk: 847" is meaningless noise |
| `motif_confidence` | TinyLlama scores each motif | Slow, adds latency, scores arbitrary |
| `beliefs` | "I keep noticing X" when count > 7 | Shallow, doesn't capture relationships |
| `estimate_novelty()` | Word overlap between captions | Misses semantic similarity |
| `cleanup_motifs()` | Prunes low-count motifs | Fighting symptoms, not cause |

---

## New Architecture: Activation Network

### Core Concept

Inspired by cognitive psychology's spreading activation model:
- Concepts have **activation levels** (0-1) that rise when observed and decay over time
- Concepts build **edges** through co-occurrence (seen together = linked)
- Activation **spreads** through edges (seeing "desk" activates linked "paper")
- **Novelty** = inverse of activation (unseen things are surprising)
- **Boredom** = high average activation (everything familiar)
- **Beliefs** = strong edges (learned associations)

### Data Structures

```python
class ActivationNetwork:
    activations: Dict[str, float]      # concept -> level (0-1)
    edges: Dict[str, Dict[str, float]] # concept -> {related: weight}
    spatial_tags: Dict[str, str]       # concept -> gaze_zone
    last_seen: Dict[str, float]        # concept -> timestamp
```

### Operations

1. **observe(concepts)** - Boost activation, build edges, spread
2. **decay()** - All activations *= decay_rate
3. **spread()** - Activation flows through edges
4. **recall(context)** - Find memories matching activated concepts

---

## What Each Old Component Becomes

```
OLD                          →  NEW
─────────────────────────────────────────────────────────────
motif_counter (frequency)    →  activation_level (0-1, decaying)
motif_confidence (TinyLlama) →  REMOVED - edges capture significance
beliefs dict                 →  strong_edges (weight > 0.7)
estimate_novelty()           →  1 - avg(activation of observed)
update_boredom()             →  avg(activation of scene)
cleanup_motifs()             →  natural decay handles this
TinyLlama scoring thread     →  REMOVED entirely
```

---

## Detailed Walkthrough

### T=0: Session Start
```
activations: {}
edges: {}
memories: []
```

### T=1min: First Observation
```
Caption: "A cluttered desk with scattered notebooks"
Extract: [desk, clutter, notebook]

→ Boost:
  desk: 0.0 → 0.3
  clutter: 0.0 → 0.3
  notebook: 0.0 → 0.3

→ Build edges:
  desk↔clutter: 0.05
  desk↔notebook: 0.05
  clutter↔notebook: 0.05

→ Store memory with concepts + zone + timestamp
```

### T=5min: Person Appears
```
Caption: "Someone is sitting at the desk"
Extract: [person, sitting, desk]

→ Boost:
  person: 0.0 → 0.3
  sitting: 0.0 → 0.3
  desk: 0.27 → 0.57 (decayed + boosted)

→ Build edges:
  person↔desk: 0.05 (NEW - important later!)
```

### T=30min: Person Left, Looking at Empty Desk
```
Caption: "The desk is empty now"
Extract: [desk, empty]

→ Boost:
  desk: 0.6 (strong - keeps getting reinforced)
  empty: 0.3

→ Spread activation:
  desk (0.6) spreads to person via edge:
  person: 0.0 → 0.03 (edge_weight * spread_factor)

→ "person" now weakly activated even though not observed!
```

### T=31min: Recall for Prompt
```
Current gaze: "down"
Activated: [desk: 0.6, empty: 0.3, person: 0.03]

→ Score memories:
  Memory #2 (person + desk): overlap=2, wins!

→ Output: "I remember someone was here 25 minutes ago."
```

---

## Novelty Scoring

```python
def calculate_novelty(self, observed_concepts: List[str]) -> float:
    """Novelty = how surprised we are."""
    if not observed_concepts:
        return 0.5

    activations = [self.activations.get(c, 0.0) for c in observed_concepts]
    familiarity = sum(activations) / len(activations)

    return 1.0 - familiarity
```

**Examples:**
- See "desk" (activation 0.8) → novelty = 0.2 (boring)
- See "cat" (activation 0.0) → novelty = 1.0 (surprising!)
- See "desk" + "cat" → novelty = 0.5 (mixed)

---

## Beliefs from Strong Edges

Instead of "I keep noticing desk", beliefs become learned associations:

```python
def get_beliefs(self) -> List[str]:
    beliefs = []
    for c1, neighbors in self.edges.items():
        for c2, weight in neighbors.items():
            if weight > 0.7 and c1 < c2:  # Avoid duplicates
                beliefs.append(self._interpret_edge(c1, c2, weight))
    return beliefs

def _interpret_edge(self, c1: str, c2: str, weight: float) -> str:
    if weight > 0.9:
        return f"The {c1} and {c2} are inseparable."
    elif weight > 0.7:
        return f"I often see {c1} with {c2}."
    return f"{c1} and {c2} connect."
```

---

## Drawing Prompt Context

```python
def get_drawing_context(self) -> dict:
    return {
        "active_concepts": self.get_activated(threshold=0.5),
        "novel_observations": self.get_recent_novel(),
        "associations": self.get_strong_edges(threshold=0.7),
        "rising": self.get_trends("rising"),
        "fading": self.get_trends("fading"),
    }
```

**Output:**
> "Your mind holds: desk, paper, scattered. Something new: a person appeared.
> In your understanding, desk and paper are inseparable."

---

## Compression Feedback Loop

```
CAPTIONS → ACTIVATION NETWORK → COMPRESSION → BOOST LOOP
    ↑                                              │
    └──────────────────────────────────────────────┘
```

```python
def process_compression(self, compression_text: str):
    """Boost concepts that made it into compression."""
    concepts = extract_concepts(compression_text)

    for concept in concepts:
        # Compression boost - these concepts "stuck"
        self.activations[concept] = min(1.0,
            self.activations.get(concept, 0) + 0.15)

    # Strengthen edges between compressed concepts
    for c1 in concepts:
        for c2 in concepts:
            if c1 != c2:
                self.edges[c1][c2] = min(1.0, self.edges[c1][c2] + 0.1)
```

---

## Memory Recall with Temporal Context

```python
def recall(self, current_gaze: str, mode: str, k: int = 2) -> List[str]:
    """Recall relevant memories with temporal framing."""
    activated_set = {c for c, a in self.activations.items() if a > 0.1}

    scored = []
    for mem in self.memories:
        overlap = len(set(mem['concepts']) & activated_set)
        if overlap == 0:
            continue

        spatial_boost = 1.5 if mem['zone'] == current_gaze else 1.0
        mode_boost = self._get_mode_boost(mem, mode)
        age_hours = (time.time() - mem['timestamp']) / 3600
        recency = 1.0 / (1.0 + age_hours * 0.1)

        score = overlap * spatial_boost * mode_boost * recency
        scored.append((mem, score))

    scored.sort(key=lambda x: -x[1])
    return [self._format_memory(m) for m, _ in scored[:k]]

def _format_memory(self, mem: dict) -> str:
    time_desc = describe_time_gap(mem['timestamp'])
    text = mem['text'][:50]
    return f"I remember {time_desc}: \"{text}...\""
```

---

## Mode-Based Filtering

| Mode | Boost Concepts | Purpose |
|------|----------------|---------|
| relational | person, someone, they | Person memories |
| workspace | desk, paper, tool | Object memories |
| introspective | feeling, wonder, thought | Reflective memories |
| observational | (recent only) | What changed? |

---

## Configuration

```python
# In config/config.py
ACTIVATION_DECAY_RATE = 0.95          # Per observation
ACTIVATION_BOOST = 0.3                 # On observe
ACTIVATION_SPREAD_FACTOR = 0.3         # Edge spread
EDGE_BUILD_INCREMENT = 0.05            # Co-occurrence
EDGE_STRENGTH_THRESHOLD = 0.7          # For beliefs
ACTIVATION_RECALL_THRESHOLD = 0.1      # For memory scoring
COMPRESSION_BOOST = 0.15               # Feedback loop
MAX_MEMORIES = 200                     # LRU eviction
EDGE_PERSISTENCE_FILE = "data/activation_edges.json"
```

---

## Files to Create/Modify

### CREATE: `captioner/activation_memory.py`
- `ActivationNetwork` class
- `ContextualMemory` class
- `extract_concepts()` helper
- Edge persistence (save/load JSON)

### MODIFY: `captioner/memory.py`
- Remove: motif_counter, motif_confidence, TinyLlama scoring
- Import ActivationNetwork, ContextualMemory
- Replace observe() internals with activation calls
- Replace estimate_novelty(), update_boredom()
- Replace get_beliefs() with edge-based beliefs

### MODIFY: `captioner/prompts.py`
- Replace dump-everything with `get_contextual_recall()`
- Update drawing prompt context builder
- Mode-based memory filtering

### MODIFY: `captioner/context_compression.py`
- Add `process_compression()` callback
- Boost concepts from compression output

### MODIFY: `captioner/captioner.py`
- Pass gaze_zone to memory.observe()
- Call decay() periodically
- Wire compression feedback

### CREATE: `debug/test_activation_memory.py`
- Unit tests for activation, decay, spread
- Integration test with real captions
- Memory recall scoring tests

---

## Persistence Strategy

**Between sessions, persist:**
1. Edge weights (learned associations) → `data/activation_edges.json`
2. High-activation concepts (> 0.5) → optional core memories
3. Spatial tags (concept → zone mappings)

**Do NOT persist:**
1. Current activation levels (reset each session)
2. Individual memories (session-specific)

---

## Testing Strategy

1. **Unit tests**
   - Activation boost/decay
   - Edge building from co-occurrence
   - Spreading activation through edges
   - Memory scoring

2. **Integration test**
   - Feed 20 real captions
   - Verify recall produces sensible output
   - Check temporal formatting

3. **Live test**
   - Run machine.py 30+ minutes
   - Watch [🧠] logs for memory recall
   - Verify contextual relevance

---

## Migration Path

1. Create activation_memory.py (standalone, no dependencies)
2. Add to memory.py alongside existing system
3. Wire to prompts.py (new recall method)
4. Test in parallel with old system
5. Remove old motif code once validated
6. Remove TinyLlama scoring thread

---

## Success Criteria

- [ ] Novelty scoring correlates with actual scene changes
- [ ] Boredom increases in static scenes
- [ ] Memory recall is contextually relevant
- [ ] "I remember X from Y ago" appears naturally in prompts
- [ ] Beliefs reflect actual learned associations
- [ ] No TinyLlama calls needed for memory/motifs
- [ ] Drawing prompts receive meaningful context
