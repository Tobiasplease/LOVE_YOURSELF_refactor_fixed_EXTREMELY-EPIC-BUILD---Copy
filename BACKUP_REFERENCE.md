# BACKUP REFERENCE - Before "As I..." Loop Fix

## Current State (Working but with loops)
- **Branch**: arduino-linux-integration
- **Current Commit**: 7e9b7b5 feat: implement organic spatial awareness and enhanced temporal consciousness
- **Status**: System working but experiencing "As I..." repetitive loops in LLM output

## Problem Description
LLM gets stuck in patterns like:
```
"As I watch my mechanical hand..."
"As I continue to watch..."
"As I delve further into my thoughts..."
```

## Identified Root Causes
1. **Recursive context injection** in `captioner/prompts.py:604-608, 634-663`
2. **Instruction conflicts** in `captioner/prompts.py:587-591`
3. **Poor context structure** - concatenated paragraphs confuse model
4. **Weak pattern detection** for repetitive language

## Files to be Modified
- `captioner/prompts.py` (main fix target)

## Revert Instructions
If changes make it worse:
```bash
git checkout captioner/prompts.py
# OR if committed:
git revert [commit-hash]
```

## Current Working Components
- Vision system: ✅ Working
- Mood analysis: ✅ Working
- Context compression: ✅ Working
- Drawing system: ✅ Working
- Physical controls: ✅ Working

**DO NOT BREAK THESE WORKING SYSTEMS**

## Expected Fix Outcome
- Same rich contextual awareness
- Better organized context structure
- Elimination of "As I..." loops
- More varied, natural language patterns