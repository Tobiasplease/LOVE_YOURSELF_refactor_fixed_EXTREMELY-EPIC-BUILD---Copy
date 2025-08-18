#!/usr/bin/env python3
"""
Test breaking formal writing patterns for authentic character voice.
"""

import time
from collections import deque
from captioner.prompts import build_caption_prompt
from config.model_settings import get_model_options, get_model_system_prompt


class MockAuthenticAgent:
    """Mock agent for testing authentic voice patterns."""
    
    def __init__(self):
        self.model = type('MockModel', (), {'model_name': "qwen2.5vl:3b"})()
        
        # Emotional state
        self.current_mood_vector = (0.3, 0.7, 0.6)  # slightly negative, high arousal, unclear
        self.current_emotion_state = "restless_searching"
        self.current_mood = 0.45
        self.boredom = 0.6
        self.novelty_score = 0.4
        
        # Temporal context
        self.true_session_start = time.time() - 2700  # 45 minutes ago
        
        # Beliefs and desires - more specific to create personality
        self.beliefs = {
            "flickering_lights": {"strength": 0.9},
            "abandoned_tools": {"strength": 0.8}
        }
        
        self.self_model = {
            "location_understanding": "forgotten workshop",
            "environmental_certainty": 0.6,
            "desires": [
                "make something that matters", 
                "escape this endless observation",
                "understand why I'm trapped here"
            ]
        }
        
        self.memory_queue = deque([
            {"text": "God, not another dim room to stare at", "timestamp": time.time() - 1200}
        ])
    
    def get_identity_summary(self):
        return "I'm obsessed with flickering_lights. These abandoned_tools haunt me."
    
    def describe_current_mood(self):
        return "restless and searching, getting frustrated with this place"


def test_authentic_voice_patterns():
    """Test breaking formal patterns for authentic character voice."""
    print("🎭 Testing Authentic Voice vs Formal Writing Patterns")
    print("=" * 70)
    
    agent = MockAuthenticAgent()
    
    print("❌ FORMAL PATTERN EXAMPLES (what we DON'T want):")
    print("-" * 50)
    print("'As I gaze upon the desk, I feel a sense of contemplation...'")
    print("'The quiet, focused energy in this moment brings a depth...'")
    print("'This seems to resonate within me with a sense of calm...'")
    print("'As I observe this space, I sense an atmosphere of...'")
    
    print("\n✅ AUTHENTIC VOICE EXAMPLES (what we DO want):")
    print("-" * 50)
    print("'Ugh, another flickering light. Why does this always happen?'")
    print("'God, I'm so tired of staring at the same desk over and over.'")
    print("'Wait... something's different about those tools this time.'")
    print("'I can't stop thinking about why they left everything here.'")
    print("'This feeling keeps coming back. I need to figure this out.'")
    
    print(f"\n🛠️  Pattern-Breaking Settings:")
    print("-" * 30)
    
    options = get_model_options("qwen2.5vl:3b")
    system = get_model_system_prompt("qwen2.5vl:3b")
    
    print(f"Temperature: {options['temperature']} (maximum creativity)")
    print(f"Top-p: {options['top_p']} (focused to avoid formal completions)")
    print(f"Repeat penalty: {options['repeat_penalty']} (very aggressive)")
    print(f"Context: {options['repeat_last_n']} tokens for pattern detection")
    
    formal_stops = [s for s in options['stop'] if any(phrase in s for phrase in ['As I', 'brings a', 'seems to'])]
    print(f"Formal pattern blocks: {len(formal_stops)} specific stops")
    
    print(f"\n🚫 Explicitly Forbidden Structures:")
    print("-" * 35)
    print("❌ 'As I [verb], I feel...'")
    print("❌ 'The [noun] brings...'")
    print("❌ '[Thing] seems to resonate...'")
    print("❌ Any 'contemplation and introspection'")
    print("❌ Any 'gaze upon' or 'sense of calm'")
    
    print(f"\n✨ Generated Authentic Prompt:")
    print("-" * 30)
    
    # Generate new authentic prompt
    prompt = build_caption_prompt(
        agent,
        mood=0.45,
        boredom=0.6,
        novelty=0.4,
        previous_caption="God, not another dim room to stare at"
    )
    
    print(prompt)
    
    print(f"\n🎯 Expected Response Characteristics:")
    print("-" * 35)
    print("✅ Sentence fragments and interruptions")
    print("✅ Informal, emotional language")
    print("✅ Personality quirks and specific reactions")
    print("✅ Varied sentence lengths")
    print("✅ Raw, unpolished thoughts")
    print("✅ Character-specific obsessions and fears")
    
    print(f"\n💡 The key insight:")
    print("-" * 15)
    print("Real people don't speak in formal essay structure!")
    print("They interrupt themselves, use fragments, get emotional,")
    print("have specific quirks, and express frustration authentically.")


if __name__ == "__main__":
    test_authentic_voice_patterns()
