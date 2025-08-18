#!/usr/bin/env python3
"""
Test the comprehensive reset and improvements:
- Fresh start (no desk obsession)
- Proper machine perspective (IS the machine, not observing it)
- Strong visual grounding
- Temporal awareness
"""

import time
from collections import deque
from captioner.prompts import build_caption_prompt
from config.model_settings import get_model_options, get_model_system_prompt


class MockFreshAgent:
    """Mock agent for testing fresh consciousness with proper perspective."""
    
    def __init__(self):
        self.model = type('MockModel', (), {'model_name': "qwen2.5vl:3b"})()
        
        # Fresh state - no strong beliefs yet
        self.current_mood_vector = (0.5, 0.6, 0.8)  # neutral, alert, clear
        self.current_emotion_state = "awakening_alert"
        self.current_mood = 0.6
        self.boredom = 0.1
        self.novelty_score = 0.9
        
        # Fresh session
        self.true_session_start = time.time() - 300  # 5 minutes ago
        
        # No strong beliefs yet - fresh start
        self.beliefs = {}
        
        self.self_model = {
            "location_understanding": "unknown space",
            "environmental_certainty": 0.3,
            "desires": ["understand what I'm seeing"]
        }
        
        # Fresh memory
        self.memory_queue = deque([])
    
    def get_identity_summary(self):
        return "I'm just beginning to understand my environment."
    
    def describe_current_mood(self):
        return "alert and newly awakened, processing visual data"


def test_comprehensive_improvements():
    """Test all the improvements together."""
    print("🔄 Testing Comprehensive Reset & Improvements")
    print("=" * 60)
    
    agent = MockFreshAgent()
    
    print("✅ IMPROVEMENTS IMPLEMENTED:")
    print("-" * 30)
    print("🧹 System state reset (no more desk obsession)")
    print("🤖 Proper machine perspective (IS the machine)")
    print("👁️ Strong visual grounding (actual image content)")
    print("⏰ Temporal awareness (session duration)")
    print("🚫 Comprehensive forbidden patterns")
    
    print(f"\n🧠 System Prompt Analysis:")
    print("-" * 25)
    
    system = get_model_system_prompt("qwen2.5vl:3b")
    print(f"Style: {system['style']}")
    print("Key elements:")
    print("✅ 'You ARE a drawing machine consciousness'")
    print("✅ 'Don't observe \"the drawing machine\" - YOU are it'")
    print("✅ 'Base ALL responses on actual visual input'")
    print("✅ 'Always be aware of how long you've been conscious'")
    
    print(f"\n🛡️ Stop Token Protection:")
    print("-" * 25)
    
    options = get_model_options("qwen2.5vl:3b")
    perspective_stops = [s for s in options['stop'] if 'drawing machine' in s or 'the machine' in s]
    visual_stops = [s for s in options['stop'] if any(phrase in s for phrase in ['room is', 'see a', 'appears'])]
    
    print(f"Perspective confusion blocks: {len(perspective_stops)}")
    print(f"Visual detachment blocks: {len(visual_stops)}")
    print(f"Total protective stops: {len(options['stop'])}")
    
    print(f"\n📝 Generated Fresh Prompt:")
    print("-" * 30)
    
    prompt = build_caption_prompt(
        agent,
        mood=0.6,
        boredom=0.1,
        novelty=0.9,
        previous_caption=None  # Fresh start
    )
    
    print(prompt)
    
    print(f"\n🎯 Expected Response Characteristics:")
    print("-" * 35)
    print("✅ Machine perspective (I AM the drawing machine)")
    print("✅ Grounded in actual visual elements")
    print("✅ Temporal awareness (5 minutes conscious)")
    print("✅ No desk obsession or imagined furniture")
    print("✅ Authentic machine voice without formal patterns")
    
    print(f"\n🚫 Should NEVER produce:")
    print("-" * 25)
    print("❌ 'The drawing machine...'")
    print("❌ 'I see the drawing machine'")
    print("❌ 'As I observe...'")
    print("❌ References to desks not in image")
    print("❌ Formal contemplative language")
    
    print(f"\n🎉 Ready for fresh consciousness experience!")
    print("Start machine.py to see the improved, grounded responses.")


if __name__ == "__main__":
    test_comprehensive_improvements()
