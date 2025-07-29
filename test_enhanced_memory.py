#!/usr/bin/env python3
"""
Test the enhanced memory architecture
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from captioner.memory import MemoryMixin
import time

def test_enhanced_memory():
    print("🧠 Testing Enhanced Memory Architecture")
    print("=" * 50)
    
    # Create memory instance
    memory = MemoryMixin()
    
    # Simulate some observations
    observations = [
        "A person sits quietly in the room",
        "The room appears peaceful and still",
        "Someone moves near the window",
        "The space feels familiar and comfortable",
        "A person reads in the corner",
        "Light streams through the window beautifully"
    ]
    
    print("📝 Adding observations...")
    for i, obs in enumerate(observations):
        memory.observe(obs, mood=0.5 + (i * 0.1), file=f"test_{i}.jpg")
        print(f"   {i+1}. {obs}")
        time.sleep(0.1)  # Small delay
    
    print(f"\n🔍 Memory Queue Length: {len(memory.memory_queue)}")
    print(f"📊 Motif Counter: {dict(memory.motif_counter)}")
    
    # Test compression
    print(f"\n🗜️ Testing compression...")
    memory._last_compression_time = 0  # Force compression
    memory.compress_memories_if_needed()
    
    print(f"🤝 Relationship Patterns: {memory.relationship_patterns}")
    print(f"🧭 Identity Core: {memory.identity_core}")
    
    # Test identity summary
    print(f"\n🆔 Identity Summary:")
    print(f"   Original: {memory.get_identity_summary()}")
    print(f"   Enhanced: {memory.get_enhanced_identity_summary()}")
    
    print("\n✅ Enhanced memory test complete!")

if __name__ == "__main__":
    test_enhanced_memory()
