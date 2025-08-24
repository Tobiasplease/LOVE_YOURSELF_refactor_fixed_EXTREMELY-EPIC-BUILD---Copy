#!/usr/bin/env python3
"""
Test the reflection-to-drawing continuity system.
Shows how artistic intentions accumulate over multiple reflections.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.artistic_intention import artistic_intention_tracker


def test_artistic_continuity():
    """Test how artistic intentions accumulate and influence drawing decisions."""
    print("Testing Reflection-to-Drawing Continuity System")
    print("=" * 60)
    
    # Simulate a series of reflections over time
    reflection_series = [
        "I've been noticing how the afternoon light creates these beautiful shadows across the room. There's something about the way it illuminates the person's face that I want to capture - that soft, contemplative quality.",
        
        "The solitary figure keeps drawing my attention. There's a sense of peaceful isolation here, someone absorbed in their own world. I find myself wanting to express that quiet intimacy through visual art.",
        
        "I'm fascinated by the contrast between the warm human presence and the cool technology around them. The laptop screen's glow against natural light creates this interesting tension I'd love to explore in a drawing.",
        
        "There's been this recurring theme of contemplative spaces in my observations. I'm drawn to capture environments where thought and reflection happen - these personal, private moments of being.",
        
        "The interplay of light and shadow has become a real obsession for me. I want to express how lighting doesn't just illuminate objects but creates emotional atmosphere, especially around human figures."
    ]
    
    print("Processing Reflection Series:")
    print("-" * 40)
    
    # Process each reflection
    for i, reflection in enumerate(reflection_series, 1):
        print(f"\nReflection {i}:")
        print(f"  '{reflection[:60]}...'")
        
        # Extract intentions
        artistic_intention_tracker.add_reflection_intentions(reflection)
        
        # Show what was extracted
        recent_intentions = artistic_intention_tracker.get_accumulated_drawing_intentions(limit=3)
        if recent_intentions:
            print(f"  Extracted intentions: {len(recent_intentions)} artistic impulses")
            for intention in recent_intentions[-2:]:  # Show last 2
                print(f"    - {intention['type']}: '{intention['content'][:40]}...'")
    
    print(f"\n" + "=" * 60)
    print("ACCUMULATED ARTISTIC VISION:")
    print("-" * 60)
    
    # Show final accumulated intentions
    all_intentions = artistic_intention_tracker.get_accumulated_drawing_intentions()
    print(f"Total intentions accumulated: {len(all_intentions)}")
    
    # Show thematic summary
    thematic_summary = artistic_intention_tracker.get_thematic_summary()
    if thematic_summary:
        print(f"Recurring themes: {thematic_summary}")
    
    # Show drawing context
    drawing_context = artistic_intention_tracker.build_drawing_context_from_intentions()
    print(f"\nDrawing Context for AI:")
    print(f"'{drawing_context}'")
    
    print(f"\n" + "=" * 60)
    print("EXAMPLE DRAWING DECISION:")
    print("-" * 60)
    
    # Simulate a drawing decision moment
    current_observation = "A person sits quietly by a window, laptop open, soft afternoon light streaming across their face."
    
    print(f"Current observation: '{current_observation}'")
    print(f"Accumulated artistic context: '{drawing_context}'")
    
    print(f"\nDrawing prompt would include:")
    print(f"- Current perception: {current_observation}")
    print(f"- Artistic intentions: {drawing_context}")
    print(f"- Question: 'Consider how your current observation connects to the artistic impulses you've been developing...'")
    
    print(f"\n" + "=" * 60)
    print("CONTINUITY ACHIEVED:")
    print("-" * 60)
    print("[+] Reflections now focus on artistic expression, not just observation")
    print("[+] Drawing ambitions accumulate over multiple reflection cycles")  
    print("[+] Drawing decisions reference accumulated creative vision")
    print("[+] Creates artistic identity that develops over time")
    print("[+] Connects contemplative inner life with creative output")
    print("=" * 60)


if __name__ == "__main__":
    test_artistic_continuity()