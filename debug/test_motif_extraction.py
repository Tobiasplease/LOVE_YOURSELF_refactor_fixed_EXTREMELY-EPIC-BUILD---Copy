#!/usr/bin/env python3
"""
Test what motifs are actually extracted from captions about people.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.memory import MemoryMixin
from utils.thematic_analyzer import get_thematic_analyzer


def test_motif_extraction():
    """Test what motifs get extracted from person-related captions"""
    print("Testing Motif Extraction")
    print("=" * 50)
    
    # Create memory system
    memory = MemoryMixin()
    
    # Test captions about people
    test_captions = [
        "The person I know well is here with me",
        "A person is sitting at the desk",
        "The person seems peaceful and calm",
        "I see a person working on their laptop",
        "The familiar person brings comfort to this space"
    ]
    
    print(f"\n1. Testing {len(test_captions)} person-related captions...")
    
    for i, caption in enumerate(test_captions):
        print(f"\nCaption {i+1}: {caption}")
        
        # Store the caption
        memory.observe(
            text=caption,
            mood=(0.7, 0.2, 0.8),
            emotion_state="calm_observant",
            mood_vector=(0.7, 0.2, 0.8)
        )
        
        # Check current motifs
        current_motifs = list(memory.current_motifs)
        print(f"   Current motifs: {current_motifs}")
        
        # Check motif counter
        relevant_motifs = {k: v for k, v in memory.motif_counter.items() 
                          if 'person' in k.lower() or 'human' in k.lower() or 'people' in k.lower()}
        if relevant_motifs:
            print(f"   Person-related motifs: {relevant_motifs}")
        else:
            print("   No person-related motifs found")
    
    print(f"\n2. Final motif counter:")
    all_motifs = dict(memory.motif_counter.most_common(10))
    for motif, count in all_motifs.items():
        print(f"   {motif}: {count}")
    
    print(f"\n3. Emotional memory check:")
    if hasattr(memory, 'emotional_memory_bank'):
        motif_emotions = memory.emotional_memory_bank.motif_emotions
        print(f"   Stored motif emotions: {list(motif_emotions.keys())}")
        
        for motif_name, motif_emotion in motif_emotions.items():
            if 'person' in motif_name.lower():
                print(f"   {motif_name}: valence={motif_emotion.cumulative_valence:.2f}, comfort={motif_emotion.comfort_level:.2f}")
    
    print("\n4. Testing thematic analyzer directly:")
    analyzer = get_thematic_analyzer(interval=1)  # Process every caption
    for caption in test_captions[:3]:
        themes = analyzer.add_caption(caption)
        print(f"   '{caption[:30]}...': themes={themes}")
    
    print("=" * 50)


if __name__ == "__main__":
    test_motif_extraction()