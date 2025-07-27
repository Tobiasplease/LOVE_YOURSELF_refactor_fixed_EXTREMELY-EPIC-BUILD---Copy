#!/usr/bin/env python3
"""
Simulate the temporal awareness issues and demonstrate the fixes
"""

import sys
import os
import time
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.state_manager import state_manager
from captioner.captioner import Captioner
from mood.mood import MoodEngine
from utils.continuity import describe_duration, now

def simulate_temporal_scenario():
    """Simulate the scenario you described with temporal disconnection"""
    print("=== TEMPORAL AWARENESS SIMULATION ===\n")
    
    # Initialize components
    mood_engine = MoodEngine()
    captioner = Captioner()
    
    # Load any existing state
    previous_state = state_manager.load_session_state()
    if previous_state:
        state_manager.apply_state_to_captioner(previous_state, captioner)
        state_manager.apply_state_to_mood_engine(previous_state, mood_engine)
        captioner.memory_loaded_from_previous = True
        
        save_time = previous_state["metadata"]["save_time"]
        time_since_last = describe_duration(save_time)
        print(f"[🔄] Continuing from {time_since_last} ago")
        print(f"[🧠] Restored: {len(captioner.beliefs)} beliefs, {len(captioner.motif_counter)} motifs")
    
    # Simulate observations with temporal context
    scenarios = [
        {
            "observation": "User is drawing on unmade bed",
            "reality_context": "User stopped drawing 5 minutes ago",
            "motifs": ["creativity", "personal_space", "art", "bedroom"]
        },
        {
            "observation": "Dog is lying on bed looking at user",
            "reality_context": "Dog left the room 5 minutes ago",
            "motifs": ["companion", "dog", "loyalty", "pet"]
        },
        {
            "observation": "Room is dimly lit by bedside lamp",
            "reality_context": "Still accurate - lighting unchanged",
            "motifs": ["ambient_lighting", "peaceful", "creative_space"]
        },
        {
            "observation": "User wearing glasses, focused on work",
            "reality_context": "User removed glasses 2 minutes ago",
            "motifs": ["focus", "glasses", "concentration", "studious"]
        }
    ]
    
    print(f"\n[📊] Simulating temporal observations:\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}] System Observation: '{scenario['observation']}'")
        print(f"    Reality Context: {scenario['reality_context']}")
        print(f"    Detected Motifs: {', '.join(scenario['motifs'])}")
        
        # Simulate system processing
        captioner.observe(scenario['observation'])
        
        # Update motif counts
        for motif in scenario['motifs']:
            captioner.motif_counter[motif] += 1
            captioner.motif_last_seen[motif] = now()
            if motif not in captioner.motif_first_seen:
                captioner.motif_first_seen[motif] = now() - random.randint(3600, 86400)  # 1-24 hours ago
        
        # Check for belief formation
        captioner.update_beliefs()
        
        print(f"    → New motif counts: {dict(captioner.motif_counter)}")
        print(f"    → Current beliefs: {len(captioner.beliefs)}")
        
        # Simulate time passing (system lag)
        time.sleep(0.5)
        print()
    
    # Show temporal disconnection issue
    print(f"[⚠️] TEMPORAL DISCONNECTION ISSUE:")
    print(f"    System believes dog is still present (5 min delay)")
    print(f"    System believes user is still drawing (5 min delay)")
    print(f"    No temporal context - observations treated as 'now'")
    print(f"    Motifs accumulate without time awareness\n")
    
    # Show identity state before improvements
    print(f"[🧩] Current Beliefs ({len(captioner.beliefs)}):")
    for motif, data in captioner.beliefs.items():
        age = describe_duration(data['first_formed'])
        print(f"    - {motif}: {data['strength']:.2f} strength (formed {age} ago)")
    
    print(f"\n[🔄] Current Motifs ({len(captioner.motif_counter)}):")
    for motif, count in captioner.motif_counter.items():
        if motif in captioner.motif_first_seen:
            age = describe_duration(captioner.motif_first_seen[motif])
            print(f"    - {motif}: {count} occurrences (first seen {age} ago)")
    
    # Generate current identity summary
    if hasattr(captioner, 'get_identity_summary'):
        print(f"\n[🎭] Current Identity:")
        print(f"    {captioner.get_identity_summary()}")
    
    # Show what SHOULD happen with temporal awareness
    print(f"\n[✨] WITH TEMPORAL AWARENESS (Future Enhancement):")
    print(f"    - Observations tagged with timestamps")
    print(f"    - Recent vs historical context differentiated")
    print(f"    - 'Dog WAS here 5 minutes ago' vs 'Dog IS here now'")
    print(f"    - Motif confidence decays over time")
    print(f"    - Identity reflects 'what I remember' vs 'what I see now'")
    print(f"    - Belief formation considers recency and relevance")
    
    # Save state with improvements
    print(f"\n[💾] Saving enhanced state...")
    success = state_manager.save_session_state(captioner, mood_engine)
    if success:
        print(f"[✅] State saved with temporal context preserved")
    
    return captioner, mood_engine

if __name__ == "__main__":
    simulate_temporal_scenario()
