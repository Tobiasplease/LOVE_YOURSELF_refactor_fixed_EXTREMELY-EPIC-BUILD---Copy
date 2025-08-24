#!/usr/bin/env python3
"""
Test the environmental pressure system to see how it affects
AI response parameters based on environmental and emotional factors.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from captioner.environmental_pressure import environmental_pressure_engine
from config.model_settings import get_model_options
from config import config


def test_environmental_pressure():
    """Test environmental pressure calculations"""
    print("Testing Environmental Pressure System")
    print("=" * 60)
    
    # Get baseline model options
    baseline_options = get_model_options(config.OLLAMA_MODEL)
    print(f"Baseline Options:")
    print(f"  Temperature: {baseline_options.get('temperature', 0.7)}")
    print(f"  Tokens: {baseline_options.get('num_predict', 100)}")
    print(f"  Stop conditions: {len(baseline_options.get('stop', []))}")
    
    test_scenarios = [
        {
            "name": "Normal State",
            "novelty": 0.3,
            "repetition": {},
            "mood": (0.1, 0.0, 0.7),  # neutral, calm, clear
            "change": 0.0,
            "time": 30
        },
        {
            "name": "High Novelty - Sharp Attention",
            "novelty": 0.9,
            "repetition": {},
            "mood": (0.2, 0.8, 0.6),  # slightly positive, high arousal
            "change": 0.6,
            "time": 45
        },
        {
            "name": "Repetition Fatigue",
            "novelty": 0.1,
            "repetition": {"high_repetition_motifs": ["messy_room", "clutter", "laptop"]},
            "mood": (-0.3, 0.4, 0.5),  # negative, moderate arousal
            "change": 0.1,
            "time": 60
        },
        {
            "name": "High Emotional Intensity",
            "novelty": 0.4,
            "repetition": {},
            "mood": (-0.6, 0.9, 0.3),  # very negative, very high arousal, unclear
            "change": 0.3,
            "time": 120
        },
        {
            "name": "Long Silence Break",
            "novelty": 0.2,
            "repetition": {},
            "mood": (0.0, -0.2, 0.8),  # neutral, low arousal, clear
            "change": 0.0,
            "time": 250  # Over 3 minutes
        }
    ]
    
    print(f"\nPressure Analysis:")
    print("-" * 60)
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Input: novelty={scenario['novelty']:.1f}, mood=({scenario['mood'][0]:.1f},{scenario['mood'][1]:.1f},{scenario['mood'][2]:.1f})")
        
        # Calculate pressure
        pressure = environmental_pressure_engine.calculate_response_pressure(
            novelty_score=scenario['novelty'],
            repetition_context=scenario['repetition'],
            mood_vector=scenario['mood'],
            environmental_change=scenario['change'],
            time_since_last=scenario['time']
        )
        
        # Apply pressure to model options
        modified_options = environmental_pressure_engine.apply_pressure_to_model_options(
            baseline_options.copy(), pressure)
        
        # Get modified system prompt
        modified_prompt = environmental_pressure_engine.get_pressure_influenced_system_prompt(
            "Base system prompt.", pressure)
        
        # Show results
        temp_change = modified_options['temperature'] - baseline_options.get('temperature', 0.7)
        token_change = modified_options['num_predict'] - baseline_options.get('num_predict', 100)
        
        print(f"  Temperature: {modified_options['temperature']:.2f} ({temp_change:+.2f})")
        print(f"  Tokens: {modified_options['num_predict']} ({token_change:+d})")
        print(f"  Brevity pressure: {pressure['brevity_pressure']:.1f}")
        print(f"  Fragmentation: {pressure['fragmentation_pressure']:.1f}")
        print(f"  System prompt: {pressure['system_prompt_modifier']}")
        
        if pressure['system_prompt_modifier'] != 'normal':
            print(f"  Modified prompt includes: '{modified_prompt.split('. ')[-1][:50]}...'")
    
    print(f"\n" + "=" * 60)
    print("PRESSURE EFFECTS:")
    print("-" * 60)
    
    # Test gut voice generation
    print(f"\nGut Voice Testing:")
    print("-" * 30)
    
    gut_scenarios = [
        {"mood": (-0.8, 0.7, 0.4), "context": "same messy room again", "desc": "Frustrated + familiar mess"},
        {"mood": (0.1, 0.9, 0.2), "context": "person appeared suddenly", "desc": "High arousal + confusion"},
        {"mood": (0.2, -0.1, 0.8), "context": "quiet afternoon scene", "desc": "Calm state"}
    ]
    
    for scenario in gut_scenarios:
        gut_prompt = environmental_pressure_engine.create_gut_voice_prompt(
            scenario["mood"], scenario["context"])
        
        if gut_prompt:
            print(f"  {scenario['desc']}: GUT VOICE TRIGGERED")
            print(f"    Context: {scenario['context']}")
            print(f"    Prompt: {gut_prompt.split('Express')[0]}...")
        else:
            print(f"  {scenario['desc']}: No gut voice (not intense enough)")
    
    print(f"\n" + "=" * 60)
    print("EXPECTED BEHAVIOR:")
    print("-" * 60)
    print("[+] High novelty → Higher temp, shorter responses, sharp attention")
    print("[+] Repetition fatigue → Brief, fragmented, impatient responses")  
    print("[+] High arousal → More spontaneous, fragmented")
    print("[+] Low clarity → Confused, scattered thoughts")
    print("[+] Environmental disruption → Interrupted thoughts")
    print("[+] No pre-written phrases - all AI-generated with pressure")
    print("=" * 60)


if __name__ == "__main__":
    test_environmental_pressure()