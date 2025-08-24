#!/usr/bin/env python3
"""
Test stop conditions to ensure they don't cause mid-sentence truncation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_settings import get_model_options
from config import config
from captioner.environmental_pressure import environmental_pressure_engine


def test_stop_conditions():
    """Test different stop conditions to find problematic ones."""
    print("Testing Stop Conditions for Mid-Sentence Truncation")
    print("=" * 60)
    
    # Test different prompt types and their stop conditions
    base_options = get_model_options(config.OLLAMA_MODEL)
    print(f"Base model: {config.OLLAMA_MODEL}")
    print(f"Base stop conditions: {base_options.get('stop', [])}")
    
    test_scenarios = [
        {
            "name": "Normal Caption (default)",
            "prompt_type": "normal",
            "pressure": None
        },
        {
            "name": "Environmental Description", 
            "prompt_type": "environmental",
            "pressure": None
        },
        {
            "name": "Reflection",
            "prompt_type": "reflection", 
            "pressure": None
        },
        {
            "name": "Drawing Prompt",
            "prompt_type": "drawing",
            "pressure": None
        },
        {
            "name": "High Brevity Pressure",
            "prompt_type": "normal",
            "pressure": {"brevity_pressure": 0.9, "fragmentation_pressure": 0.3}
        },
        {
            "name": "High Fragmentation Pressure", 
            "prompt_type": "normal",
            "pressure": {"brevity_pressure": 0.3, "fragmentation_pressure": 0.8}
        }
    ]
    
    print(f"\nStop Condition Analysis:")
    print("-" * 60)
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        
        # Simulate the model option processing
        options = base_options.copy()
        
        # Apply prompt type modifications (simulating model_wrapper logic)
        if scenario['prompt_type'] == "environmental":
            options["num_predict"] = 300
            options["stop"] = ["\n\nUser:", "\n\nHuman:", "\n\nAssistant:"]
        elif scenario['prompt_type'] == "reflection":
            options["num_predict"] = 350
            options["stop"] = ["\n\nUser:", "\n\nHuman:", "\n\nAssistant:", "\n\n---"]
        elif scenario['prompt_type'] == "drawing":
            options["num_predict"] = 200
            options["stop"] = ["\n\nUser:", "\n\nHuman:", "\n\nAssistant:"]
        elif scenario['prompt_type'] == "normal":
            # New improved normal stop conditions
            options["stop"] = [
                "\n\nUser:", "\n\nHuman:", "\n\nAssistant:",
                "What could possibly be so captivating",
                "Am I simply their creation, brought to life"
            ]
        
        # Apply environmental pressure if specified
        if scenario['pressure']:
            options = environmental_pressure_engine.apply_pressure_to_model_options(
                options, scenario['pressure'])
        
        tokens = options.get('num_predict', 100)
        stops = options.get('stop', [])
        
        print(f"  Tokens: {tokens}")
        print(f"  Stop conditions: {len(stops)} total")
        
        # Analyze stop conditions for truncation risk
        risky_stops = []
        for stop in stops:
            if stop in [". ", "... ", "!", "?", " but", " and", " the"]:
                risky_stops.append(stop)
        
        if risky_stops:
            print(f"  ⚠️  RISKY STOPS: {risky_stops} (may cause mid-sentence truncation)")
        else:
            print(f"  ✅ Safe stop conditions")
        
        # Show a few example stops
        if len(stops) > 3:
            print(f"  Examples: {stops[:3]}...")
        else:
            print(f"  All stops: {stops}")
    
    print(f"\n" + "=" * 60)
    print("PROBLEMATIC STOP PATTERNS:")
    print("-" * 60)
    print("❌ '. ' - stops after ANY sentence")
    print("❌ '! ' - stops after ANY exclamation") 
    print("❌ '? ' - stops after ANY question")
    print("❌ ' but' - stops before natural transitions")
    print("❌ ' and' - stops before natural conjunctions")
    print()
    print("✅ SAFE ALTERNATIVES:")
    print("✅ '.\\n' - stops at sentence + newline")
    print("✅ '. I' - stops at sentence starting new thought")
    print("✅ '—' - stops at dash (natural interruption)")
    print("✅ '... but' - stops at ellipsis + transition")
    print("=" * 60)


if __name__ == "__main__":
    test_stop_conditions()