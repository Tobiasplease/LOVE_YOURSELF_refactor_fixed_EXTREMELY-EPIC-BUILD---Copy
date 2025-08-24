#!/usr/bin/env python3
"""
Quick system health check to identify what's blocking caption generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_system_health():
    """Test key system components"""
    print("System Health Check")
    print("=" * 40)
    
    # Test 1: Import environmental pressure system
    try:
        from captioner.environmental_pressure import environmental_pressure_engine
        print("[OK] Environmental pressure system imported")
        
        # Test basic pressure calculation
        pressure = environmental_pressure_engine.calculate_response_pressure(
            novelty_score=0.5,
            repetition_context={},
            mood_vector=(0.0, 0.0, 0.5)
        )
        print(f"[OK] Pressure calculation works: {pressure['system_prompt_modifier']}")
        
    except Exception as e:
        print(f"[ERROR] Environmental pressure system: {e}")
        return
    
    # Test 2: Simple motif extractor 
    try:
        from utils.simple_motif_extractor_fixed import get_simple_motif_extractor
        extractor = get_simple_motif_extractor()
        motifs = extractor.extract_motifs("person sitting with laptop")
        print(f"[OK] Simple motif extractor works: {list(motifs)}")
        
    except Exception as e:
        print(f"[ERROR] Simple motif extractor: {e}")
        return
    
    # Test 3: Pattern recognition with new system
    try:
        from utils.pattern_recognition import PatternRecognitionEngine
        pattern_engine = PatternRecognitionEngine()
        result = pattern_engine.analyze_caption("A person sits at a desk with a laptop")
        print(f"[OK] Pattern recognition works: novelty={result.get('novelty', 'unknown')}")
        
    except Exception as e:
        print(f"[ERROR] Pattern recognition: {e}")
        return
    
    # Test 4: Ollama connection
    try:
        from utils.ollama import query_ollama
        from config import config
        response = query_ollama(
            "Test",
            model=config.OLLAMA_MODEL,
            options={"num_predict": 5},
            timeout=10
        )
        print(f"[OK] Ollama connection works: '{response[:20]}...'")
        
    except Exception as e:
        print(f"[ERROR] Ollama connection: {e}")
        return
    
    # Test 5: Model wrapper with environmental pressure
    try:
        from captioner.model_wrapper import MultimodalModel
        
        # Create a dummy memory reference
        class DummyMemory:
            def __init__(self):
                self.novelty_score = 0.3
                self.current_mood_vector = (0.0, 0.0, 0.5)
                self.current_mood = 0.5
                self.boredom = 0.0
                self.last_caption_time = 0
                self.motif_counter = {}
        
        model = MultimodalModel(memory_ref=DummyMemory())
        print("[OK] Model wrapper initialized with environmental pressure")
        
    except Exception as e:
        print(f"[ERROR] Model wrapper: {e}")
        return
        
    print("\n" + "=" * 40)
    print("DIAGNOSIS:")
    print("- All core components working")
    print("- Issue likely in:")
    print("  1. Caption timing/intervals")
    print("  2. Threading conflicts") 
    print("  3. Memory reference issues")
    print("  4. Environmental pressure parameter passing")
    print("=" * 40)

if __name__ == "__main__":
    test_system_health()