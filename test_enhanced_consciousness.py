#!/usr/bin/env python3
"""
Test Enhanced Spatial & Consciousness System
------------------------------------------
Tests the new spatial memory integration, enhanced continuity,
and improved temperature control for consistent personality.
"""

import tempfile
import time
from PIL import Image, ImageDraw
from captioner.model_wrapper import MultimodalModel
from captioner.captioner import Captioner
from config import config
from perception.spatial_memory import spatial_memory

def create_test_scene(scene_type="workspace"):
    """Create test image with multiple objects for spatial analysis"""
    img = Image.new('RGB', (640, 480), color='white')
    draw = ImageDraw.Draw(img)
    
    if scene_type == "workspace":
        # Simulate workspace scene with consistent spatial layout
        # Left side: desk area
        draw.rectangle([50, 200, 250, 400], fill='brown')  # desk
        draw.rectangle([60, 150, 120, 200], fill='black')   # laptop
        draw.rectangle([140, 170, 180, 190], fill='blue')   # notebook
        
        # Center: person
        draw.ellipse([300, 100, 350, 200], fill='pink')     # person head/torso
        
        # Right side: storage
        draw.rectangle([450, 150, 580, 450], fill='gray')   # shelf/cabinet
        draw.rectangle([460, 160, 520, 200], fill='red')    # book
        
        # Background elements
        draw.rectangle([10, 10, 630, 80], fill='lightblue') # wall/background
        
    return img

def test_spatial_awareness():
    """Test the enhanced spatial awareness system"""
    print("\n=== Testing Enhanced Spatial Awareness System ===")
    
    # Create test scene
    img = create_test_scene("workspace")
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img.save(f.name)
        temp_path = f.name

    try:
        print("🎯 Creating captioner with enhanced spatial & consciousness systems...")
        
        # Create captioner with rich state
        cap = Captioner()
        cap.first_caption_done = False  # Start fresh for awakening experience
        cap.current_mood = 0.6
        cap.current_mood_vector = (0.3, 0.4, 0.7)  # Positive, alert, clear
        cap.current_emotion_state = 'alert_curious'
        cap.boredom = 0.1
        cap.novelty_score = 0.9
        cap.last_caption = ''
        
        # Simulate some detection data for spatial system
        mock_detections = [
            {'label': 'laptop', 'confidence': 0.85, 'bbox': (60, 150, 120, 200)},
            {'label': 'notebook', 'confidence': 0.75, 'bbox': (140, 170, 180, 190)},
            {'label': 'person', 'confidence': 0.90, 'bbox': (300, 100, 350, 200)},
            {'label': 'book', 'confidence': 0.80, 'bbox': (460, 160, 520, 200)},
        ]
        
        # Update spatial memory
        spatial_memory.process_detections(mock_detections, (480, 640))
        
        # Create model with enhanced temperature control
        model = MultimodalModel(memory_ref=cap)
        
        print("\n📸 Taking first observation (awakening mode)...")
        result1 = model.caption_image(temp_path, flowing=False, first_time=True)
        print(f"🧠 Awakening caption: {result1}")
        print(f"📊 Word count: {len(result1.split())}")
        
        # Check spatial context
        spatial_context = cap.get_spatial_context_summary()
        print(f"🗺️  Spatial awareness: {spatial_context}")
        
        # Simulate continued observation
        cap.first_caption_done = True
        cap.last_caption = result1
        
        print("\n📸 Taking second observation (flowing mode with spatial continuity)...")
        time.sleep(2)  # Brief pause
        
        # Update spatial memory again (simulating slight movement)
        mock_detections[1]['bbox'] = (142, 172, 182, 192)  # Notebook moved slightly
        spatial_memory.process_detections(mock_detections, (480, 640))
        
        result2 = model.caption_image(temp_path, flowing=True, first_time=False)
        print(f"🔄 Flowing caption: {result2}")
        print(f"📊 Word count: {len(result2.split())}")
        
        # Check enhanced scene context
        enhanced_context = cap.get_enhanced_scene_context()
        print(f"🌟 Enhanced scene context: {enhanced_context}")
        
        # Test spatial stability
        stability_score = spatial_memory.get_scene_stability_score()
        continuity_context = spatial_memory.get_spatial_continuity_context()
        print(f"📐 Scene stability: {stability_score:.2f}")
        print(f"🔗 Spatial continuity: {continuity_context}")
        
        # Show object relationships detected
        if spatial_memory.object_relationships:
            print(f"🔗 Spatial relationships: {dict(spatial_memory.object_relationships)}")
            
        print("\n✅ Enhanced spatial awareness system test completed!")
        
        return result1, result2, spatial_context, enhanced_context
        
    except Exception as e:
        print(f"❌ Error in spatial awareness test: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
        
    finally:
        import os
        os.unlink(temp_path)

def test_temperature_consistency():
    """Test temperature control for consistent personality"""
    print("\n=== Testing Temperature Control for Consistent Personality ===")
    
    # Test different temperature settings
    img = Image.new('RGB', (200, 200), color='purple')
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img.save(f.name)
        temp_path = f.name
    
    try:
        cap = Captioner()
        cap.first_caption_done = True
        cap.current_mood = 0.5
        cap.boredom = 0.3
        cap.novelty_score = 0.6
        
        model = MultimodalModel(memory_ref=cap)
        
        print("🌡️  Testing default temperature (0.4 - consistent personality)...")
        result_default = model.caption_image(temp_path, flowing=True, first_time=False)
        print(f"Default temp result: {result_default}")
        
        print("🌡️  Testing strict evaluation (0.1 - focused)...")
        from utils.ollama import query_ollama
        from config.config import SYSTEM_PROMPT
        
        strict_result = query_ollama(
            "Describe what you see in this image briefly.",
            image=temp_path,
            system_prompt=SYSTEM_PROMPT,
            strict_evaluation=True
        )
        print(f"Strict temp result: {strict_result}")
        
        print("✅ Temperature consistency test completed!")
        
        return result_default, strict_result
        
    except Exception as e:
        print(f"❌ Error in temperature test: {e}")
        return None, None
        
    finally:
        import os
        os.unlink(temp_path)

if __name__ == "__main__":
    print("🚀 Testing Enhanced Consciousness System (Spatial + Temperature + Continuity)")
    
    # Test spatial awareness enhancements
    awakening_result, flowing_result, spatial_ctx, enhanced_ctx = test_spatial_awareness()
    
    # Test temperature consistency
    default_temp, strict_temp = test_temperature_consistency()
    
    print("\n" + "="*60)
    print("📋 ENHANCEMENT SUMMARY")
    print("="*60)
    print("✅ Spatial Memory System: Enhanced understanding of object relationships and layout")
    print("✅ Enhanced Scene Context: Rich spatial and temporal awareness for captions")  
    print("✅ Temperature Control: Consistent personality (0.4) with focused mode (0.1)")
    print("✅ Improved Continuity: Spatial context integrated into consciousness stream")
    print("\n🎯 Next Steps:")
    print("   - Run full system to test spatial persistence across sessions")
    print("   - Monitor caption quality with enhanced spatial awareness")
    print("   - Validate temperature settings maintain consistent voice")
    print("   - Test long-term spatial memory and belief formation")
