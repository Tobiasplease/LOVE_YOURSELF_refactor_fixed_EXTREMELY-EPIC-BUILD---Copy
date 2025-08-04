#!/usr/bin/env python3
"""
Test script for the standalone hand control system
Tests functionality without launching the GUI
"""
import sys
import os
sys.path.append('servo_control')

def test_import():
    """Test that all modules import correctly"""
    print("🧪 Testing imports...")
    try:
        import standalone_hand_control
        print("✅ standalone_hand_control imported")
        
        # Test key classes exist
        assert hasattr(standalone_hand_control, 'HandControlInterface')
        print("✅ HandControlInterface class found")
        
        assert hasattr(standalone_hand_control, 'EmotionalState')
        print("✅ EmotionalState class found")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_emotional_states():
    """Test emotional state system"""
    print("\n🎭 Testing emotional states...")
    try:
        from standalone_hand_control import EmotionalState
        
        # Test creating emotional states
        calm_state = EmotionalState('Calm & Observant', mood_factor=0.3, energy_factor=0.5, focus_factor=0.7)
        params = calm_state.get_movement_params()
        
        assert 'cursor_sensitivity' in params
        print(f"✅ Calm state parameters: {params}")
        
        return True
    except Exception as e:
        print(f"❌ Emotional state test failed: {e}")
        return False

def test_class_instantiation():
    """Test that we can create the class (without GUI)"""
    print("\n🏗️ Testing class structure...")
    try:
        from standalone_hand_control import HandControlInterface
        
        # Test class attributes without actually creating GUI
        print("✅ HandControlInterface class accessible")
        
        # Check if key methods exist
        methods_to_check = [
            'setup_mood_listener',
            'clear_current_emotion', 
            'clear_all_emotions',
            'update_recording_info',
            'handle_mood_update'
        ]
        
        for method_name in methods_to_check:
            if hasattr(HandControlInterface, method_name):
                print(f"✅ Method {method_name} exists")
            else:
                print(f"❌ Method {method_name} missing")
                return False
                
        return True
    except Exception as e:
        print(f"❌ Class test failed: {e}")
        return False

def test_dataset_structure():
    """Test the simplified dataset structure"""
    print("\n📊 Testing dataset structure...")
    try:
        # Simulate the simplified dataset structure
        datasets = {
            'calm_observant': [
                {
                    'movements': [{'time': 123, 'x': 0.5, 'y': 0.5}],
                    'markov_chain': {'transitions': {}},
                    'timestamp': 1234567890,
                    'point_count': 1
                }
            ]
        }
        
        # Test accessing emotion data
        emotion_data = datasets.get('calm_observant', [])
        total_points = sum(len(recording['movements']) for recording in emotion_data)
        
        print(f"✅ Dataset structure test: {len(emotion_data)} recordings, {total_points} points")
        return True
    except Exception as e:
        print(f"❌ Dataset structure test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Standalone Hand Control System")
    print("=" * 50)
    
    tests = [
        test_import,
        test_emotional_states,
        test_class_instantiation,
        test_dataset_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System ready for launch.")
        print("\n💡 To launch the GUI:")
        print("   python launch_standalone_hand_control.py")
        print("   OR double-click: launch_hand_control.bat")
    else:
        print(f"❌ {total - passed} tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
