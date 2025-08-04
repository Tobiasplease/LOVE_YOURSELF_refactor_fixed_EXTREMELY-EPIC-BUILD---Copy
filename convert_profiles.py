"""
Convert old emotional profiles to use corrected parameter names
"""

import json
import os

def convert_old_profiles():
    filename = 'emotional_profiles.json'
    
    if not os.path.exists(filename):
        print("❌ No emotional profiles file found")
        return
        
    print("🔧 Converting old parameter names to corrected names...")
    
    # Parameter name mapping from old → new
    param_mapping = {
        'dampening': 'momentum_decay',
        'emotional_influence': 'mood_influence', 
        'chaos_multiplier': 'noise_amplitude',
        'base_noise_level': 'macro_noise_amplitude',
        'micro_jitter': 'micro_noise_amplitude'
    }
    
    try:
        # Load existing profiles
        with open(filename, 'r') as f:
            profiles = json.load(f)
            
        print(f"📚 Found {len(profiles)} profiles to convert")
        
        # Convert each profile
        for emotion, profile in profiles.items():
            if 'parameters' in profile:
                old_params = profile['parameters'].copy()
                new_params = {}
                
                print(f"\n🎭 Converting '{emotion}' profile:")
                
                # Convert parameters
                for old_name, value in old_params.items():
                    if old_name in param_mapping:
                        new_name = param_mapping[old_name]
                        new_params[new_name] = value
                        print(f"   • {old_name} → {new_name} = {value}")
                    else:
                        # Keep unchanged parameters
                        new_params[old_name] = value
                        print(f"   • {old_name} = {value} (unchanged)")
                
                # Update the profile
                profile['parameters'] = new_params
        
        # Save updated profiles
        backup_filename = f"{filename}.backup"
        os.rename(filename, backup_filename)
        print(f"\n💾 Saved backup to: {backup_filename}")
        
        with open(filename, 'w') as f:
            json.dump(profiles, f, indent=2)
            
        print(f"✅ Updated profiles saved to: {filename}")
        print("\n🎯 Profile conversion complete! The movement learning system should now work properly.")
        
    except Exception as e:
        print(f"❌ Error converting profiles: {e}")

if __name__ == "__main__":
    convert_old_profiles()
