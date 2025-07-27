#!/usr/bin/env python3
"""
Test the full system with persistence but without camera dependencies
"""

import sys
import os
import time
import signal

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def signal_handler(sig, frame):
    print('\n[CTRL+C] Graceful shutdown initiated...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Test imports
try:
    from utils.state_manager import state_manager
    from captioner.captioner import Captioner
    from mood.mood import MoodEngine
    from utils.continuity import describe_duration
    from config.config import MOOD_SNAPSHOT_FOLDER, CLEAN_CAPTION_OUTPUT
    from event_logging.event_logger import get_current_run_id, set_start_time, log_json_entry, LogType
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def main():
    """Test the full persistence-enabled system"""
    print("\n=== LOVE_YOURSELF Mirror System (Persistence Test) ===")
    
    # Initialize run ID and start time for this session
    start_time = time.time()
    set_start_time(start_time)
    run_id = get_current_run_id()
    print(f"[🚀] Starting session with run ID: {run_id}")

    log_json_entry(
        LogType.SESSION_START, {"run_id": run_id}, MOOD_SNAPSHOT_FOLDER, 
        auto_print=not CLEAN_CAPTION_OUTPUT, 
        print_message=f"🚀 Starting session with run ID: {run_id}" if not CLEAN_CAPTION_OUTPUT else None
    )

    # Initialize components
    print("[🧠] Initializing mood engine...")
    mood_engine = MoodEngine()
    print("[👁️] Initializing captioner...")
    captioner = Captioner()

    # Load previous session state if available
    print("[💾] Loading previous session state...")
    previous_state = state_manager.load_session_state()
    if previous_state:
        # Apply state to components
        state_manager.apply_state_to_captioner(previous_state, captioner)
        state_manager.apply_state_to_mood_engine(previous_state, mood_engine)
        
        # Generate awakening message with continuity
        save_time = previous_state["metadata"]["save_time"]
        time_since_last = describe_duration(save_time)
        previous_beliefs = previous_state["captioner"].get("beliefs", {})
        
        awakening_msg = captioner.generate_awakening_message(time_since_last, previous_beliefs)
        if not CLEAN_CAPTION_OUTPUT:
            print(f"[🌅] {awakening_msg}")
        
        log_json_entry(
            LogType.INFO,
            {"message": awakening_msg, "continuity": True, "time_since_last": time_since_last},
            MOOD_SNAPSHOT_FOLDER,
            auto_print=CLEAN_CAPTION_OUTPUT,
            print_message=f'"{awakening_msg}"' if CLEAN_CAPTION_OUTPUT else None
        )
        
        captioner.memory_loaded_from_previous = True
    else:
        # Fresh start
        awakening_msg = captioner.generate_awakening_message()
        if not CLEAN_CAPTION_OUTPUT:
            print(f"[🌅] {awakening_msg}")
        log_json_entry(
            LogType.INFO,
            {"message": awakening_msg, "continuity": False},
            MOOD_SNAPSHOT_FOLDER,
            auto_print=CLEAN_CAPTION_OUTPUT,
            print_message=f'"{awakening_msg}"' if CLEAN_CAPTION_OUTPUT else None
        )

    # Display current state
    print(f"\n[📊] Current State:")
    print(f"   - Mood: {mood_engine.current_mood:.2f}")
    print(f"   - Beliefs: {len(captioner.beliefs)}")
    print(f"   - Motifs: {len(captioner.motif_counter)}")
    print(f"   - Awakening done: {captioner.awakening_done}")
    print(f"   - Memory loaded: {captioner.memory_loaded_from_previous}")

    if captioner.beliefs:
        print(f"\n[🧩] Current Beliefs:")
        for motif, data in captioner.beliefs.items():
            print(f"   - {motif}: {data['strength']:.2f} strength")

    if captioner.motif_counter:
        print(f"\n[🔄] Current Motifs:")
        for motif, count in captioner.motif_counter.items():
            print(f"   - {motif}: {count} occurrences")

    print(f"\n[⏰] System will run for 30 seconds, then save state and exit...")
    print(f"[ℹ️] Press Ctrl+C to exit early with graceful shutdown")
    
    # Simulate system running
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    
    # Save state before shutdown
    print(f"\n[💾] Saving session state...")
    success = state_manager.save_session_state(captioner, mood_engine)
    if success:
        print("[✅] Session state saved successfully")
        
        # Show what was saved
        lifetime_stats = state_manager.get_lifetime_stats()
        print(f"[📈] Lifetime stats: {lifetime_stats['total_sessions']} sessions, {lifetime_stats['total_runtime']:.1f}s total")
    else:
        print("[❌] Failed to save session state")
    
    # Log session end
    log_json_entry(
        LogType.INFO, 
        {"message": "Session ended", "run_id": run_id, "duration": time.time() - start_time}, 
        MOOD_SNAPSHOT_FOLDER,
        auto_print=not CLEAN_CAPTION_OUTPUT,
        print_message=f"[👋] Session ended. Duration: {time.time() - start_time:.1f}s" if not CLEAN_CAPTION_OUTPUT else None
    )
    
    print(f"[👋] System shutdown complete")

if __name__ == "__main__":
    main()
