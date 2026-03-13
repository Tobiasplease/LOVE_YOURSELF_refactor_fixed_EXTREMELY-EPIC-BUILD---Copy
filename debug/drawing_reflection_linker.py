#!/usr/bin/env python3
"""
Links specific drawings to their associated reflections and thoughts.
Usage: python debug/drawing_reflection_linker.py [run_id]
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

def extract_timestamp_from_filename(filename):
    """Extract timestamp from drawing_introspection_TIMESTAMP.jpg"""
    if 'drawing_introspection_' in filename:
        return filename.split('drawing_introspection_')[1].split('.')[0]
    return None

def load_run_data(run_id):
    """Load the event log JSON for a run"""
    log_file = f"event_log/{run_id}-event-log.json"
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found")
        return None

    with open(log_file, 'r') as f:
        return json.load(f)

def find_drawing_creation_story(events, image_timestamp):
    """Find the creative decision process that led to this drawing"""
    target_time = int(image_timestamp)

    # Find the ComfyUI prompt that created this drawing
    comfy_prompt = None
    for event in events:
        if (event['type'] == 'comfy_prompt' and
            abs(event.get('timestamp', 0) - target_time) <= 300):  # Within 5 minutes
            comfy_prompt = event
            break

    if not comfy_prompt:
        return None, []

    # Find reflections and captions in the 10 minutes BEFORE the drawing decision
    decision_time = comfy_prompt['timestamp']
    leading_events = []

    for event in events:
        event_time = event.get('timestamp', 0)
        if (decision_time - 600 <= event_time <= decision_time and  # 10 minutes before
            event['type'] in ['caption', 'reflection']):
            leading_events.append(event)

    return comfy_prompt, sorted(leading_events, key=lambda x: x['timestamp'])

def display_drawing_analysis(run_id, image_file=None):
    """Display drawing analysis for a specific image or all images in a run"""
    events = load_run_data(run_id)
    if not events:
        return

    images_dir = f"event_log/{run_id}-images"
    if not os.path.exists(images_dir):
        print(f"No images directory found for run {run_id}")
        return

    if image_file:
        # Analyze specific image
        image_path = os.path.join(images_dir, image_file)
        if not os.path.exists(image_path):
            print(f"Image {image_file} not found")
            return

        timestamp = extract_timestamp_from_filename(image_file)
        if timestamp:
            print(f"\n{'='*80}")
            print(f"DRAWING CREATION STORY: {image_file}")
            print(f"Timestamp: {timestamp}")
            print(f"{'='*80}")

            comfy_prompt, leading_events = find_drawing_creation_story(events, timestamp)

            if not comfy_prompt:
                print("No ComfyUI prompt found for this drawing")
                return

            print(f"\n🎨 FINAL DRAWING PROMPT SENT TO COMFYUI:")
            print(f"📅 {comfy_prompt['iso_timestamp']}")
            print(f"{'='*60}")
            print(comfy_prompt.get('drawing_prompt', 'N/A'))

            print(f"\n🧠 LEADING THOUGHTS (10 minutes before decision):")
            print(f"{'='*60}")

            if not leading_events:
                print("No leading reflections found")
            else:
                for i, event in enumerate(leading_events, 1):
                    time_diff = comfy_prompt['timestamp'] - event['timestamp']
                    minutes_before = time_diff // 60

                    print(f"\n{i}. [{event['iso_timestamp']}] (-{minutes_before}m {time_diff % 60}s before)")
                    if event['type'] == 'caption':
                        print(f"   👁️  SAW: {event['caption']}")
                    elif event['type'] == 'reflection':
                        print(f"   💭 REFLECTED: {event.get('content', 'N/A')}")
    else:
        # List all drawings with brief summaries
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        image_files.sort()

        print(f"\n{'='*80}")
        print(f"ALL DRAWINGS IN RUN {run_id} ({len(image_files)} total)")
        print(f"{'='*80}")

        for i, image_file in enumerate(image_files[:10]):  # Show first 10
            timestamp = extract_timestamp_from_filename(image_file)
            if timestamp:
                comfy_prompt, _ = find_drawing_creation_story(events, timestamp)

                print(f"\n{i+1}. {image_file}")
                if comfy_prompt:
                    prompt_preview = comfy_prompt.get('drawing_prompt', 'N/A')[:100]
                    print(f"   🎨 Drew: {prompt_preview}...")
                else:
                    print(f"   No ComfyUI prompt found")

        if len(image_files) > 10:
            print(f"\n... and {len(image_files) - 10} more drawings")

        print(f"\nTo analyze a specific drawing, run:")
        print(f"python debug/drawing_reflection_linker.py {run_id} <image_filename>")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug/drawing_reflection_linker.py <run_id> [image_filename]")
        print("\nExample:")
        print("  python debug/drawing_reflection_linker.py f55c9e41")
        print("  python debug/drawing_reflection_linker.py f55c9e41 drawing_introspection_1759231522.jpg")
        sys.exit(1)

    run_id = sys.argv[1]
    image_file = sys.argv[2] if len(sys.argv) > 2 else None

    display_drawing_analysis(run_id, image_file)