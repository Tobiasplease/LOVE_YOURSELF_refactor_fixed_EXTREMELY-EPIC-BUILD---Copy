#!/usr/bin/env python3
"""
Test G-code segmentation for buffer optimization
SAFE: Only analyzes files, doesn't modify execution
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def analyze_gcode_segments(gcode_file):
    """Analyze how G-code would be segmented without modifying anything"""
    print(f"\n=== ANALYZING: {os.path.basename(gcode_file)} ===")

    if not os.path.exists(gcode_file):
        print(f"❌ File not found: {gcode_file}")
        return None

    segments = []
    current_segment = []
    header_lines = []
    pen_state = "unknown"

    try:
        with open(gcode_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

    print(f"📄 Total lines: {len(lines)}")

    # Separate header from drawing commands
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith(';'):
            header_lines.append(line)
            continue

        # Look for first movement or pen command to end header
        if any(line.startswith(cmd) for cmd in ['G0', 'G1', 'M3']):
            header_lines = lines[:i]  # Everything before first real command
            drawing_lines = lines[i:]  # Everything from first command onward
            break
    else:
        drawing_lines = lines  # No header found

    print(f"📋 Header lines: {len(header_lines)}")
    print(f"🎨 Drawing lines: {len(drawing_lines)}")

    # Analyze segmentation
    for line_num, line in enumerate(drawing_lines):
        line = line.strip()
        current_segment.append(line)

        # Track pen state
        if "PEN DOWN" in line or "S50" in line or "S55" in line:
            pen_state = "down"
        elif "PEN UP" in line or "S47" in line or "S20" in line:
            pen_state = "up"
            # End segment on pen up
            segments.append({
                'lines': current_segment.copy(),
                'line_count': len(current_segment),
                'start_line': line_num - len(current_segment) + 1,
                'end_line': line_num
            })
            current_segment = []

    # Add final segment if exists
    if current_segment:
        segments.append({
            'lines': current_segment.copy(),
            'line_count': len(current_segment),
            'start_line': len(drawing_lines) - len(current_segment),
            'end_line': len(drawing_lines) - 1
        })

    # Analysis results
    print(f"\n🔢 SEGMENTATION ANALYSIS:")
    print(f"   Total segments: {len(segments)}")

    if segments:
        line_counts = [seg['line_count'] for seg in segments]
        print(f"   Lines per segment: {min(line_counts)} - {max(line_counts)}")
        print(f"   Average lines per segment: {sum(line_counts) / len(line_counts):.1f}")

        # Show first few segments as examples
        print(f"\n📋 FIRST 5 SEGMENTS:")
        for i, seg in enumerate(segments[:5]):
            print(f"   Segment {i+1}: {seg['line_count']} lines (lines {seg['start_line']}-{seg['end_line']})")
            # Show first and last line of segment
            if seg['lines']:
                print(f"      First: {seg['lines'][0][:50]}...")
                print(f"      Last:  {seg['lines'][-1][:50]}...")

        # Check for any problematic segments
        large_segments = [seg for seg in segments if seg['line_count'] > 200]
        if large_segments:
            print(f"\n⚠️  LARGE SEGMENTS (>200 lines): {len(large_segments)}")
            for seg in large_segments:
                print(f"   Segment {segments.index(seg)+1}: {seg['line_count']} lines")

    return {
        'file': gcode_file,
        'total_lines': len(lines),
        'header_lines': len(header_lines),
        'drawing_lines': len(drawing_lines),
        'segments': segments,
        'segment_count': len(segments)
    }


def test_recent_files():
    """Test segmentation on recent G-code files"""
    print("🧪 TESTING G-CODE SEGMENTATION")
    print("This is SAFE - only analyzing files, not modifying execution")

    # Look for recent G-code files
    output_dir = "/home/impostor/ComfyUI/output"
    gcode_files = []

    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.gcode'):
                full_path = os.path.join(output_dir, file)
                gcode_files.append(full_path)

    # Sort by modification time, newest first
    gcode_files.sort(key=os.path.getmtime, reverse=True)

    if not gcode_files:
        print("❌ No G-code files found in output directory")
        return

    # Analyze the most recent file
    recent_file = gcode_files[0]
    print(f"🎯 Analyzing most recent: {os.path.basename(recent_file)}")

    result = analyze_gcode_segments(recent_file)

    if result:
        print(f"\n✅ SEGMENTATION LOOKS FEASIBLE:")
        print(f"   {result['segment_count']} segments")
        print(f"   Average ~{result['drawing_lines'] / result['segment_count']:.0f} lines per segment")
        print(f"   Should reduce GRBL buffer pressure significantly")
    else:
        print("❌ Analysis failed")


if __name__ == "__main__":
    test_recent_files()