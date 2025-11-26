"""
G-code Segmentation for Buffer Optimization
Splits G-code at pen lifts to prevent GRBL buffer overload while preserving image quality
"""

import os
import sys
from typing import List, Dict, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from event_logging.event_logger import log_json_entry
    from event_logging.log_type import LogType
except ImportError:
    # Fallback for testing
    def log_json_entry(log_type, data, print_message=""):
        print(print_message)
    class LogType:
        INFO = "info"
        DEBUG = "debug"


class GCodeSegment:
    """Represents a single G-code segment"""

    def __init__(self, lines: List[str], segment_id: int, start_line: int):
        self.lines = lines
        self.segment_id = segment_id
        self.start_line = start_line
        self.end_line = start_line + len(lines) - 1
        self.line_count = len(lines)

    def __len__(self):
        return self.line_count

    def is_large(self, threshold: int = 150) -> bool:
        """Check if segment is too large for optimal buffering"""
        return self.line_count > threshold


class GCodeSegmenter:
    """Segments G-code files for optimal GRBL streaming"""

    def __init__(self, max_segment_size: int = 150):
        self.max_segment_size = max_segment_size

    def segment_gcode_file(self, gcode_file: str) -> Tuple[List[str], List[GCodeSegment]]:
        """
        Segment G-code file into optimal chunks

        Returns:
            header_lines: Setup commands to send once at start
            segments: List of GCodeSegment objects
        """
        if not os.path.exists(gcode_file):
            raise FileNotFoundError(f"G-code file not found: {gcode_file}")

        log_json_entry(
            LogType.INFO,
            {"action": "gcode_segmentation_start", "file": gcode_file, "max_segment_size": self.max_segment_size},
            print_message=f"[📋] Segmenting G-code: {os.path.basename(gcode_file)}"
        )

        try:
            with open(gcode_file, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
        except Exception as e:
            raise IOError(f"Failed to read G-code file: {e}")

        # Separate header from drawing commands
        header_lines, drawing_lines = self._separate_header_and_drawing(lines)

        # Create segments at pen lift boundaries
        initial_segments = self._create_pen_lift_segments(drawing_lines)

        # Split any oversized segments
        final_segments = self._split_large_segments(initial_segments)

        log_json_entry(
            LogType.INFO,
            {
                "action": "gcode_segmentation_complete",
                "total_lines": len(lines),
                "header_lines": len(header_lines),
                "drawing_lines": len(drawing_lines),
                "segments": len(final_segments),
                "avg_segment_size": sum(len(seg) for seg in final_segments) / len(final_segments)
            },
            print_message=f"[✅] Segmented into {len(final_segments)} segments (avg {sum(len(seg) for seg in final_segments) / len(final_segments):.0f} lines)"
        )

        return header_lines, final_segments

    def _separate_header_and_drawing(self, lines: List[str]) -> Tuple[List[str], List[str]]:
        """Separate setup commands from drawing commands"""
        header_lines = []
        drawing_start_idx = 0

        for i, line in enumerate(lines):
            if not line or line.startswith(';'):
                header_lines.append(line)
                continue

            # Look for first movement or pen command
            if any(line.startswith(cmd) for cmd in ['G0', 'G1', 'M3']):
                drawing_start_idx = i
                break
            else:
                header_lines.append(line)

        drawing_lines = lines[drawing_start_idx:]
        return header_lines, drawing_lines

    def _create_pen_lift_segments(self, drawing_lines: List[str]) -> List[GCodeSegment]:
        """Create segments at natural pen lift boundaries"""
        segments = []
        current_segment = []
        line_offset = 0

        for line_num, line in enumerate(drawing_lines):
            current_segment.append(line)

            # End segment on pen up commands
            if self._is_pen_up_command(line):
                segment = GCodeSegment(
                    lines=current_segment.copy(),
                    segment_id=len(segments) + 1,
                    start_line=line_offset
                )
                segments.append(segment)
                line_offset += len(current_segment)
                current_segment = []

        # Add final segment if exists
        if current_segment:
            segment = GCodeSegment(
                lines=current_segment.copy(),
                segment_id=len(segments) + 1,
                start_line=line_offset
            )
            segments.append(segment)

        return segments

    def _is_pen_up_command(self, line: str) -> bool:
        """Check if line represents a pen up command"""
        return any(marker in line for marker in [
            "PEN UP",
            "S47",  # Common pen up servo value
            "S20",  # Another common pen up value
            "S0"    # Spindle off
        ])

    def _split_large_segments(self, segments: List[GCodeSegment]) -> List[GCodeSegment]:
        """Split segments that are too large for optimal buffering"""
        final_segments = []

        for segment in segments:
            if segment.is_large(self.max_segment_size):
                # Split large segment into smaller chunks at safe boundaries
                sub_segments = self._split_segment_safely(segment)
                final_segments.extend(sub_segments)

                log_json_entry(
                    LogType.DEBUG,
                    {
                        "action": "large_segment_split",
                        "original_size": len(segment),
                        "sub_segments": len(sub_segments)
                    },
                    print_message=f"[✂️] Split large segment ({len(segment)} lines) into {len(sub_segments)} parts"
                )
            else:
                final_segments.append(segment)

        return final_segments

    def _split_segment_safely(self, segment: GCodeSegment) -> List[GCodeSegment]:
        """Split a large segment at safe boundaries (movement commands)"""
        sub_segments = []
        current_chunk = []
        line_offset = segment.start_line

        for line in segment.lines:
            current_chunk.append(line)

            # Split at movement commands when chunk gets large enough
            if (len(current_chunk) >= self.max_segment_size // 2 and
                line.startswith(('G0', 'G1'))):

                sub_segment = GCodeSegment(
                    lines=current_chunk.copy(),
                    segment_id=len(sub_segments) + 1,
                    start_line=line_offset
                )
                sub_segments.append(sub_segment)
                line_offset += len(current_chunk)
                current_chunk = []

        # Add final chunk
        if current_chunk:
            sub_segment = GCodeSegment(
                lines=current_chunk.copy(),
                segment_id=len(sub_segments) + 1,
                start_line=line_offset
            )
            sub_segments.append(sub_segment)

        return sub_segments


def test_segmentation(gcode_file: str):
    """Test function to validate segmentation"""
    segmenter = GCodeSegmenter(max_segment_size=150)

    try:
        header, segments = segmenter.segment_gcode_file(gcode_file)

        print(f"✅ Segmentation test successful:")
        print(f"   Header lines: {len(header)}")
        print(f"   Segments: {len(segments)}")
        print(f"   Largest segment: {max(len(seg) for seg in segments)} lines")
        print(f"   Average segment: {sum(len(seg) for seg in segments) / len(segments):.0f} lines")

        return True

    except Exception as e:
        print(f"❌ Segmentation test failed: {e}")
        return False


if __name__ == "__main__":
    # Test on most recent G-code file
    import glob

    output_dir = "/home/impostor/ComfyUI/output"
    gcode_files = glob.glob(os.path.join(output_dir, "*.gcode"))

    if gcode_files:
        recent_file = max(gcode_files, key=os.path.getmtime)
        print(f"Testing segmentation on: {os.path.basename(recent_file)}")
        test_segmentation(recent_file)
    else:
        print("No G-code files found for testing")