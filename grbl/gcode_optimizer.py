"""
G-code Optimization Module
Intelligently adjusts feed rates and pen lift patterns for optimal drawing performance
"""

import math
import re
from typing import List, Tuple, Optional
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType

try:
    from config.config import (
        GRBL_PEN_UP_S,
        GRBL_PEN_DOWN_S,
        GRBL_NORMAL_PEN_UP,
        GRBL_NORMAL_PEN_DOWN,
        GRBL_FAST_PEN_UP,
        GRBL_FAST_PEN_DOWN,
    )
except ImportError:
    GRBL_PEN_UP_S, GRBL_PEN_DOWN_S = 30, 50
    GRBL_NORMAL_PEN_UP, GRBL_NORMAL_PEN_DOWN = 30, 50
    GRBL_FAST_PEN_UP, GRBL_FAST_PEN_DOWN = 25, 55


class GCodeOptimizer:
    """Main G-code optimization system with modular feed rate and pen lift optimization"""

    def __init__(self,
                 enable_feed_optimization=True,
                 enable_pen_optimization=True,
                 enable_stroke_filtering=True,
                 draw_feed_rate=450,
                 traversal_feed_rate=2000,
                 cluster_distance_threshold=5.0,
                 cluster_sequence_min=3,
                 micro_stroke_threshold=0.15,
                 continuous_path_threshold=2.0,
                 normal_pen_up=None,
                 normal_pen_down=None,
                 fast_pen_up=None,
                 fast_pen_down=None):

        self.enable_feed_optimization = enable_feed_optimization
        self.enable_pen_optimization = enable_pen_optimization
        self.enable_stroke_filtering = enable_stroke_filtering

        # Feed rates: flat deliberate speed on ink, fast pen-up traversals
        self.draw_feed_rate = draw_feed_rate
        self.traversal_feed_rate = traversal_feed_rate

        # Pen lift optimization parameters
        self.cluster_distance_threshold = cluster_distance_threshold
        self.cluster_sequence_min = cluster_sequence_min

        # Micro-stroke filtering parameters
        self.micro_stroke_threshold = micro_stroke_threshold
        self.continuous_path_threshold = continuous_path_threshold

        # Pen lift values (configurable via parameters or config settings)
        self.normal_pen_up = normal_pen_up if normal_pen_up is not None else GRBL_NORMAL_PEN_UP
        self.normal_pen_down = normal_pen_down if normal_pen_down is not None else GRBL_NORMAL_PEN_DOWN
        self.fast_pen_up = fast_pen_up if fast_pen_up is not None else GRBL_FAST_PEN_UP
        self.fast_pen_down = fast_pen_down if fast_pen_down is not None else GRBL_FAST_PEN_DOWN

        log_json_entry(
            LogType.GRBL,
            {
                "message": "G-code optimizer initialized",
                "action": "optimizer_init",
                "feed_optimization": enable_feed_optimization,
                "pen_optimization": enable_pen_optimization,
                "feed_rates": f"draw:{draw_feed_rate} traverse:{traversal_feed_rate}",
                "pen_values": f"normal:{self.normal_pen_up}/{self.normal_pen_down}, fast:{self.fast_pen_up}/{self.fast_pen_down}"
            },
            print_message=f"[🎯] G-code optimizer: feed={enable_feed_optimization}, pen={enable_pen_optimization}"
        )

    def calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def parse_coordinates(self, line: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract X,Y coordinates from G-code line"""
        x_match = re.search(r'X([\d\.-]+)', line)
        y_match = re.search(r'Y([\d\.-]+)', line)

        x = float(x_match.group(1)) if x_match else None
        y = float(y_match.group(1)) if y_match else None

        return x, y

    def calculate_optimal_feed_rate(self, distance: float, is_traversal: bool = False) -> int:
        """Pen down = one flat deliberate rate, pen up = fast. Distance-scaled
        speeds were retired Aug 10 2026 (see config FEED RATES note): at this
        drawing scale speed saves seconds and costs ink deposition, corner
        fidelity, and the shortest strokes entirely."""
        if is_traversal:
            return self.traversal_feed_rate
        return self.draw_feed_rate

    def detect_pen_clusters(self, lines: List[str]) -> List[Tuple[int, int]]:
        """Detect clusters of short pen up/down sequences for optimization"""
        if not self.enable_pen_optimization:
            return []

        clusters = []
        current_cluster_start = None
        pen_lift_count = 0
        last_pen_down_pos = None

        for i, line in enumerate(lines):
            line = line.strip()

            if "PEN UP" in line:
                if current_cluster_start is None:
                    current_cluster_start = i
                pen_lift_count += 1

            elif "PEN DOWN" in line:
                # Check if this pen down is close to the last one
                if last_pen_down_pos is not None:
                    x, y = self.parse_coordinates(line)
                    last_x, last_y = last_pen_down_pos

                    if x is not None and y is not None and last_x is not None and last_y is not None:
                        distance = self.calculate_distance(last_x, last_y, x, y)

                        if distance > self.cluster_distance_threshold:
                            # End current cluster if distance is too large
                            if current_cluster_start is not None and pen_lift_count >= self.cluster_sequence_min:
                                clusters.append((current_cluster_start, i))
                            current_cluster_start = None
                            pen_lift_count = 0

                # Find the next coordinate line after pen down
                for j in range(i+1, min(i+3, len(lines))):
                    next_line = lines[j].strip()
                    if next_line.startswith(("G01", "G1")):
                        x, y = self.parse_coordinates(next_line)
                        if x is not None and y is not None:
                            last_pen_down_pos = (x, y)
                            break

        # Close final cluster if still open
        if current_cluster_start is not None and pen_lift_count >= self.cluster_sequence_min:
            clusters.append((current_cluster_start, len(lines) - 1))

        return clusters

    def filter_micro_strokes(self, lines: List[str]) -> List[str]:
        """Filter out pen lifts for micro-strokes and connect nearby movements"""
        if not self.enable_stroke_filtering:
            return lines

        filtered_lines = []
        i = 0
        pen_is_down = False
        last_x, last_y = 0.0, 0.0

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and comments
            if not line or line.startswith(";") or line.startswith("%"):
                filtered_lines.append(line)
                i += 1
                continue

            # Track pen state
            if "PEN DOWN" in line:
                pen_is_down = True
                filtered_lines.append(line)
                i += 1
                continue
            elif "PEN UP" in line:
                # Check if the next stroke is a micro-movement that should stay connected
                if self._should_skip_pen_lift(lines, i, last_x, last_y):
                    # Skip this pen up and the following pen down
                    i = self._skip_to_next_movement(lines, i)
                    continue
                else:
                    pen_is_down = False
                    filtered_lines.append(line)
                    i += 1
                    continue

            # Handle movement commands
            if line.startswith(("G01", "G1", "G00", "G0")):
                x, y = self.parse_coordinates(line)
                if x is not None and y is not None:
                    last_x, last_y = x, y
                filtered_lines.append(line)
                i += 1
                continue

            # Pass through other commands
            filtered_lines.append(line)
            i += 1

        return filtered_lines

    def _should_skip_pen_lift(self, lines: List[str], pen_up_index: int, last_x: float, last_y: float) -> bool:
        """Determine if pen lift should be skipped for micro-stroke optimization"""
        # Look ahead to find the next drawing movement
        next_move_index = self._find_next_drawing_movement(lines, pen_up_index)
        if next_move_index == -1:
            return False

        # Get coordinates of next movement
        next_line = lines[next_move_index]
        next_x, next_y = self.parse_coordinates(next_line)
        if next_x is None or next_y is None:
            return False

        # Calculate distance to next movement
        distance = self.calculate_distance(last_x, last_y, next_x, next_y)

        # Skip pen lift if movement is smaller than threshold
        return distance < self.micro_stroke_threshold

    def _find_next_drawing_movement(self, lines: List[str], start_index: int) -> int:
        """Find the next G01 movement after pen down"""
        i = start_index + 1
        found_pen_down = False

        while i < len(lines):
            line = lines[i].strip()
            if "PEN DOWN" in line:
                found_pen_down = True
            elif found_pen_down and line.startswith(("G01", "G1")):
                return i
            i += 1

        return -1

    def _skip_to_next_movement(self, lines: List[str], pen_up_index: int) -> int:
        """Skip past pen up/down sequence to next movement"""
        i = pen_up_index + 1
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith(("G01", "G1")):
                return i
            i += 1
        return len(lines)

    def optimize_gcode(self, lines: List[str]) -> List[str]:
        """Apply optimizations to G-code lines"""
        if not self.enable_feed_optimization and not self.enable_pen_optimization and not self.enable_stroke_filtering:
            log_json_entry(
                LogType.GRBL,
                {"message": "G-code optimization skipped (disabled)", "action": "optimization_skipped"},
                print_message="[🎯] G-code optimization disabled"
            )
            return lines

        # First pass: filter micro-strokes
        if self.enable_stroke_filtering:
            lines = self.filter_micro_strokes(lines)

        optimized_lines = []
        last_x, last_y = 0.0, 0.0
        last_feed_rate = None
        pen_is_down = False

        # Detect pen lift clusters for optimization
        clusters = self.detect_pen_clusters(lines)
        cluster_ranges = set()
        for start, end in clusters:
            cluster_ranges.update(range(start, end + 1))

        log_json_entry(
            LogType.GRBL,
            {
                "message": "Starting G-code optimization",
                "action": "optimization_start",
                "total_lines": len(lines),
                "detected_clusters": len(clusters),
                "feed_optimization": self.enable_feed_optimization,
                "pen_optimization": self.enable_pen_optimization
            },
            print_message=f"[🎯] Optimizing {len(lines)} G-code lines ({len(clusters)} pen clusters detected)"
        )

        for i, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith(";") or line.startswith("%"):
                optimized_lines.append(line)
                continue

            # Handle movement commands
            if line.startswith(("G01", "G1", "G00", "G0")):
                x, y = self.parse_coordinates(line)
                is_traversal = not pen_is_down or line.startswith(("G00", "G0 "))

                if x is not None and y is not None:
                    distance = self.calculate_distance(last_x, last_y, x, y)
                    optimal_feed_rate = self.calculate_optimal_feed_rate(distance, is_traversal=is_traversal)

                    if (self.enable_feed_optimization and
                        (last_feed_rate is None or abs(optimal_feed_rate - last_feed_rate) > 200)):
                        optimized_lines.append(f"F{optimal_feed_rate}")
                        last_feed_rate = optimal_feed_rate

                    last_x, last_y = x, y

                optimized_lines.append(line)

            # Handle pen control commands with cluster optimization
            elif "PEN UP" in line or "PEN DOWN" in line:
                pen_is_down = "PEN DOWN" in line
                if self.enable_pen_optimization and i in cluster_ranges:
                    # Use optimized pen values for clustered movements
                    if "PEN UP" in line:
                        optimized_line = f"M3 S{self.fast_pen_up} ; PEN UP (fast)"
                    else:
                        optimized_line = f"M3 S{self.fast_pen_down} ; PEN DOWN (fast)"
                    optimized_lines.append(optimized_line)
                else:
                    # Use normal pen values (apply optimization to all pen commands)
                    if "PEN UP" in line:
                        optimized_line = f"M3 S{self.normal_pen_up} ; PEN UP"
                    else:
                        optimized_line = f"M3 S{self.normal_pen_down} ; PEN DOWN"
                    optimized_lines.append(optimized_line)

            else:
                # Pass through other commands unchanged
                optimized_lines.append(line)

        optimization_stats = {
            "original_lines": len(lines),
            "optimized_lines": len(optimized_lines),
            "clusters_optimized": len(clusters),
            "feed_optimization": self.enable_feed_optimization,
            "pen_optimization": self.enable_pen_optimization
        }

        log_json_entry(
            LogType.GRBL,
            {
                "message": "G-code optimization complete",
                "action": "optimization_complete",
                **optimization_stats
            },
            print_message=f"[✅] G-code optimized: {len(clusters)} pen clusters, feed rates adjusted"
        )

        return optimized_lines

    def optimize_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        """Optimize an entire G-code file"""
        if output_file is None:
            output_file = input_file.replace('.gcode', '_optimized.gcode')

        try:
            with open(input_file, 'r') as f:
                lines = f.readlines()

            # Strip whitespace but preserve line structure
            lines = [line.rstrip() for line in lines]

            optimized_lines = self.optimize_gcode(lines)

            with open(output_file, 'w') as f:
                for line in optimized_lines:
                    f.write(line + '\n')

            log_json_entry(
                LogType.GRBL,
                {
                    "message": "G-code file optimization complete",
                    "action": "file_optimization_complete",
                    "input_file": input_file,
                    "output_file": output_file,
                    "original_lines": len(lines),
                    "optimized_lines": len(optimized_lines)
                },
                print_message=f"[💾] Optimized G-code saved: {output_file}"
            )

            return output_file

        except Exception as e:
            log_json_entry(
                LogType.ERROR,
                {
                    "message": "G-code optimization failed",
                    "component": "gcode_optimizer",
                    "error": str(e),
                    "input_file": input_file
                },
                print_message=f"[❌] G-code optimization failed: {e}"
            )
            raise


def create_optimizer_from_config() -> GCodeOptimizer:
    """Create optimizer instance using configuration values"""
    try:
        from config.config import (
            GRBL_ENABLE_FEED_OPTIMIZATION,
            GRBL_ENABLE_PEN_OPTIMIZATION,
            GRBL_DRAW_FEED_RATE,
            GRBL_TRAVERSAL_FEED_RATE,
            GRBL_CLUSTER_DISTANCE_THRESHOLD,
            GRBL_CLUSTER_SEQUENCE_MIN
        )

        # Try to get stroke filtering config, default to True
        try:
            GRBL_ENABLE_STROKE_FILTERING = getattr(__import__('config.config', fromlist=['GRBL_ENABLE_STROKE_FILTERING']), 'GRBL_ENABLE_STROKE_FILTERING', True)
        except:
            GRBL_ENABLE_STROKE_FILTERING = True

        return GCodeOptimizer(
            enable_feed_optimization=GRBL_ENABLE_FEED_OPTIMIZATION,
            enable_pen_optimization=GRBL_ENABLE_PEN_OPTIMIZATION,
            enable_stroke_filtering=GRBL_ENABLE_STROKE_FILTERING,
            draw_feed_rate=GRBL_DRAW_FEED_RATE,
            traversal_feed_rate=GRBL_TRAVERSAL_FEED_RATE,
            cluster_distance_threshold=GRBL_CLUSTER_DISTANCE_THRESHOLD,
            cluster_sequence_min=GRBL_CLUSTER_SEQUENCE_MIN,
            micro_stroke_threshold=0.5,   # Skip pen lifts for moves < 0.5mm
            continuous_path_threshold=2.0
        )

    except ImportError:
        # Fallback to defaults if config values not available
        return GCodeOptimizer(enable_stroke_filtering=True)