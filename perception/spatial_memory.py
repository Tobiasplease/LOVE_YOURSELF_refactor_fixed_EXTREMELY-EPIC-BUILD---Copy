# perception/spatial_memory.py
"""
Enhanced Spatial Memory System
-----------------------------
Builds persistent understanding of 3D space, object relationships,
and environmental layout to enhance consciousness and continuity.

Key Features:
- Persistent spatial map of environment
- Object relationships and positioning
- Scene layout understanding
- Depth and layer awareness
- Movement and change tracking
- Spatial context for captions
"""

import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# import cv2


@dataclass
class SpatialObject:
    """Represents an object with spatial properties"""

    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]
    area: float
    timestamp: float
    depth_layer: str  # "foreground", "midground", "background"
    stability_score: float = 1.0  # How stable this object position is
    movement_vector: Tuple[float, float] = (0.0, 0.0)  # dx, dy movement

    def to_dict(self) -> dict:
        return asdict(self)


class SpatialMemory:
    """Persistent spatial understanding system"""

    def __init__(self):
        # Spatial object tracking
        self.current_objects: Dict[str, SpatialObject] = {}
        self.persistent_objects: Dict[str, List[SpatialObject]] = defaultdict(list)

        # Scene layout understanding
        self.scene_regions = {
            "left": [],  # Objects consistently on left side
            "right": [],  # Objects consistently on right side
            "center": [],  # Objects consistently in center
            "top": [],  # Objects consistently at top
            "bottom": [],  # Objects consistently at bottom
        }

        # Spatial relationships
        self.object_relationships: Dict[str, Dict[str, str]] = defaultdict(dict)

        # Layout memory
        self.layout_history: deque = deque(maxlen=20)  # Recent layout snapshots
        self.stable_layout: Dict[str, str] = {}  # Confirmed spatial layout

        # Movement tracking
        self.movement_history: deque = deque(maxlen=10)

        # Frame dimensions (updated when processing)
        self.frame_width = 640
        self.frame_height = 480

        # Stability thresholds
        self.STABILITY_THRESHOLD = 0.7
        self.LAYOUT_CONFIDENCE_THRESHOLD = 5  # Must see layout 5+ times

    def update_frame_dimensions(self, width: int, height: int):
        """Update frame dimensions for spatial calculations"""
        self.frame_width = width
        self.frame_height = height

    def process_detections(self, detections: List[dict], frame_shape: Tuple[int, int]) -> None:
        """Process YOLO detections and update spatial memory"""
        current_time = time.time()
        self.frame_height, self.frame_width = frame_shape[:2]

        new_objects = {}

        for detection in detections:
            label = detection.get("label", "unknown")
            confidence = detection.get("confidence", 0.0)
            bbox = detection.get("bbox", (0, 0, 0, 0))  # x1, y1, x2, y2

            # Calculate spatial properties
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            area = (x2 - x1) * (y2 - y1)

            # Determine depth layer based on size and position
            depth_layer = self._calculate_depth_layer(area, center_y)

            # Calculate movement if object existed before
            movement_vector = (0.0, 0.0)
            stability_score = 1.0

            if label in self.current_objects:
                prev_obj = self.current_objects[label]
                dx = center_x - prev_obj.center[0]
                dy = center_y - prev_obj.center[1]
                movement_vector = (dx, dy)

                # Calculate stability (how much object has moved)
                movement_magnitude = np.sqrt(dx * dx + dy * dy)
                stability_score = max(0.0, 1.0 - movement_magnitude / 100.0)

            spatial_obj = SpatialObject(
                label=label,
                confidence=confidence,
                bbox=bbox,
                center=(center_x, center_y),
                area=area,
                timestamp=current_time,
                depth_layer=depth_layer,
                stability_score=stability_score,
                movement_vector=movement_vector,
            )

            new_objects[label] = spatial_obj

        # Update current objects
        self.current_objects = new_objects

        # Update persistent tracking
        self._update_persistent_objects()

        # Update spatial relationships
        self._update_spatial_relationships()

        # Update scene layout understanding
        self._update_scene_layout()

        # Track movement patterns
        self._track_movement_patterns()

    def _calculate_depth_layer(self, area: float, center_y: float) -> str:
        """Determine depth layer based on object size and vertical position"""
        # Larger objects are usually foreground, smaller background
        # Objects lower in frame are usually foreground

        frame_area = self.frame_width * self.frame_height
        relative_size = area / frame_area
        relative_y = center_y / self.frame_height

        # Combined size and position heuristic
        depth_score = relative_size * 0.7 + (1.0 - relative_y) * 0.3

        if depth_score > 0.6:
            return "foreground"
        elif depth_score > 0.3:
            return "midground"
        else:
            return "background"

    def _update_persistent_objects(self):
        """Update persistent object tracking with temporal decay"""
        current_time = time.time()

        for label, obj in self.current_objects.items():
            # Add to persistent tracking
            self.persistent_objects[label].append(obj)

            # Keep only recent history (last 30 seconds)
            cutoff_time = current_time - 30.0
            self.persistent_objects[label] = [o for o in self.persistent_objects[label] if o.timestamp > cutoff_time]

        # Remove objects not seen recently
        for label in list(self.persistent_objects.keys()):
            if not self.persistent_objects[label]:
                del self.persistent_objects[label]

    def _update_spatial_relationships(self):
        """Update spatial relationships between objects"""
        relationships = defaultdict(dict)

        objects = list(self.current_objects.values())

        for i, obj1 in enumerate(objects):
            for obj2 in objects[i + 1 :]:
                rel = self._determine_relationship(obj1, obj2)
                if rel:
                    relationships[obj1.label][obj2.label] = rel
                    # Add inverse relationship
                    inverse_rel = self._inverse_relationship(rel)
                    relationships[obj2.label][obj1.label] = inverse_rel

        self.object_relationships = relationships

    def _determine_relationship(self, obj1: SpatialObject, obj2: SpatialObject) -> Optional[str]:
        """Determine spatial relationship between two objects"""
        x1, y1 = obj1.center
        x2, y2 = obj2.center

        dx = x2 - x1
        dy = y2 - y1

        # Distance threshold for relationships
        distance = np.sqrt(dx * dx + dy * dy)
        if distance > self.frame_width * 0.5:  # Too far apart
            return None

        # Determine primary relationship
        if abs(dx) > abs(dy):
            # Horizontal relationship dominates
            if dx > 0:
                return "right_of"
            else:
                return "left_of"
        else:
            # Vertical relationship dominates
            if dy > 0:
                return "below"
            else:
                return "above"

    def _inverse_relationship(self, relationship: str) -> str:
        """Get inverse spatial relationship"""
        inverses = {"left_of": "right_of", "right_of": "left_of", "above": "below", "below": "above"}
        return inverses.get(relationship, relationship)

    def _update_scene_layout(self):
        """Update understanding of stable scene layout"""
        layout_snapshot = {}

        for label, obj in self.current_objects.items():
            # Determine region based on position
            x_ratio = obj.center[0] / self.frame_width
            y_ratio = obj.center[1] / self.frame_height

            regions = []

            # Horizontal regions
            if x_ratio < 0.33:
                regions.append("left")
            elif x_ratio > 0.67:
                regions.append("right")
            else:
                regions.append("center")

            # Vertical regions
            if y_ratio < 0.33:
                regions.append("top")
            elif y_ratio > 0.67:
                regions.append("bottom")

            layout_snapshot[label] = regions

        self.layout_history.append(layout_snapshot)

        # Update stable layout (objects consistently in same regions)
        self._update_stable_layout()

    def _update_stable_layout(self):
        """Determine stable layout from history"""
        object_region_counts = defaultdict(lambda: defaultdict(int))

        # Count how often each object appears in each region
        for snapshot in self.layout_history:
            for obj_label, regions in snapshot.items():
                for region in regions:
                    object_region_counts[obj_label][region] += 1

        # Determine stable regions (appear consistently)
        stable_layout = {}
        for obj_label, region_counts in object_region_counts.items():
            for region, count in region_counts.items():
                if count >= self.LAYOUT_CONFIDENCE_THRESHOLD:
                    if obj_label not in stable_layout:
                        stable_layout[obj_label] = []
                    stable_layout[obj_label].append(region)

        self.stable_layout = stable_layout

    def _track_movement_patterns(self):
        """Track movement patterns for dynamic understanding"""
        movement_data = {}

        for label, obj in self.current_objects.items():
            movement_data[label] = {"vector": obj.movement_vector, "stability": obj.stability_score, "timestamp": obj.timestamp}

        self.movement_history.append(movement_data)

    def get_spatial_context(self) -> str:
        """Generate spatial context description for captions"""
        if not self.current_objects:
            return "The space appears empty in my current view"

        context_parts = []

        # Describe stable layout
        if self.stable_layout:
            layout_desc = self._describe_stable_layout()
            if layout_desc:
                context_parts.append(layout_desc)

        # Describe current spatial relationships
        rel_desc = self._describe_current_relationships()
        if rel_desc:
            context_parts.append(rel_desc)

        # Describe movement if significant
        movement_desc = self._describe_movement()
        if movement_desc:
            context_parts.append(movement_desc)

        # Describe depth layers
        depth_desc = self._describe_depth_composition()
        if depth_desc:
            context_parts.append(depth_desc)

        if context_parts:
            return " | ".join(context_parts)
        else:
            return "I can see objects but their spatial arrangement is still unclear"

    def _describe_stable_layout(self) -> str:
        """Describe stable spatial layout"""
        if not self.stable_layout:
            return ""

        layout_groups = defaultdict(list)
        for obj, regions in self.stable_layout.items():
            region_key = "+".join(sorted(regions))
            layout_groups[region_key].append(obj)

        descriptions = []
        for region_combo, objects in layout_groups.items():
            if len(objects) > 1:
                region_desc = region_combo.replace("+", " ")
                obj_list = ", ".join(objects)
                descriptions.append(f"{obj_list} consistently in {region_desc} area")
            elif len(objects) == 1:
                region_desc = region_combo.replace("+", " ")
                descriptions.append(f"{objects[0]} consistently in {region_desc}")

        return " | ".join(descriptions) if descriptions else ""

    def _describe_current_relationships(self) -> str:
        """Describe current spatial relationships"""
        if not self.object_relationships:
            return ""

        relationships = []
        for obj1, rels in self.object_relationships.items():
            for obj2, rel in rels.items():
                # Only include one direction to avoid redundancy
                if obj1 < obj2:  # Alphabetical order to ensure consistency
                    relationships.append(f"{obj1} {rel} {obj2}")

        return " | ".join(relationships[:3]) if relationships else ""  # Limit to avoid verbosity

    def _describe_movement(self) -> str:
        """Describe significant movement"""
        if not self.movement_history or len(self.movement_history) < 2:
            return ""

        moving_objects = []
        for label, obj in self.current_objects.items():
            dx, dy = obj.movement_vector
            movement_magnitude = np.sqrt(dx * dx + dy * dy)

            if movement_magnitude > 20:  # Significant movement threshold
                direction = self._movement_direction(dx, dy)
                moving_objects.append(f"{label} moving {direction}")

        return " | ".join(moving_objects) if moving_objects else ""

    def _movement_direction(self, dx: float, dy: float) -> str:
        """Determine movement direction from vector"""
        if abs(dx) > abs(dy):
            return "rightward" if dx > 0 else "leftward"
        else:
            return "downward" if dy > 0 else "upward"

    def _describe_depth_composition(self) -> str:
        """Describe depth layer composition"""
        layers = defaultdict(list)
        for obj in self.current_objects.values():
            layers[obj.depth_layer].append(obj.label)

        descriptions = []
        for layer in ["foreground", "midground", "background"]:
            if layer in layers and layers[layer]:
                obj_list = ", ".join(layers[layer])
                descriptions.append(f"{layer}: {obj_list}")

        return " | ".join(descriptions) if descriptions else ""

    def get_scene_stability_score(self) -> float:
        """Calculate overall scene stability (0.0 = very dynamic, 1.0 = very stable)"""
        if not self.current_objects:
            return 0.5

        stability_scores = [obj.stability_score for obj in self.current_objects.values()]
        return sum(stability_scores) / len(stability_scores)

    def get_spatial_continuity_context(self) -> str:
        """Provide context about spatial continuity for consciousness"""
        stability = self.get_scene_stability_score()

        if stability > 0.8:
            return "The spatial arrangement feels stable and familiar"
        elif stability > 0.5:
            return "Some elements are shifting while others remain constant"
        else:
            return "The spatial environment is actively changing"

    def save_spatial_state(self, filepath: str):
        """Save current spatial state to JSON"""
        state = {
            "timestamp": time.time(),
            "current_objects": {k: v.to_dict() for k, v in self.current_objects.items()},
            "stable_layout": dict(self.stable_layout),
            "scene_stability": self.get_scene_stability_score(),
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_spatial_state(self, filepath: str):
        """Load spatial state from JSON"""
        try:
            with open(filepath, "r") as f:
                state = json.load(f)

            # Restore stable layout
            self.stable_layout = state.get("stable_layout", {})

            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False


# Global instance for system-wide access
spatial_memory = SpatialMemory()
