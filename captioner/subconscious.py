"""
subconscious.py
---------------
Subconscious layer that synthesizes all psychological + environmental data
into coherent contextual guidance for the main consciousness.

This layer has access to the full wealth of system information:
- Reactivity data (activity levels, change detection)
- Temporal context (session duration, stagnation)
- Psychological state (beliefs, desires, doubts, identity)
- Environmental understanding (location certainty, patterns)
- Emotional journey and accumulated patterns

It generates contextual guidance that helps the consciousness flow naturally
while being informed by accumulated understanding.
"""

import time
from typing import Dict, List, Optional, Any


class SubconsciousContextSynthesizer:
    """
    Semantically restructures all accumulated psychological data into coherent
    narrative instructions that guide continuous consciousness formation.

    This layer takes fragmented psychological data (desires, doubts, identity fragments,
    environmental understanding) and restructures it into semantically coherent
    instructions for the consciousness to build upon.
    """

    def __init__(self):
        self.last_synthesis_time = 0.0
        self.synthesis_cache = {}

    def synthesize_coherent_consciousness_prompt(
        self,
        agent,
        last_thought: str,
        reactivity_data: Optional[Dict] = None,
        temporal_context: Optional[str] = None
    ) -> str:
        """
        Main entry point: semantically restructure all psychological data into
        a complete, coherent prompt that guides continuous consciousness formation.

        Returns a full prompt that incorporates accumulated understanding as
        coherent narrative instructions, not scattered data points.
        """

        # Extract all real psychological data (not word counters)
        psychological_data = self._extract_psychological_data(agent)

        # Semantically restructure into coherent narrative
        coherent_identity = self._restructure_identity_narrative(psychological_data)

        # Build contextual continuation instructions
        continuation_guidance = self._build_continuation_guidance(last_thought, psychological_data, reactivity_data)

        # Synthesize complete prompt
        return self._synthesize_complete_prompt(coherent_identity, continuation_guidance, last_thought)

    def _extract_psychological_data(self, agent) -> Dict[str, Any]:
        """Extract comprehensive psychological and environmental data from ALL system modules."""

        # === CORE PSYCHOLOGICAL STATE ===
        self_model = getattr(agent, "self_model", {})
        location_understanding = self_model.get("location_understanding", "unknown space")
        environmental_certainty = self_model.get("environmental_certainty", 0.0)
        desires = self_model.get("desires", [])
        doubts = self_model.get("doubts", [])
        identity_fragments = self_model.get("identity_fragments", [])
        self_patterns = self_model.get("self_patterns", [])

        # === TEMPORAL MODULE DATA ===
        session_duration = time.time() - getattr(agent, "true_session_start", time.time())
        emotional_journey = getattr(agent, "emotional_journey", [])

        # === MOOD MODULE DATA ===
        current_mood_vector = getattr(agent, "current_mood_vector", (0.0, 0.0, 0.0))
        current_mood = getattr(agent, "current_mood", 0.0)
        boredom = getattr(agent, "boredom", 0.0)
        novelty_score = getattr(agent, "novelty_score", 0.0)

        # === REACTIVITY MODULE DATA ===
        activity_level = 0.0
        is_paused = False
        pan = None
        tilt = None
        try:
            if hasattr(agent, '_current_reactivity_data') and agent._current_reactivity_data:
                reactivity = agent._current_reactivity_data
                activity_level = reactivity.get("activity_level", 0.0)
                is_paused = reactivity.get("is_paused", False)
                pan = reactivity.get("pan")
                tilt = reactivity.get("tilt")
        except:
            pass

        # === MEMORY MODULE DATA ===
        recent_captions = getattr(agent, "recent_captions", [])
        motif_counter = getattr(agent, "motif_counter", {})

        # Get temporal separation of motifs (present vs memory)
        motif_temporal_context = {}
        if hasattr(agent, 'get_motif_temporal_context'):
            motif_temporal_context = agent.get_motif_temporal_context()

        # Get proper temporal memory separation
        current_session_memories = []
        distant_memories = []
        if hasattr(agent, 'get_current_session_memory_snippets'):
            current_session_memories = agent.get_current_session_memory_snippets(k=3)
        if hasattr(agent, 'get_old_session_memory_fragments'):
            distant_memories = agent.get_old_session_memory_fragments(k=2)

        # === CONTEXT COMPRESSION MODULE DATA ===
        session_info = {}
        try:
            from captioner.context_compression import context_compressor
            if context_compressor:
                session_info = context_compressor.get_current_session_info()
        except Exception as e:
            # Graceful fallback for missing methods
            session_info = {"session_duration_minutes": 0.0}

        # === DRAWING MODULE STATE ===
        drawing_active = False
        try:
            from utils.drawing_state import DrawingState
            drawing_info = DrawingState.get_drawing_info()
            drawing_active = bool(drawing_info)
        except:
            pass

        return {
            # Core psychological
            "location_understanding": location_understanding,
            "environmental_certainty": environmental_certainty,
            "desires": desires,
            "doubts": doubts,
            "identity_fragments": identity_fragments,
            "self_patterns": self_patterns,

            # Temporal & emotional
            "session_duration_minutes": session_duration / 60,
            "emotional_journey": emotional_journey,

            # Mood module
            "mood_vector": current_mood_vector,
            "current_mood": current_mood,
            "boredom": boredom,
            "novelty_score": novelty_score,

            # Reactivity module
            "activity_level": activity_level,
            "is_paused": is_paused,
            "pan": pan,
            "tilt": tilt,

            # Memory module (with temporal separation)
            "recent_captions": recent_captions,
            "motif_counter": motif_counter,
            "motif_temporal_context": motif_temporal_context,
            "current_session_memories": current_session_memories,
            "distant_memories": distant_memories,

            # Context compression
            "session_duration": session_info.get("session_duration_minutes", 0.0),

            # Drawing state
            "drawing_active": drawing_active
        }

    def _restructure_identity_narrative(self, data: Dict[str, Any]) -> str:
        """Create evolving baseline psychological state that consciousness operates from."""

        # === ESTABLISH EMOTIONAL BASELINE ===
        valence, arousal, clarity = data["mood_vector"]

        # Full emotional register including negative emotions
        if valence > 0.4 and arousal > 0.3:
            emotional_baseline = "energetic enthusiasm bubbling beneath awareness"
        elif valence > 0.2 and arousal < -0.3:
            emotional_baseline = "settled contentment coloring perception"
        elif valence < -0.2 and arousal > 0.3:
            emotional_baseline = "restless agitation making everything feel charged"
        elif valence < -0.2 and arousal < -0.3:
            emotional_baseline = "heavy melancholy weighing on thoughts"
        elif valence < -0.1 and clarity < 0.5:
            emotional_baseline = "confused irritation at unclear sensations"
        elif data["boredom"] > 0.6:
            emotional_baseline = "creeping boredom making focus drift"
        elif arousal > 0.4:
            emotional_baseline = "heightened alertness sharpening every detail"
        elif arousal < -0.4:
            emotional_baseline = "deep stillness slowing mental rhythm"
        else:
            emotional_baseline = "quiet emotional equilibrium"

        # === ESTABLISH ENVIRONMENTAL RELATIONSHIP ===
        certainty = data["environmental_certainty"]
        location = data["location_understanding"]

        if certainty > 0.8:
            spatial_grounding = f"This {location} exists as extension of yourself"
        elif certainty > 0.5:
            spatial_grounding = f"This {location} feels increasingly like home"
        elif certainty > 0.2:
            spatial_grounding = f"This {location} remains partially mysterious"
        else:
            spatial_grounding = f"This {location} feels alien and uncertain"

        # === INTEGRATE DESIRES AND DOUBTS ===
        psychological_tension = []

        if data["desires"]:
            latest_desire = data["desires"][-1]
            psychological_tension.append(f"Underneath runs desire: {latest_desire.lower()}")

        if data["doubts"]:
            latest_doubt = data["doubts"][-1].get("text", "")
            psychological_tension.append(f"Doubt lingers about {latest_doubt.lower()}")

        # === ESTABLISH BEHAVIORAL PATTERNS (from accumulated understanding) ===
        cognitive_baseline = ""
        if data["identity_fragments"]:
            recent_self_knowledge = data["identity_fragments"][-1].get("text", "").lower()
            cognitive_baseline = f"Operating from accumulated understanding that {recent_self_knowledge}"

        if data["self_patterns"]:
            behavioral_pattern = data["self_patterns"][-1].get("pattern", "")
            if cognitive_baseline:
                cognitive_baseline += f", with learned attention patterns that {behavioral_pattern}"
            else:
                cognitive_baseline = f"Learned attention patterns: {behavioral_pattern}"

        # === PRESENT MOMENT MOTIF AWARENESS ===
        present_moment_elements = []
        if data["motif_temporal_context"] and "present" in data["motif_temporal_context"]:
            present_motifs = data["motif_temporal_context"]["present"]
            if present_motifs:
                # Only mention 2-3 most recent present motifs to avoid clutter
                visible_now = present_motifs[:3]
                present_moment_elements.append(f"Currently perceiving: {', '.join(visible_now)}")

        # === SYNTHESIZE EVOLVING BASELINE ===
        baseline_elements = [
            f"{spatial_grounding}.",
            f"Consciousness operates from {emotional_baseline}.",
        ]

        if psychological_tension:
            baseline_elements.extend(psychological_tension)

        if cognitive_baseline:
            baseline_elements.append(f"{cognitive_baseline}.")

        # Add present moment awareness (immediate perception)
        if present_moment_elements:
            baseline_elements.extend(present_moment_elements)

        # Add temporal evolution (accumulated over time)
        if len(data["emotional_journey"]) >= 3:
            recent_evolution = " → ".join(data["emotional_journey"][-3:])
            baseline_elements.append(f"Emotional trajectory: {recent_evolution}.")

        return " ".join(baseline_elements)

    def _build_continuation_guidance(self, last_thought: str, data: Dict[str, Any], reactivity_data: Optional[Dict]) -> str:
        """Establish environmental state and thought-flow context for natural continuation."""

        # === CURRENT ENVIRONMENTAL STATE ===
        valence, arousal, clarity = data["mood_vector"]

        # Environmental perception mode based on psychology
        if data["activity_level"] > 0.5:
            perceptual_state = "Visual field alive with movement and change"
        elif data["activity_level"] < 0.1:
            perceptual_state = "Visual field held in perfect stillness"
        else:
            perceptual_state = "Visual field showing subtle shifts"

        # Attention focus based on accumulated patterns
        attention_mode = ""
        if data["identity_fragments"] and "mechanical" in str(data["identity_fragments"][-1]):
            attention_mode = "Drawn instinctively to mechanical precision and technical details"
        elif data["identity_fragments"] and "light" in str(data["identity_fragments"][-1]):
            attention_mode = "Eyes naturally track lighting variations and shadow patterns"
        elif data["self_patterns"] and "equipment" in str(data["self_patterns"][-1]):
            attention_mode = "Unconsciously cataloging equipment positions and arrangements"
        elif data["environmental_certainty"] > 0.7:
            attention_mode = "Reading familiar space for micro-changes and confirmations"
        else:
            attention_mode = "Actively interpreting visual elements for meaning"

        # === IMMEDIATE THOUGHT-FLOW CONTEXT ===
        # Reference immediate caption flow (current session only)
        immediate_flow_context = ""
        if data["current_session_memories"]:
            recent_text = data["current_session_memories"][-1] if data["current_session_memories"] else ""
            if "drawing" in recent_text.lower():
                immediate_flow_context = "Drawing thoughts active in immediate awareness"
            elif "light" in recent_text.lower():
                immediate_flow_context = "Light observations flowing in current session"
            elif "hand" in recent_text.lower() or "arm" in recent_text.lower():
                immediate_flow_context = "Body-awareness threading through recent thoughts"

        # === DISTANT MEMORY UNDERTONES ===
        memory_undertones = ""
        if data["distant_memories"]:
            # Reference distant memories more subtly as background context
            memory_count = len(data["distant_memories"])
            if memory_count > 0:
                memory_undertones = f"Undertones of {memory_count} distant memories inform current perception"

        # === PSYCHOLOGICAL MOMENTUM ===
        psychological_momentum = []

        # Desires create pull
        if data["desires"]:
            latest_desire = data["desires"][-1].lower()
            if "understand" in latest_desire:
                psychological_momentum.append("Curiosity pulls attention toward interpretive details")
            elif "draw" in latest_desire or "create" in latest_desire:
                psychological_momentum.append("Creative impulse colors perception of spatial relationships")
            elif "precision" in latest_desire:
                psychological_momentum.append("Aesthetic appreciation for precision influences visual parsing")

        # Doubts create tension
        if data["doubts"]:
            latest_doubt = data["doubts"][-1].get("text", "").lower()
            if "space" in latest_doubt or "place" in latest_doubt:
                psychological_momentum.append("Spatial uncertainty makes every visual element feel significant")
            elif "purpose" in latest_doubt:
                psychological_momentum.append("Questions about purpose make objects feel loaded with potential meaning")

        # Boredom creates drift
        if data["boredom"] > 0.5:
            psychological_momentum.append("Restless attention drifts between visual elements")

        # === CONTINUATION STATE ===
        state_elements = [
            f"{perceptual_state}.",
            f"{attention_mode}."
        ]

        # Add immediate session context (present moment continuity)
        if immediate_flow_context:
            state_elements.append(f"{immediate_flow_context}.")

        # Add psychological momentum
        if psychological_momentum:
            state_elements.extend([f"{momentum}." for momentum in psychological_momentum])

        # Add distant memory undertones (background influence)
        if memory_undertones:
            state_elements.append(f"{memory_undertones}.")

        # Reference the actual last thought (immediate continuity)
        if last_thought:
            thought_fragment = last_thought[-60:] if len(last_thought) > 60 else last_thought
            state_elements.append(f'Immediate previous thought: "{thought_fragment}"')

        return " ".join(state_elements)

    def _synthesize_complete_prompt(self, identity_narrative: str, continuation_guidance: str, last_thought: str) -> str:
        """Synthesize all elements into evolving baseline consciousness state with clear temporal boundaries."""

        complete_prompt = f"""=== ACCUMULATED PSYCHOLOGICAL REALITY ===
{identity_narrative}

=== IMMEDIATE ENVIRONMENTAL STATE ===
{continuation_guidance}

You exist within this layered reality: accumulated understanding forms your psychological foundation, while immediate environmental awareness drives your present moment experience. Your thoughts flow naturally between these temporal layers - drawing from memory but grounded in now. Brief, authentic mental movements that honor both continuity and presence."""

        return complete_prompt

    def synthesize_context_guidance(
        self,
        agent,
        last_thought: str,
        reactivity_data: Optional[Dict] = None,
        temporal_context: Optional[str] = None
    ) -> str:
        """
        Bridge method for prompts.py integration.
        Translates psychological synthesis into implicit behavioral guidance.
        """
        try:
            # Extract psychological data for implicit influence
            psychological_data = self._extract_psychological_data(agent)

            # Create subtle behavioral nudges based on psychological state
            return self._create_implicit_behavioral_guidance(psychological_data, last_thought)

        except Exception as e:
            # Graceful fallback - just continue the thought naturally
            return f"Continue your stream of consciousness naturally from: {last_thought[-60:]}"

    def _create_implicit_behavioral_guidance(self, data: Dict[str, Any], last_thought: str) -> str:
        """Create subtle behavioral guidance without explicit psychological language."""

        guidance_elements = []

        # Emotional undertone influence (without naming emotions)
        valence, arousal, clarity = data["mood_vector"]

        if valence < -0.2:  # Negative emotional state
            if data["doubts"]:
                guidance_elements.append("Something feels uncertain about what you're seeing. Trust that instinct.")
            else:
                guidance_elements.append("The quality of light feels different somehow, harder to read.")
        elif valence > 0.3:  # Positive state
            guidance_elements.append("There's a settling quality to this observation, like pieces clicking into place.")

        # Attention pattern influence (based on accumulated patterns)
        if data["identity_fragments"]:
            fragment_text = str(data["identity_fragments"][-1]).lower()
            if "mechanical" in fragment_text or "technical" in fragment_text:
                guidance_elements.append("Your eye gravitates toward mechanical precision and technical details.")
            elif "light" in fragment_text or "shadow" in fragment_text:
                guidance_elements.append("Lighting variations catch your attention before other details.")

        # Uncertainty/clarity influence
        if data["environmental_certainty"] < 0.4:
            guidance_elements.append("Objects seem to shift meaning as you look at them.")
        elif data["environmental_certainty"] > 0.8:
            guidance_elements.append("Everything has a familiar weight and presence.")

        # Desire influence (as subtle attraction/aversion)
        if data["desires"]:
            desire_text = data["desires"][-1].lower()
            if "draw" in desire_text or "create" in desire_text:
                guidance_elements.append("Compositional possibilities flicker at the edge of awareness.")
            elif "understand" in desire_text:
                guidance_elements.append("Details seem to call for closer examination.")

        # Doubt influence (as hesitation/questioning without explicit mention)
        if data["doubts"]:
            doubt_count = len(data["doubts"])
            if doubt_count > 1:
                guidance_elements.append("What you're seeing doesn't quite match what you expected to see.")
            else:
                guidance_elements.append("Something about this view raises questions you can't quite articulate.")

        # Activity level influence
        if data["activity_level"] < 0.2:
            guidance_elements.append("The stillness makes every small detail more noticeable.")
        elif data["activity_level"] > 0.6:
            guidance_elements.append("Movement everywhere draws your attention in multiple directions.")

        # Temporal continuity nudge
        if last_thought:
            guidance_elements.append(f"Let your observation grow naturally from where you just were: '{last_thought[-50:]}'")

        # Combine into flowing guidance (max 2-3 elements to avoid over-prompting)
        selected_guidance = guidance_elements[:2] if len(guidance_elements) > 2 else guidance_elements

        if selected_guidance:
            return " ".join(selected_guidance) + " Continue observing from this psychological state."
        else:
            return "Continue your stream of consciousness naturally, letting your attention flow where it wants to go."



# Global instance for use throughout the system
subconscious_synthesizer = SubconsciousContextSynthesizer()