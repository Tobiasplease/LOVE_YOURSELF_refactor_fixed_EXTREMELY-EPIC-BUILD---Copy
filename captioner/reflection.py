import time
from typing import TYPE_CHECKING
from .prompts import build_reflection_prompt, build_drawing_prompt  # , build_awakening_prompt
from event_logging.json_logger import log_json_entry
from drawing.drawing import DrawingController

if TYPE_CHECKING:
    from .captioner import Captioner


def run_reflection(agent: "Captioner") -> None:
    try:
        if not agent.memory_queue:
            return

        mood_delta = agent.current_mood - agent.memory_queue[0]["mood"]
        time_elapsed = int(time.time() - agent.true_session_start)
        recent_entries = list(agent.memory_queue)[-5:]
        recent_summaries = "\n".join(m["text"] for m in recent_entries)

        prompt = build_reflection_prompt(
            agent=agent,
            mood_delta=mood_delta,
            time_elapsed=time_elapsed,
            recent_summaries=recent_summaries,
            identity_summary=agent.get_identity_summary(),
        )

        reflection = agent.brain.think("", extra=prompt).strip()
        print("[🪞] Reflection:", reflection)

        # ✅ Fixed: properly defined reflection log
        log_json_entry(
            "reflection",
            {
                "text": reflection,
                "mood": agent.current_mood,
                "elapsed_seconds": time_elapsed,
                "mood_delta": mood_delta,
            },
            "mood_snapshots",
        )

        # Add to memory
        image_path = agent.memory_queue[-1]["image"]
        derived_from = [m["text"] for m in recent_entries]
        agent.observe(reflection, agent.current_mood, image_path, memory_type="reflection", derived_from=derived_from)

        # ✅ Drawing trigger
        drawing_prompt = build_drawing_prompt(agent.last_caption, agent.get_recent_memory(), reflection)
        DrawingController().handle_drawing_flow(agent, drawing_prompt, image_path, reflection=reflection)

    except Exception as e:
        print(f"[⚠️] Reflection error: {e}")
