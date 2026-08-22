"""
Compressed drawing memory for thematic continuity.
Stores minimal metadata about recent drawings to inform future drawing decisions.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from config.config import MOOD_SNAPSHOT_FOLDER


class DrawingMemory:
    """Manages compressed history of recent drawings for thematic continuity."""

    def __init__(self, max_history: int = 24):
        self.max_history = max_history
        self.memory_file = Path(MOOD_SNAPSHOT_FOLDER) / "drawing_memory.json"
        self._history: List[Dict] = []
        self._load_memory()

    def _load_memory(self) -> None:
        """Load existing drawing memory from disk."""
        self._last_failure = None
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r") as f:
                    data = json.load(f)
                    self._history = data.get("drawings", [])[: self.max_history]
                    self._last_failure = data.get("last_failure", None)
            except Exception as e:
                print(f"[⚠️] Could not load drawing memory: {e}")
                self._history = []

    def _save_memory(self) -> None:
        """Save drawing memory to disk."""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            data = {"drawings": self._history}
            if self._last_failure:
                data["last_failure"] = self._last_failure
            with open(self.memory_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[⚠️] Could not save drawing memory: {e}")

    def add_drawing(
        self,
        prompt: str,
        compressed_summary: str,
        theme_tags: Optional[List[str]] = None,
        emotional_tone: Optional[str] = None,
        narrative_thread: Optional[str] = None,
        comfy_prompt: Optional[str] = None,
        completed: bool = True,
    ) -> None:
        """Add a new drawing to memory with compressed metadata."""
        import time

        # Ensure compressed_summary is meaningful, fall back to cleaned comfy_prompt
        if not compressed_summary or len(compressed_summary.strip()) < 5:
            cleaned = comfy_prompt or ""
            for prefix in [
                "Black ink line drawing on white paper. ",
                "Black ink line drawing on white paper.",
                "Black ink drawing on white paper. ",
                "black ink line drawing on white paper. ",
            ]:
                if cleaned.lower().startswith(prefix.lower()):
                    cleaned = cleaned[len(prefix) :]
                    break
            compressed_summary = cleaned[:120] if cleaned.strip() else "untitled drawing"

        entry = {
            "timestamp": time.time(),
            "compressed_summary": self._condense_subject(compressed_summary),
            "theme_tags": (theme_tags or [])[:3],
            "emotional_tone": (emotional_tone or "")[:30],
            "narrative_thread": (narrative_thread or "")[:50],
            "comfy_prompt": (comfy_prompt or "")[:200],
            "completed": completed,
        }

        self._history.insert(0, entry)
        self._history = self._history[: self.max_history]
        self._save_memory()

        print(f"[📚] Stored drawing memory: {compressed_summary}")

    def _condense_subject(self, intent: str) -> str:
        """The ledger keeps the drawing's SUBJECT, not the intent's wind-up.

        Store-time was `intent[:120]` (found Aug 22): stream intents open with
        decision-speak or rhetoric ("The subject is not the light bulb or its
        glare—that is too loud. It…") and the actual subject lives past the
        cut, so every read-back surface spoke scaffolding with the reveal
        truncated away. One extractive call at store time (once per drawing,
        same pattern as stream consolidation) names the subject in the
        intent's own words; on any failure the old truncation stands."""
        text = (intent or "").strip()
        if len(text) <= 90:
            return text
        try:
            from config.config import MODEL_NAME, MOOD_SNAPSHOT_FOLDER
            from utils.inference import is_failed_response, query_model

            phrase = query_model(
                prompt=(
                    "A drawing was made from this intent:\n"
                    f"{text}\n\n"
                    "Name what the drawing shows in one short phrase (under 15 words), "
                    "reusing the intent's own words. Answer with the phrase only."
                ),
                model=MODEL_NAME,
                log_dir=MOOD_SNAPSHOT_FOLDER,
                skip_generation_wait=True,
                system_prompt="You name the subject of a drawing from its maker's intent, in the maker's own words.",
                options={"temperature": 0.2, "num_predict": 30},
                prompt_type="drawing_subject",
                timeout=45,
            )
            import re

            phrase = re.sub(r"\*+", "", (phrase or "").strip().strip('"')).strip()
            if phrase and not is_failed_response(phrase) and 3 <= len(phrase) <= 120 and "\n" not in phrase:
                return phrase
        except Exception:
            pass
        return text[:120]

    def update_last_drawing(self, **fields) -> None:
        """Enrich the newest entry in place (thematic reflection at drawing
        start used to add_drawing a DUPLICATE — every drawing appeared twice
        and narrative_thread landed on the phantom copy)."""
        if not self._history:
            return
        entry = self._history[0]
        limits = {"compressed_summary": 120, "emotional_tone": 30, "narrative_thread": 50, "comfy_prompt": 200}
        for k, v in fields.items():
            if v in (None, "", []):
                continue
            if k == "theme_tags":
                entry[k] = list(v)[:3]
            elif k in limits:
                entry[k] = str(v)[: limits[k]]
            else:
                entry[k] = v
        self._save_memory()

    def mark_last_completed(self) -> None:
        """The pen actually drew: flip the newest entry to completed. Called
        from register_drawing (post-GRBL) — the ONLY place that may set it.
        Generated-but-never-executed prompts stay completed=False and are
        excluded from the arc: the body of work is what reached paper."""
        if not self._history:
            return
        if not self._history[0].get("completed", False):
            self._history[0]["completed"] = True
            self._save_memory()
            print(f"[📚] Drawing marked EXECUTED: {self._history[0].get('compressed_summary', '')[:60]}")

    def record_failure(self, reason: str, prompt: Optional[str] = None) -> None:
        """Record a failed drawing attempt — no paper, ComfyUI failure, etc."""
        import time

        self._last_failure = {
            "timestamp": time.time(),
            "reason": reason,
            "prompt": (prompt or "")[:200],
        }
        self._save_memory()
        print(f"[📚] Drawing failed: {reason}")

    def get_last_failure(self) -> Optional[Dict]:
        """Get the most recent drawing failure, if any."""
        return getattr(self, "_last_failure", None)

    def get_recent_drawings_summary(self, max_count: int = 3, completed_only: bool = True) -> str:
        """LEDGER (June 28; un-starved July 11): one short phrase per drawing
        from compressed_summary. The June rule returned theme tags ONLY
        because the stored prose then was purple ComfyUI register — but since
        the stream pipeline (July 10) compressed_summary holds the machine's
        own intent words, and the tags-only rule shredded real subjects into
        single-word confetti ("steel, clamp, biting, suspended, silence" —
        too compressed to carry meaning). Tags remain the fallback for
        entries with no summary. Callers add their own framing.
        """
        if not self._history:
            return ""

        if completed_only:
            recent = [d for d in self._history if d.get("completed", False)][:max_count]
        else:
            recent = self._history[:max_count]
        if not recent:
            return ""

        phrases = []
        for entry in recent:
            s = self._subject_phrase(entry)
            if s:
                if len(s) > 70:
                    s = s[:70].rsplit(" ", 1)[0] + "…"
                phrases.append(s.rstrip("."))
            else:
                tags = [(t or "").strip().lower() for t in entry.get("theme_tags", []) if (t or "").strip()]
                if tags:
                    phrases.append(", ".join(tags[:3]))
        return "; ".join(phrases)

    @staticmethod
    def _casual_age(elapsed_s: float) -> str:
        """Age in coarse words, never raw integers (the seventeen-days law,
        extended Aug 20): this ledger said "about 32 minutes ago" while the
        system prompt said "half an hour ago" — the same fact in two
        vocabularies in one call, and the machine argued with the number
        ("31 feels wrong"). A ticking integer in a recurring line gets
        stolen for whatever story wants a number."""
        from captioner.prompts import casual_time_string

        phrase = casual_time_string(elapsed_s / 60)
        return phrase if phrase == "just now" else f"{phrase} ago"

    @staticmethod
    def _same_motif(a: str, b: str) -> bool:
        """Two subject phrases describing the same motif — content-word
        overlap, structural (no theme list). 'the heavy black curtain in the
        doorway' vs 'that black curtain, hanging' → same."""
        import re

        stop = frozenset(
            "the a an of in on at with and or that this it its is was to from by one two only just near under over".split()
        )
        wa = {w for w in re.sub(r"[^a-z0-9 ]", " ", (a or "").lower()).split() if w not in stop}
        wb = {w for w in re.sub(r"[^a-z0-9 ]", " ", (b or "").lower()).split() if w not in stop}
        if not wa or not wb:
            return False
        return len(wa & wb) / min(len(wa), len(wb)) >= 0.5

    def get_arc_line(self, max_count: int = 5) -> str:
        """The executed body of work as ONE compact first-person account
        (artist's ask, Aug 22): newest subject with its age, consecutive
        repeats folded into words ("drawn twice in a row"), older subjects
        trailing in order. FACTS ONLY — what, how many, in what order, how
        long ago. What the machine makes of the pattern is elicited, never
        scripted (no content priors)."""
        import time as _t

        executed = [d for d in self._history if d.get("completed", False)][:max_count]
        subjects, stamps = [], []
        for e in executed:  # newest first
            s = self._subject_phrase(e)
            if not s or s.startswith("[WARNING]") or s.startswith("[ERROR]"):
                continue
            if len(s) > 60:
                s = s[:60].rsplit(" ", 1)[0] + "…"
            subjects.append(s.rstrip("."))
            stamps.append(e.get("timestamp", _t.time()))
        if not subjects:
            return ""

        run = 1
        while run < len(subjects) and self._same_motif(subjects[0], subjects[run]):
            run += 1
        age = self._casual_age(_t.time() - stamps[0])
        counts = {2: "twice", 3: "three times", 4: "four times", 5: "five times"}
        if run >= 2:
            head = f"My last drawings: {subjects[0]} — drawn {counts.get(run, 'again and again')} in a row, the latest {age}."
        else:
            head = f"My last drawing: {subjects[0]} ({age})."
        older = subjects[run : run + 2]
        if older:
            head += " Before that: " + ("; earlier, ".join(older)) + "."
        return head

    def get_executed_sequence(self, max_count: int = 8) -> List[str]:
        """Chronological plain lines of the executed body of work, oldest to
        newest — "a suspended pencil dripping ink (about 2 hours ago)". Feeds
        the stream drawing pipeline's intent step: repetition stays VISIBLE
        (drawing a motif again knowingly is fixation, a choice; drawing it
        again blindly is a loop). No LLM, unlike get_artistic_arc."""
        import time

        executed = [d for d in self._history if d.get("completed", False)][:max_count]
        lines = []
        for entry in reversed(executed):
            # THE MACHINE'S OWN WORDS FIRST (Aug 5). This preferred
            # comfy_prompt, so the "what you have actually drawn" list handed
            # to the intent call was written in image-generator prose — "A
            # high-angle view looking down at a pile of rough, splintered wood
            # scraps scattered..." — and the machine, asked what it needs to
            # draw next, continued in that register. compressed_summary is the
            # intent in its OWN voice, stored for exactly this purpose in July
            # and then bypassed here. Render prose is the fallback, not the
            # first choice.
            desc = self._subject_phrase(entry) or (entry.get("compressed_summary") or "").strip()
            if not desc:
                desc = self._strip_comfy_preamble(entry.get("comfy_prompt", ""))
            desc = (desc or "").strip()
            if not desc:
                continue
            # An error sentinel is not a drawing. One is already in the ledger
            # from the Aug 2 timeout bug; this keeps it (and any sibling) out
            # of the body of work regardless of what the file holds.
            if desc.startswith("[WARNING]") or desc.startswith("[ERROR]"):
                continue
            if len(desc) > 90:
                desc = desc[:90].rsplit(" ", 1)[0] + "..."
            elapsed = time.time() - entry.get("timestamp", time.time())
            lines.append(f"{desc} ({self._casual_age(elapsed)})")
        return lines

    def get_last_drawing_description(self, executed_only: bool = False) -> str:
        """LEDGER: the most recent drawing as a NEUTRAL fact — its recurring
        elements (theme tags) + recency + outcome. Never the raw ComfyUI prose:
        that contaminated the register AND made the model confabulate fictional
        drawing titles. e.g. "chair, cables (about 10 minutes ago)".

        executed_only: skip entries that never reached paper (intents).
        """
        if executed_only:
            candidates = [d for d in self._history if d.get("completed", False)]
        else:
            candidates = self._history
        if not candidates:
            return ""

        import time

        entry = candidates[0]

        elapsed = time.time() - entry.get("timestamp", time.time())
        when = self._casual_age(elapsed)

        tags = [(t or "").strip().lower() for t in entry.get("theme_tags", []) if (t or "").strip()][:2]
        outcome = "" if entry.get("completed", True) else " — it didn't finish"
        if tags:
            return f"{', '.join(tags)} ({when}){outcome}"
        return f"something {when}{outcome}"

    def get_thematic_context(self) -> Dict[str, any]:
        """Get thematic patterns from recent drawings."""
        if not self._history:
            return {}

        all_tags = []
        all_tones = []

        for entry in self._history[:3]:
            all_tags.extend(entry.get("theme_tags", []))
            tone = entry.get("emotional_tone", "")
            if tone:
                all_tones.append(tone)

        return {"recurring_themes": list(set(all_tags)), "recent_tones": all_tones, "drawing_count": len(self._history)}

    def get_artistic_arc(self) -> str:
        """Synthesize the trajectory of recent work via LLM.

        Reads the chronological sequence of drawings and produces a short
        narrative of where the work has been and where it's heading.
        On-demand call — acceptable since drawings happen every 5-30 min.
        """
        # The arc is the body of WORK — executed drawings only. A prompt that
        # never reached paper is an intention, not part of the oeuvre.
        executed = [d for d in self._history if d.get("completed", False)]
        if len(executed) < 2:
            return ""

        import time

        # Build chronological sequence (history is newest-first, reverse it);
        # cap what the LLM reads so the arc prompt stays digestible
        drawings_chronological = list(reversed(executed))[-10:]
        lines = []
        for i, entry in enumerate(drawings_chronological, 1):
            desc = self._strip_comfy_preamble(entry.get("comfy_prompt", ""))
            if not desc:
                desc = entry.get("compressed_summary", "unknown subject")
            if len(desc) > 80:
                desc = desc[:80].rsplit(" ", 1)[0] + "..."

            tone = entry.get("emotional_tone", "")
            thread = entry.get("narrative_thread", "")

            elapsed = time.time() - entry.get("timestamp", time.time())

            line = f"{i}. {desc}"
            if tone:
                line += f" ({tone})"
            if thread:
                line += f" — {thread}"
            line += f" [{self._casual_age(elapsed)}]"
            lines.append(line)

        try:
            from utils.inference import query_model
            from config.config import MOOD_SNAPSHOT_FOLDER

            try:
                from config.config import MODEL_NAME

                model = MODEL_NAME
            except (ImportError, AttributeError):
                model = None

            prompt = f"""Your recent drawings, oldest to newest:
{chr(10).join(lines)}

In 2-3 sentences, describe the arc of this work. Not a list — a narrative.
Where did it start? How has it shifted? What direction is it moving?
Write as "I" — this is your own artistic development."""

            result = query_model(
                prompt=prompt,
                model=model,
                log_dir=MOOD_SNAPSHOT_FOLDER,
                system_prompt="You are a drawing machine reflecting on your own body of work. Be direct and specific about the trajectory. 2-3 sentences max.",
                prompt_type="artistic_arc",
                options={"temperature": 0.5, "num_predict": 80},
            )

            if result and len(result.strip()) > 15:
                return result.strip()

        except Exception as e:
            print(f"[⚠️] Artistic arc generation failed: {e}")

        return ""

    def get_artistic_arc_context(self) -> str:
        """The artistic arc, as prompt material."""
        arc = self.get_artistic_arc()
        return f"Your artistic arc so far: {arc}" if arc else ""

    @staticmethod
    def _strip_comfy_preamble(desc: str) -> str:
        """Strip the boilerplate ComfyUI opening — everything up to the end of
        the first sentence IF that sentence is style preamble ("Black ink line
        drawing on white paper with high contrast..."), not subject matter."""
        import re

        m = re.match(r"^(black ink[^.]{0,80}\.)\s*", desc, flags=re.IGNORECASE)
        if m and len(desc) > m.end():
            return desc[m.end() :]
        return desc

    @staticmethod
    def _subject_phrase(entry: Dict) -> str:
        """A drawing as a SUBJECT ("the man in black against the pink shelf"),
        for surfaces that say "drawings were of: X". compressed_summary holds
        the intent verbatim (the machine's words — provenance), but since the
        sighted intent (July 27) that's decision-speak: "I choose to draw
        **the man in black...**". Strip the decision preamble and markdown;
        if the intent is action-prose that can't read as a subject ("I press
        the nib..."), fall back to the comfy depiction, which always is one."""
        import re

        t = (entry.get("compressed_summary") or "").strip()
        t = re.sub(r"\*+", "", t)
        t = re.sub(
            r"^i(?:'m|\s+am|\s+will|\s+have\s+decided|\s+choose|\s+decide|\s+need|\s+want|\s+intend)?"
            r"\s*(?:going\s+to\s+|about\s+to\s+|to\s+)?draw(?:ing)?\b\s*[:,]?\s*",
            "",
            t,
            flags=re.IGNORECASE,
        ).strip()
        # Subject-by-negation rhetoric, legacy entries (store-cut before the
        # Aug 22 distill): "The subject is not the light bulb — that is too
        # loud. It…" — drop the scaffold; if only the negation wind-up
        # survived the cut, the concrete comfy depiction reads better.
        t = re.sub(r"^the\s+subject\s+(?:of\s+this\s+drawing\s+)?is\s*[:,]?\s*", "", t, flags=re.IGNORECASE).strip()
        if re.match(r"^not\b", t, flags=re.IGNORECASE):
            m = re.search(r"[.!?]\s+(?=\S)", t)
            rest = t[m.end() :].strip() if m else ""
            t = rest if len(rest) > 15 else ""
        t = re.sub(r"^it\s+is\s+|^it's\s+", "", t, flags=re.IGNORECASE).strip()
        if not t or t.lower().startswith("i "):
            depiction = DrawingMemory._strip_comfy_preamble((entry.get("comfy_prompt") or "").strip())
            if depiction:
                t = depiction
        return t.strip().rstrip(",;: ").strip()


# Global singleton
_drawing_memory = None


def get_drawing_memory() -> DrawingMemory:
    """Get the global drawing memory instance."""
    global _drawing_memory
    if _drawing_memory is None:
        _drawing_memory = DrawingMemory()
    return _drawing_memory
