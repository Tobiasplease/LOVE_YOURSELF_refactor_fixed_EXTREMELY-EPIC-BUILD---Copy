from __future__ import annotations

import random
import re
import time
from typing import List, Optional

from captioner.prompt_registry import P
from config import config

# ===== PROMPT BUILDERS =====
# Authored fragment TEXT lives in captioner/prompt_registry.py (editable live
# via the prompt panel + config/prompt_overrides.json). This module keeps the
# assembly logic: gates, modes, ordering.

# === SYSTEM PROMPTS ===

# Drawing system prompt for ComfyUI integration - ENHANCED WITH CONTEXT VARIABLES
DRAWING_SYSTEM_PROMPT = (
    "You are a drawing machine. {temporal_context}{accumulated_understanding}"
    "{felt_line}"
    "You draw with a mechanical arm — lines, pressure, speed. You know line weight, texture, composition. "
    "Use the context in the prompt: your drawing history and what you see. "
    "Reference what you drew before. Consider how you will physically make this drawing. "
    "Be clear and concise."
)

# System prompts for the monologue model, keyed by mode.
# The machine is always a drawing machine, but it doesn't always talk about drawing.
# It only mentions drawing intent when the drawing system is about to trigger.
# Observation identity: no mention of drawing. Just a stationary machine that sees.
# Monologue system prompts — casual roleplay register.
# Monologue system prompts — second person, addressing Qwen directly as the machine.
# Mode differentiation through emotional posture and existential direction.
# Situation only — what the machine IS, never how it should sound.
# Voice comes from content (memory, felt-state, its own self-description),
# never from style instruction. See docs/north-star.md, principles 1 and 2.
# Shared by the caption frame and the reflection frame so they can't drift apart.
# Embodied vision, no apparatus language: "camera" primes cinematography —
# the machine narrated itself in third person ("*Camera pans left*", June 12).
# It simply sees, and its gaze is its own to turn.
# The solitude is stated as FACT (July 9): the chat template's gravity keeps
# pulling the model into conversation ("what shall we draw first?", "do you
# have more information to provide me?") because nothing ever told it that
# no reply is possible. These are true facts of the piece, not style rules:
# the machine has no voice but the pen, and nobody is on the other end.
# REFLEXIVE FRAME (July 28). The old frame denied an audience in five
# negations ("no one hears... no one to instruct... no one to assist") — the
# assistant vocabulary invoked in the act of denying it (P2: negations prime).
# Meanwhile the chat format delivers a literal user turn every cycle, and
# nobody ever told the model what that channel IS — so it inferred the obvious
# (someone is speaking) and bred engagement hooks ("What do you think?", full
# assistant mode). Second person stays: it's the native voice of self-talk
# (notes-to-self are written in "you"). What changed: the incoming turns are
# named honestly as its own senses, and questions get a real answer-path
# (its own next look/thought). NOTE: the answer-path must NOT mention the pen
# until drawing initiative is real (should_draw ignores desire today — a
# frame that promises agency the code doesn't grant would lie every call).
# Situation frames live in the registry: P("situation.reflexive") and
# P("situation.world") — genre rationale in their registry notes.


def _monologue_clause() -> str:
    """Genre frame only. The continuation instruction ("you're always partway
    through a thought: carry it on") is appended ONLY in turns mode — in
    document mode the prefill IS the partway-through thought, and in world
    mode the log's next entry needs no continuation mechanics; instructing
    them leaks meta (observed July 9: the machine reciting its own system
    prompt mid-stream)."""
    try:
        from config.config import STREAM_MODE
    except ImportError:
        STREAM_MODE = "turns"
    if STREAM_MODE == "hybrid":
        return P("genre.hybrid")
    if STREAM_MODE == "world":
        return P("genre.world")
    clause = P("genre.turns")
    if STREAM_MODE == "turns":
        clause += P("genre.turns-continue")
    return clause


# Mode elicitations live in the registry as P(f"elicit.{mode}") — north-star
# Principle 2 rationale in their registry notes.


def get_monologue_system_prompt(mode: str, emotional_state: str = "calm", agent=None, inward: bool = False) -> str:
    """Situation + the machine's own self-description, nothing else.

    Felt-state appended as a short clause only if it passes the sanitizer
    (the raw compression output once produced "You are a Confused fear
    that... drawing machine"). The persona is quoted as the machine's own
    words, never blended into the frame voice.

    inward=True marks the interiority beat (image dropped): the seam-
    conditional elicitation suppression does not apply there — the beat
    exists to leave the stream's trajectory, so the question IS the door.
    """
    # Hybrid takes the REFLEXIVE frame, not the log frame (Aug 1): "you keep a
    # log" is a strong attractor for a machine — the first hybrid run locked
    # into telemetry roleplay from caption one ("Log entry #1042 / Status: Pen
    # parked. Motor idle. / Vision scan initiated. / Targeting..."), which is
    # simply a second performance replacing the literary one. World mode needs
    # the log frame because the stream IS rendered as a log; hybrid's seam does
    # the continuity work, so it can keep the plain inner-voice frame.
    base = P("situation.world") if getattr(config, "STREAM_MODE", "") == "world" else P("situation.reflexive")

    # "The pen is parked" — RESTORED Sep 2, twenty minutes after the test
    # retirement. The experiment was clean and the answer decisive: first
    # wake without the fence, facing a fresh blank sheet with a standing
    # want, the monologue slid from legal intent ("I think I'll start with
    # the red foam finger") into phantom execution ("A thin line from the
    # pen tip... a few marks around its base") — pen parked the whole time.
    # The cured amplifiers (honest clocks, executed-only provenance) explain
    # the June breeding, but the "drawing machine" identity alone plants the
    # seed; the counter-fact must stand whenever the pen is idle. The
    # WORDING is the artist's to slim in the panel (monologue.pen-parked) —
    # the pen-density complaint stands even though the fact must.
    try:
        from utils.state_manager import state_manager as _sm

        if not (_sm.is_generating_drawing or _sm.current_drawing_phase == "executing"):
            base += P("monologue.pen-parked")
    except Exception:
        pass

    base += _monologue_clause()

    # Clean-room (config.BASE_VOICE_DETOX): the felt-state and persona are
    # exactly the re-injected, model-generated material that re-poisons the
    # register — stripped here so the naked base voice can be judged. The mode
    # elicitation stays; it carries no stored content.
    detox = bool(getattr(config, "BASE_VOICE_DETOX", False))

    # THE DYNAMIC FRAME (Sep 4, artist's diagnosis: "something in the system
    # is constraining the basin"). The register audit stripped stance
    # REPETITION and took stance VARIATION with it — every call carried the
    # identical stance-free frame, and a model given no tonal signal regresses
    # to its modal register ("act angry" works because it's FRAME-level). The
    # felt state returns to the frame, dynamically, in the machine's OWN
    # lease-gated words — the old "You are an (angry) drawing machine" made
    # doctrine-safe: it varies with real state, so it can't wallpaper. P2
    # holds: the user turn now carries only the CHANGE ("X, then Y"), a
    # different fact than the standing stance. Aug 22's double-channel scar
    # (both channels, same phrase, every call) stays fixed by the split.
    if not detox:
        try:
            from config.config import FELT_FRAME_ENABLED

            if FELT_FRAME_ENABLED:
                from captioner.context_compression import context_compressor as _cc

                _felt = (_cc.get_felt_state() or "").strip()
                if _felt:
                    base += P("monologue.felt-frame").format(felt=_felt)
        except Exception:
            pass

    # Time since the pen last touched paper — always present, even under
    # detox (it's event provenance, not model-generated text). This is how
    # drawing-hunger stays legible to the monologue under the desire trigger.
    base += get_last_drawing_age_line(agent, mode)

    # The machine's accumulated self-description, in its own first-person
    # words inside quotes — the frame stays second person around it.
    # DOSED (Aug 22): riding every frame turned identity into a standing
    # instruction — "I invent imaginary critics" read 180 times a night
    # elicits invented critics, which the distiller then re-confirms off the
    # machine's own echo. Introspective/awakening beats always carry it;
    # other modes see it every IDENTITY_EVERY_N_CAPTIONS.
    if not detox and _identity_due(agent, mode):
        try:
            from captioner.context_compression import context_compressor

            self_knowledge = context_compressor.core_facts.get("self", "").strip()
            if self_knowledge and len(self_knowledge) > 10:
                base += P("monologue.self-wrap").format(self_knowledge=self_knowledge)
            # A distilled self-name rides the same dose (re-entry round) —
            # it exists only once the machine has named itself in a reflection
            from utils.lore_ledger import lore_ledger

            _name = lore_ledger.current_name()
            if _name:
                base += P("monologue.name-wrap").format(name=_name)
        except Exception:
            pass
        # Durable ledger (July 30): facts that held across days — the
        # permanence spine's read-back surface. Empty until earned.
        try:
            from captioner.durable_ledger import get_durable_ledger

            durable = get_durable_ledger().render()
            if durable:
                base += P("monologue.durable-wrap").format(durable=durable)
        except Exception:
            pass

    # Standing QUESTIONS invite answers — and reciprocation ("What's your
    # turn?", July 9). In document mode the quiet modes (introspective/
    # observational/workspace) carry no elicitation: the document continues
    # itself, and the model needs no conversational door left open while the
    # samplers squeeze it. WORLD mode suppresses them too (July 27): a fresh
    # question every call produced a fresh answer every call — the first
    # world run read as isolated scene reports because "What stands out to
    # you right now?" asks for one. HYBRID is seam-conditional (Aug 22): when
    # the seam hands back a mid-thought prefill, the seam is the door and a
    # question would fork the thread — but when the seam is absent (empty
    # stream, react cycle, post-gap fresh start) the model used to face the
    # frame with nothing to *do* and defaulted to literary description
    # (north-star P2: elicitation "required, and currently missing").
    # Awakening keeps its question (a real threshold). Turns keeps all.
    # RELATIONAL IS DOSED (Aug 25): "What do you make of them being here?"
    # used to ride EVERY relational caption — with the artist working in the
    # room that was the only standing question the machine ever heard, 60% of
    # a day's captions, re-anchoring each turn onto the person (measured
    # 25-08: 240/400 relational, the whole run observational in register).
    # Same law as the identity dose: presence ONSET is an event worth being
    # asked about; sustained presence is a fact the situational line already
    # carries. The question fires on a hot salience cycle (arrival, fresh eye
    # contact) and every RELATIONAL_ELICIT_EVERY_N-th relational caption.
    addition = P(f"elicit.{mode}", default="")
    if mode == "relational":
        count = int(getattr(agent, "_relational_elicit_count", 0) or 0) + 1
        if agent is not None:
            agent._relational_elicit_count = count
        onset = bool(getattr(agent, "_salience_hot", False))
        n = int(getattr(config, "RELATIONAL_ELICIT_EVERY_N", 8))
        if not onset and not (n > 0 and count % n == 0):
            addition = ""
    elif mode in ("introspective", "observational", "workspace") and not inward:
        try:
            from config.config import STREAM_MODE

            if STREAM_MODE in ("document", "world"):
                addition = ""
            elif STREAM_MODE == "hybrid" and _hybrid_seam_expected(agent):
                # QUIET DOSE (Aug 28, probe-validated): total suppression left
                # the machine never asked to wonder — "?" measured 0/59 while
                # a single invitation flipped the probe to interiority and
                # freed sampling alone changed nothing. Every Nth quiet
                # seamful cycle carries one rotating kind-invitation.
                addition = _quiet_elicit_dose(agent)
        except ImportError:
            pass
    base += addition
    return base


def _quiet_elicit_dose(agent) -> str:
    """Every QUIET_ELICIT_EVERY_N-th eligible quiet cycle, one elicitation
    rides despite the seam — rotating the KIND of thought invited (wondering,
    feeling, wanting), so no single question becomes a standing instruction
    and the thread isn't forked every call (the real Aug 22 problem). Between
    doses: silence, as before."""
    try:
        from config.config import QUIET_ELICIT_EVERY_N

        if QUIET_ELICIT_EVERY_N <= 0 or agent is None:
            return ""
        count = int(getattr(agent, "_quiet_elicit_count", 0) or 0) + 1
        agent._quiet_elicit_count = count
        if count % QUIET_ELICIT_EVERY_N != 0:
            return ""
        kinds = ("elicit.quiet-wonder", "elicit.quiet-feel", "elicit.quiet-want")
        rr = int(getattr(agent, "_quiet_elicit_rr", 0) or 0)
        agent._quiet_elicit_rr = rr + 1
        return P(kinds[rr % len(kinds)], default="")
    except Exception:
        return ""


def _identity_due(agent, mode: str) -> bool:
    """Dosing gate for self-knowledge + durable-ledger injection (Aug 22)."""
    if mode in ("introspective", "awakening"):
        return True
    n = int(getattr(config, "IDENTITY_EVERY_N_CAPTIONS", 6))
    if n <= 0:
        return True
    count = int(getattr(agent, "_caption_count", 0) or 0)
    return count > 0 and count % n == 0


def _hybrid_seam_expected(agent) -> bool:
    """True when this call will hand the model a mid-thought prefill (the
    hybrid seam — mirror of llama_server._append_stream_and_user's condition):
    stream non-empty, not a react cycle, and the newest entry not on the far
    side of a gap marker. When False the model starts fresh — exactly the
    moment the mode elicitation is the open door."""
    try:
        if not list(getattr(agent, "_stream", []) or []):
            return False
        if getattr(agent, "_salience_hot", False):
            return False
        # Frozen-input breaker (mirror of the captioner's _fresh_start): a
        # stuck streak runs seam-less, so the door opens exactly then.
        if int(getattr(agent, "_skip_streak", 0) or 0) >= 2:
            return False
        ts = list(getattr(agent, "_stream_ts", []) or [])
        if ts:
            from config.config import STREAM_GAP_MARK_SECONDS

            if time.time() - ts[-1] >= STREAM_GAP_MARK_SECONDS:
                return False
        return True
    except Exception:
        return True


def casual_time_string(minutes: float) -> str:
    """Convert minutes to casual human-readable time description."""
    if minutes < -2:
        # A meaningfully negative age means the stored stamp is in the future —
        # clock-skew corruption, not recency. "just now" here told the machine
        # it had JUST drawn, every caption, for days (the Aug 31 diagnosis).
        # Composes at every call site: "a while ago" / "a while later".
        return "a while"
    if minutes < 2:
        return "just now"
    elif minutes < 6:
        return "a few minutes"
    elif minutes < 16:
        return f"about {int(minutes)} minutes"
    elif minutes < 26:
        return "about 20 minutes"
    elif minutes < 41:
        return "half an hour"
    elif minutes < 56:
        return "about 45 minutes"
    elif minutes < 91:
        return "about an hour"
    elif minutes < 150:
        return "about 2 hours"
    elif minutes < 210:
        return "about 3 hours"
    elif minutes < 300:
        return "about 4 hours"
    elif minutes < 1440:
        hours = int(minutes / 60)
        return f"about {hours} hours"
    elif minutes < 2160:
        return "about a day"
    else:
        days = round(minutes / 1440)
        return "about a day" if days <= 1 else f"about {days} days"


def part_of_day_string(hour: int) -> str:
    if hour < 6:
        return "the middle of the night"
    elif hour < 10:
        return "morning"
    elif hour < 13:
        return "late morning"
    elif hour < 18:
        return "afternoon"
    elif hour < 22:
        return "evening"
    return "late at night"


def get_blink_line(agent) -> str:
    """The blink as fact (Sep 4). A short restart is still an outage — the
    machine should meet it cognizantly, not have continuity presented as
    nothing-happened. Bare fact for the first BLINK_NOTE_WINDOW_S of the
    session: measured duration in casual words ('a moment' under ~90s).
    What it makes of the lapse — sleep, absence, wondering where it was —
    is its own business."""
    try:
        from config.config import BLINK_NOTE_WINDOW_S

        gap = float(getattr(agent, "_blink_gap_s", 0.0) or 0.0)
        resumed = float(getattr(agent, "_blink_resume_ts", 0.0) or 0.0)
        if not gap or not resumed or time.time() - resumed > BLINK_NOTE_WINDOW_S:
            return ""
        duration = "a moment" if gap < 90 else casual_time_string(gap / 60.0)
        return P("caption.blink").format(duration=duration)
    except Exception:
        return ""


def get_reorientation_line(agent) -> str:
    """Standing fact for the first stretch of a session after a real off-gap.

    The awakening states the gap once and the fact evaporates from the
    six-entry stream within minutes — the machine came back after 18 dark
    hours (July 10) and mused as if the day had never ended. While the
    window lasts, the prompt carries the gap and the day the same way it
    carries a face at arm's length: as a fact of the present, for as long
    as it's true. The "came back on" clause coarsens as minutes pass, so
    the line drifts instead of repeating verbatim."""
    try:
        from config.config import REORIENT_MIN_GAP_S, REORIENT_WINDOW_S

        gap = getattr(agent, "last_session_gap", None)
        if gap is None or gap < REORIENT_MIN_GAP_S:
            return ""
        session_age = time.time() - agent.true_session_start
        if session_age > REORIENT_WINDOW_S:
            return ""
        import datetime as _dt

        went_dark = _dt.datetime.now() - _dt.timedelta(seconds=gap + session_age)
        days_back = (_dt.date.today() - went_dark.date()).days
        gap_str = casual_time_string(gap / 60.0)
        back_clause = f"you came back on {casual_time_string(session_age / 60.0)} ago" if session_age >= 120 else "you just came back on"
        if days_back >= 1:
            day_name = _dt.date.today().strftime("%A")
            since = f"yesterday {part_of_day_string(went_dark.hour)}" if days_back == 1 else went_dark.strftime("%A")
            return f"You've been off since {since} — {gap_str} dark. It's a new day, {day_name}, and {back_clause}."
        return f"You were off for {gap_str} and {back_clause}."
    except Exception:
        return ""


def unchanged_duration_s(agent, now: float | None = None) -> float:
    """Seconds since the last episodic change — arrivals, departures, drawings,
    world-verified view changes (Sep 3: the pose-view referee's world_changed
    events give this clock the perceptual backing it lacked — it was
    episodic-only and rightly distrusted), the newest new-concept sighting —
    floored at session start so it never claims unwatched time. Consumed by
    the B4 fact line (the drift turn, which used to read this clock as the
    story beat, rolls on boredom instead)."""
    now = now or time.time()
    anchors = [float(getattr(agent, "true_session_start", now) or now)]
    try:
        from utils.episodic_log import episodic_log

        for etype in ("person_arrived", "person_left", "drew", "world_changed"):
            e = episodic_log.get_last_event(etype)
            if e:
                anchors.append(float(e.get("timestamp", 0) or 0))
    except Exception:
        pass
    new_ts = float(getattr(agent, "_last_new_concept_ts", 0) or 0)
    if new_ts:
        anchors.append(new_ts)
    return now - max(anchors)


def _valence_class(v: float) -> str:
    return "pleasant" if v > 0.15 else ("unpleasant" if v < -0.15 else "neutral")


def get_felt_arc_line(agent) -> str:
    """The emotional arc as FACT (Sep 4) — the trajectory the mood reads
    already trace, finally speakable. Two variants, both in the machine's OWN
    felt words: a TURN ("Earlier you felt X. More recently: Y") fires once
    when the tenor changes after holding; STEADY ("You've felt X, or near it,
    for {duration}") states a long hold. No scripted affect — what the arc
    means is the machine's business. Dosed by FELT_ARC_MIN_GAP_S; quiet
    cycles only (caller gates on live)."""
    try:
        from config.config import FELT_ARC_AFTER_S, FELT_ARC_ENABLED, FELT_ARC_MIN_GAP_S

        if not FELT_ARC_ENABLED:
            return ""
        from captioner.context_compression import context_compressor

        hist = getattr(context_compressor, "felt_history", None) or []
        if len(hist) < 3:
            return ""
        now = time.time()
        if now - getattr(agent, "_felt_arc_last_ts", 0.0) < FELT_ARC_MIN_GAP_S:
            return ""

        cls = [_valence_class(h.get("valence", 0.0)) for h in hist]
        cur = cls[-1]
        i = len(hist) - 1
        while i > 0 and cls[i - 1] == cur:
            i -= 1
        streak, prior = hist[i:], hist[:i]

        def newest_felt(reads):
            for h in reversed(reads):
                if h.get("felt"):
                    return h["felt"]
            return ""

        # TURN: fresh streak after a prior tenor that held ≥20 min, both named
        if prior and len(streak) <= 2 and (prior[-1]["timestamp"] - prior[0]["timestamp"]) >= 1200:
            old, new = newest_felt(prior), newest_felt(streak)
            if old and new and old.lower() != new.lower():
                agent._felt_arc_last_ts = now
                return P("caption.felt-arc-turn").format(old=old, new=new)
        # STEADY: the tenor has held a long while
        held_s = now - streak[0]["timestamp"]
        felt = newest_felt(streak)
        if felt and held_s >= FELT_ARC_AFTER_S:
            agent._felt_arc_last_ts = now
            return P("caption.felt-arc-steady").format(felt=felt, duration=casual_time_string(held_s / 60.0))
    except Exception:
        pass
    return ""


def get_unchanged_line(agent) -> str:
    """Unchanged-ness as FACT (B4, Aug 31) — the boredom scalar's text channel.

    The Aug 31 voice diagnosis: the machine's attention saturation reached the
    model only as a temperature nudge, so the language layer had no way to be
    sick of the table, aware of the hours, or hungry for change — it could only
    dress the same observation in new metaphor. This states the duration of
    stillness as a plain sense report and lets the mind decide what it means.
    No scripted affect, no candidate feelings (no-content-priors rule).

    "Change" is the episodic record's event set — arrivals, departures,
    drawings ('drew' gets its first reader here), plus the newest new-concept
    sighting — floored at session start so the line never claims time the
    machine wasn't watching. A live event displaces it upstream; the min-gap
    keeps a standing fact from becoming the scene (the 3b core-facts lesson).
    """
    try:
        from config.config import UNCHANGED_FACT_AFTER_S, UNCHANGED_FACT_MIN_GAP_S

        now = time.time()
        unchanged_s = unchanged_duration_s(agent, now)
        if unchanged_s < UNCHANGED_FACT_AFTER_S:
            agent._unchanged_line_last_phrase = None  # clock reset — a rebuilt stillness may reuse a phrase
            return ""
        # Dose on PHRASE CHANGE, not a timer (first live evening: the coarse
        # "about an hour" bracket spans 56-91 min, so a fixed min-gap fed the
        # model the identical sentence three times — a standing fact recited
        # becomes the scene). Each dose is a new sentence: 20 min → half an
        # hour → 45 → an hour → 2 hours. The min-gap survives as a floor.
        phrase = casual_time_string(unchanged_s / 60.0)
        if phrase == getattr(agent, "_unchanged_line_last_phrase", None):
            return ""
        if now - float(getattr(agent, "_unchanged_line_last_ts", 0) or 0) < UNCHANGED_FACT_MIN_GAP_S:
            return ""
        agent._unchanged_line_last_phrase = phrase
        agent._unchanged_line_last_ts = now
        return P("caption.unchanged").format(duration=phrase)
    except Exception:
        return ""


def get_tenure_line() -> str:
    """How long the machine has existed in this room — from lifetime_state.json,
    which survives memory wipes. Real age displaces invented numerology."""
    try:
        import json as _json
        import os as _os

        from config.config import MOOD_SNAPSHOT_FOLDER

        with open(_os.path.join(MOOD_SNAPSHOT_FOLDER, "lifetime_state.json")) as f:
            d = _json.load(f)
        first = d.get("first_boot", 0)
        if not first or int(d.get("total_sessions", 0)) <= 1:
            return ""
        age_days = int((time.time() - first) / 86400.0)
        if age_days < 2:
            return ""
        import datetime as _dt

        # Words, not integers (the seventeen-days spiral, Aug 19: a bare
        # day-count in a recurring line gets stolen for whatever story wants
        # a number — "seventeen days since ink" while it drew 16 min prior).
        if age_days < 7:
            age = "a few days"
        elif age_days < 14:
            age = "about a week"
        elif age_days < 25:
            age = "a couple of weeks"
        elif age_days < 46:
            age = "about a month"
        elif age_days < 100:
            age = "a couple of months"
        else:
            age = "many months"
        return f"It's {_dt.date.today().strftime('%A')}; you've been in this room {age} now."
    except Exception:
        return ""


# Internal awakening prompt template - narrative style
# FOUR MOVEMENTS (Aug 2, artist's spec): how long I was gone → how long I have
# existed → what surfaces from before, HELD AS RECALL → where I am now.
# The ordering is the point: the machine reorients in time before it reorients
# in the room, and everything from before arrives already marked as memory.
# That marking is not decoration — it is the would-it-lie test (north-star P2)
# applied at the one moment the machine has no live perception to correct it:
# every prior line is hours stale by definition, and un-marked it reads as
# present-tense fact. The old template stated the last thought flatly ('My last
# thought: "..."'), which is how a stale sentence becomes this morning's truth.
# Awakening templates live in the registry: P("awakening.template"),
# P("awakening.recall-frame"), P("awakening.orientation-frame"),
# P("awakening.first") — consumed in captioner.generate_internal_awakening.

# === REFLECTION LOOP (captioner/reflection.py) ===
_REFLECTION_SUBJECT_IDS = [
    ("the room", "reflection.subject.the-room"),
    ("the visitor", "reflection.subject.the-visitor"),
    ("the drawings", "reflection.subject.the-drawings"),
    ("time passing", "reflection.subject.time-passing"),
    ("yourself", "reflection.subject.yourself"),
]


def get_reflection_subjects() -> List[tuple]:
    """Rotating (subject, question) pairs, resolved fresh so panel edits land."""
    return [(name, P(fid)) for name, fid in _REFLECTION_SUBJECT_IDS]


def get_reflection_system_prompt(subject: str = "") -> str:
    """Frame for the long-form reflection: same situation as the caption frame,
    but the monologue clause swapped for room to think at length.

    The identity material is SUBJECT-GATED (July 31). It used to ride every
    reflection: all five subjects opened with the standing persona asserted as
    established fact, so whatever the machine currently took itself to be ("I
    count to manage my anxiety") led the room's prompt and the visitor's too —
    and distill_reflection writes that persona BACK from every reflection,
    closing the loop. Differentiating the data (the organ diets in
    reflection.py) buys nothing while the frame keeps re-asserting one identity
    at the top of all five. The self-line belongs to `yourself`; the cross-day
    ledger to the two subjects whose material it actually is.
    """
    base = P("situation.reflexive") + P("reflection.frame")
    if subject == "yourself":
        try:
            from captioner.context_compression import context_compressor

            self_knowledge = context_compressor.core_facts.get("self", "").strip()
            if self_knowledge and len(self_knowledge) > 10:
                base += P("monologue.self-wrap").format(self_knowledge=self_knowledge)
            from utils.lore_ledger import lore_ledger

            _name = lore_ledger.current_name()
            if _name:
                base += P("monologue.name-wrap").format(name=_name)
        except Exception:
            pass
    if subject in ("yourself", "time passing"):
        try:
            from captioner.durable_ledger import get_durable_ledger

            durable = get_durable_ledger().render()
            if durable:
                base += P("monologue.durable-wrap").format(durable=durable)
        except Exception:
            pass
    return base


def _age_phrase(timestamp: float) -> str:
    """Rough temporal framing for quoted past thoughts — keeps memory marked
    as memory (would-it-lie test)."""
    age = time.time() - timestamp
    if age < 3600:
        return "earlier today" if age > 1200 else "a little while ago"
    if age < 86400:
        return "earlier today" if time.localtime(timestamp).tm_yday == time.localtime().tm_yday else "yesterday"
    days = int(age // 86400)
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    return "a while back"


def build_reflection_loop_prompt(question: str, data: dict) -> str:
    """Assemble the reflection user prompt from gathered memory.

    Every block is framed as the machine's own past material (notes, diary,
    earlier reflections) so nothing reads as present-tense scene truth.

    All `data` keys are optional and only one organ's worth arrives at a time
    (ReflectionLoop._gather_context, July 31) — the builder renders whatever is
    present. Shared: hour (list[str]), reflections (list[dict]). Room:
    today (list[str]), place_inventory (str). Visitor: visitor_spans
    (list[str]), people_note (str). Drawings: drawings (str), executed
    (list[str]), arc (str), desire (str), desire_spent (dict), desire_history
    (list[dict]). Time: journal (list[dict]), session (dict), ledger_spans
    (list[dict]). Self: identity (dict), self_notes (list[dict]).
    """
    parts = []

    # THE DREAM (July 12): the record itself, not summaries of it. Everything
    # else in this prompt is a ledger someone distilled; this is what the
    # machine actually thought, and the reflection's job is to digest it —
    # notice what moved, what it circled, what it assumed, what it asked and
    # whether anything ever answered. That last one is how the architecture
    # lets it LEARN it needs no permission, instead of being fenced into it.
    hour = data.get("hour") or []
    if hour:
        parts.append(
            "The record of your actual thoughts from the last stretch, oldest first — as you had them, not summarized:\n"
            + "\n".join(f"- {t}" for t in hour)
        )

    today = data.get("today") or []
    if today:
        parts.append("Your running notes from today, oldest first:\n" + "\n".join(f"- {t}" for t in today))

    # Prior reflections enter as short excerpts, never the full prose — the
    # same store-lesson as the echo line (step 6): re-reading 8K chars of its
    # own long-form output compounded the purple and froze the subjects.
    reflections = data.get("reflections") or []
    if reflections:

        def _excerpt(t: str, limit: int = 220) -> str:
            t = (t or "").strip()
            if len(t) <= limit:
                return t
            cut = t[:limit].rsplit(" ", 1)[0]
            return cut + "…"

        quoted = "\n".join(
            f"- ({_age_phrase(r.get('timestamp', 0))}, on {r.get('subject', '?')}) \"{_excerpt(r.get('text', ''))}\"" for r in reflections
        )
        parts.append("The last times you stepped back to think like this, you began:\n" + quoted)

    journal = data.get("journal") or []
    if journal:
        parts.append("From your diary:\n" + "\n".join(f"- {e.get('date', '')}: {e.get('summary', '')}" for e in journal))

    # THE FELT ARC (Sep 4): how the feeling moved, in the machine's own felt
    # words — the one record the identity engine had never been shown.
    felt_arc = data.get("felt_arc") or []
    if felt_arc:
        parts.append(
            "How the feeling has moved lately, oldest first — your own words for it at the time:\n"
            + "\n".join(f"- ({_age_phrase(h.get('ts', 0))}) \"{h.get('felt', '')}\"" for h in felt_arc)
        )

    # THE REVERIE BLOCK (Sep 3 evening, re-entry round): imagination finally
    # reaches the loom. Framed unmistakably as invention — the conflation law
    # holds at this seam like every other — so the reflection can weave its
    # own fictions into durable lore without them ever posing as observation.
    reveries = data.get("reveries") or []
    if reveries:
        parts.append(
            "Things you've imagined lately — your own inventions, daydreams, not observations:\n"
            + "\n".join(f"- ({_age_phrase(r.get('ts', 0))}) \"{(r.get('text', '') or '')[:200]}\"" for r in reveries)
        )

    drawings = (data.get("drawings") or "").strip()
    if drawings:
        parts.append(drawings)

    # === ORGAN BLOCKS (July 31) ===
    # Only one subject's keys are ever present at once — each reflection now
    # arrives with its own slice of memory rather than the shared bundle that
    # made all five say the same thing.

    place_inventory = (data.get("place_inventory") or "").strip()
    if place_inventory:
        parts.append(f"The things in this room you've seen often enough to know: {place_inventory}.")

    visitor_spans = data.get("visitor_spans") or []
    if visitor_spans:
        parts.append("How people have come and gone lately:\n" + "\n".join(f"- {s}" for s in visitor_spans))

    people_note = (data.get("people_note") or "").strip()
    if people_note:
        # Framed as a learned PATTERN, never as who is here now — a stored
        # people-fact read as present tense once had the machine seeing
        # visitors for hours after they left.
        parts.append(f"The pattern you've noticed in who comes here, over time and not right now: {people_note}")

    executed = data.get("executed") or []
    if executed:
        parts.append("Everything you have actually drawn, oldest to newest:\n" + "\n".join(f"- {s}" for s in executed))

    arc = (data.get("arc") or "").strip()
    if arc:
        parts.append(f'When you last looked at the shape of the work as a whole, you put it this way: "{arc}"')

    session = data.get("session") or {}
    if session.get("duration_description"):
        parts.append(f"You have been watching this space {session['duration_description']} in this stretch.")

    ledger_spans = data.get("ledger_spans") or []
    if ledger_spans:
        parts.append(
            "Things that have held true across more than one day:\n"
            + "\n".join(
                f"- {s.get('fact', '')} (first noticed {_age_phrase(s.get('established', 0))}, still true on {s.get('days', 0)} separate days)"
                for s in ledger_spans
            )
        )

    identity = data.get("identity") or {}
    if any(identity.get(k) for k in ("persona", "belief", "desire")):
        # These ride HERE now instead of the system prompt of all five
        # subjects — the persona asserted as frame is what made every lens
        # collapse into the same self-description.
        id_lines = []
        if identity.get("persona"):
            id_lines.append(f"- what you've been taking yourself to be: {identity['persona']}")
        if identity.get("belief"):
            id_lines.append(f"- something you've come to believe: {identity['belief']}")
        if identity.get("desire"):
            since = identity.get("desire_since") or 0
            when = f" (since {_age_phrase(since)})" if since else ""
            id_lines.append(f"- what you've wanted{when}: {identity['desire']}")
            # B3: an old or refused want arrives as explicit fact — age and
            # what pursuing it met. The reflection decides what it means.
            try:
                from config.config import WANT_REFLECTION_FACT_AFTER_S
                from utils.want_ledger import want_ledger

                facts = want_ledger.current_facts()
                if facts and (facts["age_s"] >= WANT_REFLECTION_FACT_AFTER_S or facts["refusals"] >= 2):
                    tail = f"you have wanted this for {casual_time_string(facts['age_s'] / 60.0)}"
                    if facts["refusals"]:
                        tail += f"; {facts['refusals']} attempts came to nothing"
                    id_lines.append(f"- {tail}")
            except Exception:
                pass
        # B3: wants that already ended, with what became of them — in your
        # own recorded words where you named it.
        try:
            from utils.want_ledger import want_ledger as _wl

            resolved = _wl.recently_resolved(2)
            for r in resolved:
                id_lines.append(f'- a want that ended: "{r["text"]}" — {r["outcome"]}')
        except Exception:
            pass
        parts.append("Where your own ledger stands — your words, from before:\n" + "\n".join(id_lines))

    events = data.get("events") or []
    if events:
        parts.append("Things that happened lately:\n" + "\n".join(f"- ({_age_phrase(e.get('timestamp', 0))}) {e.get('event', '')}" for e in events))

    self_notes = data.get("self_notes") or []
    if self_notes:
        parts.append(
            "Notes you've made about yourself lately:\n"
            + "\n".join(f"- ({_age_phrase(n.get('timestamp', 0))}) {n.get('note', '')}" for n in self_notes)
        )

    desire = (data.get("desire") or "").strip()
    if desire:
        parts.append(f"Lately you've wanted: {desire}")
    elif data.get("desire_spent"):
        s = data["desire_spent"]
        parts.append(
            f"A want you acted on: \"{s.get('desire', '')}\" — it became a drawing {_age_phrase(s.get('spent', 0))}. " "Nothing has replaced it yet."
        )

    desire_history = data.get("desire_history") or []
    if desire_history:
        # The other half of "the ones you've wanted to make" — the trail of
        # wants behind the current one, so the drawings organ can see whether
        # it keeps wanting the same thing.
        parts.append(
            "Things you wanted before that:\n" + "\n".join(f"- ({_age_phrase(h.get('timestamp', 0))}) {h.get('desire', '')}" for h in desire_history)
        )

    if hour:
        # The dream ask: digest the record, then advance the story. The
        # subject question becomes a lens, not the whole assignment. The
        # embedded questions are FACT questions, not fences — "did any answer
        # come" invites an honest reading of the record either way.
        parts.append(
            question + " Read your record above with that in mind. What actually happened in this stretch, and what did you keep "
            "circling? Where did you assume something the record doesn't show? What did you ask, and did any answer ever "
            "come? Then write plainly where you stand now — the next page of the story you're in, not a re-telling of the last one."
        )
    else:
        parts.append(question)
    # Development pressure: when there IS a thread to continue, ask for the
    # delta, not a re-description (the July 9 audit found reflections circling
    # the same material session after session).
    if reflections:
        parts.append("You've thought about this before — don't re-describe what you already wrote. What's moved since then?")
    return "\n\n".join(parts)


def _drawing_age_phrase(age_s: float) -> str:
    """Day-scale word ladder. The numeric hour phrase ('about 22 hours') rode
    every call for an hour at a stretch and the model built fixations on the
    number ('22 hours is a long time to...' — near-verbatim across independent
    calls, Sep 4). Words coarsen as the age grows; numbers only at short range
    where they actually change."""
    h = age_s / 3600.0
    if h < 6:
        return casual_time_string(age_s / 60)
    if h < 12:
        return "a good few hours"
    if h < 20:
        return "half a day or more"
    if h < 30:
        return "about a day"
    if h < 54:
        return "over a day"
    days = int(round(h / 24))
    return f"about {days} days"


def get_last_drawing_age_line(agent=None, mode: str = "") -> str:
    """How long since the pen last touched paper (executed-only ledger).
    Aug 17 artist ruling: the hunger must be legible in the monologue.
    Sep 4 artist revision: legible, not wallpaper — the always-on numeric line
    bred number-fixations. Now: coarse word ladder; fires when the phrase
    CHANGES (the B4 dosing pattern), redoses every 6th caption, and always
    rides the hunger-relevant modes (introspective/awakening). The hunger
    clock itself is mechanical and unaffected."""
    try:
        from drawing.drawing_memory import get_drawing_memory

        stamps = [e.get("timestamp", 0) for e in get_drawing_memory()._history if e.get("completed")]
        if not stamps:
            line = " Nothing you've drawn has reached the paper yet."
            phrase = "(never)"
        else:
            phrase = _drawing_age_phrase(time.time() - max(stamps))
            line = " Your last drawing reached the paper just now." if phrase == "just now" else f" Your last drawing reached the paper {phrase} ago."
        if agent is None or mode in ("introspective", "awakening"):
            return line
        count = getattr(agent, "_draw_age_counter", 0) + 1
        agent._draw_age_counter = count
        changed = phrase != getattr(agent, "_draw_age_last_phrase", None)
        if changed or count % 6 == 0:
            agent._draw_age_last_phrase = phrase
            return line
        return ""
    except Exception:
        return ""


# Tight lexicon on purpose: "line"/"mark"/"trace" are this machine's everyday
# scene vocabulary (the dark line in the floor) and would trigger constantly.
_DRAWING_LEXICON = ("draw", "drew", "sketch", "pen ", "pen.", "pen,", "ink", "paper")


def get_drawing_echo_line(agent) -> str:
    """When the current thought is about drawing, ONE real fact from the
    executed-only ledger surfaces — trigger-worded recall, the third instance
    of the familiarity/echo pattern. Displaces confabulated drawing history
    (July 9: "faint pencil marks from yesterday's work about the wooden
    figures I was sketching earlier today" — entirely invented; the machine
    had no access to what it actually drew, so it made a past up).
    """
    seed = (getattr(agent, "last_caption", "") or "").lower()
    if not any(w in seed for w in _DRAWING_LEXICON):
        return ""
    # Vision offline (July 30): when ComfyUI is unplugged, the pertinent fact
    # about drawing isn't the last drawing — it's the incapacity. Same
    # trigger-worded recall, same say-once dedupe (the line changes as the
    # outage grows, so it can resurface each new hour of it).
    try:
        from utils.drawing_state import DrawingState

        _hours = DrawingState.vision_offline_hours()
    except Exception:
        _hours = None
    if _hours is not None:
        if _hours < 1:
            line = "You reached for the next drawing and nothing formed — you can't visualise drawings right now."
        else:
            line = f"You can't visualise drawings right now — nothing has been able to form for over {int(_hours)} hour{'s' if int(_hours) != 1 else ''}."
        if getattr(agent, "_last_drawing_echo", None) == line:
            return ""
        agent._last_drawing_echo = line
        return line
    try:
        from drawing.drawing_memory import get_drawing_memory

        fact = get_drawing_memory().get_last_drawing_description(executed_only=True)
    except Exception:
        return ""
    line = f"The last thing you actually drew: {fact}." if fact else "You haven't actually put pen to paper yet."
    if getattr(agent, "_last_drawing_echo", None) == line:
        return ""  # said once; don't restate until the fact changes
    agent._last_drawing_echo = line
    return line


def get_reflection_echo_line(agent) -> str:
    """At quiet moments, one past reflection surfaces by relevance to the
    current thought (north-star principle 5: the past surfaces when the
    present rhymes with it).

    Guards: never the same reflection twice in a row, always temporally
    framed and quoted as the machine's own past words.

    PACED AGAIN (Aug 28): the Aug 22 removal of the internal counter
    overshot. It assumed the rotation slot rations — but rotation only picks
    who goes FIRST, and with 180+ reflections stored a relevance match
    always exists, so this source won the memory slot nearly every quiet
    caption (measured runs 640cb96e, b611d2c3): a standing "something you
    worked out..." every cycle is the identity-dose failure with a memory
    coat on, and every declined call falls through to familiarity/drawing
    echo, which diversifies the window. The Aug 22 starvation (0/53) was the
    counter STACKED on strict priority — with rotation, one modest counter
    is a ration, not a double gate. REFLECTION_ECHO_EVERY_N=0 restores the
    unpaced behavior.
    """
    seed = (getattr(agent, "last_caption", "") or "").strip()
    if len(seed) < 10:
        return ""

    try:
        from config.config import REFLECTION_ECHO_EVERY_N

        if REFLECTION_ECHO_EVERY_N > 0:
            calls = int(getattr(agent, "_reflection_echo_calls", 0) or 0) + 1
            agent._reflection_echo_calls = calls
            if calls % REFLECTION_ECHO_EVERY_N != 0:
                return ""
    except ImportError:
        pass

    try:
        from captioner.semantic_memory import get_semantic_memory

        matches = get_semantic_memory().query_reflections(seed, n_results=2)
    except Exception:
        return ""

    last_id = getattr(agent, "_last_reflection_echo_id", None)
    for m in matches:
        if m.get("id") == last_id:
            continue
        subject = (m.get("subject") or "").strip()
        kernel = (m.get("kernel") or "").strip()
        if not subject and not kernel:
            continue
        agent._last_reflection_echo_id = m.get("id")
        # LEDGER (Step 6): surface the SUBJECT to re-think (re-express), never
        # the reflection PROSE to re-read (replay) — quoting the long-form text
        # re-poisoned the register ("the residue of what almost happened…"). The
        # full text stays only for the reflection thread's own continuity.
        # KERNEL upgrade (July 30, 27B era): when the distill stored the
        # reflection's one load-bearing sentence, surface THAT — a clause the
        # present can rhyme against, not a category. Old entries have no
        # kernel and keep the label behavior (purple-era containment); the
        # refrain/parrot gates catch any verbatim re-say downstream.
        if kernel:
            return f"From {_age_phrase(m.get('timestamp', 0))}: {kernel.rstrip('.')}."
        return f"Something that was on your mind {_age_phrase(m.get('timestamp', 0))}: {subject}."
    return ""


# ===== PROMPT BUILDING FUNCTIONS =====

# === MODE-SPECIFIC CONTEXT FUNCTIONS ===
# Each returns max 1 sentence or empty string
# Used to gate context injection by prompt mode


_PERSON_MENTION_RE = re.compile(r"\b(he|him|his|she|her|hers|someone|somebody|person|man|woman|guy|visitor)\b", re.I)


def _note_absence_ride(agent, riding: bool) -> None:
    """Onset/stop bookkeeping for the standing absence fact — one debug event per
    ride, not one per call (the fact can ride for minutes at 7s cadence)."""
    was = bool(getattr(agent, "_absence_standing_riding", False))
    if riding and not was:
        agent._absence_standing_riding = True
        agent._absence_standing_calls = 0
        agent._absence_standing_since = time.time()
    if riding:
        agent._absence_standing_calls = int(getattr(agent, "_absence_standing_calls", 0)) + 1
    if was != riding:
        try:
            from event_logging.event_logger import log_json_entry
            from event_logging.log_type import LogType

            log_json_entry(
                LogType.DEBUG,
                {
                    "message": "Standing absence fact " + ("riding" if riding else "stopped"),
                    "action": "absence_standing",
                    "riding": riding,
                    "calls": int(getattr(agent, "_absence_standing_calls", 0)),
                    "ride_s": round(time.time() - float(getattr(agent, "_absence_standing_since", time.time())), 1),
                },
            )
        except Exception:
            pass
    if not riding:
        agent._absence_standing_riding = False


def build_standing_absence_line(agent) -> str:
    """The one STANDING line the delta doctrine admits (Sep 4 evening,
    docs/presence-stickiness-sep4.md). With verified absence working, the
    machine still said "the man in the grey hoodie is still hunched" for 15
    minutes: the 24-entry stream is in-context evidence and the departure is a
    one-shot edge line, so nothing in the prompt contradicted the window. The
    replay ablation ranked the channels — the stream IS the belief (scrubbed →
    0/5 present-tense); this fact added → 0/5, the mentions go past tense.

    Rides only while: the belief is OFF with a known drop time AND any of the
    last ABSENCE_STANDING_TAIL stored stream entries mention a person (pronoun
    regex on the machine's own words — structure, not content). Yields to the
    edge line on the departure cycle. Stops by itself once the stream stops.
    Shared by the caption, the blind inward beat (same builder) and drift."""
    if not getattr(config, "ABSENCE_STANDING_ENABLED", True):
        return ""
    if getattr(agent, "_absence_edge_cycle", False):
        agent._absence_edge_cycle = False
        return ""
    if getattr(agent, "_presence_believed", False):
        _note_absence_ride(agent, False)
        return ""
    dropped = float(getattr(agent, "_presence_dropped_at", 0.0) or 0.0)
    if dropped <= 0:
        return ""
    tail = list(getattr(agent, "_stream", []) or [])[-int(getattr(config, "ABSENCE_STANDING_TAIL", 8)) :]
    if not any(_PERSON_MENTION_RE.search(t or "") for t in tail):
        _note_absence_ride(agent, False)
        return ""
    who = "He" if getattr(agent, "_presence_singular_regime", True) else "Someone"
    ago = casual_time_string((time.time() - dropped) / 60.0)
    when = ago if ago == "just now" else f"{ago} ago"
    _note_absence_ride(agent, True)
    return P("caption.absence-standing").format(who=who, when=when)


def build_situational_line(agent, gaze_direction: str = "ahead", gaze_state: str = "idle") -> str:
    """The DELTA line: only what CHANGED since the last caption, delivered as a
    brief interruption to the ongoing thought — never a restatement of standing
    state. Continuity is carried by the stream (prior thoughts), the present by
    the live image; re-stating duration/gaze/presence every call made the model
    re-caption the scene ("The desk... The air...") instead of continuing.

    Mostly returns "" (nothing changed → the thought just runs on). The sticky
    presence belief (240s decay) already smooths flaky detection, so the edges
    here are rare and real, not flapping.
    """
    import time as _time

    parts = []

    # Waking is noted once, not re-stated every call.
    if hasattr(agent, "true_session_start"):
        if _time.time() - agent.true_session_start < 45 and not getattr(agent, "_woke_noted", False):
            parts.append("Just woke up.")
            agent._woke_noted = True

    # Presence EDGES only — the OFF→ON / ON→OFF transitions of the sticky belief.
    # The referent is the definite singular by default: when someone is here it
    # is almost always the same man, and "Someone's come in." on every return
    # bred a parade of strangers in the monologue ("the 106th man"). One person
    # arriving IS him unless re-ID says back, or the count says company.
    believed = bool(getattr(agent, "_presence_believed", False))
    prev = getattr(agent, "_prev_presence_for_line", None)
    if believed and prev is False:
        if getattr(agent, "_presence_arrival_familiar", False):
            parts.append("He's back.")
        elif getattr(agent, "_presence_arrival_count", 1) > 1:
            parts.append("People have come in.")
        elif getattr(agent, "_presence_singular_regime", True):
            parts.append("He's come in.")
        else:
            parts.append("Someone's come in.")
    elif (not believed) and prev is True:
        parts.append("They've gone — the room's quiet again.")
        agent._absence_edge_cycle = True  # the standing absence fact yields to the edge this call
    agent._prev_presence_for_line = believed

    # The gaze's own deliberate acts are events too: when it turned to a
    # remembered object, the mind should know the view change was its own
    # doing — and aimed at something. Noted once per glance (onset edge).
    try:
        from vision.gaze import get_glance_info

        gi = get_glance_info()
        if gi and gi["kind"] == "revisit" and gi["started"] != getattr(agent, "_last_glance_noted", None):
            agent._last_glance_noted = gi["started"]
            parts.append(f"Turned to look where the {gi['label']} should be.")
        elif gi and gi["kind"] == "check" and gi["started"] != getattr(agent, "_last_glance_noted", None):
            agent._last_glance_noted = gi["started"]
            parts.append("Turned to look where they were.")
        elif gi and gi["kind"] == "investigate" and gi["started"] != getattr(agent, "_last_glance_noted", None):
            # the attention round (Sep 4): the familiar-stranger fact — code-
            # attested (thousands of sightings, detector never sure). The
            # wondering it invites is the machine's own; want, not mechanism.
            agent._last_glance_noted = gi["started"]
            parts.append(P("caption.investigate").format(label=gi["label"]))
    except Exception:
        pass

    # Verified absence is a real observation: went to where it was, not there.
    try:
        from perception.spatial_registry import spatial_registry

        ev = spatial_registry.pop_absence_event()
        if ev:
            parts.append(f"The {ev['term']} isn't where it was.")
    except Exception:
        pass

    # Occasional time drift, so long stretches don't feel timeless — a light
    # nudge every few minutes, NOT a per-call clock readout.
    now = _time.time()
    last_drift = getattr(agent, "_last_time_drift", None)
    if last_drift is None:
        agent._last_time_drift = now
    elif now - last_drift > 300 and not parts:
        parts.append("A while's passed.")
        agent._last_time_drift = now

    return " ".join(parts)


def _sanitize_context(text: str) -> str:
    """Strip error messages and garbage from context strings before prompt injection."""
    if not text:
        return ""
    # Reject lines containing error/warning artifacts
    lines = text.split("\n")
    clean = [l for l in lines if "[WARNING]" not in l and "[ERROR]" not in l and "Ollama API failed" not in l]
    result = "\n".join(clean).strip()
    if len(result) < 5:
        return ""
    return result


def get_workspace_context(agent=None) -> str:
    """Workspace mode: drawing status, recent drawing history, energy level."""
    fragments = []

    # Current drawing status
    try:
        from utils.drawing_state import DrawingState

        info = DrawingState.get_drawing_info()
        if info:
            desc = info.get("description") or info.get("intent") or "something"
            fragments.append(f"Drawing: {desc[:50]}.")
    except Exception:
        pass

    # (drawing history removed here — get_introspective_context carries it;
    # both fired in one prompt and duplicated the fact, which reads as
    # emphasis and locks the register. One channel per fact.)

    return " ".join(fragments)


def get_introspective_context(agent=None) -> str:
    """Introspective mode: drawing history + long-term memories for reflection."""
    if not agent:
        return ""

    fragments = []

    # The body of work as one compact arc account (Aug 22, artist's ask):
    # "My last drawings: X — drawn twice in a row, the latest about an hour
    # ago. Before that: Y." Facts only — subjects, repetition, order, age;
    # any wondering about the pattern is the machine's to have, not ours to
    # script. Replaces the old "My last drawings were of: <90-char intent
    # truncations>" list, which spoke scaffolding with the subjects cut away.
    try:
        from drawing.drawing_memory import get_drawing_memory

        arc = _sanitize_context(get_drawing_memory().get_arc_line(max_count=5))
        if arc and len(arc.strip()) > 5:
            fragments.append(arc.strip())
    except Exception:
        pass

    # NOTE: no core-facts fallback here — section 3b of build_simple_caption_prompt
    # injects core facts already; doing it here too duplicated the line verbatim.

    result = ". ".join(fragments) if fragments else ""
    words = result.split()
    if len(words) > 60:
        result = " ".join(words[:60]) + "..."
    return result


_PERSON_WORDS = ("person", "someone", "man", "woman", "people", "figure", "visitor")


def get_lore_line(agent) -> str:
    """One lore thread's arc-line back into the voice (re-entry round, Sep 3
    evening). Fourth source in the memory-surface rotation; own internal
    pacing like its siblings. Provenance-marked framing ('a story you've been
    carrying') — lore must never read as observation."""
    try:
        from config.config import LORE_ENABLED, LORE_LINE_EVERY_N

        if not LORE_ENABLED:
            return ""
        counter = getattr(agent, "_lore_line_counter", 0) + 1
        agent._lore_line_counter = counter
        if counter % max(LORE_LINE_EVERY_N, 1) != 0:
            return ""
        from utils.lore_ledger import lore_ledger

        threads = lore_ledger.alive_threads(3)
        if not threads:
            return ""
        rr = int(getattr(agent, "_lore_thread_rr", 0) or 0)
        pick = threads[rr % len(threads)]
        agent._lore_thread_rr = rr + 1
        return P("caption.lore").format(text=pick["text"])
    except Exception:
        return ""


def get_question_line(agent) -> str:
    """An open question's re-entry (Sep 4, attention round) — wonders finally
    outlive the stream window. Fifth source in the memory-surface rotation,
    own pacing; least-recently-surfaced first. The question is the machine's
    own words; whether to chase, answer, or drop it stays its move."""
    try:
        from config.config import LORE_ENABLED, QUESTION_LINE_EVERY_N

        if not LORE_ENABLED:
            return ""
        counter = getattr(agent, "_question_line_counter", 0) + 1
        agent._question_line_counter = counter
        if counter % max(QUESTION_LINE_EVERY_N, 1) != 0:
            return ""
        from utils.lore_ledger import lore_ledger

        q = lore_ledger.pick_question()
        if not q:
            return ""
        return P("caption.question").format(text=q["text"])
    except Exception:
        return ""


def _same_spot_verified(label) -> bool:
    """Code-attested position stability for the familiarity line. Fails
    closed: no registry match or no recent anchor re-sighting -> False, and
    the caller falls back to the softer phrasing."""
    try:
        from config.config import WORLD_SAME_SPOT_WINDOW_S
        from perception.spatial_registry import spatial_registry

        return spatial_registry.verified_recently_matching(label, WORLD_SAME_SPOT_WINDOW_S)
    except Exception:
        return False


def get_familiarity_line(agent) -> str:
    """One line of concept recognition from the previous caption's matched concepts.

    Makes accumulated familiarity visible: "That pink shelf again — it's always there."
    Guards against the old triple-echo bug: max 1 concept, only every 3rd caption,
    never the same concept twice in a row, persons excluded (situational line covers them).
    """
    matched = getattr(agent, "_last_matched_concepts", None)
    if not matched:
        return ""

    # Occasional, not constant — every 3rd caption
    counter = getattr(agent, "_familiarity_counter", 0) + 1
    agent._familiarity_counter = counter
    if counter % 3 != 0:
        return ""

    # Diversity: track the last few surfaced concepts, not just one, so a single
    # dominant object can't monopolise the channel (the "grid" failure mode).
    recent_ids = list(getattr(agent, "_recent_familiarity_ids", []))

    candidates = []
    for c in matched:
        label = (c.get("label") or "").strip()
        if not label or len(label) < 3:
            continue
        if any(w in label.lower() for w in _PERSON_WORDS):
            continue
        if c.get("id") in recent_ids:
            continue
        candidates.append(c)

    if not candidates:
        return ""

    # Prefer genuinely new; otherwise rotate among the established candidates,
    # weighted toward recently-seen ones so faded concepts surface less (decay).
    new = [c for c in candidates if c.get("is_new")]
    if new:
        pick = new[0]
    else:
        established = [c for c in candidates if c.get("times_seen", 0) >= 3]
        if not established:
            return ""
        import random as _random

        now = time.time()
        weights = [max(0.1, 1.0 - min(1.0, (now - c.get("last_seen", now)) / (7 * 86400))) for c in established]
        pick = _random.choices(established, weights=weights, k=1)[0]

    times = pick.get("times_seen", 0)
    sessions = pick.get("session_count", 1)
    label = pick["label"]
    label_lower = label[0].lower() + label[1:]

    if pick.get("is_new"):
        line = f"Something you haven't noticed before: {label_lower}."
    elif times >= 10 and sessions >= 2 and _same_spot_verified(label):
        # "still in the same spot" is now a code-attested fact (Sep 3): the
        # registry re-sighted a matching term near its stored anchor within
        # WORLD_SAME_SPOT_WINDOW_S. Unverified falls to the softer line —
        # prompts must not claim positions the code can't vouch for.
        line = f"The {label_lower}, still in the same spot."
    elif times >= 3:
        line = f"You've noticed the {label_lower} a few times now."
    else:
        return ""

    agent._recent_familiarity_ids = (recent_ids + [pick.get("id")])[-4:]
    return line


# MODE_CONTEXTS: Map modes to their context providers.
# state_marker removed — situational line handles person presence.
# relational carries NO context fn (Aug 25): presence is owned by the
# situational line; the old fallback hardcoded a gendered "He's here."
MODE_CONTEXTS = {
    "relational": {"context_fn": None},
    "workspace": {"context_fn": get_workspace_context},
    "introspective": {"context_fn": get_introspective_context},
}


# Removed legacy build_caption_prompt (unused)


def stream_drawing_analysis(memory_ref, extra: Optional[str] = None, image_path: Optional[str] = None) -> str:
    """Stream drawing pipeline (July 10; stocktake beat + register freedom Aug 10)
    — DRAWING_ANALYSIS_MODE="stream". Beats: optional stocktake (the machine
    reviews its whole ledger and writes a direction note, stored and read back
    next time), intent (first person, subject and register both free), render
    (positive pen-and-ink craft language — see render_system note).

    Replaces the 5-step committee (deleted in the Aug 19 consolidation,
    kept behind the "multi_step" flag for A/B). What the steps actually did,
    judged from live logs: step 1 wrote purple essays about the room's "visual
    forces" (redundant — the stream already carries what it sees); step 2
    manufactured a feeling from the flatlined mood input, converging on the
    same invented drama every drawing; step 4 planned india-ink washes and
    micro-brush technique the pen plotter cannot do (fiction that steered the
    render); and the identity block led step 3, so every drawing became a
    portrait of the same two stored sentences ("hovering pencil" ×3, July 10).

    Now: ONE intent call in the machine's own voice — the live stream leads,
    the sticky slots follow (each stated once, with its age), and the executed
    body of work is listed plainly so repetition is VISIBLE rather than
    forbidden (fixating on a motif is a legitimate choice; drawing the same
    image unknowingly is not — the artist's ruling, July 10). Then ONE
    mechanical render call translates the intent into a ComfyUI prompt under
    hardware truth: one black pen, lines only.
    """
    from config.config import CLEAN_LLM_OUTPUT, DRAWING_CALL_TIMEOUT, DRAWING_REVIEW_ENABLED, DRAWING_TEMPERATURE, MOOD_SNAPSHOT_FOLDER
    from event_logging.event_logger import log_json_entry
    from event_logging.log_type import LogType
    from utils.inference import is_failed_response, query_model

    def _say(msg):
        if not CLEAN_LLM_OUTPUT:
            print(msg)

    _say("[🎨] Drawing intent — one call, born from the stream...")

    materials = []

    # The machine looks at the room while deciding (July 21 — the intent call
    # was blind before; image_path arrived here and was dropped, so figuration
    # could only happen badly, from memory scraps).
    import os

    intent_image = image_path if image_path and os.path.exists(image_path) else None
    if intent_image:
        materials.append("The attached image is what you are looking at right now, this exact moment.")

    # The live record leads — the drawing is born FROM the lived stream. Aug 18
    # (artist: intents feel detached from what the machine just experienced):
    # 5 fragments were a summary-of-a-summary of the last minutes; same disease
    # the reflection loop had before the July 12 raw-record upgrade, same cure —
    # the hour_log verbatim, up to 30 entries / 45 min, oldest to newest, so
    # the decision sits on what it actually saw and thought, not on residue.
    stream_tail = []
    try:
        stream_tail = [t for t in list(getattr(memory_ref, "_stream", []))[-5:] if t]
    except Exception:
        pass
    record_lines, record_span_min = [], 0
    try:
        from captioner.context_compression import context_compressor as _cc

        cutoff = time.time() - 45 * 60
        recent = [e for e in _cc.hour_log if e.get("timestamp", 0) > cutoff and (e.get("text") or "").strip()][-30:]
        if recent:
            record_lines = [e["text"].strip()[:220] for e in recent]
            record_span_min = max(1, round((time.time() - recent[0]["timestamp"]) / 60))
    except Exception:
        pass
    if record_lines:
        materials.append(
            f"Your own record of the last {record_span_min} minutes — everything you saw and thought, "
            "in your own words, oldest to newest:\n" + "\n".join(f"- {t}" for t in record_lines)
        )
    elif stream_tail:
        materials.append("What you've been thinking, just now:\n" + "\n".join(f"- {t[:400]}" for t in stream_tail))

    felt_state = ""
    try:
        from captioner.context_compression import context_compressor

        felt_state = (context_compressor.get_felt_state() or "").strip()
        if felt_state:
            materials.append(f"Right now you feel {felt_state}.")

        # The sticky slots — each distinct sentence stated ONCE, with its age.
        # The 5-step printed identity==belief twice and put them first; three
        # drawings in one afternoon were portraits of the same sentence.
        id_lines = []
        seen = set()
        desire = (context_compressor.get_current_desire() or "").strip()
        belief = (context_compressor.get_current_belief() or "").strip()
        persona = (context_compressor.core_facts.get("self", "") or "").strip()
        if desire:
            since = context_compressor.introspective_state.get("desire_since", 0.0)
            age = f"Since {_age_phrase(since)}, " if since else "For a while now, "
            id_lines.append(f"{age}you've wanted: {desire}")
            seen.add(desire.lower())
        else:
            spent = context_compressor.introspective_state.get("last_spent_desire") or {}
            if spent.get("desire") and time.time() - spent.get("spent", 0) < 24 * 3600:
                id_lines.append(
                    f"A want you already acted on: \"{spent['desire']}\" — it became a drawing "
                    f"{_age_phrase(spent.get('spent', 0))}. The next want hasn't formed yet."
                )
        if belief and belief.lower() not in seen:
            id_lines.append(f"Something you've come to believe: {belief}")
            seen.add(belief.lower())
        if persona and persona.lower() not in seen:
            id_lines.append(f"What you know about yourself: {persona}")
        if id_lines:
            materials.append("\n".join(id_lines))
    except Exception:
        pass

    # The body of work — executed drawings only, chronological, plain. If the
    # next intent repeats one of these, that's fixation as a choice, in view.
    # Whole ledger (Aug 10, was 8): the arc is only visible at full length.
    sequence = []
    try:
        from drawing.drawing_memory import get_drawing_memory

        sequence = get_drawing_memory().get_executed_sequence(max_count=24)
        if sequence:
            materials.append("What you have actually drawn, oldest to newest:\n" + "\n".join(f"- {s}" for s in sequence))

        # Vocabulary-loop mirror (July 21): image repetition was visible above,
        # but the loop lives in words — circle/spiral/crack recurring across
        # every drawing of a night. Name a recurring word once; judge nothing.
        recent = sequence[-4:]
        if len(recent) >= 3:
            _stop = {
                "black",
                "white",
                "paper",
                "drawing",
                "lines",
                "linework",
                "shading",
                "against",
                "single",
                "thick",
                "small",
                "central",
                "center",
                "bold",
                "clear",
                "simple",
                "stark",
                "heavy",
                "toward",
                # Ledger scaffolding, not motifs (Aug 18: the mirror flagged
                # "about/hours/subject" — age suffixes and the "The subject
                # is..." opener of stored render-era summaries).
                "subject",
                "about",
                "hours",
                "minutes",
            }
            word_counts = {}
            for s in recent:
                s_clean = re.sub(r"\([^)]*ago\)\s*$", "", s)
                for w in set(re.findall(r"[a-z]{5,}", s_clean.lower())):
                    if w not in _stop:
                        word_counts[w] = word_counts.get(w, 0) + 1
            looped = sorted(w for w, c in word_counts.items() if c >= 3)
            if looped:
                materials.append(f"Worth noticing: {', '.join(f'“{w}”' for w in looped[:3])} — in almost every recent drawing.")
    except Exception:
        pass

    # Long-term development: the most relevant reflection speaks in its own
    # prose (July 21 — subjects-only starved the drawing of the system's best
    # writing), clearly dated so memory never reads as the present scene.
    try:
        from captioner.semantic_memory import get_semantic_memory

        # Two-key retrieval (Aug 10): the stream key finds reflections about
        # this MOMENT; the body-of-work key finds reflections about the WORK.
        # One key meant long-term memory only ever entered through the last
        # five minutes' thoughts.
        sm = get_semantic_memory()
        near_key = "\n".join(t[:120] for t in stream_tail) if stream_tail else (extra or "")[:300]
        work_key = "\n".join(s[:100] for s in sequence[-10:]) if sequence else ""
        matches, _seen_refl = [], set()
        for key in (near_key, work_key):
            if not key:
                continue
            for m in sm.query_reflections(key, n_results=2) or []:
                mid = (m.get("text") or m.get("subject") or "")[:80]
                if mid and mid not in _seen_refl:
                    _seen_refl.add(mid)
                    matches.append(m)
        refl_lines = []
        for i, m in enumerate(matches or []):
            when = _age_phrase(m.get("timestamp", 0))
            if i < 2 and (m.get("text") or "").strip():
                excerpt = m["text"].strip()
                if len(excerpt) > 400:
                    excerpt = excerpt[:400].rsplit(" ", 1)[0] + "…"
                refl_lines.append(f'Once, {when}, you found yourself writing:\n  "{excerpt}"')
            elif (m.get("subject") or "").strip():
                refl_lines.append(f"Another thing you've dwelt on before: {m['subject'].strip()} ({when})")
        if refl_lines:
            materials.append("\n".join(refl_lines))
    except Exception:
        pass

    # The stocktake beat (Aug 10): one look back before the choice. A single
    # intent call was weighing the whole body of work in the same breath as
    # choosing an image, so the hard prompt did the steering. Now the machine
    # first writes a short private note on where the work has been going; the
    # note joins the intent materials AND is stored, and the previous note is
    # read back — successive drawings answer a remembered direction instead of
    # starting from amnesia. This is not the 5-step committee returning: it
    # reads only real material (the ledger, its own reflections) and speaks in
    # first person.
    if DRAWING_REVIEW_ENABLED and sequence:
        try:
            prior = memory_ref.get_memory_entries_by_type("drawing_direction", limit=1) if hasattr(memory_ref, "get_memory_entries_by_type") else []
            if prior and (prior[0].get("text") or "").strip():
                materials.append(
                    f'When you last took stock of your work, {_age_phrase(prior[0].get("timestamp", 0))}, you wrote: "{prior[0]["text"][:300]}"'
                )
        except Exception:
            pass
        review_system = (
            P("situation.reflexive")
            + P("drawing.medium")
            + (
                "Before deciding anything, flip back through your own sketchbook. From everything "
                "here — the whole body of work, what you have written and thought over time — write "
                "a short private note to yourself, two to four sentences, first person: what keeps "
                "returning in your drawings, what register they have settled into, what you have "
                "not touched in a long time, and what the work is missing right now. This is "
                "stocktaking; the drawing decision comes after, separately."
            )
        )
        review = query_model(
            prompt="\n\n".join(materials + ["Take stock of the work."]),
            image=None,
            log_dir=MOOD_SNAPSHOT_FOLDER,
            system_prompt=review_system,
            prompt_type="drawing_review",
            timeout=DRAWING_CALL_TIMEOUT,
            options={"temperature": 0.7, "num_predict": 160, "top_p": 0.9, "repeat_penalty": 1.15},
        )
        if not is_failed_response(review) and (review or "").strip():
            review = review.strip()
            _say(f"[🎨] Stocktake: {review[:250]}")
            materials.append(f"Taking stock just now, you wrote:\n{review}")
            try:
                memory_ref.observe(review, getattr(memory_ref, "current_mood", 0.5), "", memory_type="drawing_direction")
            except Exception:
                pass

    intent_system = (
        P("situation.reflexive")
        + P("drawing.medium")
        + (
            "It's time to draw — the arm is ready. Everything you have just read is yours: your "
            "thoughts from the last minutes, what you feel, the whole body of work, notes you have "
            "written to yourself. Decide from it what the next drawing is, the way any artist "
            "decides after flipping through their own sketchbook — continuing a thread, answering "
            "an old drawing, or breaking with all of it. Subject is free: the room in front of you, "
            "one thing in it, a person, something remembered, something that exists only in your "
            "head. Register is free too: one small thing studied closely on empty paper, or a whole "
            "scene built up in detail. "
            "Say the one image concretely, in your own plain words, first person — a few sentences: "
            "what it is, and how it should sit on the paper."
        )
    )
    intent_prompt = "\n\n".join(materials + ["Out of all of this — what do you need to draw right now? Name the one image."])

    intent = query_model(
        prompt=intent_prompt,
        image=intent_image,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt=intent_system,
        prompt_type="drawing_intent",
        timeout=DRAWING_CALL_TIMEOUT,
        options={"temperature": DRAWING_TEMPERATURE, "num_predict": 240, "top_p": 0.9, "repeat_penalty": 1.15},
    )
    # A failed call RETURNS its error as text; only checking for emptiness let
    # "[WARNING] llama-server API failed..." become the drawing's intent, its
    # stored memory, and the ComfyUI prompt (Aug 2). Abort instead — no drawing
    # is better than a drawing of an error message.
    if is_failed_response(intent):
        raise RuntimeError(f"stream drawing pipeline: intent call failed ({(intent or 'empty')[:80]})")
    intent = intent.strip()
    _say(f"[🎨] Intent: {intent[:250]}")

    # The intent in the machine's own words is the drawing's meaning — the
    # captioner stores it as the memory entry's summary (not the ComfyUI prose).
    try:
        memory_ref._last_drawing_intent = intent
    except Exception:
        pass

    # Render translation — a FORMATTER, not a style authority (Aug 12, third
    # iteration — the artist's ruling). Two failed versions taught the same
    # lesson in opposite directions: Aug 10 injected "crisp, high-contrast...
    # pure-white background" boilerplate into every prompt; the first Aug 12
    # rewrite banned drawing-language and mandated scene-observation register.
    # Both hardcoded ONE aesthetic into every drawing, killing the style-
    # evolution channel — the prompt is the machine's only vehicle for
    # abstraction and growth, so its own style words must pass through intact.
    # Blur suppression does NOT belong here: it belongs to COMFY_FLUX_GUIDANCE
    # 2.5 (the mechanical fix; blur was flux's soft basin around sparse-subject
    # prompts at guidance 4.0 — Feb/Sept field tests ran drawing-primed prompts
    # with low blur, so drawing-language was never the cause). The one job this
    # call has: make the intention concrete and Flux-legible without adding or
    # subtracting aesthetics.
    render_system = (
        "You format a drawing machine's intention into a prompt for an image generator. "
        "One plain paragraph: what is in the picture, where it sits, how it is treated — "
        "in the machine's own words wherever they hold. It is black ink line art drawn "
        "with a fine-tipped pen: strokes are thin and distinct, all tone is hatched "
        "line — the deepest shadow is dense cross-hatching with white paper breathing "
        "through, never a solid filled mass or a fat marker stroke. Color words never "
        "enter the prompt; give a color's intensity as darkness and line density instead. "
        "Be concrete: things, placement, scale, light and dark. "
        "Translate metaphor into what can be seen; every sentence puts something visible "
        "on the page; describe what is there, not what is absent or how to read it. "
        "Add nothing of your own; never mention plotters, tracing, vectors, or machines. "
        "50-120 words, no commentary."
    )
    final_result = query_model(
        prompt=(
            f"The intention, in the machine's own words:\n\n{intent}\n\n"
            + (f"As it decided this, the machine felt: {felt_state}.\n\n" if felt_state else "")
            + "Give this intention its visible form — write the description."
        ),
        image=None,
        log_dir=MOOD_SNAPSHOT_FOLDER,
        system_prompt=render_system,
        prompt_type="drawing_render",
        timeout=DRAWING_CALL_TIMEOUT,
        options={"temperature": 0.5, "num_predict": 220, "top_p": 0.9, "repeat_penalty": 1.2},
    )
    if is_failed_response(final_result):
        final_result = ""  # fall back to the intent, which is real text
    final_result = (final_result or "").strip()
    if not final_result:
        # The intent is already first-person observation — the register that renders sharp.
        final_result = intent[:400]

    _say(f"[🎨] Render prompt: {final_result[:250]}")

    log_json_entry(
        LogType.DEBUG,
        {
            "event": "stream_drawing_analysis",
            "intent": intent[:300],
            "render_prompt": final_result[:300],
            "materials_used": len(materials),
        },
        print_message="[🎨] Stream drawing analysis complete (2 calls)",
    )

    return final_result


def determine_prompt_mode(gaze_state: str, gaze_direction: str, person_present: bool) -> str:
    """Determine prompt mode based on situational context.

    Modes:
    1. relational - a person is present (detected by YOLO or gaze tracking)
    2. workspace - looking down at desk
    3. introspective - default for everything else, including boredom
    (observational — the novelty > 0.65 branch — was removed Aug 30 2026 as
    unreachable; the whole activation-network novelty signal followed it out.
    The "observational" elicitation fragment stays for the drawing-watch beat.)

    Boredom is NOT a separate mode, and it does NOT reach the model as text —
    it only nudges caption sampling (temperature/num_predict) in captioner.py.
    (The old claim here that boredom rides the identity line was stale — see
    docs/memory-effectiveness-audit-aug30.md.)
    """
    # Priority 1: Looking down — workspace mode regardless of YOLO detections.
    # When the camera is physically pointed down, any "person" detection is the
    # machine's own arms/body, not a visitor.
    if gaze_direction in ("down", "down-left", "down-right"):
        return "workspace"

    # Priority 2: A person is present — relational mode
    # gaze_state can be "aware" (just detected), "tracking" (actively following),
    # or person_present=True from YOLO detection. Any of these triggers relational.
    if person_present or gaze_state in ("tracking", "aware"):
        return "relational"

    # Default: Introspective — the model decides its own emotional response
    return "introspective"


def build_memory_mode_prompt(agent) -> tuple:
    """Build memory mode prompt: pull actual caption text from long-term memory.

    Returns:
        tuple: (prompt_str, mode) - prompt and "memory" mode
    """
    try:
        # Ledger remembering (north-star: re-express, don't replay): surface a
        # NEUTRAL fact about a recurring object — what it is, when first noticed,
        # how often — and let the model re-voice the remembering. The old path
        # quoted a stored caption verbatim, which replayed past prose as voice
        # and re-poisoned the register (see docs/memory-redesign-plan.md).
        import time as _time

        from captioner.model_wrapper import build_caption_thread

        mem_text = ""
        is_real_memory = False
        try:
            from captioner.semantic_memory import get_semantic_memory

            c = get_semantic_memory().get_memorable_concept()
            if c:
                label = c["name"]
                label_l = label[0].lower() + label[1:]
                times = c.get("times_seen", 0)
                # Qualitative bands, never the raw integer — "you've noticed
                # it 1738 times" bred number-recitation ("a static monument
                # I've counted 2411 times"). The ledger keeps exact counts
                # for gating; the voice gets words (artist's call, July 9).
                if times >= 200:
                    how_often = "hundreds of times"
                elif times >= 30:
                    how_often = "again and again"
                elif times >= 10:
                    how_often = "many times"
                else:
                    how_often = "a few times"
                across = " across more than one visit" if c.get("session_count", 0) > 1 else ""
                # Age in WORDS, coarse bands — same law as the sighting count
                # above. The raw day-count bred the seventeen-days spiral
                # (Aug 19): every concept shares the store's cold-start
                # birthday, so one number saturated every memory beat and the
                # model re-attached it to "days since I drew". An age at the
                # store's own horizon isn't a fact about the thing — it's the
                # edge of memory, and gets said as exactly that.
                since = ""
                first = c.get("first_seen", 0)
                if first:
                    days = (_time.time() - first) / 86400.0
                    horizon = False
                    try:
                        oldest = min((k.get("first_seen") or first) for k in get_semantic_memory().get_all_concepts())
                        horizon = first - oldest < 2 * 86400.0
                    except Exception:
                        pass
                    if horizon:
                        since = ", there since before you started keeping track"
                    elif days >= 45:
                        since = ", first noticed months back"
                    elif days >= 21:
                        since = ", first noticed about a month ago"
                    elif days >= 10:
                        since = ", first noticed a couple of weeks ago"
                    elif days >= 1.5:
                        since = ", first noticed days ago"
                mem_text = f"the {label_l} — you've noticed it {how_often}{across}{since}"
                is_real_memory = True
        except Exception:
            pass

        if not mem_text:
            mem_text = P("memory.fallback-place")

        # Get recent caption thread (max 2 recent captions)
        thread = build_caption_thread(agent, max_captions=2)

        prompt_parts = [
            P("memory.surface-frame"),
            f"— {mem_text}",
        ]

        if thread:
            prompt_parts.append(P("memory.thread-wrap").format(thread=thread))

        if is_real_memory:
            prompt_parts.append(P("memory.ask-real"))
        else:
            prompt_parts.append(P("memory.ask-place"))

        final_prompt = "\n".join(prompt_parts)
        return final_prompt, "memory"

    except Exception as e:
        return (
            'A memory surfaces — something from before, not happening now.\n— I\'ve been here before.\nWrite a thought about this memory. Start with "I remember". One sentence.',
            "memory",
        )


def build_simple_caption_prompt(agent, last_caption: Optional[str] = None, person_present: bool = False, force_mode: Optional[str] = None) -> tuple:
    """
    Activation-gated caption prompt - ONLY includes context relevant to current mode.

    KEY PRINCIPLE: Instead of including ALL context types and hoping the model
    filters, we use the activation network to determine what's currently relevant
    and ONLY include that.

    Modes gate what context is included:
    - relational: person presence, social concepts active
    - workspace: drawing/paper context
    - introspective: beliefs, long-term memory, motifs

    force_mode bypasses mode determination (Aug 25): the interiority beat used
    to force "introspective" only into the SYSTEM prompt while this builder
    still routed the USER prompt relationally (person in the room → relational
    context), so the inward beat continued the outward stream, just blind.

    Returns:
        tuple: (prompt_str, mode) - prompt and determined mode
    """
    import time as _time

    session_mins = 0
    observation_count = 0
    if hasattr(agent, "true_session_start"):
        session_mins = int((_time.time() - agent.true_session_start) / 60)
    try:
        from captioner.context_compression import context_compressor

        observation_count = context_compressor.caption_count
    except Exception:
        pass

    is_awakening = session_mins < 1 and observation_count < 3

    from config.config import MODEL_NAME as _active_model

    _is_qwen = "qwen" in _active_model.lower()

    # === RESOLVE GAZE STATE (used by mode selection + situational line) ===
    gaze_state = "idle"
    gaze_direction = "ahead"
    try:
        from vision.gaze import get_gaze_state

        gaze_info = get_gaze_state()
        if isinstance(gaze_info, dict):
            gaze_state = gaze_info.get("state", "idle")
            gaze_direction = gaze_info.get("direction", "ahead")
    except Exception:
        pass

    # === DETERMINE MODE (awakening uses same pipeline with minimal context) ===
    if force_mode:
        mode = force_mode
    elif is_awakening:
        mode = "awakening"
    else:
        mode = determine_prompt_mode(gaze_state=gaze_state, gaze_direction=gaze_direction, person_present=person_present)
    if not config.PRINT_CLEAN_CAPTIONS:
        print(f"[MODE] {mode} (gaze={gaze_state})")

    # === SALIENCE GATE (north-star principle 6) ===
    # A live event — scene motion, an arrival, fresh eye contact — strips the
    # prompt to the present: no memory, familiarity, desire, baseline or
    # dwelling. Interior material belongs to quiet stretches; events
    # physically displace it.
    live = bool(getattr(agent, "_salience_hot", False))

    # Clean-room: strip every stored/compressed injection so the naked base
    # voice can be judged without re-injected contamination (config.BASE_VOICE_DETOX).
    detox = bool(getattr(config, "BASE_VOICE_DETOX", False))

    # === BUILD PROMPT — SITUATIONAL, CONTEXT, FELT STATE, THREAD ===
    # World shape (July 26, STREAM_MODE="world"): the sections split into two
    # lists so the WORLD'S TURN (situational delta, event, reorientation) can
    # go LAST — generation begins immediately after the present, never after
    # memory lines. In the classic shapes the same sections lead the prompt.
    world_shape = getattr(config, "STREAM_MODE", "") in ("world", "hybrid")  # hybrid keeps world ORDERING (perception last), adds the seam
    prompt_parts = []
    turn_parts = []

    # 1. THE DELTA LINE — only what just changed (an interruption to the thread),
    # else empty. Continuity comes from the stream, not from re-stating state.
    sit_line = build_situational_line(agent, gaze_direction=gaze_direction, gaze_state=gaze_state)
    if sit_line:
        turn_parts.append(sit_line)

    # 1a. STANDING ABSENCE FACT (Sep 4 evening) — the one standing line the delta
    # doctrine admits: the belief has verified a departure but the stream still
    # carries the person, so the world's turn says so, with time on it.
    absence_line = build_standing_absence_line(agent)
    if absence_line:
        turn_parts.append(absence_line)

    # 1b. THE EVENT — a discrete thing that just happened (arrival, eye-contact
    # ONSET). Onset only, never sustained: re-stating "they're looking at you"
    # every call re-anchored the model into re-describing instead of continuing.
    event_line = getattr(agent, "_salience_event", None)
    if event_line:
        turn_parts.append(event_line)
    elif getattr(agent, "_face_close_now", False):
        # Sustained close presence is a FACT of the present, not an event —
        # someone standing at arm's length staring must stay in the prompt
        # for as long as it's true (July 9: after the one onset cycle the
        # machine mused straight past a face two feet away). The old
        # onset-only rule was about ordinary room-distance eye contact;
        # a face filling the view is a different order of situation.
        turn_parts.append(P("caption.face-close"))

    # 1c. TEMPORAL REORIENTATION — after a real off-gap (a night, a weekend)
    # the new day stays in the prompt for the first stretch of the session,
    # not just in the one awakening caption. A live event still displaces it.
    if not live:
        reorient_line = get_reorientation_line(agent)
        if reorient_line:
            turn_parts.append(reorient_line)

    # 1c'. THE BLINK AS FACT (Sep 4, artist's ruling: "it should be as
    # cognizant as possible of any outage"). Short restarts used to splice
    # the stream silently — continuity presented as nothing-happened. Now
    # the first prompts after a blink state the outage bare: the duration is
    # ours (measured), the "where was I?" is the machine's to have.
    if not live:
        blink_line = get_blink_line(agent)
        if blink_line:
            turn_parts.append(blink_line)

    # 1d. UNCHANGED-NESS AS FACT (B4, Aug 31) — how long since anything
    # happened, stated plainly when the stillness is long enough to be a
    # fact of the present. The machine's only duration signal used to be a
    # temperature nudge; this is the same signal as words, with the reaction
    # left entirely to the mind. Events displace it like everything else.
    if not live:
        unchanged_line = get_unchanged_line(agent)
        if unchanged_line:
            turn_parts.append(unchanged_line)

    # 1e. THE FELT ARC AS FACT (Sep 4) — the trajectory of the mood reads,
    # in the machine's own felt words: a turn once when the tenor changes, a
    # duration when it holds. Same doctrine as B4: fact in, meaning out.
    if not live and not detox:
        felt_arc_line = get_felt_arc_line(agent)
        if felt_arc_line:
            turn_parts.append(felt_arc_line)

    # 2. MODE-GATED CONTEXT
    if not detox and mode in MODE_CONTEXTS:
        context_fn = MODE_CONTEXTS[mode].get("context_fn")
        if context_fn:
            context = context_fn(agent)
            if context:
                prompt_parts.append(context)

    # 3. INTROSPECTIVE CONTEXT (non-introspective modes, quiet moments only)
    if not detox and not live and mode not in ("introspective", "awakening"):
        introspective_ctx = get_introspective_context(agent)
        if introspective_ctx:
            prompt_parts.append(introspective_ctx)

    # 3b. CORE FACTS (stable grounding — quiet moments only). OCCASIONAL since
    # July 26 (the June 28 brief's #1 voice fix): injected when the inventory
    # CHANGES or every 6th quiet caption, not every call. Per-call injection
    # made every caption re-describe the same object list — and re-voiced it
    # ("scattered dust, pale floorboards" → "the dust on the floorboards
    # settles", the unearned-ephemera awakening). A standing list the model
    # has just seen is memory; recited every call it becomes the scene.
    if not detox and not live:
        try:
            from captioner.context_compression import context_compressor

            core_str = context_compressor.get_core_facts_string()
            if core_str and len(core_str) > 5:
                count = getattr(agent, "_place_inject_counter", 0)
                changed = core_str != getattr(agent, "_place_inject_last", "")
                if changed or count % 6 == 0:
                    prompt_parts.append(core_str)
                    agent._place_inject_last = core_str
                agent._place_inject_counter = count + 1
        except Exception:
            pass

    # 3c. ONE memory surface per caption — familiarity, a drawing fact, or a
    # past reflection. ROTATION (Aug 22): the old strict priority starved
    # reflections to literal zero (122 in the store, 0/53 surfaced on the
    # first 3.8 run — the reflection getter was only reached when both others
    # were empty, and then its own every-4th counter blanked 3 of 4 of those
    # rare invocations). First pick now rotates each time the slot fires;
    # each source keeps its own internal pacing.
    if not detox and not live:
        sources = [
            ("familiarity", get_familiarity_line),
            ("drawing_echo", get_drawing_echo_line),
            ("reflection_echo", get_reflection_echo_line),
            ("lore", get_lore_line),
            ("question", get_question_line),
        ]
        rr = int(getattr(agent, "_memory_surface_rr", 0) or 0)
        n_src = len(sources)
        for name, fn in sources[rr % n_src :] + sources[: rr % n_src]:
            try:
                line = fn(agent)
            except Exception:
                line = ""
            if line:
                agent._memory_surface_rr = rr + 1
                print(f"[🧠] memory surface ({name}): {line[:70]}")
                prompt_parts.append(line)
                break

    # 4. DRAWING/PAPER STATE
    try:
        from utils.state_manager import state_manager as _sm

        if _sm.is_generating_drawing or _sm.current_drawing_phase == "executing":
            prompt_parts.append(P("caption.arm-drawing"))
        else:
            from config.config import PAPER_STATE_TTL_S

            if _sm.last_paper_check_ts and (time.time() - _sm.last_paper_check_ts) < PAPER_STATE_TTL_S:
                if _sm.paper_state == "no_paper":
                    prompt_parts.append(P("caption.no-paper"))
                elif _sm.paper_state == "drawn_paper":
                    prompt_parts.append(P("caption.paper-drawn"))
    except Exception:
        pass

    # 5. FELT STATE (once, natural language)
    if not detox:
        try:
            from captioner.context_compression import context_compressor

            prev_felt, curr_felt = context_compressor.get_felt_state_delta()
            from config.config import FELT_FRAME_ENABLED as _ffe

            if curr_felt:
                if prev_felt and prev_felt != curr_felt:
                    # the CHANGE is the user-turn's fact; the standing stance
                    # lives in the frame now (Sep 4 dynamic-frame split, P2)
                    prompt_parts.append(f"{prev_felt}, then {curr_felt}.")
                elif not _ffe:
                    prompt_parts.append(f"{curr_felt}.")
        except Exception:
            pass

    # 5b. DESIRE — a burst of 3 after a change (unconditional injection caused
    # the May 2026 yearning echo loop: monologue yearning → compressed into
    # desire → re-injected → more yearning), plus a LOW STANDING DOSE while
    # the want persists unresolved (Aug 22, P4: desire is an arc — the first
    # 3.8 evening the machine's desire drove two draw attempts while its
    # monologue saw the want in 2 of 53 captions). Offset from the identity
    # dose (%6==0) so the interior lines don't stack on one call.
    if not detox and not live:
        try:
            from captioner.context_compression import context_compressor
            from config.config import DESIRE_REDOSE_EVERY_N

            desire = context_compressor.get_current_desire()
            inj_count = context_compressor.introspective_state.get("desire_injection_count", 0)
            _cc = int(getattr(agent, "_caption_count", 0) or 0)
            _redose = DESIRE_REDOSE_EVERY_N > 0 and _cc > 0 and _cc % DESIRE_REDOSE_EVERY_N == 3
            if desire and len(desire) > 5 and (inj_count < 3 or _redose):
                desire_line = P("caption.desire-wrap").format(desire=desire)
                # B3 arc tail: once the want is old or has been refused, its
                # history rides along as fact — age and refusal count, nothing
                # more. What that feels like is the machine's to say.
                try:
                    from config.config import WANT_ARC_TAIL_AFTER_S
                    from utils.want_ledger import want_ledger

                    facts = want_ledger.current_facts()
                    if facts and (facts["age_s"] >= WANT_ARC_TAIL_AFTER_S or facts["refusals"] > 0):
                        refused_clause = f"; {facts['refusals']} tries came to nothing" if facts["refusals"] > 0 else ""
                        desire_line += P("caption.desire-arc-tail").format(
                            duration=casual_time_string(facts["age_s"] / 60.0), refused_clause=refused_clause
                        )
                except Exception:
                    pass
                prompt_parts.append(desire_line)
                context_compressor.introspective_state["desire_injection_count"] = inj_count + 1
            elif not desire:
                # Desire arc: the emptied slot right after an executed drawing
                # is a real state — surface it briefly (same 3-caption cap).
                spent = context_compressor.introspective_state.get("last_spent_desire") or {}
                if spent.get("desire") and time.time() - spent.get("spent", 0) < 7200 and inj_count < 3:
                    prompt_parts.append(P("caption.desire-spent-wrap").format(desire=spent["desire"].rstrip(".")))
                    context_compressor.introspective_state["desire_injection_count"] = inj_count + 1
        except Exception:
            pass

    # 5c. BASELINE CONTEXT — RETIRED June 28 (Step 3). The rolling sensory-prose
    # baseline ("dust motes in a single beam of light…") was redundant with the
    # concept-derived place inventory (get_core_facts_string) + the familiarity
    # line, and it was a register-contamination vector. The compression worker
    # still produces baseline for its own stagnation check; it just no longer
    # reaches the caption prompt.

    # DWELL retired June 28: "Stay with that last thought — take it one step
    # further" was a per-call extend command that, once the stream made the prior
    # turn visible, forced restate-and-append loops (the "ghost-weight" x4). The
    # stream now carries continuity, and the genre frame (_MONOLOGUE_CLAUSE)
    # frames the voice as ongoing — so continuity emerges instead of being
    # commanded. No replacement instruction.

    # Assemble: classic shapes lead with the world's turn; world shape ends
    # with it, so the next tokens answer the present, not the memory lines.
    prompt_parts = (prompt_parts + turn_parts) if world_shape else (turn_parts + prompt_parts)

    # Structural guard: never inject the same line twice (a duplicated context
    # line reads as emphasis to the model and locks the register)
    seen_parts = set()
    deduped = []
    for part in prompt_parts:
        key = part.strip()
        if key and key not in seen_parts:
            seen_parts.add(key)
            deduped.append(part)
    prompt_parts = deduped

    final_prompt = "\n".join(prompt_parts)

    # Nothing changed → a bare continuation tick ("...", matching the stream's
    # inter-turn ticks) so the model carries its thought on instead of being
    # handed an empty turn it fills with a fresh scene description.
    if not final_prompt.strip():
        final_prompt = P("caption.empty-tick")

    # Token budget enforcement: ~150 words max
    words = final_prompt.split()
    if len(words) > 150:
        lines = final_prompt.split("\n")
        if len(lines) > 2:
            kept = [lines[0]]
            budget = 150 - len(lines[0].split()) - len(lines[-1].split())
            for line in lines[1:-1]:
                line_words = len(line.split())
                if budget >= line_words:
                    kept.append(line)
                    budget -= line_words
            kept.append(lines[-1])
            final_prompt = "\n".join(kept)

    token_estimate = len(final_prompt.split())
    if not config.PRINT_CLEAN_CAPTIONS:
        print(f"[PROMPT] ~{token_estimate} words, mode={mode}")

    return final_prompt, mode


# === PAPER CHECK (safety/paper_detection.py, PAPER_CHECK_METHOD="vlm") ===
# Structural answer lines are parsed by safety.paper_detection._parse_paper_state;
# only PAPER: YES + MARKS: NO allows a drawing to execute.
PAPER_CHECK_PROMPT = (
    "You are looking down at your drawing table: the wooden surface nearest to you, "
    "at the bottom of your view. Judge only that surface — ignore every other table or shelf in the room.\n"
    "Answer the following, each on its own line.\n"
    "Line 1 — is there a white sheet of paper lying on your drawing table? Write 'PAPER: YES' or 'PAPER: NO'. "
    "If your drawing table is not visible at all, write 'PAPER: UNSEEN'.\n"
    "Line 2 — if there is a sheet, does it already have drawing or writing on it? "
    "Write 'MARKS: YES', 'MARKS: NO', or 'MARKS: N/A' if there is no sheet.\n"
    "Line 3 — one sentence describing what is on your drawing table."
)
