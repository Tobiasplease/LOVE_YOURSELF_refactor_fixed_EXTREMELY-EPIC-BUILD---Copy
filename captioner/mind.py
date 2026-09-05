"""The mind (Sep 5 evening) — a conversation with itself over a life.

Diagnosis (docs/architecture-diagnosis-sep5.md): the machine's continuity was
a log of its own output. Its mind at the moment of speaking was its last 24
stamped sentences plus a status board; its identity a distillation of that
log's failure theme; every store the same theme in five formats. No organ
bolted on could widen it, because every organ wrote into the same log.

This module replaces the log with a conversation:
  - the last few thoughts ride as REAL assistant turns; the world speaks in
    user turns; the clock is the cue ("18:41. Eyes resting.");
  - two turn kinds — LOOK (frame + what changed + what's known to be in view)
    and THINK (no frame; wondering is legal, seeing is not);
  - a compact LIFE block opens the conversation instead of a status board:
    when, the room as known, people today, drawings, the want, questions,
    where recent threads got to, a couple of dated past thoughts;
  - the deepening mechanic: a subject's last conclusion is its POSITION and
    rides as a premise; the reframe move ("it's not X; it's Y") on one subject
    with no new words counts as a PIVOT, and after N pivots the machine hears
    it (mind.pivot-notice);
  - memories surface by choice: old enough, novel against the recent thread,
    never a reframe, never person-tinged while the room is believed empty.

Wordings live in captioner/prompt_registry.py (mind.*) — the artist's to
finalize. Structure only here: kinds are named, contents never.
"""
import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

from captioner.prompt_registry import P
from config import config

_NEG_FRAME_RE = re.compile(r"\b(it.s not|isn.t|not (?:a|an|the|just)\b|no longer|used to|not .{1,25} anymore)\b", re.I)
_WORD_RE = re.compile(r"[a-z']+")
_STOP = set(
    "the a an and or but of to in on at for with from by as is are was were be been being it its this that these those there here "
    "just still now then than into over under about like my me i you your it's i'm i've that's what when where which who how "
    "have has had do does did not no yes so if because while very more most some any all each".split()
)
_SENT_END_RE = re.compile(r"(?<=[.!?…])\s+")
_KEEP_KINDS = ("look", "think", "memory", "wake")


def clock(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


def content_words(text: str) -> set:
    return {w for w in _WORD_RE.findall((text or "").lower()) if len(w) > 3 and w not in _STOP}


def when_words(age_s: float) -> str:
    """Words, never integers: how long ago a thought was."""
    m = age_s / 60.0
    if m < 2:
        return "a moment ago"
    if m < 20:
        return "a few minutes ago"
    if m < 50:
        return "half an hour ago"
    if m < 100:
        return "about an hour ago"
    if m < 300:
        return "a few hours ago"
    if m < 20 * 60:
        return "earlier today"
    if m < 44 * 60:
        return "yesterday"
    d = int(m / 1440)
    if d < 7:
        return "a few days ago"
    if d < 14:
        return "about a week ago"
    return "a while ago"


def daypart(ts: float) -> str:
    h = time.localtime(ts).tm_hour
    if h < 5:
        return "night"
    if h < 12:
        return "morning"
    if h < 18:
        return "afternoon"
    if h < 22:
        return "evening"
    return "night"


def dur_words(seconds: float) -> str:
    m = seconds / 60.0
    if m < 3:
        return "a minute or two"
    if m < 15:
        return "a few minutes"
    from captioner.prompts import casual_time_string

    return casual_time_string(m)


def title_of(desc: str, max_words: int = 9) -> str:
    """A drawing's description as a short title: first clause, no dangling verb."""
    d = re.split(r"[.;:]", (desc or "").replace("finished a drawing of ", "").strip())[0]
    parts = [p.strip() for p in d.split(",") if p.strip()]
    out = parts[0] if parts else d
    if len(parts) > 1 and len((out + ", " + parts[1]).split()) <= max_words:
        out += ", " + parts[1]
    words = out.split()[:max_words]
    return " ".join(words).strip(" ,")


def moved_recently(recent_meta: list, now: float, settle_s: float) -> bool:
    """Did the head move within settle_s of now? (frame_buffer flags ego_motion
    per frame; the flare/exposure settling that follows a pan is not motion.)"""
    if settle_s <= 0:
        return False
    return any(f.get("detection", {}).get("ego_motion") and now - float(f.get("timestamp", 0)) <= settle_s for f in recent_meta or [])


def steady_jpeg(recent_meta: list):
    """The newest frame captured with the head still, or None."""
    for f in reversed(recent_meta or []):
        if not f.get("detection", {}).get("ego_motion") and f.get("jpeg"):
            return f["jpeg"]
    return None


def last_sentence(text: str) -> str:
    parts = [p.strip() for p in _SENT_END_RE.split((text or "").strip()) if p.strip()]
    return parts[-1] if parts else (text or "").strip()


class Mind:
    def __init__(self, agent=None, path: Optional[str] = None):
        self.agent = agent
        self.path = path or os.path.join(config.MOOD_SNAPSHOT_FOLDER, "mind_thread.json")
        self.thread: List[dict] = []
        self.positions: Dict[str, dict] = {}
        self.last_look_ts = 0.0
        self.think_count = 0
        self.pending_notice: Optional[Tuple[str, int]] = None
        self._last_believed: Optional[bool] = None
        self._load()

    # ---- persistence -----------------------------------------------------
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.thread = list(d.get("thread") or [])[-int(config.MIND_THREAD_MAX) :]
            self.positions = dict(d.get("positions") or {})
            self.last_look_ts = float(d.get("last_look_ts") or 0.0)
        except Exception:
            self.thread, self.positions = [], {}

    def _save(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"thread": self.thread[-int(config.MIND_THREAD_MAX) :], "positions": self.positions, "last_look_ts": self.last_look_ts}, f, indent=1)
            os.replace(tmp, self.path)
        except Exception:
            pass

    # ---- the thread --------------------------------------------------------
    def has_session(self, session_start: float) -> bool:
        return any(e.get("ts", 0) >= session_start for e in self.thread)

    def absorb(self, text: str, kind: str, cue: str, now: Optional[float] = None) -> dict:
        now = now or time.time()
        text = (text or "").strip()
        subject = self.subject_of(text)
        entry = {"ts": now, "kind": kind, "cue": (cue or "").strip(), "text": text, "subject": subject}
        self.thread.append(entry)
        if len(self.thread) > int(config.MIND_THREAD_MAX):
            self.thread = self.thread[-int(config.MIND_THREAD_MAX) :]
        if kind == "look":
            self.last_look_ts = now
        self._update_position(subject, text, now)
        self._save()
        return entry

    def premise(self, now: Optional[float] = None) -> str:
        """The machine's own last sentence, quoted back as the thing to go on
        from (the continuation mechanic — see mind.cue-premise)."""
        now = now or time.time()
        if not self.thread:
            return ""
        last = self.thread[-1]
        if now - last.get("ts", 0) > int(config.MIND_TURN_MAX_AGE_S):
            return ""
        return last_sentence(last.get("text", ""))[:200]

    def recent_turns(self, now: Optional[float] = None) -> List[dict]:
        now = now or time.time()
        fresh = [e for e in self.thread if now - e.get("ts", 0) <= int(config.MIND_TURN_MAX_AGE_S) and e.get("text")]
        return fresh[-int(config.MIND_TURNS) :]

    # ---- subjects, positions, pivots ----------------------------------------
    def _terms(self) -> List[str]:
        try:
            from perception.spatial_registry import spatial_registry

            entries = spatial_registry.get_entries() or {}
            return sorted(entries.keys(), key=lambda k: -(entries[k].get("hits", 0) or 0))
        except Exception:
            return []

    def subject_of(self, text: str) -> str:
        """The registry term the thought is about (the machine's own vocabulary);
        failing that, a recurring abstract noun — the lemma that this thought
        shares with one of the last three (so a fixation on 'silence' or
        'absence' is tracked for positions and pivots, not only room objects)."""
        term = self._registry_subject(text)
        if term:
            return term
        return self._recurring_noun(text)

    def _recurring_noun(self, text: str) -> str:
        try:
            from utils.nlp import nlp

            def nouns(t):
                return [tok.lemma_.lower() for tok in nlp(t or "") if tok.pos_ in ("NOUN", "PROPN") and len(tok.lemma_) > 3 and tok.lemma_.lower() not in _STOP]

            here = nouns(text)
            if not here:
                return ""
            prior = set()
            for e in self.thread[-3:]:
                prior |= set(nouns(e.get("text", "")))
            shared = [n for n in here if n in prior]
            if not shared:
                return ""
            prev = (self.thread[-1].get("subject") if self.thread else "") or ""
            if prev and (prev in shared or set(here) & set(nouns(self.thread[-1].get("text", "")))):
                return prev  # continuity: a reframe renames the thing; the chain keeps its subject
            return max(dict.fromkeys(shared), key=lambda n: (shared.count(n), -here.index(n)))
        except Exception:
            return ""

    def _registry_subject(self, text: str) -> str:
        stems = {re.sub(r"'s?$", "", w).rstrip("s") for w in _WORD_RE.findall((text or "").lower())}
        low = (text or "").lower()
        best, best_score = "", 0
        for term in self._terms():
            tw = [w for w in term.lower().split() if w not in _STOP]
            if not tw:
                continue
            if term.lower() in low:
                score = 10 + len(tw)
            else:
                head = tw[-1].rstrip("s")
                hits = sum(1 for w in tw if w.rstrip("s") in stems)
                score = hits if (head in stems and hits * 2 >= len(tw)) else 0
            if score > best_score:
                best, best_score = term, score
        return best

    def _update_position(self, subject: str, text: str, now: float) -> None:
        if not subject or not text:
            return
        pos = self.positions.get(subject)
        ttl = int(config.MIND_POSITION_TTL_S)
        if pos and now - float(pos.get("last_ts", 0)) < ttl:
            new_words = content_words(text) - content_words(pos.get("text", "")) - set(subject.lower().split())
            if _NEG_FRAME_RE.search(text) and len(new_words) < 6:  # a reframe with little new material
                pos["pivots"] = int(pos.get("pivots", 0)) + 1
            else:
                pos["pivots"] = 0
            if pos["pivots"] >= int(config.MIND_PIVOTS_BEFORE_NOTICE):
                self.pending_notice = (subject, pos["pivots"])
                pos["pivots"] = 0
        else:
            pos = {"pivots": 0, "ts": now}
        pos["text"] = last_sentence(text)[:200]
        pos["last_ts"] = now
        self.positions[subject] = pos
        if len(self.positions) > 40:
            for k in sorted(self.positions, key=lambda k: self.positions[k].get("last_ts", 0))[:-40]:
                self.positions.pop(k, None)

    def fresh_positions(self, now: float, exclude: str = "", n: int = 2) -> List[Tuple[str, str]]:
        ttl = int(config.MIND_POSITION_TTL_S)
        items = [(k, v) for k, v in self.positions.items() if now - float(v.get("last_ts", 0)) < ttl and k != exclude and v.get("text")]
        items.sort(key=lambda kv: -float(kv[1].get("last_ts", 0)))
        return [(k, v["text"]) for k, v in items[:n]]

    # ---- memory surfacing --------------------------------------------------
    def choose_memory(self, now: float, believed: bool = False, exclude: Optional[set] = None) -> Optional[dict]:
        recent = set()
        for e in self.thread[-6:]:
            recent |= content_words(e.get("text", ""))
        min_age = int(config.MIND_MEMORY_MIN_AGE_S)
        cands = [e for e in self.thread if now - e.get("ts", 0) >= min_age and e.get("kind") in _KEEP_KINDS and len(e.get("text", "")) > 30]
        cands = [e for e in cands if not _NEG_FRAME_RE.search(e["text"])]
        if not believed:
            try:
                from utils.presence_text import PERSON_RE

                cands = [e for e in cands if not PERSON_RE.search(e["text"])]
            except Exception:
                pass
        if exclude:
            cands = [e for e in cands if e["text"] not in exclude]
        if not cands:
            return None

        def novelty(e):
            cw = content_words(e["text"])
            return len(cw - recent) / max(1, len(cw))

        top = sorted(cands, key=novelty, reverse=True)[:6]
        return random.choice(top)

    # ---- turn kind + cadence -------------------------------------------------
    def next_kind(self, now: float, scene: Optional[dict], agent) -> str:
        hot = bool(getattr(agent, "_salience_hot", False))
        believed = bool(getattr(agent, "_presence_believed", False))
        edge = self._last_believed is not None and believed != self._last_believed
        self._last_believed = believed
        since_look = now - self.last_look_ts
        if not self.last_look_ts:
            return "look"
        if hot or edge or (scene or {}).get("view_changed"):
            return "look" if since_look >= 3 else "think"
        if since_look < float(config.MIND_LOOK_MIN_GAP_S):
            return "think"
        try:
            from utils import chosen_glance

            if chosen_glance.current() or chosen_glance.pending():
                return "look"
        except Exception:
            pass
        if since_look >= float(config.MIND_LOOK_EVERY_S):
            return "look"
        return "think"

    def interval(self, now: float, agent) -> float:
        if getattr(agent, "_salience_hot", False):
            return float(config.CAPTION_INTERVAL_LIVE)
        base = float(config.MIND_THINK_INTERVAL_S)
        try:
            from utils import felt_loop

            base *= float(felt_loop.cadence_mult())
        except Exception:
            pass
        return base

    # ---- the life block ------------------------------------------------------
    def life_block(self, now: float, agent) -> str:
        lines = []
        try:
            with open(os.path.join(config.MOOD_SNAPSHOT_FOLDER, "lifetime_state.json"), encoding="utf-8") as f:
                life = json.load(f)
            first = time.strftime("%B %Y", time.localtime(float(life.get("first_boot", now))))
        except Exception:
            first = "some time ago"
        woke = clock(float(getattr(agent, "true_session_start", now) or now))
        lines.append(P("mind.life-when").format(clock=clock(now), weekday=time.strftime("%A", time.localtime(now)), daypart=daypart(now), first=first, woke=woke))
        terms = self._terms()[: int(config.MIND_ROOM_TERMS)]
        if terms:
            lines.append(P("mind.life-room").format(terms=", ".join(terms)))
        lines.append(self._people_line(now, agent))
        lines.append(self._drawings_line(now))
        try:
            from utils.state_manager import state_manager as _sm

            if _sm.paper_state == "no_paper" and _sm.last_paper_check_ts and now - _sm.last_paper_check_ts < float(config.PAPER_STATE_TTL_S):
                lines.append(P("caption.no-paper"))
        except Exception:
            pass
        try:
            from captioner.context_compression import context_compressor
            from utils.want_ledger import want_ledger

            facts = want_ledger.current_facts()
            want = (context_compressor.get_current_desire() or (facts or {}).get("text") or "").strip()
            if want and facts:
                from captioner.prompts import casual_time_string

                lines.append(P("mind.life-want").format(age=casual_time_string(facts["age_s"] / 60.0), want=want.rstrip(".") + "."))
        except Exception:
            pass
        try:
            from utils.lore_ledger import lore_ledger

            qs = [(q.get("text") or q.get("words") or "").strip() for q in lore_ledger.open_questions(2)]
            qs = [q for q in qs if q]
            if qs:
                lines.append(P("mind.life-questions").format(questions=" ".join(qs)))
        except Exception:
            pass
        last_subject = (self.thread[-1].get("subject") if self.thread else "") or ""
        for subject, text in self.fresh_positions(now, exclude=last_subject):
            lines.append(P("mind.life-position").format(subject=subject, text=text))
        chosen: set = set()
        believed = bool(getattr(agent, "_presence_believed", False))
        for _ in range(int(config.MIND_PAST_THOUGHTS)):
            m = self.choose_memory(now, believed=believed, exclude=chosen)
            if not m:
                break
            chosen.add(m["text"])
            lines.append(P("mind.life-past").format(when=when_words(now - m["ts"]), text=m["text"][:220]))
        return " ".join(x for x in lines if x)

    def _people_line(self, now: float, agent) -> str:
        if getattr(agent, "_presence_believed", False):
            return P("mind.life-people-now")
        try:
            from utils.episodic_log import episodic_log

            lt = time.localtime(now)
            midnight = now - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec)
            pairs = episodic_log.get_pairs_in_window("person_arrived", "person_left", window_seconds=int(now - midnight) + 1)
            pairs = [p for p in pairs if p.get("start", {}).get("timestamp", 0) >= midnight and p.get("end")]
            if not pairs:
                return P("mind.life-people-none")
            last = max(pairs, key=lambda p: p["end"]["timestamp"])
            times = {1: "once", 2: "twice", 3: "three times"}.get(len(pairs), "several times")
            return P("mind.life-people-today").format(times=times, last_ago=when_words(now - last["end"]["timestamp"]), duration=dur_words(last["duration_seconds"]))
        except Exception:
            return ""

    def _drawings_line(self, now: float) -> str:
        try:
            from utils.episodic_log import episodic_log

            drew = episodic_log.get_recent_events(window_seconds=10 * 365 * 86400, types=["drew"])
            if not drew:
                return P("mind.life-drawings-none")
            last = max(drew, key=lambda e: e.get("timestamp", 0))
            desc = title_of(last.get("description") or "") or "something"
            desc = desc[0].lower() + desc[1:]
            return P("mind.life-drawings").format(count=len(drew), age=when_words(now - last.get("timestamp", now)), desc=desc[:120])
        except Exception:
            return ""

    # ---- what a look lands on --------------------------------------------------
    def in_view(self, agent) -> List[str]:
        try:
            from perception.spatial_registry import spatial_registry
            from vision.gaze import get_gaze_state

            g = get_gaze_state()
            pan, tilt = float(g.get("pan", 90)), float(g.get("tilt", 90))
            entries = spatial_registry.get_entries() or {}
            tp, tt = float(config.MIND_VIEW_TOL_PAN), float(config.MIND_VIEW_TOL_TILT)
            near = [(k, v) for k, v in entries.items() if abs(float(v.get("pan", 999)) - pan) <= tp and abs(float(v.get("tilt", 999)) - tilt) <= tt]
            near.sort(key=lambda kv: -float(kv[1].get("hits", 0)))  # familiarity, not detector confidence
            return [k for k, _ in near[: int(config.MIND_VIEW_TERMS)]]
        except Exception:
            return []

    def note_look(self, now: float) -> None:
        """A look happened, stored or not — the look timer advances either way
        (Sep 5 23:25–23:39: gated looks left the timer stale, so every phantom
        was followed by a second look a minute later)."""
        self.last_look_ts = now
        self._save()

    def _seen_this_session(self, terms: List[str], agent) -> bool:
        """Seen within the thread's own window (not the process session — a
        restart must not make the room new again)."""
        try:
            from perception.spatial_registry import spatial_registry

            entries = spatial_registry.get_entries() or {}
            start = time.time() - int(config.MIND_TURN_MAX_AGE_S)
            return any(float((entries.get(t) or {}).get("last_seen", 0)) >= start for t in terms)
        except Exception:
            return True

    # ---- the call ----------------------------------------------------------------
    def build(self, kind: str, now: float, agent, scene: Optional[dict], img_path: Optional[str]) -> dict:
        system = P("mind.system")
        try:
            from utils.state_manager import state_manager as _sm

            if not (_sm.is_generating_drawing or _sm.current_drawing_phase == "executing"):
                system += P("monologue.pen-parked")
        except Exception:
            pass
        try:
            if config.FELT_FRAME_ENABLED:
                from captioner.context_compression import context_compressor as _cc

                felt = (_cc.get_felt_state() or "").strip()
                if felt:
                    system += P("monologue.felt-frame").format(felt=felt)
        except Exception:
            pass

        believed = bool(getattr(agent, "_presence_believed", False))
        someone = P("mind.someone") if believed else ""
        memory = None
        if kind == "look":
            event = getattr(agent, "_salience_event", None)
            if event:
                cue = P("mind.cue-look-event").format(clock=clock(now), event=str(event).strip(), someone=someone)
            else:
                terms = self.in_view(agent)
                where = P("mind.where").format(terms=", the ".join(terms[:-1]) + " and the " + terms[-1] if len(terms) > 1 else terms[0]) if terms else ""
                verdict = getattr(agent, "_last_view_verdict", None)
                change = {"unchanged": P("mind.change-none"), "changed": P("mind.change-yes")}.get(verdict or "", "")
                if verdict in ("new", "baselined") and not self._seen_this_session(terms, agent):
                    change = P("mind.change-new")  # honest only when nothing in view was seen this session (the referee keys by gaze cell)
                cue = P("mind.cue-look").format(clock=clock(now), where=where, change=change, someone=someone)
        else:
            self.think_count += 1
            n = int(config.MIND_MEMORY_EVERY_N)
            if n > 0 and self.think_count % n == 0:
                memory = self.choose_memory(now, believed=believed)
            if memory:
                cue = P("mind.cue-think-memory").format(clock=clock(now), when=when_words(now - memory["ts"]), memory=memory["text"][:220])
            else:
                cue = P("mind.cue-think").format(clock=clock(now))
                prem = self.premise(now)
                if prem:
                    cue += P("mind.cue-premise").format(premise=prem)
        if self.thread and now - self.thread[-1].get("ts", now) >= float(config.STREAM_GAP_MARK_SECONDS):
            from captioner.prompts import casual_time_string

            cue += P("mind.gap").format(gap=casual_time_string((now - self.thread[-1]["ts"]) / 60.0).capitalize())
        if self.pending_notice and kind == "think":
            subject, n_piv = self.pending_notice
            self.pending_notice = None
            cue += P("mind.pivot-notice").format(subject=subject, n=n_piv)

        prior = self.recent_turns(now)
        turns: List[dict] = []
        life = self.life_block(now, agent)
        if prior:
            first = prior[0]
            turns.append({"role": "user", "content": f"{life}\n\n{first.get('cue') or ''}".strip()})
            turns.append({"role": "assistant", "content": first["text"]})
            for e in prior[1:]:
                turns.append({"role": "user", "content": e.get("cue") or clock(e["ts"]) + "."})
                turns.append({"role": "assistant", "content": e["text"]})
            user = cue
        else:
            user = f"{life}\n\n{cue}".strip()
        return {"system": system, "turns": turns, "user": user, "image": img_path if kind == "look" else None, "cue": cue, "memory": memory, "life": life}
