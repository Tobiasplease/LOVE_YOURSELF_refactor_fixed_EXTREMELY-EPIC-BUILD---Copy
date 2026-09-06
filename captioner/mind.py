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
_KEEP_KINDS = ("look", "think", "memory", "wake", "past", "reflection", "record", "dream")


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
    def __init__(self, agent=None, path: Optional[str] = None, backfill: bool = True):
        self.agent = agent
        self.path = path or os.path.join(config.MOOD_SNAPSHOT_FOLDER, "mind_thread.json")
        self.thread: List[dict] = []
        self.positions: Dict[str, dict] = {}
        self.last_look_ts = 0.0
        self.think_count = 0
        self.pending_notice: Optional[Tuple[str, int]] = None
        self._last_believed: Optional[bool] = None
        self._last_felt: str = ""
        self._edges: Dict[str, list] = {}
        self._recalled: Dict[str, float] = {}
        self._index = None  # the "thoughts" ChromaDB collection (lazy); tests inject a fake
        self._load()
        if backfill:
            try:
                n_added = self.backfill()
                if n_added:
                    print(f"[MIND] backfilled {n_added} past thoughts from earlier days")
            except Exception:
                pass

    # ---- persistence -----------------------------------------------------
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.thread = list(d.get("thread") or [])[-int(config.MIND_THREAD_MAX) :]
            self.positions = dict(d.get("positions") or {})
            self.last_look_ts = float(d.get("last_look_ts") or 0.0)
            self._edges = dict(d.get("edges") or {})
            self._recalled = dict(d.get("recalled") or {})
            self.last_dream_ts = float(d.get("last_dream_ts") or 0.0)
            self._restore_read(d.get("mood_read") or {})
            try:
                from utils import mood as _mood

                _mood.load(d.get("mood") or {})
            except Exception:
                pass
        except Exception:
            self.thread, self.positions = [], {}

    def _save(self) -> None:
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "thread": self.thread[-int(config.MIND_THREAD_MAX) :],
                        "positions": self.positions,
                        "last_look_ts": self.last_look_ts,
                        "edges": self._edges,
                        "recalled": self._recalled,
                        "mood": self._mood_state(),
                        "mood_read": self._last_read(),
                        "last_dream_ts": float(getattr(self, "last_dream_ts", 0.0) or 0.0),
                    },
                    f,
                    indent=1,
                )
            os.replace(tmp, self.path)
        except Exception:
            pass

    # ---- recall by association (ChromaDB "thoughts") ------------------------------
    def index(self):
        """The "thoughts" collection. A failure never latches: retried after
        MIND_INDEX_RETRY_S (03:05 Sep 6: 304 of 404 entries were missing —
        one early error had switched indexing off for whole runs). On first
        success the index is reconciled against the thread."""
        if self._index is None and getattr(config, "MIND_RECALL_ENABLED", True) and time.time() >= getattr(self, "_index_retry_at", 0.0):
            try:
                from captioner.semantic_memory import get_semantic_memory

                self._index = get_semantic_memory()._client.get_or_create_collection(name="thoughts", metadata={"hnsw:space": "cosine"})
                self.reconcile_index()
            except Exception as e:  # noqa: BLE001
                self._index = None
                self._index_retry_at = time.time() + float(getattr(config, "MIND_INDEX_RETRY_S", 60))
                print(f"[MIND] thoughts index unavailable ({e}); retrying in a minute")
        return self._index or None

    def reconcile_index(self) -> int:
        """Add every eligible thread entry the index doesn't hold yet."""
        idx = self._index
        if not idx:
            return 0
        try:
            have = set(idx.get(include=[])["ids"]) if idx.count() else set()
        except Exception:
            have = set()
        missing = [e for e in self.thread if e.get("text") and e.get("kind") in _KEEP_KINDS and self._tid(e) not in have and self._tid_old(e) not in have]
        for i in range(0, len(missing), 200):
            self._index_add(missing[i : i + 200])
        if missing:
            print(f"[MIND] thoughts index reconciled: +{len(missing)} (now {idx.count()})")
        return len(missing)

    @staticmethod
    def _tid(entry: dict) -> str:
        """Index id: second + kind initial (a look and a think can share a second)."""
        return f"t{int(float(entry.get('ts', 0)))}{(entry.get('kind') or 'x')[0]}"

    @staticmethod
    def _tid_old(entry: dict) -> str:
        return f"t{int(float(entry.get('ts', 0)))}"  # ids written before 03:10 Sep 6 — still count as present

    def _index_add(self, entries: List[dict]) -> None:
        idx = self.index()
        if not idx:
            return
        entries = [e for e in entries if e.get("text") and e.get("kind") in _KEEP_KINDS]
        seen: set = set()
        entries = [e for e in entries if not (self._tid(e) in seen or seen.add(self._tid(e)))]  # one id per batch
        if not entries:
            return
        try:
            idx.upsert(
                ids=[self._tid(e) for e in entries],
                documents=[e["text"][:600] for e in entries],
                metadatas=[{"ts": float(e.get("ts", 0)), "kind": e.get("kind", "")} for e in entries],
            )
        except Exception as e:  # noqa: BLE001
            print(f"[MIND] thoughts index write failed ({e})")

    def reindex(self) -> int:
        entries = [e for e in self.thread if e.get("text") and e.get("kind") in _KEEP_KINDS]
        for i in range(0, len(entries), 200):
            self._index_add(entries[i : i + 200])
        return len(entries)

    def recall_similar(self, query: str, now: float, believed: bool = False) -> Optional[dict]:
        """The past thought nearest to the current one — surfaces by
        association, never by schedule. Old enough, close enough, not in the
        turns, not recalled within the cooldown, never person-tinged while the
        room is believed empty."""
        idx = self.index()
        if not idx or not (query or "").strip():
            return None
        if now - float(getattr(self, "_last_recall_ts", 0.0) or 0.0) < int(getattr(config, "MIND_RECALL_MIN_GAP_S", 0)):
            return None
        try:
            n = min(8, idx.count())
            if n <= 0:
                return None
            res = idx.query(query_texts=[query[:400]], n_results=n, include=["documents", "metadatas", "distances"])
        except Exception:
            return None
        turn_texts = {e.get("text") for e in self.recent_turns(now)}
        min_age = int(config.MIND_MEMORY_MIN_AGE_S)
        cool = int(config.MIND_RECALL_COOLDOWN_S)
        maxd = float(config.MIND_RECALL_MAX_DIST)
        try:
            from utils.presence_text import PERSON_RE
        except Exception:
            PERSON_RE = None  # noqa: N806
        for rid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            ts = float((meta or {}).get("ts", 0))
            if dist > maxd or now - ts < min_age or doc in turn_texts:
                continue
            key = " ".join(_WORD_RE.findall(doc.lower()))[:160]  # the same sentence under many ids cools as one
            if now - float(self._recalled.get(key, 0)) < cool:
                continue
            if _NEG_FRAME_RE.search(doc) or (not believed and PERSON_RE and PERSON_RE.search(doc)):
                continue
            self._recalled[key] = now
            self._last_recall_ts = now
            if len(self._recalled) > 500:
                for k in sorted(self._recalled, key=self._recalled.get)[:-500]:
                    self._recalled.pop(k, None)
            self._save()  # a gated thought absorbs nothing — the cooldown must survive on its own
            return {"ts": ts, "text": doc, "kind": (meta or {}).get("kind", ""), "distance": float(dist)}
        return None

    def before_chain(self, now: float) -> Optional[dict]:
        """The last thought of the previous chain — quoted in the life block as continuity."""
        start = self.thread_start(now)
        if not start:
            return None
        for e in reversed(self.thread):
            ts = float(e.get("ts", 0))
            if ts < start and e.get("kind") in ("think", "look", "reflection", "wake", "dream") and e.get("text"):
                if now - ts <= int(config.MIND_LIFE_BEFORE_MAX_AGE_S):
                    return e
                return None
        return None

    # ---- its own past, days deep -----------------------------------------------
    def backfill(self, log_dir: Optional[str] = None, days: int = 4, per_day: int = 40, now: Optional[float] = None) -> int:
        """Seed the thread with kept thoughts from earlier days' event logs so
        memories can surface from nights ago, not only from tonight (Sep 6:
        'the accumulated info must reach back to the prompting'). Runs once —
        skipped when the thread already reaches back more than a day. The
        same filters as memory choice apply: no phantom presence, no reframe."""
        import glob

        now = now or time.time()
        if any(now - e.get("ts", 0) > 86400 for e in self.thread):
            return 0
        log_dir = log_dir or config.MOOD_SNAPSHOT_FOLDER
        try:
            from utils.presence_text import is_phantom_presence
        except Exception:
            is_phantom_presence = lambda t: False  # noqa: E731
        by_day: Dict[str, list] = {}
        for path in glob.glob(os.path.join(log_dir, "*-event-log.json")):
            try:
                if now - os.path.getmtime(path) > days * 86400:
                    continue
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        if '"type": "caption"' not in line or '"caption":' not in line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        t = (r.get("caption") or "").strip()
                        ts = float(r.get("timestamp", 0))
                        if len(t) < 40 or now - ts < 3600 or r.get("mode") in ("awakening", "error") or r.get("duplicate"):
                            continue
                        if _NEG_FRAME_RE.search(t) or is_phantom_presence(t) or re.match(r"\s*\d\d?:\d\d", t):
                            continue
                        by_day.setdefault(time.strftime("%Y-%m-%d", time.localtime(ts)), []).append((ts, t))
            except Exception:
                continue
        added = 0
        for day, items in by_day.items():
            items.sort()
            step = max(1, len(items) // per_day)
            for ts, t in items[::step][:per_day]:
                self.thread.append({"ts": ts, "kind": "past", "cue": "", "text": t[:400], "subject": ""})
                added += 1
        if added:
            self.thread.sort(key=lambda e: e.get("ts", 0))
            self.thread = self.thread[-int(config.MIND_THREAD_MAX) :]
            self._save()
            self._index_add([e for e in self.thread if e.get("kind") == "past"])
        return added

    # ---- the thread --------------------------------------------------------
    def has_session(self, session_start: float) -> bool:
        return any(e.get("ts", 0) >= session_start for e in self.thread)

    def absorb(self, text: str, kind: str, cue: str, now: Optional[float] = None, uneventful: bool = False) -> dict:
        now = now or time.time()
        text = (text or "").strip()
        subject = self.subject_of(text)
        entry = {"ts": now, "kind": kind, "cue": (cue or "").strip(), "text": text, "subject": subject}
        if kind == "look" and uneventful:
            entry["uneventful"] = True
        self._gate_streak = 0
        self._spoken_tail = None
        self.thread.append(entry)
        if len(self.thread) > int(config.MIND_THREAD_MAX):
            self.thread = self.thread[-int(config.MIND_THREAD_MAX) :]
        if kind == "look":
            self.last_look_ts = now
        self._update_position(subject, text, now)
        self._save()
        self._index_add([entry])
        return entry

    def note_spoken(self, text: str, now: float) -> None:
        """A thought that was spoken but not kept still moves the premise —
        otherwise the next turn gets the identical context and says the
        identical thing (14:37–14:40 Sep 6: four "A black stick in the dark"
        openings in a row, each gated, the premise never moving)."""
        self._spoken_tail = (last_sentence(text)[:200], now)
        self._gate_streak = int(getattr(self, "_gate_streak", 0) or 0) + 1

    def strip_restated_premise(self, text: str, premise: str) -> str:
        """The model often opens by restating the quoted premise; the
        continuation is the entry, the restatement is not."""
        t = (text or "").strip()
        pz = (premise or "").strip().rstrip(".!?…").strip()
        if not pz or len(pz.split()) < 3:
            return t
        head = t[: len(pz) + 3].lower()
        if head.startswith(pz.lower()):
            rest = t[len(pz):].lstrip(" .!?…,;:—–-")
            if len(rest.split()) >= 3:
                return rest[0].upper() + rest[1:] if rest[0].islower() else rest
        return t

    def premise(self, now: Optional[float] = None) -> str:
        """The machine's own last sentence, quoted back as the thing to go on
        from (the continuation mechanic — see mind.cue-premise)."""
        now = now or time.time()
        if int(getattr(self, "_gate_streak", 0) or 0) >= 2:
            self._gate_streak = 0
            return ""  # two refusals in a row: one turn with no premise, so the context changes
        spoken = getattr(self, "_spoken_tail", None)
        if spoken and (not self.thread or spoken[1] > float(self.thread[-1].get("ts", 0))) and now - spoken[1] < 600:
            return spoken[0]
        if not self.thread:
            return ""
        last = self.thread[-1]
        if now - last.get("ts", 0) > int(config.MIND_TURN_MAX_AGE_S):
            return ""
        for e in self.thread[-3:]:
            # a reflection that just settled is the thing to go on from, even if a thought landed after it (Sep 6 03:00)
            if e.get("kind") == "reflection" and now - e.get("ts", 0) <= 240 and not e.get("premised"):
                e["premised"] = True
                self._save()
                return last_sentence(e.get("text", ""))[:200]
        if last.get("kind") == "look" and last.get("uneventful"):
            # an uneventful glance is not a new thought — the chain continues from the last one (Sep 6 01:00)
            for e in reversed(self.thread[:-1]):
                if now - e.get("ts", 0) > int(config.MIND_TURN_MAX_AGE_S):
                    break
                if e.get("kind") in ("think", "reflection", "wake", "memory"):
                    return last_sentence(e.get("text", ""))[:200]
        prem = last_sentence(last.get("text", ""))[:200]
        if len(prem.split()) <= 3:
            # a one-word premise + "go on" reads as "define it" (12:36 Sep 6: "Scattering." → "Scattering is a loss of coherence.");
            # a beat carries the two sentences before it, so there is a thought to continue
            parts = [p for p in _SENT_END_RE.split((last.get("text") or "").strip()) if p.strip()]
            tail = " ".join(parts[-3:]) if len(parts) > 1 else ""
            if len(tail.split()) <= 4:
                for e in reversed(self.thread[:-1]):
                    if now - e.get("ts", 0) > int(config.MIND_TURN_MAX_AGE_S):
                        break
                    if e.get("kind") in ("think", "look", "reflection", "wake", "memory") and e.get("text"):
                        prev = [p for p in _SENT_END_RE.split(e["text"].strip()) if p.strip()]
                        tail = (" ".join(prev[-2:]) + " " + prem).strip()
                        break
            if tail:
                prem = tail[-240:]
        return prem

    def recent_turns(self, now: Optional[float] = None) -> List[dict]:
        now = now or time.time()
        fresh = [e for e in self.thread if now - e.get("ts", 0) <= int(config.MIND_TURN_MAX_AGE_S) and e.get("text") and e.get("kind") not in ("past", "record")]
        n = int(config.MIND_TEXT_ENTRIES) if getattr(config, "MIND_SHAPE", "text") == "text" else int(config.MIND_TURNS)
        return fresh[-n:]

    @staticmethod
    def running_text(entries: List[dict]) -> str:
        """The thread as journal text: paragraphs, no stamps, no cues. A new
        paragraph at a gap of three minutes or more, at a look, at a reflection
        (the same rule debug/journal.py prints with). Sep 6 morning, artist:
        'put together it should form pages of what looks like an actual journal'."""
        paras: List[List[str]] = []
        last_ts = None
        for e in entries:
            t = (e.get("text") or "").strip()
            if not t or e.get("kind") in ("record", "past"):
                continue
            ts = float(e.get("ts", 0))
            if not paras or (last_ts is not None and (ts - last_ts >= 180 or e.get("kind") in ("reflection", "wake", "dream"))):
                # a look does NOT break the paragraph (Sep 6 12:20): entries follow each other; only a gap or a settling does
                paras.append([])
            paras[-1].append(t)
            last_ts = ts
        return "\n\n".join(" ".join(p) for p in paras)

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
        edge = (self._last_believed is None and believed) or (self._last_believed is not None and believed != self._last_believed)  # a belief restored at boot is an edge too (15:56 Sep 6)
        self._last_believed = believed
        since_look = now - self.last_look_ts
        if not self.last_look_ts:
            return "look"
        if hot or edge or (scene or {}).get("view_changed"):
            return "look" if since_look >= 3 else "think"
        if believed and since_look >= float(getattr(config, "MIND_LOOK_EVERY_BELIEVED_S", 60)):
            return "look"
        if since_look < float(config.MIND_LOOK_MIN_GAP_S):
            return "think"
        try:
            from utils import chosen_glance

            if chosen_glance.current() or chosen_glance.pending():
                return "look"
        except Exception:
            pass
        look_mult = 1.0
        try:
            from utils import felt_loop as _fl

            mc = _fl._mood_cadence()
            look_mult = float(mc["look_mult"]) if mc else 1.0
        except Exception:
            pass
        if since_look >= float(config.MIND_LOOK_EVERY_S) / max(0.25, look_mult):
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
        if getattr(agent, "_presence_believed", False):
            base *= float(getattr(config, "MIND_INTERVAL_BELIEVED_MULT", 0.6))
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
        woke = self.woke_words(now, agent)
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
        name = self._name()
        if name:
            lines.append(P("mind.life-name").format(name=name))
        events = self.events_today(now, agent)
        if events:
            lines.append(P("mind.life-events").format(events="; ".join(events)))
        before = self.before_chain(now)
        if before:
            lines.append(P("mind.life-before").format(when=when_words(now - float(before["ts"])), text=before["text"][:220]))
        if getattr(config, "MIND_LIFE_FULL", False):
            try:
                from utils.lore_ledger import lore_ledger

                qs = [(q.get("text") or q.get("words") or "").strip() for q in lore_ledger.open_questions(2)]
                qs = [q for q in qs if q]
                if qs:
                    lines.append(P("mind.life-questions").format(questions=" ".join(qs)))
            except Exception:
                pass
            belief = self._belief()
            if belief and len(belief) > 8:
                lines.append(P("mind.life-belief").format(belief=belief.rstrip(".")))
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

    def events_today(self, now: float, agent, max_events: int = 3) -> List[str]:
        """What has happened since the machine woke, as events with clocks, in
        words: arrivals and departures, changes the referee saw, drawings, the
        night's page. Memory material — the machine can say 'this morning'."""
        out: List[tuple] = []
        start = self.woke_at(now, agent)
        try:
            from utils.episodic_log import episodic_log

            pairs = episodic_log.get_pairs_in_window("person_arrived", "person_left", window_seconds=int(now - start) + 1)
            paired_ends = set()
            for p in pairs:
                st = p.get("start") or {}
                en = p.get("end") or {}
                if en:
                    paired_ends.add(float(en.get("timestamp", 0)))
                if float(st.get("timestamp", 0)) < start:
                    continue
                if en:
                    out.append((float(en["timestamp"]), f"someone came in at {clock(float(st['timestamp']))} and left after {dur_words(float(p.get('duration_seconds', 0)))}"))
                else:
                    out.append((float(st["timestamp"]), f"someone came in at {clock(float(st['timestamp']))}"))
            for e in episodic_log.get_recent_events(window_seconds=int(now - start) + 1, types=["person_left"]):
                ts = float(e.get("timestamp", 0))
                if ts >= start and ts not in paired_ends:
                    out.append((ts, f"someone left at {clock(ts)}"))  # a departure with no logged arrival (they were here when you woke)
            for e in episodic_log.get_recent_events(window_seconds=int(now - start) + 1, types=["world_changed", "drew"]):
                ts = float(e.get("timestamp", 0))
                if ts < start:
                    continue
                if e.get("type") == "drew":
                    out.append((ts, f"you drew at {clock(ts)}"))
                else:
                    out.append((ts, f"the view changed at {clock(ts)}"))
        except Exception:
            pass
        for e in self.thread:
            if e.get("kind") == "dream" and now - float(e.get("ts", 0)) < 36 * 3600:
                out.append((float(e["ts"]), f"you wrote a page {when_words(now - float(e['ts']))}"))
        out.sort()
        return [t for _, t in out[-max_events:]]

    def conclusions_today(self, now: float, max_items: int = 6) -> List[str]:
        """The day's reflections' last sentences with their clocks, and last
        night's page ending — the spine of the reflection (Sep 6)."""
        lt = time.localtime(now)
        midnight = now - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec)
        items = []
        for e in self.thread:
            ts = float(e.get("ts", 0))
            if e.get("kind") == "reflection" and ts >= midnight and e.get("text"):
                items.append(f"{clock(ts)} — {last_sentence(e['text'])[:200]}")
            elif e.get("kind") == "dream" and now - ts < 36 * 3600 and e.get("text"):
                items.append(f"{when_words(now - ts)}, the night's page ended — {last_sentence(e['text'])[:200]}")
        return items[-max_items:]

    def person_since(self, now: float) -> str:
        try:
            from utils.episodic_log import episodic_log

            ev = episodic_log.get_last_event("person_arrived")
            return clock(float(ev["timestamp"])) if ev else "a moment ago"
        except Exception:
            return "a moment ago"

    def person_history(self, now: float) -> str:
        """Visits over the last days and what the machine has come to know of people — its own words."""
        parts = []
        try:
            from utils.episodic_log import episodic_log

            pairs = [p for p in episodic_log.get_pairs_in_window("person_arrived", "person_left", window_seconds=72 * 3600) if p.get("end")]
            if pairs:
                typical = dur_words(sorted(float(p.get("duration_seconds", 0)) for p in pairs)[len(pairs) // 2])
                times = {1: "once", 2: "twice", 3: "three times"}.get(len(pairs), "several times")
                parts.append(f"They've been by {times} in the last few days, usually for {typical}.")
        except Exception:
            pass
        try:
            from captioner.context_compression import context_compressor

            ppl = (context_compressor.core_facts.get("people") or "").strip()
            if len(ppl) > 8:
                parts.append(f'What you know of them: "{ppl[:220]}"')
        except Exception:
            pass
        return (" " + " ".join(parts)) if parts else ""

    def _people_line(self, now: float, agent) -> str:
        if getattr(agent, "_presence_believed", False):
            return P("mind.life-people-now").format(since=self.person_since(now), history=self.person_history(now))
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

    # ---- the body: direction words in the gaze code's own convention -----------------
    @staticmethod
    def _dir_words(pan: float, tilt: float, pan_center: float = 90.0, tilt_center: float = 107.5, thr: float = 8.0) -> str:
        h = "left" if pan - pan_center < -thr else ("right" if pan - pan_center > thr else "")
        v = "down" if tilt - tilt_center < -thr else ("up" if tilt - tilt_center > thr else "")
        return f"{v}-{h}" if v and h else (v or h or "ahead")

    @staticmethod
    def placement_words(pan: float, tilt: float, thr: float = 8.0) -> str:
        """Where a thing sits relative to the body: 'high to your right', 'low ahead', 'to your left'."""
        h = "to your left" if pan - 90.0 < -thr else ("to your right" if pan - 90.0 > thr else "ahead")
        v = "low" if tilt - 107.5 < -thr else ("high" if tilt - 107.5 > thr else "")
        return f"{v} {h}".strip()

    def turn_report(self, pose: Optional[tuple]) -> str:
        """Since the last look: turned which way, or not at all. A sense report, one clause."""
        if not pose:
            return ""
        last = getattr(self, "_last_look_pose", None)
        self._last_look_pose = pose
        if not last:
            return ""
        dp, dt = float(pose[0]) - float(last[0]), float(pose[1]) - float(last[1])
        thr = float(getattr(config, "MIND_TURN_MIN_DEG", 8))
        if abs(dp) < thr and abs(dt) < thr:
            return P("mind.head-still")
        h = "to your left" if dp < -thr else ("to your right" if dp > thr else "")
        v = "up" if dt > thr else ("down" if dt < -thr else "")
        direction = " and ".join(x for x in (h, v) if x)
        return P("mind.turned").format(direction=direction)

    def in_view_placed(self, agent) -> List[tuple]:
        """(term, placement words) for the things in view, most familiar first."""
        try:
            from perception.spatial_registry import spatial_registry

            entries = spatial_registry.get_entries() or {}
            return [(t, self.placement_words(float(entries[t].get("pan", 90)), float(entries[t].get("tilt", 107.5)))) for t in self.in_view(agent) if t in entries]
        except Exception:
            return []

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

    @staticmethod
    def _grams(text: str, n: int = 6) -> set:
        w = _WORD_RE.findall((text or "").lower())
        return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}

    def is_recall(self, text: str, call: Optional[dict] = None, now: Optional[float] = None) -> bool:
        """A thought that reproduces a line it was shown (a quoted past thought,
        a surfaced memory, a position) or any older thread entry, six words in a
        row — recall dressed as a new thought (Sep 6 00:31/00:35: two verbatim
        copies of 22:01 and 22:57 lines quoted in the life block). Turns in the
        current conversation are exempt: continuing them is the point."""
        g = self._grams(text)
        if not g:
            return False
        g8 = self._grams(text, 8)
        now = now or time.time()
        turn_texts = {e.get("text") for e in self.recent_turns(now)}
        sources = []
        if call:
            m = call.get("memory")
            if m:
                sources.append(m.get("text", ""))
            sources.append(call.get("life", ""))
        for e in self.thread:
            if e.get("text") not in turn_texts:
                sources.append(e.get("text", ""))
        for src in sources:
            if not src:
                continue
            shared = len(g & self._grams(src))
            ratio = shared / len(g)
            if shared and (ratio >= 0.5 or (ratio >= 0.25 and g8 & self._grams(src, 8))):
                return True  # half the thought, or a quarter plus eight words in a row — a copy, not a phrase the room keeps producing
        return False

    # ---- the mood with dynamics ---------------------------------------------------
    @staticmethod
    def _mood_state() -> dict:
        try:
            from utils import mood as _mood

            return _mood.state()
        except Exception:
            return {}

    @staticmethod
    def _last_read() -> dict:
        try:
            from captioner.context_compression import context_compressor as _cc

            return dict(getattr(_cc, "last_mood_read", None) or {})
        except Exception:
            return {}

    @staticmethod
    def _restore_read(read: dict) -> None:
        """A restart must not empty the frame: the last read (felt, tone) survives if fresh."""
        try:
            if not read or time.time() - float(read.get("timestamp", 0)) > 900:
                return
            from captioner.context_compression import context_compressor as _cc

            if not getattr(_cc, "last_mood_read", None):
                restored = dict(read)
                restored["tone"] = ""  # never restore a tone across a restart: the standing line is a directive and a stale one is worse
                _cc.last_mood_read = restored
                if read.get("felt"):
                    _cc.set_felt_state(read["felt"])
        except Exception:
            pass

    def note_scare(self, now: float) -> None:
        self._scare_ts = now

    def situation(self, now: float, agent) -> dict:
        """The situation as numbers: what the mood is pulled by."""
        alone_h = 0.0
        if not getattr(agent, "_presence_believed", False):
            left = 0.0
            try:
                from utils.episodic_log import episodic_log

                ev = episodic_log.get_last_event("person_left")
                left = float(ev.get("timestamp", 0)) if ev else 0.0
            except Exception:
                pass
            left = max(left, float(getattr(agent, "_presence_dropped_at", 0.0) or 0.0))
            alone_h = (now - left) / 3600.0 if left else 0.0
        start = self.thread_start(now)
        still = float(getattr(agent, "_world_change_ts", 0.0) or 0.0)
        hits = [h for h in (getattr(agent, "_loop_hits", None) or []) if now - float(h[0]) < 600]
        settled = any(e.get("kind") == "reflection" and now - e.get("ts", 0) < 600 for e in self.thread[-8:])
        scare = (now - float(getattr(self, "_scare_ts", 0.0) or 0.0) < 300) or ("moved" in str(getattr(agent, "_salience_event", "") or ""))
        hour = time.localtime(now).tm_hour
        session = float(getattr(agent, "true_session_start", 0.0) or 0.0)
        awake_since = max(start, session) if (start and session) else (start or session)  # the chain bridges an hour off; fatigue counts the shorter
        return {
            "awake_h": (now - awake_since) / 3600.0 if awake_since else 0.0,
            "alone_h": alone_h,
            "still_h": (now - still) / 3600.0 if still else 0.0,
            "night": hour < 6,
            "refusals": len(hits) + (1 if int(getattr(agent, "_skip_streak", 0) or 0) >= 2 else 0),
            "settled": settled,
            "scare": scare,
            "presence": bool(getattr(agent, "_presence_believed", False)),
        }

    def situation_words(self, inputs: dict) -> str:
        """The same situation in words for the compressor's FELT ask (structure: durations and facts only)."""
        from captioner.prompts import casual_time_string

        parts = []
        if inputs.get("awake_h", 0) >= 0.5:
            parts.append(f"awake {casual_time_string(inputs['awake_h'] * 60)}")
        if inputs.get("alone_h", 0) >= 0.5:
            parts.append(f"no one here for {casual_time_string(inputs['alone_h'] * 60)}")
        if inputs.get("presence"):
            parts.append("someone in the room")
        if inputs.get("night"):
            parts.append("the middle of the night")
        if inputs.get("still_h", 0) >= 1:
            parts.append(f"nothing changed for {casual_time_string(inputs['still_h'] * 60)}")
        if inputs.get("refusals", 0) >= 2:
            parts.append("your last thoughts kept circling")
        if inputs.get("scare"):
            parts.append("something startled you just now")
        return ", ".join(parts)

    def tick_mood(self, now: float, agent) -> dict:
        try:
            from utils import mood as _mood

            if not getattr(config, "MOOD_ENABLED", True):
                return {}
            read = None
            try:
                from captioner.context_compression import context_compressor as _cc

                read = _cc.get_last_mood_read()
                inputs = self.situation(now, agent)
                _cc.situation_line = self.situation_words(inputs)
            except Exception:
                inputs = self.situation(now, agent)
            st = _mood.tick(now, read, inputs)
            self._save()
            return st
        except Exception:
            return {}

    @staticmethod
    def beat_of(raw: str) -> Optional[str]:
        """Rhythm (Sep 6, artist: 'some captions should be a single word or even just …'):
        an all-punctuation reply, or a short boundary-less fragment, is kept as a
        beat in the text instead of being dropped as silence. Returns the beat
        text, or None when the reply is a normal thought."""
        t = (raw or "").strip()
        if not t:
            return "…"
        if all(c in ".…·-— " for c in t):
            return "…"
        has_end = any(c in t for c in ".!?…")
        words = t.split()
        if not has_end and len(words) <= int(getattr(config, "MIND_BEAT_MAX_WORDS", 6)):
            return t
        if not has_end:
            return "…"
        return None

    def note_look(self, now: float) -> None:
        """A look happened, stored or not — the look timer advances either way
        (Sep 5 23:25–23:39: gated looks left the timer stale, so every phantom
        was followed by a second look a minute later)."""
        self.last_look_ts = now
        self._save()

    def woke_at(self, now: float, agent) -> float:
        """When the machine woke: the first thought after the last gap of
        MIND_WAKE_GAP_S or more in the thread (it was off), not the last
        process restart and not midnight (15:26 Sep 6: 'you woke at 15:26'
        after a restart; then 'at 00:00' when the chain crossed midnight)."""
        gap = int(getattr(config, "MIND_WAKE_GAP_S", 2700))
        entries = [e for e in self.thread if e.get("kind") in ("wake", "look", "think", "reflection", "dream") and float(e.get("ts", 0)) <= now]
        woke = 0.0
        last_ts = None
        for e in entries:
            ts = float(e.get("ts", 0))
            if last_ts is None or ts - last_ts >= gap:
                woke = ts
            last_ts = ts
        session = float(getattr(agent, "true_session_start", 0.0) or 0.0)
        return woke or session or now

    def woke_words(self, now: float, agent) -> str:
        """'today at 11:22' or 'yesterday at 21:41' or 'a few days ago'."""
        w = self.woke_at(now, agent)
        lt = time.localtime(now)
        midnight = now - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec)
        if w >= midnight:
            return f"today at {clock(w)}"
        if w >= midnight - 86400:
            return f"yesterday at {clock(w)}"
        return when_words(now - w)

    def thread_start(self, now: float) -> float:
        """Start of the continuous chain of thought (gaps under MIND_TURN_MAX_AGE_S)."""
        start = 0.0
        last = now
        for e in reversed(self.thread):
            ts = float(e.get("ts", 0))
            if e.get("kind") == "past" or last - ts > int(config.MIND_TURN_MAX_AGE_S):
                break
            start, last = ts, ts
        return start

    def time_edges(self, now: float, agent) -> str:
        """Time as an event: one line when a duration threshold is crossed —
        since anyone was here, since the room last changed, since the chain of
        thought began. Each threshold fires once per anchor."""
        from captioner.prompts import casual_time_string

        thresholds = sorted(int(x) for x in getattr(config, "DURATION_EDGE_THRESHOLDS_MIN", [30, 60, 120, 240, 480]))
        anchors = []
        if not getattr(agent, "_presence_believed", False):
            left = 0.0
            try:
                from utils.episodic_log import episodic_log

                ev = episodic_log.get_last_event("person_left")
                left = float(ev.get("timestamp", 0)) if ev else 0.0
            except Exception:
                pass
            left = max(left, float(getattr(agent, "_presence_dropped_at", 0.0) or 0.0))
            if left:
                anchors.append(("alone", left, "mind.edge-alone"))
        still = float(getattr(agent, "_world_change_ts", 0.0) or 0.0)
        if still:
            anchors.append(("still", still, "mind.edge-still"))
        start = self.thread_start(now)
        if start:
            anchors.append(("awake", start, "mind.edge-awake"))
        for name, anchor, frag in anchors:
            mins = (now - anchor) / 60.0
            crossed = [t for t in thresholds if mins >= t]
            if not crossed:
                continue
            top = crossed[-1]
            prev = self._edges.get(name) or [0.0, 0]
            if abs(float(prev[0]) - anchor) > 60 or int(prev[1]) < top:
                self._edges[name] = [anchor, top]
                self._save()
                return P(frag).format(duration=casual_time_string(mins).capitalize() if frag == "mind.edge-alone" else casual_time_string(mins))
        return ""

    def _loop_line(self, agent) -> str:
        """Its own repetition, named — the loop notice (gate hits / the
        compressor's REPEATING slot) reaches the cue in mind mode too."""
        try:
            from captioner.prompts import build_loop_notice_line

            line = (build_loop_notice_line(agent) or "").strip()
            return (" " + line) if line else ""
        except Exception:
            return ""

    def _name(self) -> str:
        try:
            from utils.lore_ledger import lore_ledger

            return (lore_ledger.current_name() or "").strip()
        except Exception:
            return ""

    def _belief(self) -> str:
        try:
            from captioner.context_compression import context_compressor

            return (context_compressor.get_current_belief() or "").strip()
        except Exception:
            return ""

    def _mirror_locked(self, channel: str, phrase: str) -> bool:
        """A standing mirror of the machine's own output is a directive. When
        one word runs through MIND_TONE_LOCK_READS consecutive reads of a
        channel (tone, felt), the line leaves the frame for
        MIND_TONE_SUPPRESS_S and the cue says it back once as a noticing."""
        now = time.time()
        st = getattr(self, "_mirror", None)
        if st is None:
            st = self._mirror = {}
        c = st.setdefault(channel, {"hist": [], "until": 0.0, "pending": "", "hist_max": 3})
        if now < float(c["until"]):
            return True
        if not c["hist"] or c["hist"][-1] != phrase:
            c["hist"] = (c["hist"] + [phrase])[-3:]
        n = max(2, int(getattr(config, "MIND_TONE_LOCK_READS", 2)))
        words = [set(w for w in _WORD_RE.findall(t.lower()) if len(w) > 3) for t in c["hist"][-n:]]
        locked = len(words) >= n and bool(set.intersection(*words))
        if locked and not c["pending"]:
            c["pending"] = phrase
            c["until"] = now + int(getattr(config, "MIND_TONE_SUPPRESS_S", 900))
            c["hist"] = []
        return locked

    def _mirror_notice(self, channel: str, frag: str, key: str) -> str:
        c = (getattr(self, "_mirror", None) or {}).get(channel)
        if c and c.get("pending"):
            phrase, c["pending"] = c["pending"], ""
            return P(frag).format(**{key: phrase})
        return ""

    def _tone_locked(self, tone: str) -> bool:
        return self._mirror_locked("tone", tone)

    def _tone_notice(self) -> str:
        return self._mirror_notice("tone", "mind.tone-held", "tone") + self._mirror_notice("felt", "mind.felt-lock", "felt")

    def _felt_shift(self) -> str:
        """The felt loop as an event: rides once when the compressor's felt word changes."""
        try:
            from captioner.context_compression import context_compressor as _cc

            curr = (_cc.get_felt_state() or "").strip()
        except Exception:
            curr = ""
        prev, self._last_felt = self._last_felt, curr
        if curr and prev and curr != prev:
            return P("mind.felt-shift").format(prev=prev, curr=curr)
        return ""

    def _elicit_dose(self) -> str:
        """Every MIND_ELICIT_EVERY_N-th think turn, one elicit line — the kind
        leaning by the felt loop's valence (feel when low, want when high),
        otherwise rotating wonder / feel / want. The artist's existing lines."""
        n = int(getattr(config, "MIND_ELICIT_EVERY_N", 0) or 0)
        if n <= 0 or self.think_count % n != 0:
            return ""
        lean = None
        try:
            from utils import felt_loop

            lean = felt_loop.elicit_lean()
        except Exception:
            pass
        if lean == "feel":
            return P("elicit.quiet-feel", default="")
        if lean == "want":
            return P("elicit.quiet-want", default="")
        kinds = ("elicit.quiet-wonder", "elicit.quiet-feel", "elicit.quiet-want")
        return P(kinds[(self.think_count // n) % 3], default="")

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
        self.tick_mood(now, agent)
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
                if felt and not self._mirror_locked("felt", felt):
                    system += P("monologue.felt-frame").format(felt=felt)
                    try:
                        from utils import mood as _mood

                        held = _mood.felt_held_s(now)
                        if getattr(config, "MOOD_ENABLED", True) and held >= float(config.MOOD_FELT_HELD_MIN_S):
                            from captioner.prompts import casual_time_string

                            system += P("monologue.felt-held").format(duration=casual_time_string(held / 60.0))
                    except Exception:
                        pass
                try:
                    tone = (_cc.get_tone() or "").strip()
                    if tone and self._tone_locked(tone):
                        pass  # the read still feeds the noticing (mind.tone-held) even with the standing line off
                    elif tone and getattr(config, "MIND_TONE_FRAME", False):
                        system += P("monologue.tone-frame").format(tone=tone)
                except Exception:
                    pass
        except Exception:
            pass

        believed = bool(getattr(agent, "_presence_believed", False))
        someone = ""
        lead = ""
        if believed:
            seen = ""
            try:
                from perception.presence_adjudicator import presence_adjudicator as _pa

                d = (getattr(_pa, "last_person_desc", "") or "").strip().rstrip(".")
                if d and now - float(getattr(_pa, "last_person_ts", 0.0) or 0.0) < 90:
                    seen = " — " + (d[0].lower() + d[1:])  # only on the arrival turn: a snapshot of what someone is doing is stale a minute later (artist, 16:20)
            except Exception:
                pass
            lead = P("mind.someone-here").format(since=self.person_since(now), seen=seen)  # the person leads the cue
        memory = None
        uneventful = False
        if kind == "look":
            event = getattr(agent, "_salience_event", None)
            uneventful = not event and not believed and getattr(agent, "_last_view_verdict", None) == "unchanged"
            if event:
                cue = P("mind.cue-look-event").format(clock=clock(now), event=str(event).strip(), someone=someone)
            else:
                placed = self.in_view_placed(agent)[: int(getattr(config, "MIND_VIEW_NAMED", 2))]
                terms = [t for t, _ in placed] if placed else self.in_view(agent)
                if placed:
                    groups: Dict[str, List[str]] = {}
                    for t, w in placed:
                        groups.setdefault(w, []).append(t)
                    parts = []
                    for w, ts in groups.items():
                        names = ", the ".join(ts[:-1]) + " and the " + ts[-1] if len(ts) > 1 else ts[0]
                        parts.append(f"{names} {w}")
                    where = P("mind.where").format(terms="; the ".join(parts))  # "You look at the lamp and the bag high to your right; the shelf to your right."
                else:
                    where = P("mind.where").format(terms=", the ".join(terms[:-1]) + " and the " + terms[-1] if len(terms) > 1 else terms[0]) if terms else ""
                verdict = getattr(agent, "_last_view_verdict", None)
                change = {"unchanged": P("mind.change-none"), "changed": P("mind.change-yes")}.get(verdict or "", "")
                if verdict in ("new", "baselined") and not self._seen_this_session(terms, agent):
                    change = P("mind.change-new")  # honest only when nothing in view was seen this session (the referee keys by gaze cell)
                cue = P("mind.cue-look").format(clock=clock(now), where=where, change=change, someone=someone)
            try:
                from vision.gaze import get_gaze_state

                g = get_gaze_state()
                pose = (float(g.get("pan", 90)), float(g.get("tilt", 107.5)))
            except Exception:
                pose = None
            cue += self.turn_report(pose)
        else:
            self.think_count += 1
            n = int(config.MIND_MEMORY_EVERY_N)
            if n > 0 and self.think_count % n == 0:
                memory = self.choose_memory(now, believed=believed)  # scheduled fallback (off by default)
            if memory:
                cue = P("mind.cue-think-memory").format(clock=clock(now), when=when_words(now - memory["ts"]), memory=memory["text"][:220])
            else:
                cue = P("mind.cue-think").format(clock=clock(now))
                prem = self.premise(now)
                if believed:
                    try:
                        from utils.episodic_log import episodic_log as _el

                        _arr = _el.get_last_event("person_arrived")
                        if _arr and now - float(_arr.get("timestamp", 0)) < int(getattr(config, "MIND_PERSON_FRESH_S", 600)):
                            prem = ""  # a person after days alone outranks the thread (Sep 6 16:15)
                    except Exception:
                        pass
                self._premise_used = prem
                if prem:
                    cue += P("mind.cue-premise").format(premise=prem)
                    memory = self.recall_similar(prem, now, believed=believed)
                    if memory:
                        cue += P("mind.cue-recall").format(when=when_words(now - memory["ts"]), memory=memory["text"][:220])
                        print(f"[MIND] recall by association (d={memory.get('distance', 0):.2f}, {when_words(now - memory['ts'])}): {memory['text'][:70]}")
        if lead:
            cue = re.sub(r"^(\d\d:\d\d\.)", r"\1" + lead.replace("\\", "\\\\"), cue, count=1) if re.match(r"^\d\d:\d\d\.", cue) else lead.strip() + " " + cue
        if kind == "think":
            cue += self._felt_shift()
            cue += self._tone_notice()
            cue += self.time_edges(now, agent)
            cue += self._loop_line(agent)
            if not memory:
                cue += self._elicit_dose()
        if self.thread and now - self.thread[-1].get("ts", now) >= float(config.STREAM_GAP_MARK_SECONDS):
            from captioner.prompts import casual_time_string

            cue += P("mind.gap").format(gap=casual_time_string((now - self.thread[-1]["ts"]) / 60.0).capitalize())
        if self.pending_notice and kind == "think":
            subject, n_piv = self.pending_notice
            self.pending_notice = None
            cue += P("mind.pivot-notice").format(subject=subject, n=n_piv)

        # TEMPO FOLLOWS THE EVENT (Sep 6): what just happened sets the room, the heat and the odds of a beat
        tempo = "plain"
        if kind == "look" and (getattr(agent, "_salience_event", None) or believed != getattr(self, "_last_believed_for_tempo", believed) or now - float(getattr(self, "_scare_ts", 0.0) or 0.0) < 120):
            tempo = "surprise"
        elif memory or (self.thread and self.thread[-1].get("kind") in ("reflection", "dream")) or ("since" in cue and ("anyone" in cue or "awake" in cue or "changed for" in cue)):
            tempo = "new_thread"
        else:
            try:
                sit = self.situation(now, agent)
                if sit.get("still_h", 0) >= 1 and not sit.get("presence") and not sit.get("scare"):
                    tempo = "still"
            except Exception:
                pass
        self._last_believed_for_tempo = believed
        prior = self.recent_turns(now)
        turns: List[dict] = []
        life = self.life_block(now, agent)
        if prior and getattr(config, "MIND_SHAPE", "text") == "text":
            # the journal shape: the world's cues are ephemeral; only the machine's own text persists, as pages
            turns.append({"role": "user", "content": life})
            turns.append({"role": "assistant", "content": self.running_text(prior)})
            user = cue
        elif prior:
            first = prior[0]
            turns.append({"role": "user", "content": f"{life}\n\n{first.get('cue') or ''}".strip()})
            turns.append({"role": "assistant", "content": first["text"]})
            for e in prior[1:]:
                turns.append({"role": "user", "content": e.get("cue") or clock(e["ts"]) + "."})
                turns.append({"role": "assistant", "content": e["text"]})
            user = cue
        else:
            user = f"{life}\n\n{cue}".strip()
        return {"system": system, "turns": turns, "user": user, "image": img_path if kind == "look" else None, "cue": cue, "memory": memory, "life": life, "uneventful": uneventful, "tempo": tempo}
