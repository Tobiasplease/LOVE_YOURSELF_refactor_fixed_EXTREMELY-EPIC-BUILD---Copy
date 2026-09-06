"""The dream pass (Sep 6 2026) — the long overnight read of the whole day.

The artist: "with the overnight runs we should be able to run a long and
thorough dreaming compression pass that meaningfully reasons through large
context and perhaps rewrites some fundamentals that can have a noticeable
impact on the following day." Measured first: one llama-server slot takes a
13k-token prompt in 15 s, so the day rides raw — no summaries of summaries.

Two calls, once a night when the room has been still and empty for a while
(or on demand: debug/run_dream.py):
  1. RECORDS — the day as threads: where a thought ran on, what it was about,
     where it got to. One line each, the machine's words. Stored as "record"
     entries in the thread (never as turns) and embedded in the thoughts index,
     so recall by association can return a whole thread's conclusion.
  2. THE NIGHT'S PAGE — what stayed, what to carry, what to let go. Stored as
     a "dream" entry: the morning's continuity quote is its last sentence, and
     it is text the machine can read back as its own.
Kinds only in the asks; contents never. Wordings: prompt_registry dream.*.
"""
import re
import time
from typing import Dict, List, Optional

from captioner.prompt_registry import P
from config import config

_KINDS = ("wake", "look", "think", "reflection", "memory")


def _tokens(text: str) -> int:
    return int(len(text) / 3.6) + 1  # rough, conservative for English prose


def gather_day(thread: List[dict], since: float, until: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
    """The day's entries as journal pages with hour headings (the same
    paragraph rule as the running text). Trimmed from the OLDEST end to fit."""
    from captioner.mind import Mind

    until = until or time.time()
    max_tokens = max_tokens or int(config.DREAM_MAX_TOKENS)
    entries = [e for e in thread if e.get("text") and e.get("kind") in _KINDS and since <= float(e.get("ts", 0)) <= until]
    while entries:
        by_hour: Dict[str, List[dict]] = {}
        for e in entries:
            by_hour.setdefault(time.strftime("%H:00", time.localtime(e["ts"])), []).append(e)
        blocks = [f"— {h} —\n\n{Mind.running_text(es)}" for h, es in by_hour.items()]
        day = "\n\n".join(blocks)
        if _tokens(day) <= max_tokens:
            return day
        entries = entries[max(1, len(entries) // 10):]  # drop the oldest tenth and try again
    return ""


def _query(system: str, user: str, num_predict: int) -> str:
    from utils.inference import query_model

    return query_model(
        prompt=user,
        model=config.MODEL_NAME,
        image=None,
        system_prompt=system,
        timeout=600,
        log_dir=config.MOOD_SNAPSHOT_FOLDER,
        options={"temperature": float(config.DREAM_TEMPERATURE), "top_p": 0.95, "num_predict": num_predict, "repeat_penalty": 1.05, "num_ctx": 16384},
        prompt_type="dream",
        history=None,
        turns=[],  # a clean call: no stream, no prefill
    )


def parse_records(text: str, max_records: int) -> List[str]:
    out = []
    for line in (text or "").split("\n"):
        line = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", line).strip()
        if len(line.split()) >= 6 and not line.endswith(":"):
            out.append(line)
        if len(out) >= max_records:
            break
    return out


def run_dream(mind, now: Optional[float] = None, since: Optional[float] = None, dry: bool = False) -> Dict:
    """The pass. Returns {"day_tokens", "records", "page"}; stores unless dry."""
    now = now or time.time()
    since = since or (now - 24 * 3600)
    day = gather_day(mind.thread, since, now)
    if not day or _tokens(day) < 300:
        return {"day_tokens": _tokens(day), "records": [], "page": "", "skipped": "too little day to read"}
    system = P("dream.system")
    rec_text = _query(system, P("dream.records").format(day=day, max_records=int(config.DREAM_MAX_RECORDS)), int(config.DREAM_NUM_PREDICT_RECORDS))
    records = parse_records(rec_text, int(config.DREAM_MAX_RECORDS))
    page = (_query(system, P("dream.page").format(day=day), int(config.DREAM_NUM_PREDICT_PAGE)) or "").strip()
    page = re.sub(r"^\s*#+.*$", "", page, flags=re.M).strip()
    result = {"day_tokens": _tokens(day), "records": records, "page": page}
    if dry:
        return result
    for i, r in enumerate(records):
        mind.thread.append({"ts": now - 1 + i * 0.001, "kind": "record", "cue": "", "text": r[:400], "subject": ""})
    mind._index_add([e for e in mind.thread[-len(records):]] if records else [])
    if page and len(page.split()) >= 30:
        mind.absorb(page[:2000], "dream", P("mind.cue-reflection").format(clock=time.strftime("%H:%M", time.localtime(now))), now)
    mind.last_dream_ts = now
    mind._save()
    try:
        from event_logging.event_logger import log_json_entry
        from event_logging.log_type import LogType

        log_json_entry(LogType.DEBUG, {"message": "Dream pass", "action": "dream_pass", "day_tokens": result["day_tokens"], "records": len(records), "page_words": len(page.split())}, print_message=f"[🌙] dream: {len(records)} records, page {len(page.split())} words over {result['day_tokens']} tokens")
    except Exception:
        pass
    return result


def due(mind, now: float, agent) -> bool:
    """Once a night, in the DREAM_HOUR window, still and alone for DREAM_STILL_MIN_S."""
    if not getattr(config, "DREAM_ENABLED", True):
        return False
    h = time.localtime(now).tm_hour
    if not (int(config.DREAM_HOUR) <= h < int(config.DREAM_HOUR_END)):
        return False
    last = float(getattr(mind, "last_dream_ts", 0.0) or 0.0)
    if now - last < 20 * 3600:
        return False
    if getattr(agent, "_presence_believed", False) or getattr(agent, "_salience_hot", False):
        return False
    still = float(getattr(agent, "_world_change_ts", 0.0) or 0.0)
    if still and now - still < int(config.DREAM_STILL_MIN_S):
        return False
    return True
