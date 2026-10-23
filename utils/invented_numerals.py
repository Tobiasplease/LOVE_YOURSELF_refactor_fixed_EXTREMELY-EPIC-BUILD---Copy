"""A storage gate: a bare number is never somebody, and never a measurement
this machine makes.

Written Oct 23 2026, after the third occurrence of the same failure.

The chain, each time: the machine emits a number — from a count, from a
grounding box, or from nothing at all — the number lands in a sentence where a
subject belongs, an extractor stores that sentence as an event or a fact, and
the frame reads it back as something the machine knows. From then on it is a
person in the room.

  Sep 13  the stretch line reported counts; "a hundred times or more" became
          "104 looks now…", then "I feel 104's awareness as an indifference",
          and by evening the reveries held 102 and 103 sitting in chairs in an
          empty room.
  Sep 14  the same failure in Qwen's grounding notation — "1416x1828", then an
          incrementing "2440x-1420, 2460x-1440" — 761 of 5420 captions
          overnight, 64% of the last ninety minutes.
  Oct 22  the machine wrote "Blue hat. 1928." unprompted. The event extractor
          took 1928 for a name: "1928 left the room after turning away from the
          desk", "1928 in the grey shirt stood by the door". It reached
          machine_identity.json and lore_ledger.json, and rode the prompt on 36
          of 140 calls — the machine reading its own hallucination back as fact.

Each time the response was to remove that era's SOURCE of numbers and purge the
stores (debug/purge_numbered_people.py). No gate was ever added, so the next
source started the chain again. This is the gate. The detection is the purge
tool's regex, which two incidents' worth of tuning has already made careful:
matched against individual strings, never a JSON dump, and only where a number
is used AS somebody — "100 years" is fine, "102 is hunched over that desk" is
not.

Call it on the WRITE path, not the read path. A thing that was never stored
needs no purging.
"""

import re

# A number used as somebody: possessive, subject of a verb, paired with
# another, or a grounding box. Kept identical to debug/purge_numbered_people.py
# so the gate and the cleanup agree about what counts.
# Widened to {3,4} on Oct 23: September's pattern was tuned on 102/103/104 and
# did not match 1928, so the gate would have missed the very case that prompted
# it. Any bare 3-or-4 digit number is in scope; what makes it a REFUSAL is the
# company it keeps — a verb, a possessive, a description. "100 years of dust"
# and "twenty-five days" stay; "1928 left the room" and "104's awareness" go.
_AS_SOMEBODY = re.compile(
    r"\b\d{3,4}\s*[x×]\s*-?\d{3,4}\b"
    r"|\b\d{3,4}'s\b"
    r"|\b\d{3,4}\s+and\s+\d{3,4}\b"
    r"|\b\d{3,4}\s+(?:in|with|at)\s+(?:the|a|an|his|her|their)\b"
    # A bare number standing alone as a sentence's subject. This is the SEED
    # form — the Oct 22 chain began with the machine writing "Blue hat. 1928."
    # and nothing else. Anchored to a sentence boundary on purpose: "the shelf
    # holds 200, maybe 300 books" is a count mid-clause and must survive.
    r"|(?:^|[.!?]\s+)\d{3,4}\s*[,.]"
    r"|\b\d{3,4}\s+(?:is|was|are|were|has|have|had|left|stayed|sat|sits|sitting|stands|standing|stood|moved|disappeared|vanished|keeps|kept|hunched|leaned|shifted|turned|looked|looking|walked|came|went|waits|waited|watches|watched|made|makes|making|said|says|put|placed|holds|held|holding|wore|wearing|brought|took|picked|reached|pointed|gave|wants|wanted|seems|seemed|appears|appeared|likes|knows|knew|returns|returned|arrives|arrived)\b",
    re.I,
)


def reads_as_a_person(text) -> bool:
    """True if a bare number is standing in for somebody in this text.

    Accepts a string, or a dict/list of them (an entry about to be stored), so
    a caller can hand over the whole record it is about to append.
    """
    if isinstance(text, str):
        return bool(_AS_SOMEBODY.search(text))
    if isinstance(text, dict):
        return any(reads_as_a_person(v) for v in text.values())
    if isinstance(text, (list, tuple)):
        return any(reads_as_a_person(v) for v in text)
    return False


def refuse(text, where: str) -> bool:
    """reads_as_a_person, and says so on the way past.

    Loud on purpose: this failure is silent by nature — a stored sentence looks
    exactly like a real memory — and it took three occurrences and a four-day
    gap to notice the third. A line in the log is what makes the fourth cheap.
    """
    if not reads_as_a_person(text):
        return False
    snippet = (text if isinstance(text, str) else str(text))[:90]
    print(f"[🔢] refused by the numeral gate ({where}): {snippet}")
    return True
