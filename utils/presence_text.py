"""Structure-only text tests for presence claims (Sep 5 2026, presence
stickiness — docs/presence-stickiness-sep4.md).

A third-person PRESENT-TENSE claim ("he's still hunched", "his head is down")
versus memory, absence or wondering ("since he left", "he used to", "maybe
he's still there") — judged sentence by sentence on the machine's own words.
Shared by the phantom_presence storage gate (captioner), the standing absence
fact (prompts) and debug/scrub_phantom_presence.py. Never content: pronouns,
tense cues and absence words only.
"""

import re
from typing import List

# "(?<!the )": a cut word ("i didn't draw the she[lf]", "the he[adset]") is not a pronoun.
PERSON_RE = re.compile(r"(?<!the )\b(he|him|his|she|her|hers)\b|\b(the|that|this) (man|woman|guy|person|visitor)\b", re.I)

# Within the SAME sentence as the person mention: absence, past tense, or wondering.
NOT_PRESENT_RE = re.compile(
    r"\b(gone|empty|no one|nobody|not here|isn.t here|used to|since|before|earlier|ago|anymore|remember|came back|come back|comes back|yesterday)\b"
    r"|\blast (night|time|week)\b"
    r"|\b(he|she|they|him|her)('d|'s been| was| were| had| has been| did| went| came| sat| stood| took| said| typed| kept| got| left| looked| moved| walked| stopped| turned| leaned| stayed|\s+\w+ed)\b"
    r"|\b(if|whether|maybe|perhaps|when|wonder|wondering|unless|imagine|imagining|pretend|pretending)\s+(he|she|they|him|her)\b"
    r"|\b(he|she|they)('s| is| are| has| have)?\s+(gone|left|not here|missing)\b"
    r"|\bwhere (he|she|they) (was|were|sat|stood|used)\b",
    re.I,
)

_SENT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


def sentences(text: str) -> List[str]:
    """Sentence split with short fragments ("that man.") folded into the next
    sentence, so a two-word fragment is judged with what it opens."""
    raw = [s.strip() for s in _SENT_RE.split(text or "") if s and s.strip()]
    out: List[str] = []
    carry = ""
    for s in raw:
        s = (carry + " " + s).strip() if carry else s
        carry = ""
        if len(s.split()) <= 3:
            carry = s
            continue
        out.append(s)
    if carry:
        out.append(carry)
    return out


def phantom_sentences(text: str) -> List[str]:
    """The sentences that claim a third person in the present tense."""
    return [s for s in sentences(text) if PERSON_RE.search(s) and not NOT_PRESENT_RE.search(s)]


def is_phantom_presence(text: str) -> bool:
    return bool(phantom_sentences(text))
