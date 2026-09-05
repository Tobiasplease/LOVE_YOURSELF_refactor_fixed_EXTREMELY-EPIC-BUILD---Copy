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
PERSON_RE = re.compile(
    r"(?<!the )\b(he|him|his|she|her|hers)\b"
    r"|\b(the|that|this) (man|woman|guy|person|visitor)\b"
    r"|\b(a|an|some|one) (man|woman|guy|person|visitor) (is|sits|stands|sitting|standing|leaning|hunched|crouching|working|reading|looking|typing|at the|in the|on the|by the)\b"
    r"|\bsomeone('s| is| sits| stands| sitting| standing| leaning| hunched| crouching| working| reading| looking)\b",
    re.I,
)  # Sep 5 23:19: "I see a person sitting in that wooden chair. They're looking down…" (the mannequin head at the desk) slipped through on "a person"

# Within the SAME sentence as the person mention: absence, past tense, or wondering.
NOT_PRESENT_RE = re.compile(
    r"\b(gone|empty|no one|nobody|not here|isn.t here|used to|since|before|earlier|ago|anymore|remember|came back|come back|comes back|yesterday)\b"
    r"|\blast (night|time|week)\b"
    r"|\b(he|she|they|him|her)('d| was| were| had| did| went| came| sat| stood| took| said| typed| kept| got| left| looked| moved| walked| stopped| turned| leaned| stayed|\s+\w+ed)\b"  # NOT "'s been": "he's been sitting there" is present
    r"|\b(if|whether|maybe|perhaps|when|wonder|wondering|unless|imagine|imagining|pretend|pretending)\s+(he|she|they|him|her|someone|a person|a man|a woman)\b"
    r"|\b(could|should|would|might|want to|draw|drawing|sketch|sketching)\b[^.?!]{0,24}\b(a|an|some|one) (man|woman|guy|person|visitor)\b"  # "I could draw a person inside it" is not a sighting
    r"|\b(a|an|some|one) (man|woman|guy|person|visitor) (should|would|could|might|used to|ought to)\b"
    r"|\b(he|she|they)('s| is| are| has| have)?\s+(gone|left|not here|missing)\b"
    r"|\bwhere (he|she|they) (was|were|sat|stood|used)\b",
    re.I,
)

# Clause boundaries too: "He's been sitting there, or maybe he just arrived" is a claim AND a wondering — judge them apart.
_SENT_RE = re.compile(r"(?<=[.!?…])\s+|\n+|,\s+or\s+(?=maybe|perhaps)|;\s+|\s+—\s+")


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
    """The sentences that CLAIM a third person in the present tense. A question
    ("Is someone sitting there?") is wondering, not a claim — never gated
    (Sep 5 23:37: the machine asked exactly that about a dark shape, honestly)."""
    return [s for s in sentences(text) if PERSON_RE.search(s) and not NOT_PRESENT_RE.search(s) and not _is_question(s)]


_Q_TAIL_RE = re.compile(r"\?\s*(?:\w+[.!]?\s*){0,3}$")


def _is_question(s: str) -> bool:
    """Ends with '?', or a '?' followed by a word or three ('Is he back? No')."""
    return bool(_Q_TAIL_RE.search(s.rstrip()))


def is_phantom_presence(text: str) -> bool:
    return bool(phantom_sentences(text))
