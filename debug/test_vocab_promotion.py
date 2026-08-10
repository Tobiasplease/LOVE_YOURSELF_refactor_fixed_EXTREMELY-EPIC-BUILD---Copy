"""Standalone check of Phase 2 vocabulary promotion (no camera, no detector).

Replays real captions from live_captions.txt through a fresh promoter and
reports what gets promoted, in order, plus which candidate phrases the filters
blocked — so threshold/window/stoplist tuning is visible before a live run.

Usage:
    python debug/test_vocab_promotion.py [n_captions]
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.vocab_promotion import VocabularyPromoter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StubDetector:
    def __init__(self):
        self.vocab = None

    def set_vocabulary(self, terms):
        self.vocab = terms

    def get_term_hit_counts(self):
        return {}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    lines = [l.strip() for l in open(os.path.join(REPO, "event_log", "live_captions.txt"), encoding="utf-8", errors="replace") if l.strip()]
    captions = lines[-n:]
    print(f"replaying {len(captions)} captions\n")

    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab_promotion_test.json")
    if os.path.exists(state_path):
        os.remove(state_path)
    promoter = VocabularyPromoter(state_path=state_path, log_events=False)
    stub = StubDetector()
    promoter.attach_detector(stub)

    seen_candidates = Counter()
    for caption in captions:
        seen_candidates.update(promoter._extract_candidates(caption))
        promoter.observe_caption(caption)

    print(f"\npromoted ({len(promoter.promoted)}):")
    for p in promoter.promoted:
        print(f"  {p['mentions']:3d} mentions  {p['term']}")

    promoted_terms = {p["term"] for p in promoter.promoted}
    print("\nnear misses (candidates below threshold):")
    for term, c in seen_candidates.most_common(20):
        if term not in promoted_terms and not promoter._in_vocabulary(term):
            print(f"  {c:3d}  {term}")

    if stub.vocab:
        print(f"\nfinal compiled vocabulary would be {len(stub.vocab)} terms:")
        print("  " + ", ".join(stub.vocab))
    os.remove(state_path)


if __name__ == "__main__":
    main()
