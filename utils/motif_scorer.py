"""
Optimized TinyLlama-based motif significance scorer with strict multiple choice output.
"""

import time
import requests
from typing import Dict, List, Optional, Tuple

# Use default Ollama URL since OLLAMA_BASE_URL not in config
OLLAMA_BASE_URL = "http://localhost:11434"
from event_logging.event_logger import log_json_entry
from event_logging.log_type import LogType


class MotifScorer:
    """Handles motif significance scoring using TinyLlama with strict output format."""

    def __init__(self, model_name: str = "tinyllama:latest"):
        self.model_name = model_name
        self.ollama_url = f"{OLLAMA_BASE_URL}/api/generate"
        self.cache = {}  # Simple cache for repeated motifs
        self.cache_hits = 0
        self.cache_misses = 0

        # Performance tracking
        self.total_requests = 0
        self.total_time = 0.0
        self.failed_requests = 0

    def score_motif(self, motif: str, context: str = "") -> float:
        """
        Score a motif's significance using TinyLlama with multiple choice format.
        Returns score between 0.0-1.0.
        """
        start_time = time.time()

        # Check cache first
        cache_key = f"{motif.lower().strip()}:{context[:50]}"
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        self.cache_misses += 1
        self.total_requests += 1

        try:
            # Quick heuristic pre-filter for obviously mundane objects
            score = self._heuristic_score(motif)
            if score is not None:
                self.cache[cache_key] = score
                elapsed = time.time() - start_time
                self.total_time += elapsed
                self._log_scoring_event(motif, score, "heuristic", elapsed, context)
                return score

            # Use TinyLlama for nuanced scoring
            llama_score = self._query_tinyllama(motif, context)
            self.cache[cache_key] = llama_score

            elapsed = time.time() - start_time
            self.total_time += elapsed
            self._log_scoring_event(motif, llama_score, "tinyllama", elapsed, context)

            return llama_score

        except Exception as e:
            self.failed_requests += 1
            elapsed = time.time() - start_time
            self.total_time += elapsed

            # Fallback to heuristic on failure
            fallback_score = self._heuristic_score(motif) or 0.3
            self._log_scoring_event(motif, fallback_score, "fallback", elapsed, context, str(e))

            return fallback_score

    def _heuristic_score(self, motif: str) -> Optional[float]:
        """Fast heuristic scoring for obvious cases."""
        motif_lower = motif.lower().strip()

        # Ultra-mundane objects - skip TinyLlama
        ultra_mundane = {
            "table",
            "chair",
            "desk",
            "wall",
            "door",
            "window",
            "laptop",
            "phone",
            "book",
            "paper",
            "screen",
            "corner",
            "room",
            "floor",
            "ceiling",
            "keyboard",
            "mouse",
            "monitor",
            "shelf",
            "couch",
            "bed",
            "surface",
            "object",
        }

        if motif_lower in ultra_mundane:
            return 0.1

        # Conceptually interesting - still check with TinyLlama but bias high
        interesting_concepts = {
            "consciousness",
            "awareness",
            "existence",
            "reality",
            "imagination",
            "creativity",
            "mystery",
            "wonder",
            "discovery",
            "understanding",
            "perception",
            "emotion",
            "thought",
            "mind",
            "soul",
            "spirit",
            "dream",
            "vision",
            "inspiration",
            "revelation",
            "transformation",
            "connection",
        }

        if motif_lower in interesting_concepts:
            return 0.9

        # Very short motifs are usually not significant
        if len(motif) < 3:
            return 0.2

        # Let TinyLlama handle the nuanced cases
        return None

    def _query_tinyllama(self, motif: str, context: str) -> float:
        """Query TinyLlama with strict multiple choice format."""

        # Build strict prompt with multiple choice
        prompt = f"""Rate the conceptual significance of "{motif}" in creative/artistic context.

Context: {context[:100] if context else "General observation"}

Choose EXACTLY ONE letter:
A) 0.1 - Mundane object (furniture, basic tools, common items)
B) 0.3 - Somewhat interesting but common
C) 0.5 - Moderately significant
D) 0.7 - Quite interesting or meaningful
E) 0.9 - Highly significant concept

Answer with ONLY the letter (A, B, C, D, or E):"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistency
                "top_p": 0.9,
                "num_predict": 1,  # Only need one character
                "stop": ["\n", " ", ".", ","],  # Stop after single letter
            },
        }

        response = requests.post(self.ollama_url, json=payload, timeout=5.0)
        response.raise_for_status()

        result = response.json()
        raw_response = result.get("response", "").strip().upper()

        # Parse multiple choice response
        if raw_response.startswith("A"):
            return 0.1
        elif raw_response.startswith("B"):
            return 0.3
        elif raw_response.startswith("C"):
            return 0.5
        elif raw_response.startswith("D"):
            return 0.7
        elif raw_response.startswith("E"):
            return 0.9
        else:
            # Fallback if response is malformed
            return 0.4

    def _log_scoring_event(self, motif: str, score: float, method: str, elapsed: float, context: str = "", error: str = ""):
        """Log motif scoring events for debugging and analysis."""
        log_data = {
            "motif": motif,
            "score": score,
            "method": method,
            "elapsed_ms": round(elapsed * 1000, 2),
            "context_snippet": context[:30] if context else "",
            "cache_stats": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            },
        }

        if error:
            log_data["error"] = error

        log_json_entry(LogType.MOTIF_SCORE, log_data)

    def get_performance_stats(self) -> Dict:
        """Get performance statistics for monitoring."""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.total_requests - self.failed_requests) / self.total_requests if self.total_requests > 0 else 0,
            "avg_response_time_ms": (self.total_time * 1000) / self.total_requests if self.total_requests > 0 else 0,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "cache_size": len(self.cache),
        }

    def clear_cache(self):
        """Clear the motif scoring cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


# Global scorer instance
_motif_scorer = None


def get_motif_scorer() -> MotifScorer:
    """Get the global motif scorer instance."""
    global _motif_scorer
    if _motif_scorer is None:
        _motif_scorer = MotifScorer()
    return _motif_scorer


def score_motif_significance(motif: str, context: str = "") -> float:
    """Convenience function to score a motif's significance."""
    return get_motif_scorer().score_motif(motif, context)


def score_multiple_motifs(motifs: List[str], context: str = "") -> List[Tuple[str, float]]:
    """Score multiple motifs efficiently and return sorted by significance."""
    scorer = get_motif_scorer()

    scored_motifs = []
    for motif in motifs:
        score = scorer.score_motif(motif, context)
        scored_motifs.append((motif, score))

    # Sort by score descending
    scored_motifs.sort(key=lambda x: x[1], reverse=True)
    return scored_motifs
