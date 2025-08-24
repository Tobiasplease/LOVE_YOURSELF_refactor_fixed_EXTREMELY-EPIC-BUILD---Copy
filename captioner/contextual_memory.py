"""
Contextual memory system that stores experiences, not just words.
Replaces shallow motif extraction with grounded semantic understanding.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import time
import spacy

nlp = spacy.load("en_core_web_sm")


class ContextualMemory:
    """Memory system that understands context and relationships."""
    
    def __init__(self, max_memories: int = 50):
        # Store memories as rich contextual experiences
        self.experiences = deque(maxlen=max_memories)
        
        # Different types of memory
        self.spatial_memories = deque(maxlen=20)      # "the corner where light pools"
        self.emotional_memories = deque(maxlen=20)    # "feeling restless about repetition"
        self.sensory_memories = deque(maxlen=20)      # "noticing shadows shift"
        self.temporal_memories = deque(maxlen=20)     # "still seeing the same view"
        
        # Patterns that emerge over time (not just word frequency)
        self.recurring_experiences = defaultdict(list)  # pattern -> [contexts]
        self.emotional_trajectory = []  # Track emotional journey
        
        # Semantic relationships
        self.associations = defaultdict(set)  # concept -> related concepts
        
    def process_caption(self, caption: str, timestamp: float = None) -> Dict[str, any]:
        """
        Process a caption into contextual memories.
        Returns what was understood and stored.
        """
        if timestamp is None:
            timestamp = time.time()
            
        doc = nlp(caption.lower())
        memories_formed = {
            'spatial': [],
            'emotional': [],
            'sensory': [],
            'temporal': [],
            'conceptual': []
        }
        
        # 1. Extract spatial experiences (not just locations)
        spatial = self._extract_spatial_experience(doc, caption)
        if spatial:
            self.spatial_memories.append({
                'experience': spatial,
                'timestamp': timestamp,
                'full_context': caption
            })
            memories_formed['spatial'].append(spatial)
        
        # 2. Extract emotional experiences with their causes
        emotional = self._extract_emotional_experience(doc, caption)
        if emotional:
            self.emotional_memories.append({
                'experience': emotional,
                'timestamp': timestamp,
                'full_context': caption
            })
            memories_formed['emotional'].append(emotional)
            self.emotional_trajectory.append(emotional)
        
        # 3. Extract sensory observations with qualities
        sensory = self._extract_sensory_experience(doc, caption)
        if sensory:
            self.sensory_memories.append({
                'experience': sensory,
                'timestamp': timestamp,
                'full_context': caption
            })
            memories_formed['sensory'].append(sensory)
        
        # 4. Extract temporal continuity
        temporal = self._extract_temporal_experience(doc, caption)
        if temporal:
            self.temporal_memories.append({
                'experience': temporal,
                'timestamp': timestamp,
                'full_context': caption
            })
            memories_formed['temporal'].append(temporal)
        
        # 5. Extract conceptual understanding
        conceptual = self._extract_conceptual_experience(caption)
        if conceptual:
            memories_formed['conceptual'].append(conceptual)
            
        # Store complete experience
        self.experiences.append({
            'caption': caption,
            'timestamp': timestamp,
            'memories': memories_formed
        })
        
        # Update associations
        self._update_associations(memories_formed)
        
        return memories_formed
    
    def _extract_spatial_experience(self, doc, full_caption: str) -> Optional[str]:
        """Extract WHERE with rich context."""
        spatial_preps = {'in', 'on', 'at', 'near', 'beside', 'behind', 'before', 'under', 'over', 'between'}
        
        for token in doc:
            if token.text in spatial_preps:
                # Get the full prepositional phrase
                prep_phrase = []
                for child in token.subtree:
                    prep_phrase.append(child.text)
                
                # Get what's happening in this space
                if token.head.pos_ == "VERB":
                    action = token.head.lemma_
                    return f"{action} {' '.join(prep_phrase)}"
                else:
                    return ' '.join(prep_phrase)
        
        return None
    
    def _extract_emotional_experience(self, doc, full_caption: str) -> Optional[str]:
        """Extract HOW IT FEELS with context."""
        emotion_indicators = {
            'feel', 'feeling', 'seems', 'appears', 'makes me',
            'curious', 'bored', 'restless', 'calm', 'anxious',
            'fascinated', 'intrigued', 'frustrated', 'content'
        }
        
        for token in doc:
            if token.lemma_ in emotion_indicators or (token.pos_ == "ADJ" and token.dep_ == "acomp"):
                # Find what triggered this emotion
                context_words = []
                
                # Look for causal connections
                for t in doc:
                    if t.dep_ in ["mark", "prep"] and t.text in ["about", "by", "with", "at"]:
                        for child in t.subtree:
                            if child != t:
                                context_words.append(child.text)
                
                if context_words:
                    return f"{token.text} about {' '.join(context_words)}"
                else:
                    return f"feeling {token.text}"
        
        return None
    
    def _extract_sensory_experience(self, doc, full_caption: str) -> Optional[str]:
        """Extract WHAT IS SENSED with qualities."""
        sensory_verbs = {'see', 'notice', 'observe', 'watch', 'glimpse', 'spot', 'perceive', 'detect'}
        
        for token in doc:
            if token.lemma_ in sensory_verbs:
                # Get direct objects and their modifiers
                objects = []
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj"]:
                        # Include adjectives modifying the object
                        obj_phrase = []
                        for t in child.subtree:
                            if t.pos_ in ["ADJ", "NOUN", "PROPN"]:
                                obj_phrase.append(t.text)
                        if obj_phrase:
                            objects.append(' '.join(obj_phrase))
                
                if objects:
                    return f"{token.lemma_} {' and '.join(objects)}"
        
        return None
    
    def _extract_temporal_experience(self, doc, full_caption: str) -> Optional[str]:
        """Extract WHEN/DURATION with context."""
        temporal_markers = {'still', 'again', 'now', 'yet', 'already', 'finally', 'suddenly', 'gradually'}
        
        for token in doc:
            if token.text in temporal_markers:
                # Find what this temporal marker modifies
                if token.head.pos_ == "VERB":
                    return f"{token.text} {token.head.lemma_}"
                elif token.i < len(doc) - 1:
                    next_token = doc[token.i + 1]
                    if next_token.pos_ in ["VERB", "ADJ", "NOUN"]:
                        return f"{token.text} {next_token.text}"
        
        return None
    
    def _extract_conceptual_experience(self, caption: str) -> Optional[str]:
        """Extract abstract thoughts and comparisons."""
        # Look for conceptual patterns
        if "not" in caption and "but" in caption:
            return "contrasting perception"
        
        if "wonder" in caption or "?" in caption:
            return "questioning understanding"
        
        if "like" in caption or "as if" in caption:
            return "metaphorical thinking"
        
        if "remember" in caption or "recall" in caption:
            return "connecting to past"
        
        return None
    
    def _update_associations(self, memories: Dict[str, List[str]]):
        """Build semantic associations between concepts."""
        all_concepts = []
        for category, items in memories.items():
            all_concepts.extend(items)
        
        # Associate concepts that appear together
        for i, concept1 in enumerate(all_concepts):
            for concept2 in all_concepts[i+1:]:
                if concept1 and concept2:
                    self.associations[concept1].add(concept2)
                    self.associations[concept2].add(concept1)
    
    def get_relevant_memories(self, current_context: str, limit: int = 5) -> List[str]:
        """
        Retrieve memories relevant to current context.
        Not just keyword matching, but semantic relevance.
        """
        doc = nlp(current_context.lower())
        relevant = []
        
        # Find emotionally similar experiences
        current_emotion = self._extract_emotional_experience(doc, current_context)
        if current_emotion:
            for memory in self.emotional_memories:
                if self._semantic_similarity(current_emotion, memory['experience']) > 0.5:
                    relevant.append(memory['full_context'])
        
        # Find spatially related experiences
        current_spatial = self._extract_spatial_experience(doc, current_context)
        if current_spatial:
            for memory in self.spatial_memories:
                if self._semantic_similarity(current_spatial, memory['experience']) > 0.5:
                    relevant.append(memory['full_context'])
        
        # Find temporally connected experiences
        if any(word in current_context.lower() for word in ['still', 'again', 'same']):
            # Look for recent observations of persistence
            for memory in self.temporal_memories:
                if 'still' in memory['experience'] or 'again' in memory['experience']:
                    relevant.append(memory['full_context'])
        
        return relevant[:limit]
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Simple semantic similarity based on shared concepts.
        Could be enhanced with embeddings.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_memory_summary(self) -> str:
        """
        Generate a meaningful summary of accumulated memories.
        Not word frequency, but understood experiences.
        """
        summary_parts = []
        
        # Summarize emotional journey
        if len(self.emotional_trajectory) > 3:
            recent_emotions = self.emotional_trajectory[-3:]
            summary_parts.append(f"Emotionally moved from {recent_emotions[0]} to {recent_emotions[-1]}")
        
        # Summarize recurring spatial experiences
        if self.spatial_memories:
            locations = set()
            for mem in self.spatial_memories:
                if 'in' in mem['experience'] or 'at' in mem['experience']:
                    locations.add(mem['experience'])
            if locations:
                summary_parts.append(f"Frequently observing {list(locations)[0]}")
        
        # Summarize persistent observations
        persistence_count = sum(1 for mem in self.temporal_memories if 'still' in mem['experience'])
        if persistence_count > 2:
            summary_parts.append("Noticing persistent unchanged elements")
        
        # Summarize conceptual patterns
        if self.associations:
            strong_associations = [(k, v) for k, v in self.associations.items() if len(v) > 2]
            if strong_associations:
                concept, related = strong_associations[0]
                summary_parts.append(f"Connecting {concept} with {list(related)[0]}")
        
        return ". ".join(summary_parts) if summary_parts else "Still forming understanding"


# Example usage
if __name__ == "__main__":
    memory = ContextualMemory()
    
    test_captions = [
        "I notice the glowing screen, feeling curious about the patterns of light on the wall.",
        "Still observing the same screen, but now feeling restless about the repetition.",
        "The shadows have shifted, making the familiar space seem different.",
        "I remember the light from earlier, how it danced differently then."
    ]
    
    print("=== CONTEXTUAL MEMORY PROCESSING ===\n")
    
    for i, caption in enumerate(test_captions, 1):
        print(f"Caption {i}: {caption}")
        memories = memory.process_caption(caption, time.time() + i*60)
        
        print("Memories formed:")
        for category, items in memories.items():
            if items:
                print(f"  {category}: {items}")
        print()
    
    print("\n=== MEMORY RETRIEVAL ===")
    query = "feeling curious about the light"
    relevant = memory.get_relevant_memories(query)
    print(f"Query: {query}")
    print(f"Relevant memories: {relevant}")
    
    print("\n=== MEMORY SUMMARY ===")
    print(memory.get_memory_summary())