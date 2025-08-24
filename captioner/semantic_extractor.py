"""
Semantic motif extraction with contextual grounding.
Extracts meaningful experiences, not disconnected objects.
"""

import spacy
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re

nlp = spacy.load("en_core_web_sm")


class SemanticMotifExtractor:
    """Extract semantically meaningful motifs with context."""
    
    def __init__(self):
        # Categories of meaningful extraction
        self.categories = {
            'emotional_states': [],
            'sensory_experiences': [],
            'spatial_relationships': [],
            'temporal_markers': [],
            'actions_processes': [],
            'qualities_attributes': [],
            'conceptual_thoughts': []
        }
        
        # Emotional indicators
        self.emotion_words = {
            'curious', 'fascinated', 'bored', 'restless', 'calm', 'anxious',
            'intrigued', 'confused', 'settled', 'unsettled', 'drawn', 'repelled',
            'comfortable', 'uncomfortable', 'engaged', 'detached', 'focused', 'distracted'
        }
        
        # Sensory markers
        self.sensory_markers = {
            'see', 'notice', 'observe', 'glimpse', 'spot', 'detect',
            'hear', 'sense', 'feel', 'perceive', 'glow', 'shine', 'shadow'
        }
        
        # Temporal context words
        self.temporal_markers = {
            'now', 'still', 'again', 'before', 'after', 'suddenly',
            'gradually', 'always', 'never', 'sometimes', 'recently', 'earlier'
        }
        
    def extract_grounded_motifs(self, text: str, previous_context: Optional[str] = None) -> Dict[str, List[Tuple[str, str]]]:
        """
        Extract motifs with their contextual grounding.
        Returns tuples of (motif, context) for each category.
        """
        doc = nlp(text.lower())
        grounded_motifs = defaultdict(list)
        
        # 1. Extract emotional experiences with context
        for token in doc:
            if token.text in self.emotion_words or token.pos_ == "ADJ" and any(e in token.text for e in ['ed', 'ing']):
                # Find what the emotion is about
                context = self._find_emotional_context(token, doc)
                if context:
                    grounded_motifs['emotional_states'].append((token.text, context))
        
        # 2. Extract sensory experiences with spatial grounding
        for token in doc:
            if token.lemma_ in self.sensory_markers:
                # What is being sensed and where?
                object_context = self._find_sensory_object(token, doc)
                if object_context:
                    grounded_motifs['sensory_experiences'].append((token.lemma_, object_context))
        
        # 3. Extract spatial relationships (not just objects)
        for chunk in doc.noun_chunks:
            spatial_prep = self._find_spatial_preposition(chunk, doc)
            if spatial_prep:
                # "light on the wall" -> ('light', 'on the wall')
                grounded_motifs['spatial_relationships'].append((chunk.text, spatial_prep))
        
        # 4. Extract temporal continuity markers
        for token in doc:
            if token.text in self.temporal_markers:
                # What is the temporal marker referring to?
                temporal_context = self._find_temporal_reference(token, doc)
                if temporal_context:
                    grounded_motifs['temporal_markers'].append((token.text, temporal_context))
        
        # 5. Extract action-object relationships
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                # What action and on what?
                action_context = self._find_action_context(token, doc)
                if action_context:
                    grounded_motifs['actions_processes'].append((token.lemma_, action_context))
        
        # 6. Extract qualities with their subjects
        for token in doc:
            if token.pos_ == "ADJ" and not token.text in self.emotion_words:
                # What is being described?
                quality_subject = self._find_quality_subject(token, doc)
                if quality_subject:
                    grounded_motifs['qualities_attributes'].append((token.text, quality_subject))
        
        # 7. Extract conceptual thoughts (comparative structures, negations)
        conceptual_patterns = self._extract_conceptual_patterns(text)
        grounded_motifs['conceptual_thoughts'].extend(conceptual_patterns)
        
        return dict(grounded_motifs)
    
    def _find_emotional_context(self, emotion_token, doc) -> Optional[str]:
        """Find what the emotion is about."""
        # Look for prepositions after emotion
        for child in emotion_token.children:
            if child.dep_ == "prep":
                prep_phrase = self._get_prep_phrase(child)
                return f"about {prep_phrase}"
        
        # Look for subject if emotion is predicate
        if emotion_token.dep_ == "acomp":
            for token in doc:
                if token.dep_ == "nsubj":
                    return f"feeling toward {token.text}"
        
        return None
    
    def _find_sensory_object(self, sense_token, doc) -> Optional[str]:
        """Find what is being sensed."""
        objects = []
        for child in sense_token.children:
            if child.dep_ in ["dobj", "pobj"]:
                # Include modifiers for rich context
                obj_phrase = " ".join([t.text for t in child.subtree])
                objects.append(obj_phrase)
        
        return " and ".join(objects) if objects else None
    
    def _find_spatial_preposition(self, chunk, doc) -> Optional[str]:
        """Find spatial relationships."""
        for token in chunk:
            for child in token.children:
                if child.dep_ == "prep":
                    return self._get_prep_phrase(child)
        return None
    
    def _find_temporal_reference(self, temporal_token, doc) -> Optional[str]:
        """Find what the temporal marker refers to."""
        # Look at surrounding context
        if temporal_token.i > 0:
            prev_token = doc[temporal_token.i - 1]
            if prev_token.pos_ == "VERB":
                return f"{prev_token.lemma_}"
        
        if temporal_token.i < len(doc) - 1:
            next_token = doc[temporal_token.i + 1]
            if next_token.pos_ in ["VERB", "NOUN"]:
                return f"{next_token.text}"
        
        return None
    
    def _find_action_context(self, verb_token, doc) -> Optional[str]:
        """Find what action is being performed on what."""
        objects = []
        preps = []
        
        for child in verb_token.children:
            if child.dep_ in ["dobj", "pobj"]:
                objects.append(child.text)
            elif child.dep_ == "prep":
                preps.append(self._get_prep_phrase(child))
        
        context_parts = objects + preps
        return " ".join(context_parts) if context_parts else None
    
    def _find_quality_subject(self, adj_token, doc) -> Optional[str]:
        """Find what has this quality."""
        # Check if modifying a noun
        if adj_token.head.pos_ == "NOUN":
            return adj_token.head.text
        
        # Check for copula structure (X is ADJ)
        for token in doc:
            if token.dep_ == "nsubj" and adj_token.head == token.head:
                return token.text
        
        return None
    
    def _get_prep_phrase(self, prep_token) -> str:
        """Get full prepositional phrase."""
        phrase_tokens = []
        for child in prep_token.subtree:
            phrase_tokens.append(child.text)
        return " ".join(phrase_tokens)
    
    def _extract_conceptual_patterns(self, text: str) -> List[Tuple[str, str]]:
        """Extract higher-level conceptual patterns."""
        patterns = []
        
        # "Not X but Y" pattern
        not_but = re.findall(r'not\s+(\w+).*?but\s+(\w+)', text, re.IGNORECASE)
        for match in not_but:
            patterns.append(("contrast", f"not {match[0]} but {match[1]}"))
        
        # "More X than Y" pattern
        more_than = re.findall(r'more\s+(\w+)\s+than\s+(\w+)', text, re.IGNORECASE)
        for match in more_than:
            patterns.append(("comparison", f"more {match[0]} than {match[1]}"))
        
        # Questions to self
        if '?' in text:
            patterns.append(("questioning", "self-inquiry"))
        
        return patterns


def transform_motifs_to_memories(grounded_motifs: Dict[str, List[Tuple[str, str]]]) -> List[str]:
    """
    Transform grounded motifs into meaningful memory fragments.
    Instead of "table" -> "the cluttered table beneath warm light"
    """
    memories = []
    
    # Combine spatial and sensory for rich memories
    for sense_motif, sense_context in grounded_motifs.get('sensory_experiences', []):
        for spatial_motif, spatial_context in grounded_motifs.get('spatial_relationships', []):
            if spatial_motif in sense_context or sense_context in spatial_context:
                memory = f"{sense_motif} {spatial_motif} {spatial_context}"
                memories.append(memory)
    
    # Combine emotional states with their contexts
    for emotion, context in grounded_motifs.get('emotional_states', []):
        if context:
            memory = f"feeling {emotion} {context}"
            memories.append(memory)
    
    # Combine actions with qualities
    for action, action_context in grounded_motifs.get('actions_processes', []):
        for quality, quality_subject in grounded_motifs.get('qualities_attributes', []):
            if quality_subject in action_context:
                memory = f"{action} the {quality} {quality_subject}"
                memories.append(memory)
    
    # Add temporal continuity
    for temporal, reference in grounded_motifs.get('temporal_markers', []):
        if reference:
            memory = f"{temporal} {reference}"
            memories.append(memory)
    
    return memories


# Example usage
if __name__ == "__main__":
    extractor = SemanticMotifExtractor()
    
    test_texts = [
        "I notice the glowing screen, feeling curious about the patterns of light dancing on the wall behind it.",
        "The table sits there, same as always, but today shadows make it seem different somehow.",
        "Not the laptop itself, but the warmth it generates, creating this bubble of electrical presence."
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        motifs = extractor.extract_grounded_motifs(text)
        print("Grounded motifs:")
        for category, items in motifs.items():
            if items:
                print(f"  {category}:")
                for motif, context in items:
                    print(f"    - {motif}: {context}")
        
        # Transform to memories
        memories = transform_motifs_to_memories(motifs)
        print("Meaningful memories:")
        for memory in memories:
            print(f"  - {memory}")