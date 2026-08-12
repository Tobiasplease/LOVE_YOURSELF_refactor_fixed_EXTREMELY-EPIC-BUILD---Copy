"""The spaCy singleton. Lived in utils/pattern_recognition until Aug 12 2026,
when the pattern engine (motif extraction + saturated novelty) was retired;
vocab promotion's noun-chunk extraction is the surviving consumer."""

import spacy

nlp = spacy.load("en_core_web_sm")
