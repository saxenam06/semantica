"""
Vocabulary and Controlled Vocabulary Module

This module provides vocabulary management for hierarchical controlled terms
that supplement ontologies with relevant terminology. Vocabularies connect
to ontology classes to provide structured, pre-defined term hierarchies.
"""

from .vocabulary_manager import VocabularyManager
from .controlled_vocabulary import (
    ControlledVocabulary,
    VocabularyTerm,
    TermRelation,
)

__all__ = [
    "VocabularyManager",
    "ControlledVocabulary",
    "VocabularyTerm",
    "TermRelation",
]
