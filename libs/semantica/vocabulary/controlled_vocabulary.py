"""
Controlled Vocabulary for Semantica framework.

Represents hierarchical collections of controlled terms that can be
connected to ontology classes. Useful when classes have many known
subtypes that share attributes but don't need separate modeling.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from ..utils.exceptions import ValidationError
from ..utils.logging import get_logger


class TermRelation(str, Enum):
    """Term relationship types."""
    BROADER = "broader"
    NARROWER = "narrower"
    RELATED = "related"
    EXACT_MATCH = "exactMatch"
    CLOSE_MATCH = "closeMatch"
    BROAD_MATCH = "broadMatch"
    NARROW_MATCH = "narrowMatch"


@dataclass
class VocabularyTerm:
    """Controlled vocabulary term."""
    uri: str
    label: str
    definition: Optional[str] = None
    notation: Optional[str] = None
    broader_terms: List[str] = field(default_factory=list)
    narrower_terms: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    alt_labels: List[str] = field(default_factory=list)
    hidden_labels: List[str] = field(default_factory=list)
    scope_note: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    in_scheme: Optional[str] = None
    top_concept_of: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ControlledVocabulary:
    """
    Controlled vocabulary with hierarchical term structure.
    
    • Manages hierarchical term structures (broader/narrower)
    • Supports SKOS vocabulary standards
    • Enables vocabulary connection to ontology classes
    • Handles term updates without ontology changes
    """
    
    def __init__(
        self,
        uri: str,
        title: str,
        description: Optional[str] = None,
        **metadata
    ):
        """
        Initialize controlled vocabulary.
        
        Args:
            uri: Vocabulary URI
            title: Vocabulary title
            description: Vocabulary description
            **metadata: Additional metadata
        """
        self.logger = get_logger("controlled_vocabulary")
        self.uri = uri
        self.title = title
        self.description = description
        self.metadata = metadata
        
        self.terms: Dict[str, VocabularyTerm] = {}
        self.top_concepts: List[str] = []
        self.ontology_classes: Dict[str, str] = {}  # term_uri -> class_uri
    
    def add_term(
        self,
        term: VocabularyTerm,
        top_concept: bool = False
    ) -> bool:
        """
        Add term to vocabulary.
        
        Args:
            term: Vocabulary term
            top_concept: Whether this is a top-level concept
            
        Returns:
            True if term added successfully
        """
        if term.uri in self.terms:
            self.logger.warning(f"Term '{term.uri}' already exists, updating")
        
        self.terms[term.uri] = term
        term.in_scheme = self.uri
        
        if top_concept:
            term.top_concept_of = self.uri
            if term.uri not in self.top_concepts:
                self.top_concepts.append(term.uri)
        
        self.logger.info(f"Added term: {term.uri}")
        return True
    
    def remove_term(self, term_uri: str) -> bool:
        """
        Remove term from vocabulary.
        
        Args:
            term_uri: Term URI
            
        Returns:
            True if term removed successfully
        """
        if term_uri not in self.terms:
            return False
        
        term = self.terms[term_uri]
        
        # Remove from broader terms' narrower lists
        for broader_uri in term.broader_terms:
            if broader_uri in self.terms:
                if term_uri in self.terms[broader_uri].narrower_terms:
                    self.terms[broader_uri].narrower_terms.remove(term_uri)
        
        # Remove from narrower terms' broader lists
        for narrower_uri in term.narrower_terms:
            if narrower_uri in self.terms:
                if term_uri in self.terms[narrower_uri].broader_terms:
                    self.terms[narrower_uri].broader_terms.remove(term_uri)
        
        # Remove from top concepts
        if term_uri in self.top_concepts:
            self.top_concepts.remove(term_uri)
        
        del self.terms[term_uri]
        self.logger.info(f"Removed term: {term_uri}")
        return True
    
    def add_broader_term(
        self,
        term_uri: str,
        broader_uri: str
    ) -> bool:
        """
        Add broader term relationship.
        
        Args:
            term_uri: Term URI
            broader_uri: Broader term URI
            
        Returns:
            True if relationship added successfully
        """
        if term_uri not in self.terms or broader_uri not in self.terms:
            raise ValidationError("Both terms must exist in vocabulary")
        
        if broader_uri not in self.terms[term_uri].broader_terms:
            self.terms[term_uri].broader_terms.append(broader_uri)
        
        if term_uri not in self.terms[broader_uri].narrower_terms:
            self.terms[broader_uri].narrower_terms.append(term_uri)
        
        return True
    
    def add_narrower_term(
        self,
        term_uri: str,
        narrower_uri: str
    ) -> bool:
        """
        Add narrower term relationship.
        
        Args:
            term_uri: Term URI
            narrower_uri: Narrower term URI
            
        Returns:
            True if relationship added successfully
        """
        return self.add_broader_term(narrower_uri, term_uri)
    
    def add_related_term(
        self,
        term_uri: str,
        related_uri: str
    ) -> bool:
        """
        Add related term relationship.
        
        Args:
            term_uri: Term URI
            related_uri: Related term URI
            
        Returns:
            True if relationship added successfully
        """
        if term_uri not in self.terms or related_uri not in self.terms:
            raise ValidationError("Both terms must exist in vocabulary")
        
        if related_uri not in self.terms[term_uri].related_terms:
            self.terms[term_uri].related_terms.append(related_uri)
        
        if term_uri not in self.terms[related_uri].related_terms:
            self.terms[related_uri].related_terms.append(term_uri)
        
        return True
    
    def get_term(self, term_uri: str) -> Optional[VocabularyTerm]:
        """
        Get term by URI.
        
        Args:
            term_uri: Term URI
            
        Returns:
            Vocabulary term or None if not found
        """
        return self.terms.get(term_uri)
    
    def get_broader_terms(
        self,
        term_uri: str,
        transitive: bool = False
    ) -> List[str]:
        """
        Get broader terms.
        
        Args:
            term_uri: Term URI
            transitive: Include transitive broader terms
            
        Returns:
            List of broader term URIs
        """
        if term_uri not in self.terms:
            return []
        
        broader = list(self.terms[term_uri].broader_terms)
        
        if transitive:
            for broader_uri in broader:
                broader.extend(self.get_broader_terms(broader_uri, transitive=True))
        
        return list(set(broader))
    
    def get_narrower_terms(
        self,
        term_uri: str,
        transitive: bool = False
    ) -> List[str]:
        """
        Get narrower terms.
        
        Args:
            term_uri: Term URI
            transitive: Include transitive narrower terms
            
        Returns:
            List of narrower term URIs
        """
        if term_uri not in self.terms:
            return []
        
        narrower = list(self.terms[term_uri].narrower_terms)
        
        if transitive:
            for narrower_uri in narrower:
                narrower.extend(self.get_narrower_terms(narrower_uri, transitive=True))
        
        return list(set(narrower))
    
    def connect_to_ontology_class(
        self,
        term_uri: str,
        class_uri: str
    ) -> bool:
        """
        Connect vocabulary term to ontology class.
        
        Args:
            term_uri: Term URI
            class_uri: Ontology class URI
            
        Returns:
            True if connection successful
        """
        if term_uri not in self.terms:
            raise ValidationError(f"Term '{term_uri}' not found")
        
        self.ontology_classes[term_uri] = class_uri
        self.logger.info(f"Connected term {term_uri} to class {class_uri}")
        return True
    
    def search_terms(
        self,
        query: str,
        search_labels: bool = True,
        search_definitions: bool = True
    ) -> List[str]:
        """
        Search for terms by query.
        
        Args:
            query: Search query
            search_labels: Search in labels
            search_definitions: Search in definitions
            
        Returns:
            List of matching term URIs
        """
        query_lower = query.lower()
        matches = []
        
        for term_uri, term in self.terms.items():
            matched = False
            
            if search_labels:
                if query_lower in term.label.lower():
                    matched = True
                for alt_label in term.alt_labels:
                    if query_lower in alt_label.lower():
                        matched = True
            
            if search_definitions and term.definition:
                if query_lower in term.definition.lower():
                    matched = True
            
            if matched:
                matches.append(term_uri)
        
        return matches
    
    def to_skos(self) -> Dict[str, Any]:
        """
        Convert vocabulary to SKOS format.
        
        Returns:
            SKOS vocabulary dictionary
        """
        skos = {
            "@context": {
                "skos": "http://www.w3.org/2004/02/skos/core#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            },
            "@type": "skos:ConceptScheme",
            "@id": self.uri,
            "skos:prefLabel": {
                "@value": self.title,
                "@language": "en"
            },
            "skos:hasTopConcept": [{"@id": uri} for uri in self.top_concepts],
            "skos:concept": []
        }
        
        if self.description:
            skos["skos:definition"] = {
                "@value": self.description,
                "@language": "en"
            }
        
        # Convert terms to SKOS
        for term_uri, term in self.terms.items():
            concept = {
                "@id": term_uri,
                "@type": "skos:Concept",
                "skos:prefLabel": {
                    "@value": term.label,
                    "@language": "en"
                },
                "skos:inScheme": {"@id": self.uri}
            }
            
            if term.definition:
                concept["skos:definition"] = {
                    "@value": term.definition,
                    "@language": "en"
                }
            
            if term.notation:
                concept["skos:notation"] = term.notation
            
            if term.broader_terms:
                concept["skos:broader"] = [{"@id": uri} for uri in term.broader_terms]
            
            if term.narrower_terms:
                concept["skos:narrower"] = [{"@id": uri} for uri in term.narrower_terms]
            
            if term.related_terms:
                concept["skos:related"] = [{"@id": uri} for uri in term.related_terms]
            
            if term.alt_labels:
                concept["skos:altLabel"] = [
                    {"@value": label, "@language": "en"}
                    for label in term.alt_labels
                ]
            
            if term.scope_note:
                concept["skos:scopeNote"] = {
                    "@value": term.scope_note,
                    "@language": "en"
                }
            
            skos["skos:concept"].append(concept)
        
        return skos
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get vocabulary statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "uri": self.uri,
            "title": self.title,
            "term_count": len(self.terms),
            "top_concept_count": len(self.top_concepts),
            "ontology_connections": len(self.ontology_classes),
            "terms_with_broader": sum(1 for t in self.terms.values() if t.broader_terms),
            "terms_with_narrower": sum(1 for t in self.terms.values() if t.narrower_terms),
            "terms_with_related": sum(1 for t in self.terms.values() if t.related_terms)
        }
