"""
Vocabulary Manager for Semantica framework.

Manages controlled vocabularies that supplement ontologies with hierarchical
terminology, enabling concise modeling when many concepts share attributes
but don't need unique modeling.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .controlled_vocabulary import ControlledVocabulary, VocabularyTerm
from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.helpers import read_json_file, write_json_file


class VocabularyManager:
    """
    Manager for controlled vocabularies.
    
    • Creates and manages multiple controlled vocabularies
    • Builds hierarchical term structures
    • Connects vocabularies to ontology classes
    • Supports multi-language terms
    • Manages synonyms and alternative labels
    • Exports vocabularies in SKOS format
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize vocabulary manager.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("vocabulary_manager")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.vocabularies: Dict[str, ControlledVocabulary] = {}
    
    def create_vocabulary(
        self,
        uri: str,
        title: str,
        description: Optional[str] = None,
        **metadata
    ) -> ControlledVocabulary:
        """
        Create a new controlled vocabulary.
        
        Args:
            uri: Vocabulary URI
            title: Vocabulary title
            description: Vocabulary description
            **metadata: Additional metadata
            
        Returns:
            Created ControlledVocabulary
            
        Raises:
            ValidationError: If vocabulary with URI already exists
        """
        if uri in self.vocabularies:
            raise ValidationError(f"Vocabulary with URI '{uri}' already exists")
        
        vocabulary = ControlledVocabulary(
            uri=uri,
            title=title,
            description=description,
            **metadata
        )
        
        self.vocabularies[uri] = vocabulary
        self.logger.info(f"Created vocabulary: {title} ({uri})")
        
        return vocabulary
    
    def get_vocabulary(self, uri: str) -> Optional[ControlledVocabulary]:
        """
        Get vocabulary by URI.
        
        Args:
            uri: Vocabulary URI
            
        Returns:
            ControlledVocabulary or None if not found
        """
        return self.vocabularies.get(uri)
    
    def remove_vocabulary(self, uri: str) -> bool:
        """
        Remove vocabulary.
        
        Args:
            uri: Vocabulary URI
            
        Returns:
            True if vocabulary removed successfully
        """
        if uri not in self.vocabularies:
            return False
        
        del self.vocabularies[uri]
        self.logger.info(f"Removed vocabulary: {uri}")
        return True
    
    def list_vocabularies(self) -> List[str]:
        """
        List all vocabulary URIs.
        
        Returns:
            List of vocabulary URIs
        """
        return list(self.vocabularies.keys())
    
    def add_term_to_vocabulary(
        self,
        vocabulary_uri: str,
        term: VocabularyTerm,
        top_concept: bool = False
    ) -> bool:
        """
        Add term to vocabulary.
        
        Args:
            vocabulary_uri: Vocabulary URI
            term: Vocabulary term
            top_concept: Whether this is a top-level concept
            
        Returns:
            True if term added successfully
        """
        vocabulary = self.get_vocabulary(vocabulary_uri)
        if not vocabulary:
            raise ValidationError(f"Vocabulary '{vocabulary_uri}' not found")
        
        return vocabulary.add_term(term, top_concept=top_concept)
    
    def search_across_vocabularies(
        self,
        query: str,
        vocabulary_uris: Optional[List[str]] = None,
        search_labels: bool = True,
        search_definitions: bool = True
    ) -> Dict[str, List[str]]:
        """
        Search for terms across vocabularies.
        
        Args:
            query: Search query
            vocabulary_uris: List of vocabulary URIs to search (None = all)
            search_labels: Search in labels
            search_definitions: Search in definitions
            
        Returns:
            Dictionary mapping vocabulary URI to list of matching term URIs
        """
        results = {}
        
        vocab_uris = vocabulary_uris or self.list_vocabularies()
        
        for vocab_uri in vocab_uris:
            vocabulary = self.get_vocabulary(vocab_uri)
            if vocabulary:
                matches = vocabulary.search_terms(
                    query,
                    search_labels=search_labels,
                    search_definitions=search_definitions
                )
                if matches:
                    results[vocab_uri] = matches
        
        return results
    
    def connect_term_to_class(
        self,
        vocabulary_uri: str,
        term_uri: str,
        class_uri: str
    ) -> bool:
        """
        Connect vocabulary term to ontology class.
        
        Args:
            vocabulary_uri: Vocabulary URI
            term_uri: Term URI
            class_uri: Ontology class URI
            
        Returns:
            True if connection successful
        """
        vocabulary = self.get_vocabulary(vocabulary_uri)
        if not vocabulary:
            raise ValidationError(f"Vocabulary '{vocabulary_uri}' not found")
        
        return vocabulary.connect_to_ontology_class(term_uri, class_uri)
    
    def export_vocabulary(
        self,
        vocabulary_uri: str,
        file_path: Union[str, Path],
        format: str = "skos"
    ) -> None:
        """
        Export vocabulary to file.
        
        Args:
            vocabulary_uri: Vocabulary URI
            file_path: Output file path
            format: Export format ('skos', 'json')
        """
        vocabulary = self.get_vocabulary(vocabulary_uri)
        if not vocabulary:
            raise ValidationError(f"Vocabulary '{vocabulary_uri}' not found")
        
        file_path = Path(file_path)
        
        if format == "skos":
            skos_data = vocabulary.to_skos()
            write_json_file(skos_data, file_path)
        
        elif format == "json":
            vocab_data = {
                "uri": vocabulary.uri,
                "title": vocabulary.title,
                "description": vocabulary.description,
                "metadata": vocabulary.metadata,
                "top_concepts": vocabulary.top_concepts,
                "terms": [
                    {
                        "uri": term.uri,
                        "label": term.label,
                        "definition": term.definition,
                        "notation": term.notation,
                        "broader_terms": term.broader_terms,
                        "narrower_terms": term.narrower_terms,
                        "related_terms": term.related_terms,
                        "alt_labels": term.alt_labels,
                        "scope_note": term.scope_note,
                        "examples": term.examples,
                        "metadata": term.metadata
                    }
                    for term in vocabulary.terms.values()
                ],
                "ontology_classes": vocabulary.ontology_classes
            }
            write_json_file(vocab_data, file_path)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported vocabulary {vocabulary_uri} to {file_path}")
    
    def import_vocabulary(
        self,
        file_path: Union[str, Path],
        format: str = "json"
    ) -> ControlledVocabulary:
        """
        Import vocabulary from file.
        
        Args:
            file_path: Input file path
            format: Import format ('skos', 'json')
            
        Returns:
            Imported ControlledVocabulary
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")
        
        if format == "json":
            data = read_json_file(file_path)
            
            vocabulary = ControlledVocabulary(
                uri=data["uri"],
                title=data["title"],
                description=data.get("description"),
                **data.get("metadata", {})
            )
            
            # Import terms
            for term_data in data.get("terms", []):
                term = VocabularyTerm(
                    uri=term_data["uri"],
                    label=term_data["label"],
                    definition=term_data.get("definition"),
                    notation=term_data.get("notation"),
                    broader_terms=term_data.get("broader_terms", []),
                    narrower_terms=term_data.get("narrower_terms", []),
                    related_terms=term_data.get("related_terms", []),
                    alt_labels=term_data.get("alt_labels", []),
                    scope_note=term_data.get("scope_note"),
                    examples=term_data.get("examples", []),
                    metadata=term_data.get("metadata", {})
                )
                vocabulary.add_term(term, top_concept=term_data["uri"] in data.get("top_concepts", []))
            
            # Import ontology connections
            for term_uri, class_uri in data.get("ontology_classes", {}).items():
                vocabulary.ontology_classes[term_uri] = class_uri
            
            self.vocabularies[vocabulary.uri] = vocabulary
            self.logger.info(f"Imported vocabulary: {vocabulary.title}")
            
            return vocabulary
        
        elif format == "skos":
            # SKOS import would require parsing SKOS JSON-LD
            raise NotImplementedError("SKOS import not yet fully implemented")
        
        else:
            raise ValueError(f"Unsupported import format: {format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all vocabularies.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "vocabulary_count": len(self.vocabularies),
            "total_terms": sum(len(v.terms) for v in self.vocabularies.values()),
            "vocabularies": {}
        }
        
        for vocab_uri, vocabulary in self.vocabularies.items():
            stats["vocabularies"][vocab_uri] = vocabulary.get_statistics()
        
        return stats
