"""
Semantic Extraction Module

This module provides comprehensive semantic extraction capabilities for knowledge
engineering, enabling extraction of entities, relations, events, triples, and
semantic networks from text.

Key Features:
    - Named entity recognition (NER) with multiple implementations
    - Relationship extraction between entities
    - Event detection and temporal processing
    - Coreference resolution for pronouns and entity references
    - RDF triple extraction and serialization
    - Semantic analysis and role labeling
    - Semantic network construction
    - LLM-based extraction enhancement
    - Extraction validation and quality assessment

Main Classes:
    - NamedEntityRecognizer: Main NER coordinator
    - NERExtractor: Core NER implementation
    - RelationExtractor: Relationship extraction
    - EventDetector: Event detection and classification
    - CoreferenceResolver: Coreference resolution
    - TripleExtractor: RDF triple extraction
    - SemanticAnalyzer: Semantic analysis engine
    - SemanticNetworkExtractor: Semantic network construction
    - LLMEnhancer: LLM-based enhancement
    - ExtractionValidator: Quality validation

Example Usage:
    >>> from semantica.semantic_extract import build
    >>> result = build("Apple Inc. was founded by Steve Jobs in 1976.")
    >>> print(f"Extracted {len(result['entities'])} entities")
    >>> from semantica.semantic_extract import NamedEntityRecognizer
    >>> ner = NamedEntityRecognizer()
    >>> entities = ner.extract_entities("Steve Jobs founded Apple.")

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional, Union

from .named_entity_recognizer import (
    NamedEntityRecognizer,
    EntityClassifier,
    EntityConfidenceScorer,
    CustomEntityDetector
)
from .ner_extractor import (
    NERExtractor,
    Entity
)
from .relation_extractor import (
    RelationExtractor,
    Relation
)
from .event_detector import (
    EventDetector,
    Event,
    EventClassifier,
    TemporalEventProcessor,
    EventRelationshipExtractor
)
from .coreference_resolver import (
    CoreferenceResolver,
    Mention,
    CoreferenceChain,
    PronounResolver,
    EntityCoreferenceDetector,
    CoreferenceChainBuilder
)
from .triple_extractor import (
    TripleExtractor,
    Triple,
    TripleValidator,
    RDFSerializer,
    TripleQualityChecker
)
from .semantic_analyzer import (
    SemanticAnalyzer,
    SemanticRole,
    SemanticCluster,
    SimilarityAnalyzer,
    RoleLabeler,
    SemanticClusterer
)
from .semantic_network_extractor import (
    SemanticNetworkExtractor,
    SemanticNode,
    SemanticEdge,
    SemanticNetwork
)
from .llm_enhancer import (
    LLMEnhancer,
    LLMResponse
)
from .extraction_validator import (
    ExtractionValidator,
    ValidationResult
)
from .providers import (
    BaseProvider,
    OpenAIProvider,
    GeminiProvider,
    GroqProvider,
    AnthropicProvider,
    OllamaProvider,
    HuggingFaceLLMProvider,
    HuggingFaceModelLoader,
    create_provider
)
from .registry import (
    ProviderRegistry,
    MethodRegistry,
    provider_registry,
    method_registry
)
from .config import Config, config
from .methods import (
    get_entity_method,
    get_relation_method,
    get_triple_method
)

__all__ = [
    # Named Entity Recognition
    "NamedEntityRecognizer",
    "EntityClassifier",
    "EntityConfidenceScorer",
    "CustomEntityDetector",
    "NERExtractor",
    "Entity",
    
    # Relation Extraction
    "RelationExtractor",
    "Relation",
    
    # Event Detection
    "EventDetector",
    "Event",
    "EventClassifier",
    "TemporalEventProcessor",
    "EventRelationshipExtractor",
    
    # Coreference Resolution
    "CoreferenceResolver",
    "Mention",
    "CoreferenceChain",
    "PronounResolver",
    "EntityCoreferenceDetector",
    "CoreferenceChainBuilder",
    
    # Triple Extraction
    "TripleExtractor",
    "Triple",
    "TripleValidator",
    "RDFSerializer",
    "TripleQualityChecker",
    
    # Semantic Analysis
    "SemanticAnalyzer",
    "SemanticRole",
    "SemanticCluster",
    "SimilarityAnalyzer",
    "RoleLabeler",
    "SemanticClusterer",
    
    # Semantic Network
    "SemanticNetworkExtractor",
    "SemanticNode",
    "SemanticEdge",
    "SemanticNetwork",
    
    # LLM Enhancement
    "LLMEnhancer",
    "LLMResponse",
    
    # Validation
    "ExtractionValidator",
    "ValidationResult",
    
    # Providers
    "BaseProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "GroqProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "HuggingFaceLLMProvider",
    "HuggingFaceModelLoader",
    "create_provider",
    
    # Registry
    "ProviderRegistry",
    "MethodRegistry",
    "provider_registry",
    "method_registry",
    
    # Config
    "Config",
    "config",
    
    # Methods
    "get_entity_method",
    "get_relation_method",
    "get_triple_method",
    
    # Convenience
    "build",
]


def build(
    text: Union[str, List[str]],
    extract_entities: bool = True,
    extract_relations: bool = True,
    extract_events: bool = False,
    extract_triples: bool = False,
    resolve_coreferences: bool = False,
    **options
) -> Dict[str, Any]:
    """
    Extract semantic information from text (module-level convenience function).
    
    This is a user-friendly wrapper that performs comprehensive semantic extraction
    including entities, relations, events, and triples.
    
    Args:
        text: Input text or list of texts to process
        extract_entities: Whether to extract named entities (default: True)
        extract_relations: Whether to extract relationships (default: True)
        extract_events: Whether to extract events (default: False)
        extract_triples: Whether to extract RDF triples (default: False)
        resolve_coreferences: Whether to resolve coreferences (default: False)
        **options: Additional extraction options
        
    Returns:
        Dictionary containing:
            - entities: List of extracted entities
            - relations: List of extracted relationships
            - events: List of extracted events (if enabled)
            - triples: List of extracted triples (if enabled)
            - coreferences: Coreference resolution results (if enabled)
            - metadata: Extraction metadata
            - statistics: Extraction statistics
            
    Examples:
        >>> import semantica
        >>> result = semantica.semantic_extract.build(
        ...     text="Apple Inc. was founded by Steve Jobs in 1976.",
        ...     extract_entities=True,
        ...     extract_relations=True
        ... )
        >>> print(f"Extracted {len(result['entities'])} entities")
    """
    # Normalize text to list
    is_single = isinstance(text, str)
    if is_single:
        texts = [text]
    else:
        texts = text
    
    results = {
        "entities": [],
        "relations": [],
        "events": [],
        "triples": [],
        "coreferences": [],
        "metadata": {},
        "statistics": {}
    }
    
    # Initialize extractors
    if extract_entities:
        ner = NamedEntityRecognizer(config=options.get("ner_config", {}), **options)
    
    if extract_relations:
        from .relation_extractor import RelationExtractor
        rel_extractor = RelationExtractor(**options.get("relation_config", {}), **options)
    
    if extract_events:
        from .event_detector import EventDetector
        event_detector = EventDetector(**options.get("event_config", {}), **options)
    
    if extract_triples:
        from .triple_extractor import TripleExtractor
        triple_extractor = TripleExtractor(**options.get("triple_config", {}), **options)
    
    if resolve_coreferences:
        from .coreference_resolver import CoreferenceResolver
        coref_resolver = CoreferenceResolver(**options.get("coref_config", {}), **options)
    
    # Process texts
    all_entities = []
    all_relations = []
    all_events = []
    all_triples = []
    
    for txt in texts:
        if extract_entities:
            entities = ner.extract_entities(txt, **options)
            all_entities.extend(entities)
        
        if extract_relations:
            # Relations typically need entities, so extract if not already done
            if not extract_entities:
                entities = ner.extract_entities(txt, **options) if 'ner' in locals() else []
            relations = rel_extractor.extract_relations(txt, entities=entities if extract_entities else [], **options)
            all_relations.extend(relations)
        
        if extract_events:
            events = event_detector.detect_events(txt, **options)
            all_events.extend(events)
        
        if extract_triples:
            triples = triple_extractor.extract_triples(txt, **options)
            all_triples.extend(triples)
        
        if resolve_coreferences:
            corefs = coref_resolver.resolve(txt, **options)
            results["coreferences"].append(corefs)
    
    results["entities"] = all_entities
    results["relations"] = all_relations
    results["events"] = all_events
    results["triples"] = all_triples
    
    results["statistics"] = {
        "texts_processed": len(texts),
        "entities_extracted": len(all_entities),
        "relations_extracted": len(all_relations),
        "events_extracted": len(all_events),
        "triples_extracted": len(all_triples)
    }
    
    results["metadata"] = {
        "extract_entities": extract_entities,
        "extract_relations": extract_relations,
        "extract_events": extract_events,
        "extract_triples": extract_triples,
        "resolve_coreferences": resolve_coreferences
    }
    
    return results
