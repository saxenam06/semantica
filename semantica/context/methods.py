"""
Context Methods Module

This module provides all context engineering methods as simple, reusable functions for
context graph construction, agent memory management, context retrieval, and entity linking.
It supports multiple context engineering approaches and integrates with the method registry
for extensibility.

Supported Methods:

Context Graph Construction:
    - "entities_relationships": Build graph from entities and relationships
    - "conversations": Build graph from conversations
    - "hybrid": Hybrid graph construction combining multiple sources

Agent Memory Management:
    - "store": Store memory items with RAG integration
    - "retrieve": Retrieve memories using vector search
    - "conversation": Conversation history management
    - "hybrid": Hybrid memory retrieval (vector + graph)

Context Retrieval:
    - "vector": Vector-based context retrieval
    - "graph": Graph-based context retrieval
    - "memory": Memory-based context retrieval
    - "hybrid": Hybrid retrieval combining all sources

Entity Linking:
    - "uri": URI assignment for entities
    - "similarity": Similarity-based entity linking
    - "knowledge_graph": Knowledge graph-based linking
    - "cross_document": Cross-document entity linking

Algorithms Used:

Context Graph Construction:
    - Graph Building: Node and edge construction from entities and relationships
    - Entity Extraction: Entity extraction from conversations and text
    - Relationship Extraction: Relationship extraction from conversations
    - Intent Extraction: Intent classification from conversations
    - Sentiment Analysis: Sentiment extraction from conversations
    - Graph Traversal: BFS/DFS for neighbor discovery and multi-hop traversal
    - Graph Indexing: Type-based indexing for efficient node/edge lookup

Agent Memory Management:
    - Vector Embedding: Embedding generation for memory items
    - Vector Search: Similarity search in vector space
    - Keyword Search: Fallback keyword-based search
    - Retention Policy: Time-based memory retention and cleanup
    - Memory Indexing: Deque-based memory index for efficient access
    - Knowledge Graph Integration: Entity and relationship updates to knowledge graph

Context Retrieval:
    - Vector Similarity Search: Cosine similarity in vector space
    - Graph Traversal: Multi-hop graph expansion for related entities
    - Memory Search: Vector and keyword search in memory store
    - Result Ranking: Score-based ranking and merging
    - Deduplication: Content-based result deduplication
    - Hybrid Scoring: Weighted combination of multiple retrieval sources

Entity Linking:
    - URI Generation: Hash-based and text-based URI assignment
    - Text Similarity: Word overlap-based similarity calculation
    - Knowledge Graph Lookup: Entity matching in knowledge graph
    - Cross-Document Linking: Entity linking across multiple documents
    - Bidirectional Linking: Symmetric relationship creation
    - Entity Web Construction: Graph-based entity connection web

Key Features:
    - Multiple context graph construction methods
    - Multiple agent memory management methods
    - Multiple context retrieval methods
    - Multiple entity linking methods
    - Method dispatchers with registry support
    - Custom method registration capability
    - Consistent interface across all methods

Main Functions:
    - build_context_graph: Context graph construction wrapper
    - store_memory: Memory storage wrapper
    - retrieve_context: Context retrieval wrapper
    - link_entities: Entity linking wrapper
    - get_context_method: Get context method by name

Example Usage:
    >>> from semantica.context.methods import build_context_graph, retrieve_context
    >>> graph = build_context_graph(entities, relationships, method="entities_relationships")
    >>> results = retrieve_context("Python programming", method="hybrid", max_results=5)
    >>> from semantica.context.methods import get_context_method
    >>> method = get_context_method("graph", "custom_method")

Author: Semantica Contributors
License: MIT
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..utils.exceptions import ConfigurationError, ProcessingError
from ..utils.logging import get_logger
from .agent_memory import AgentMemory, MemoryItem
from .context_graph import ContextEdge, ContextGraph, ContextNode
from .context_retriever import ContextRetriever, RetrievedContext
from .entity_linker import EntityLink, EntityLinker, LinkedEntity
from .registry import method_registry

logger = get_logger("context_methods")


def build_context_graph(
    entities: Optional[List[Dict[str, Any]]] = None,
    relationships: Optional[List[Dict[str, Any]]] = None,
    conversations: Optional[List[Union[str, Dict[str, Any]]]] = None,
    method: str = "entities_relationships",
    **kwargs,
) -> Dict[str, Any]:
    """
    Build context graph from various sources (convenience function).

    This is a user-friendly wrapper that builds context graphs using the specified method.

    Args:
        entities: List of entity dictionaries
        relationships: List of relationship dictionaries
        conversations: List of conversation files or dictionaries
        method: Graph construction method (default: "entities_relationships")
            - "entities_relationships": Build from entities and relationships
            - "conversations": Build from conversations
            - "hybrid": Hybrid construction combining multiple sources
        **kwargs: Additional options passed to ContextGraph

    Returns:
        Context graph dictionary containing:
            - nodes: List of context nodes
            - edges: List of context edges
            - statistics: Graph statistics

    Examples:
        >>> from semantica.context.methods import build_context_graph
        >>> entities = [{"id": "e1", "text": "Python", "type": "PROGRAMMING_LANGUAGE"}]
        >>> relationships = [{"source_id": "e1", "target_id": "e2", "type": "related_to"}]
        >>> graph = build_context_graph(entities, relationships, method="entities_relationships")
        >>> print(f"Graph has {graph['statistics']['node_count']} nodes")
    """
    # Check for custom method in registry
    custom_method = method_registry.get("graph", method)
    if custom_method:
        try:
            return custom_method(entities, relationships, conversations, **kwargs)
        except Exception as e:
            logger.warning(
                f"Custom method {method} failed: {e}, falling back to default"
            )

    try:
        # Use ContextGraph as the builder
        builder = ContextGraph(**kwargs)

        if method == "entities_relationships":
            if not entities or not relationships:
                raise ProcessingError(
                    "entities and relationships required for entities_relationships method"
                )
            return builder.build_from_entities_and_relationships(
                entities, relationships, **kwargs
            )

        elif method == "conversations":
            if not conversations:
                raise ProcessingError("conversations required for conversations method")
            return builder.build_from_conversations(conversations, **kwargs)

        elif method == "hybrid":
            graph = {}
            if entities and relationships:
                graph1 = builder.build_from_entities_and_relationships(
                    entities, relationships, **kwargs
                )
                graph = graph1
            if conversations:
                graph2 = builder.build_from_conversations(conversations, **kwargs)
                # Merge graphs
                if graph:
                    graph["nodes"].extend(graph2.get("nodes", []))
                    graph["edges"].extend(graph2.get("edges", []))
                else:
                    graph = graph2
            return graph

        else:
            raise ProcessingError(f"Unknown graph construction method: {method}")

    except Exception as e:
        logger.error(f"Failed to build context graph: {e}", exc_info=True)
        raise ProcessingError(f"Context graph construction failed: {e}") from e


def store_memory(
    content: str,
    vector_store: Optional[Any] = None,
    knowledge_graph: Optional[Any] = None,
    method: str = "store",
    **kwargs,
) -> str:
    """
    Store memory item (convenience function).

    This is a user-friendly wrapper that stores memory using the specified method.

    Args:
        content: Memory content
        vector_store: Vector store instance
        knowledge_graph: Knowledge graph instance
        method: Memory storage method (default: "store")
            - "store": Standard memory storage with RAG
            - "conversation": Conversation memory storage
        **kwargs: Additional options passed to AgentMemory

    Returns:
        Memory ID

    Examples:
        >>> from semantica.context.methods import store_memory
        >>> memory_id = store_memory("User asked about Python", vector_store=vs, method="store")
        >>> print(f"Stored memory: {memory_id}")
    """
    # Check for custom method in registry
    custom_method = method_registry.get("memory", method)
    if custom_method:
        try:
            return custom_method(content, vector_store, knowledge_graph, **kwargs)
        except Exception as e:
            logger.warning(
                f"Custom method {method} failed: {e}, falling back to default"
            )

    try:
        memory = AgentMemory(
            vector_store=vector_store, knowledge_graph=knowledge_graph, **kwargs
        )

        metadata = kwargs.get("metadata", {})
        if method == "conversation":
            metadata["type"] = "conversation"

        return memory.store(
            content,
            metadata=metadata,
            entities=kwargs.get("entities"),
            relationships=kwargs.get("relationships"),
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["metadata", "entities", "relationships"]
            },
        )

    except Exception as e:
        logger.error(f"Failed to store memory: {e}", exc_info=True)
        raise ProcessingError(f"Memory storage failed: {e}") from e


def retrieve_context(
    query: str,
    memory_store: Optional[Any] = None,
    knowledge_graph: Optional[Any] = None,
    vector_store: Optional[Any] = None,
    method: str = "hybrid",
    max_results: int = 5,
    **kwargs,
) -> List[RetrievedContext]:
    """
    Retrieve relevant context (convenience function).

    This is a user-friendly wrapper that retrieves context using the specified method.

    Args:
        query: Search query
        memory_store: Memory store instance
        knowledge_graph: Knowledge graph instance
        vector_store: Vector store instance
        method: Retrieval method (default: "hybrid")
            - "vector": Vector-based retrieval only
            - "graph": Graph-based retrieval only
            - "memory": Memory-based retrieval only
            - "hybrid": Hybrid retrieval combining all sources
        max_results: Maximum number of results
        **kwargs: Additional options passed to ContextRetriever

    Returns:
        List of RetrievedContext objects

    Examples:
        >>> from semantica.context.methods import retrieve_context
        >>> results = retrieve_context("Python programming", vector_store=vs, method="hybrid")
        >>> for result in results:
        ...     print(f"{result.content}: {result.score:.2f}")
    """
    # Check for custom method in registry
    custom_method = method_registry.get("retrieval", method)
    if custom_method:
        try:
            return custom_method(
                query,
                memory_store,
                knowledge_graph,
                vector_store,
                max_results,
                **kwargs,
            )
        except Exception as e:
            logger.warning(
                f"Custom method {method} failed: {e}, falling back to default"
            )

    try:
        retriever = ContextRetriever(
            memory_store=memory_store,
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            **kwargs,
        )

        if method == "vector":
            retriever.use_graph_expansion = False
            retriever.memory_store = None
        elif method == "graph":
            retriever.vector_store = None
            retriever.memory_store = None
        elif method == "memory":
            retriever.vector_store = None
            retriever.use_graph_expansion = False

        return retriever.retrieve(query, max_results=max_results, **kwargs)

    except Exception as e:
        logger.error(f"Failed to retrieve context: {e}", exc_info=True)
        raise ProcessingError(f"Context retrieval failed: {e}") from e


def link_entities(
    entities: List[Dict[str, Any]],
    knowledge_graph: Optional[Any] = None,
    method: str = "similarity",
    **kwargs,
) -> List[LinkedEntity]:
    """
    Link entities across sources (convenience function).

    This is a user-friendly wrapper that links entities using the specified method.

    Args:
        entities: List of entity dictionaries
        knowledge_graph: Knowledge graph instance
        method: Linking method (default: "similarity")
            - "uri": URI assignment only
            - "similarity": Similarity-based linking
            - "knowledge_graph": Knowledge graph-based linking
            - "cross_document": Cross-document linking
        **kwargs: Additional options passed to EntityLinker

    Returns:
        List of LinkedEntity objects

    Examples:
        >>> from semantica.context.methods import link_entities
        >>> entities = [{"id": "e1", "text": "Python", "type": "PROGRAMMING_LANGUAGE"}]
        >>> linked = link_entities(entities, knowledge_graph=kg, method="similarity")
        >>> for entity in linked:
        ...     print(f"{entity.text}: {entity.uri}")
    """
    # Check for custom method in registry
    custom_method = method_registry.get("linking", method)
    if custom_method:
        try:
            return custom_method(entities, knowledge_graph, **kwargs)
        except Exception as e:
            logger.warning(
                f"Custom method {method} failed: {e}, falling back to default"
            )

    try:
        linker = EntityLinker(knowledge_graph=knowledge_graph, **kwargs)

        if method == "uri":
            # Just assign URIs
            linked = []
            for entity in entities:
                entity_id = entity.get("id") or entity.get("entity_id")
                entity_text = (
                    entity.get("text") or entity.get("label") or entity.get("name", "")
                )
                entity_type = entity.get("type") or entity.get("entity_type")
                uri = linker.assign_uri(entity_id, entity_text, entity_type)
                linked.append(
                    LinkedEntity(
                        entity_id=entity_id,
                        uri=uri,
                        text=entity_text,
                        type=entity_type or "UNKNOWN",
                        linked_entities=[],
                        context=entity.get("metadata", {}),
                        confidence=entity.get("confidence", 1.0),
                    )
                )
            return linked

        else:
            # Use full linking
            return linker.link(
                text="",  # Not used for entity list
                entities=entities,
                context=kwargs.get("context"),
            )

    except Exception as e:
        logger.error(f"Failed to link entities: {e}", exc_info=True)
        raise ProcessingError(f"Entity linking failed: {e}") from e


def get_context_method(task: str, name: str) -> Optional[Callable]:
    """
    Get a registered context method.

    Args:
        task: Task type ("graph", "memory", "retrieval", "linking")
        name: Method name

    Returns:
        Registered method or None if not found

    Examples:
        >>> from semantica.context.methods import get_context_method
        >>> method = get_context_method("graph", "custom_method")
        >>> if method:
        ...     result = method(entities, relationships)
    """
    return method_registry.get(task, name)


def list_available_methods(task: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List all available context methods.

    Args:
        task: Optional task type filter

    Returns:
        Dictionary mapping task types to method names

    Examples:
        >>> from semantica.context.methods import list_available_methods
        >>> all_methods = list_available_methods()
        >>> graph_methods = list_available_methods("graph")
    """
    return method_registry.list_all(task)


# Register default methods
method_registry.register("graph", "entities_relationships", build_context_graph)
method_registry.register("graph", "conversations", build_context_graph)
method_registry.register("graph", "hybrid", build_context_graph)
method_registry.register("memory", "store", store_memory)
method_registry.register("memory", "conversation", store_memory)
method_registry.register("retrieval", "vector", retrieve_context)
method_registry.register("retrieval", "graph", retrieve_context)
method_registry.register("retrieval", "memory", retrieve_context)
method_registry.register("retrieval", "hybrid", retrieve_context)
method_registry.register("linking", "uri", link_entities)
method_registry.register("linking", "similarity", link_entities)
method_registry.register("linking", "knowledge_graph", link_entities)
method_registry.register("linking", "cross_document", link_entities)
