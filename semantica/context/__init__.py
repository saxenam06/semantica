"""
Context Engineering Module

This module provides comprehensive context engineering infrastructure for agents,
formalizing context as a graph of connections to enable meaningful agent
understanding and memory. It integrates RAG with knowledge graphs to provide
persistent context for intelligent agents.

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
    - High-level interface (AgentContext) for easy use
    - Context graph construction from entities, relationships, and conversations
    - Agent memory management with RAG integration
    - Entity linking across sources with URI assignment
    - Hybrid context retrieval (vector + graph + memory)
    - Conversation history management
    - Context accumulation and synthesis
    - Graph-based context traversal and querying
    - Method registry for custom context methods
    - Configuration management with environment variables and config files
    - Boolean flags for common options (user-friendly)
    - Auto-detection of content types and retrieval strategies

Main Classes:
    - AgentContext: High-level interface for agent context management (store, retrieve, forget, conversation)
    - ContextGraphBuilder: Builds context graphs from various sources
    - ContextNode: Context graph node data structure
    - ContextEdge: Context graph edge data structure
    - AgentMemory: Manages persistent agent memory with RAG
    - MemoryItem: Memory item data structure
    - EntityLinker: Links entities across sources with URIs
    - EntityLink: Entity link data structure
    - LinkedEntity: Linked entity with context
    - ContextRetriever: Retrieves relevant context from multiple sources
    - RetrievedContext: Retrieved context item data structure
    - MethodRegistry: Registry for custom context methods (accessed via registry submodule)
    - ContextConfig: Configuration manager for context module (accessed via config submodule)

Submodules:
    - methods: Context engineering methods (build_context_graph, store_memory, retrieve_context, etc.)
    - registry: Method registry for custom methods (method_registry, MethodRegistry)
    - config: Configuration management (context_config, ContextConfig)

Example Usage:
    >>> # High-level interface (recommended for most users)
    >>> from semantica.context import AgentContext
    >>> context = AgentContext(vector_store=vs, knowledge_graph=kg)
    >>> memory_id = context.store("User asked about Python", conversation_id="conv1")
    >>> results = context.retrieve("Python programming")
    >>> stats = context.store(["Doc 1", "Doc 2"], extract_entities=True)
    
    >>> # Low-level classes (for advanced use cases)
    >>> from semantica.context import ContextGraphBuilder, AgentMemory, methods
    >>> builder = ContextGraphBuilder()
    >>> graph = builder.build_from_entities_and_relationships(entities, relationships)
    >>> memory = AgentMemory(vector_store=vs, knowledge_graph=kg)
    >>> memory_id = memory.store("User asked about Python", metadata={"type": "conversation"})
    >>> results = memory.retrieve("Python", max_results=5)
    >>> # Using methods submodule
    >>> graph = methods.build_context_graph(entities, relationships, method="entities_relationships")

Author: Semantica Contributors
License: MIT
"""

from .agent_context import AgentContext
from .agent_memory import AgentMemory, MemoryItem
from .context_graph import ContextEdge, ContextGraph, ContextNode
from .context_retriever import ContextRetriever, RetrievedContext
from .entity_linker import EntityLink, EntityLinker, LinkedEntity
from . import methods
from . import registry
from . import config

__all__ = [
    # High-level interface
    "AgentContext",
    # Main classes
    "ContextGraph",
    "ContextNode",
    "ContextEdge",
    "EntityLinker",
    "EntityLink",
    "LinkedEntity",
    "AgentMemory",
    "MemoryItem",
    "ContextRetriever",
    "RetrievedContext",
    # Submodules
    "methods",
    "registry",
    "config",
]

# Backward compatibility alias
ContextGraphBuilder = ContextGraph
