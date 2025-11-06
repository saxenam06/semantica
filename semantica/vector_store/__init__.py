"""
Vector Store Management Module

This module provides comprehensive vector storage and retrieval capabilities for the
Semantica framework, including support for multiple vector store backends (FAISS,
Pinecone, Weaviate, Qdrant, Milvus), hybrid search combining vector similarity and
metadata filtering, metadata management, and namespace isolation.

Key Features:
    - Multi-backend vector store support (FAISS, Pinecone, Weaviate, Qdrant, Milvus)
    - Vector indexing and similarity search
    - Metadata indexing and filtering
    - Hybrid search combining vector and metadata queries
    - Namespace isolation and multi-tenant support
    - Vector store management and optimization
    - Batch operations and performance optimization

Main Classes:
    - VectorStore: Main vector store interface
    - VectorIndexer: Vector indexing engine
    - VectorRetriever: Vector retrieval and similarity search
    - VectorManager: Vector store management and operations
    - FAISSAdapter: FAISS integration for local vector storage
    - PineconeAdapter: Pinecone cloud vector database integration
    - WeaviateAdapter: Weaviate vector database integration
    - QdrantAdapter: Qdrant vector database integration
    - MilvusAdapter: Milvus vector database integration
    - HybridSearch: Hybrid vector and metadata search
    - MetadataStore: Metadata indexing and management
    - NamespaceManager: Namespace isolation and management

Example Usage:
    >>> from semantica.vector_store import VectorStore, FAISSAdapter
    >>> store = VectorStore(backend="faiss", dimension=768)
    >>> vector_ids = store.store_vectors(vectors, metadata=metadata_list)
    >>> results = store.search_vectors(query_vector, k=10)
    >>> 
    >>> from semantica.vector_store import PineconeAdapter
    >>> adapter = PineconeAdapter(api_key="your-key")
    >>> adapter.connect()
    >>> index = adapter.create_index("my-index", dimension=768)
    >>> adapter.upsert_vectors(vectors, ids, metadata)
    >>> 
    >>> from semantica.vector_store import HybridSearch, MetadataFilter
    >>> search = HybridSearch()
    >>> filter = MetadataFilter().eq("category", "science")
    >>> results = search.search(query_vector, vectors, metadata, vector_ids, filter=filter)

Author: Semantica Contributors
License: MIT
"""

from .vector_store import (
    VectorStore,
    VectorIndexer,
    VectorRetriever,
    VectorManager
)

from .faiss_adapter import (
    FAISSAdapter,
    FAISSIndex,
    FAISSSearch,
    FAISSIndexBuilder
)

from .pinecone_adapter import (
    PineconeAdapter,
    PineconeIndex,
    PineconeQuery,
    PineconeMetadata
)

from .weaviate_adapter import (
    WeaviateAdapter,
    WeaviateClient,
    WeaviateSchema,
    WeaviateQuery
)

from .qdrant_adapter import (
    QdrantAdapter,
    QdrantClient,
    QdrantCollection,
    QdrantSearch
)

from .milvus_adapter import (
    MilvusAdapter,
    MilvusClient,
    MilvusCollection,
    MilvusSearch
)

from .hybrid_search import (
    HybridSearch,
    MetadataFilter,
    SearchRanker
)

from .metadata_store import (
    MetadataStore,
    MetadataIndex,
    MetadataSchema
)

from .namespace_manager import (
    NamespaceManager,
    Namespace
)

__all__ = [
    # Core vector store
    "VectorStore",
    "VectorIndexer",
    "VectorRetriever",
    "VectorManager",
    # FAISS
    "FAISSAdapter",
    "FAISSIndex",
    "FAISSSearch",
    "FAISSIndexBuilder",
    # Pinecone
    "PineconeAdapter",
    "PineconeIndex",
    "PineconeQuery",
    "PineconeMetadata",
    # Weaviate
    "WeaviateAdapter",
    "WeaviateClient",
    "WeaviateSchema",
    "WeaviateQuery",
    # Qdrant
    "QdrantAdapter",
    "QdrantClient",
    "QdrantCollection",
    "QdrantSearch",
    # Milvus
    "MilvusAdapter",
    "MilvusClient",
    "MilvusCollection",
    "MilvusSearch",
    # Hybrid search
    "HybridSearch",
    "MetadataFilter",
    "SearchRanker",
    # Metadata store
    "MetadataStore",
    "MetadataIndex",
    "MetadataSchema",
    # Namespace manager
    "NamespaceManager",
    "Namespace",
]
