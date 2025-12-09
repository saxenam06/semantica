"""
Context Retriever for Agents

This module provides comprehensive context retrieval capabilities for agents,
retrieving relevant context from memory, knowledge graphs, and vector stores
to inform decision-making. It supports hybrid retrieval combining multiple
sources for optimal context relevance.

Algorithms Used:

Vector Retrieval:
    - Vector Similarity Search: Cosine similarity search in vector space
    - Query Embedding: Embedding generation for search queries
    - Top-K Retrieval: Top-K result selection based on similarity scores

Graph Retrieval:
    - Keyword Matching: Word overlap-based node matching
    - Graph Traversal: Multi-hop graph expansion for related entities
    - Relevance Scoring: Word overlap-based relevance calculation
    - Entity Expansion: BFS-based entity relationship traversal

Memory Retrieval:
    - Memory Search: Vector and keyword search in memory store
    - Conversation History: Temporal-based memory retrieval

Result Processing:
    - Result Ranking: Score-based ranking and merging
    - Deduplication: Content-based result deduplication
    - Score Aggregation: Maximum score selection for duplicate results
    - Metadata Merging: Dictionary-based metadata merging
    - Entity Merging: Set-based entity deduplication

Key Features:
    - Retrieve context from multiple sources (memory, graph, vector)
    - Hybrid retrieval (vector + graph + memory) with weighted combination
    - Context relevance ranking and scoring
    - Context aggregation and synthesis
    - Ontology-aware context retrieval
    - Real-time context updates
    - Graph expansion for related entities
    - Multi-hop relationship traversal
    - Result deduplication and merging
    - Configurable retrieval strategies

Main Classes:
    - RetrievedContext: Retrieved context item data structure with content, score, source, metadata, related_entities, related_relationships
    - ContextRetriever: Context retriever for hybrid retrieval

Example Usage:
    >>> from semantica.context import ContextRetriever
    >>> retriever = ContextRetriever(memory_store=mem, knowledge_graph=kg, vector_store=vs)
    >>> results = retriever.retrieve("Python programming", max_results=5)
    >>> for result in results:
    ...     print(f"{result.content}: {result.score:.2f}")
    ...     print(f"Related entities: {len(result.related_entities)}")

Author: Semantica Contributors
License: MIT
"""

from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker


@dataclass
class RetrievedContext:
    """Retrieved context item."""

    content: str
    score: float
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_entities: List[Dict[str, Any]] = field(default_factory=list)
    related_relationships: List[Dict[str, Any]] = field(default_factory=list)


class ContextRetriever:
    """
    Context retriever for hybrid retrieval.

    • Retrieve context from multiple sources
    • Hybrid retrieval (vector + graph)
    • Context relevance ranking
    • Context aggregation and synthesis
    • Ontology-aware context retrieval
    • Real-time context updates
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize context retriever.

        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - memory_store: Memory store instance
                - knowledge_graph: Knowledge graph instance
                - vector_store: Vector store instance
                - use_graph_expansion: Use graph expansion (default: True)
                - max_expansion_hops: Maximum graph expansion hops (default: 2)
                - hybrid_alpha: Weight for hybrid retrieval (0=vector only, 1=graph only, default: 0.5)
        """
        self.logger = get_logger("context_retriever")
        self.config = config or {}
        self.config.update(kwargs)

        self.memory_store = self.config.get("memory_store")
        self.knowledge_graph = self.config.get("knowledge_graph")
        self.vector_store = self.config.get("vector_store")

        self.use_graph_expansion = self.config.get("use_graph_expansion", True)
        self.max_expansion_hops = self.config.get("max_expansion_hops", 2)
        self.hybrid_alpha = self.config.get("hybrid_alpha", 0.5)

        # Initialize progress tracker
        self.progress_tracker = get_progress_tracker()

    def retrieve(
        self,
        query: str,
        max_results: int = 5,
        use_graph_expansion: Optional[bool] = None,
        min_relevance_score: float = 0.0,
        **options,
    ) -> List[RetrievedContext]:
        """
        Retrieve relevant context for query.

        Args:
            query: Search query
            max_results: Maximum number of results
            use_graph_expansion: Use graph expansion (overrides config)
            min_relevance_score: Minimum relevance score
            **options: Additional options:
                - entity_ids: Filter by entity IDs
                - node_types: Filter by node types
                - max_hops: Maximum expansion hops

        Returns:
            List of retrieved context items
        """
        # Track context retrieval
        tracking_id = self.progress_tracker.start_tracking(
            file=None,
            module="context",
            submodule="ContextRetriever",
            message=f"Retrieving context for: {query[:50]}...",
        )

        try:
            use_expansion = (
                use_graph_expansion
                if use_graph_expansion is not None
                else self.use_graph_expansion
            )

            all_results = []

            # Vector-based retrieval
            self.progress_tracker.update_tracking(
                tracking_id, message="Retrieving from vector store..."
            )
            vector_results = self._retrieve_from_vector(query, max_results * 2)
            all_results.extend(vector_results)

            # Graph-based retrieval
            if self.knowledge_graph and use_expansion:
                self.progress_tracker.update_tracking(
                    tracking_id, message="Retrieving from knowledge graph..."
                )
                graph_results = self._retrieve_from_graph(
                    query,
                    max_results * 2,
                    max_hops=options.get("max_hops", self.max_expansion_hops),
                )
                all_results.extend(graph_results)

            # Memory-based retrieval
            if self.memory_store:
                self.progress_tracker.update_tracking(
                    tracking_id, message="Retrieving from memory..."
                )
                memory_results = self._retrieve_from_memory(query, max_results * 2)
                all_results.extend(memory_results)

            # Combine and rank results
            self.progress_tracker.update_tracking(
                tracking_id, message="Ranking and merging results..."
            )
            ranked_results = self._rank_and_merge(all_results, query)

            # Filter by minimum score
            filtered_results = [
                r for r in ranked_results if r.score >= min_relevance_score
            ]

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Retrieved {len(filtered_results[:max_results])} results",
            )
            # Return top results
            return filtered_results[:max_results]

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def _retrieve_from_vector(
        self, query: str, max_results: int
    ) -> List[RetrievedContext]:
        """Retrieve from vector store."""
        if not self.vector_store:
            return []

        try:
            # Simulate vector search (actual implementation would use vector_store)
            results = []

            # If vector_store has a search method
            if hasattr(self.vector_store, "search"):
                search_results = self.vector_store.search(
                    query=query, limit=max_results
                )

                for result in search_results:
                    # Handle VectorSearchResult object or dict
                    if hasattr(result, "content"):
                        content = result.content
                        score = result.score
                        source = f"vector:{result.id}"
                        metadata = result.metadata or {}
                    else:
                        content = result.get("content", "")
                        score = result.get("score", 0.0)
                        source = result.get("source")
                        metadata = result.get("metadata", {})

                    results.append(
                        RetrievedContext(
                            content=content,
                            score=score,
                            source=source,
                            metadata=metadata,
                        )
                    )

            return results

        except Exception as e:
            self.logger.warning(f"Vector retrieval failed: {e}")
            return []

    def _retrieve_from_graph(
        self, query: str, max_results: int, max_hops: int = 2
    ) -> List[RetrievedContext]:
        """Retrieve from knowledge graph."""
        if not self.knowledge_graph:
            return []

        results = []

        try:
            # Check if knowledge_graph implements GraphStore protocol (has query method)
            if hasattr(self.knowledge_graph, "query"):
                graph_results = self.knowledge_graph.query(query)
                
                for res in graph_results:
                    # Handle both interface dicts and raw dicts
                    node = res.get("node")
                    if hasattr(node, "id"): # GraphNodeInterface
                        node_id = node.id
                        node_type = node.type
                        content = node.properties.get("content", "")
                        metadata = node.properties
                    else: # Raw dict
                        node_id = res.get("id") or res.get("node", {}).get("id")
                        node_type = res.get("type") or res.get("node", {}).get("type")
                        content = res.get("content") or res.get("node", {}).get("content")
                        metadata = res.get("metadata") or res.get("node", {}).get("metadata")

                    score = res.get("score", 0.0)
                    
                    # Get related entities
                    related_entities = self._get_related_entities(
                        node_id, max_hops=max_hops
                    )

                    results.append(
                        RetrievedContext(
                            content=content,
                            score=score,
                            source=f"graph:{node_id}",
                            metadata={
                                "node_type": node_type,
                                "node_id": node_id,
                                **(metadata or {}),
                            },
                            related_entities=related_entities,
                        )
                    )
                
                # Sort by score
                results.sort(key=lambda x: x.score, reverse=True)
                return results[:max_results]

            # Fallback to dictionary-based graph retrieval
            # Extract entities from query (simplified)
            query_lower = query.lower()

            # Search for nodes matching query
            nodes = self.knowledge_graph.get("nodes", [])

            for node in nodes:
                node_content = node.get("content", "").lower()
                node_type = node.get("type", "")

                # Simple keyword matching
                if any(word in node_content for word in query_lower.split()):
                    score = self._calculate_graph_relevance(node, query)

                    # Get related entities
                    related_entities = self._get_related_entities(
                        node.get("id"), max_hops=max_hops
                    )

                    results.append(
                        RetrievedContext(
                            content=node.get("content", ""),
                            score=score,
                            source=f"graph:{node.get('id')}",
                            metadata={
                                "node_type": node_type,
                                "node_id": node.get("id"),
                                **node.get("metadata", {}),
                            },
                            related_entities=related_entities,
                        )
                    )

            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)

            return results[:max_results]

        except Exception as e:
            self.logger.warning(f"Graph retrieval failed: {e}")
            return []

    def _retrieve_from_memory(
        self, query: str, max_results: int
    ) -> List[RetrievedContext]:
        """Retrieve from memory store."""
        if not self.memory_store:
            return []

        try:
            # If memory_store has a retrieve method
            if hasattr(self.memory_store, "retrieve"):
                memory_results = self.memory_store.retrieve(
                    query=query, max_results=max_results
                )

                results = []
                for result in memory_results:
                    results.append(
                        RetrievedContext(
                            content=result.get("content", ""),
                            score=result.get("score", 0.0),
                            source=result.get("source"),
                            metadata=result.get("metadata", {}),
                        )
                    )

                return results

            return []

        except Exception as e:
            self.logger.warning(f"Memory retrieval failed: {e}")
            return []

    def _rank_and_merge(
        self, results: List[RetrievedContext], query: str
    ) -> List[RetrievedContext]:
        """Rank and merge results from multiple sources."""
        # Deduplicate by content
        seen_content = {}

        for result in results:
            content_key = result.content[:100]  # First 100 chars as key

            if content_key not in seen_content:
                seen_content[content_key] = result
            else:
                # Merge scores (take maximum or average)
                existing = seen_content[content_key]
                existing.score = max(existing.score, result.score)

                # Merge metadata
                existing.metadata.update(result.metadata)

                # Merge related entities
                existing_entity_ids = {e.get("id") for e in existing.related_entities}
                for entity in result.related_entities:
                    if entity.get("id") not in existing_entity_ids:
                        existing.related_entities.append(entity)

        # Sort by score
        ranked = sorted(seen_content.values(), key=lambda x: x.score, reverse=True)

        return ranked

    def _calculate_graph_relevance(self, node: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for graph node."""
        content = node.get("content", "").lower()
        query_lower = query.lower()

        # Simple word overlap
        query_words = set(query_lower.split())
        content_words = set(content.split())

        if not query_words:
            return 0.0

        overlap = len(query_words & content_words)
        return overlap / len(query_words)

    def _get_related_entities(
        self, node_id: str, max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """Get related entities from graph."""
        if not self.knowledge_graph:
            return []

        # Check if knowledge_graph implements GraphStore protocol
        if hasattr(self.knowledge_graph, "get_neighbors"):
            return self.knowledge_graph.get_neighbors(node_id, hops=max_hops)

        related = []
        visited = set()
        current_level = {node_id}

        for hop in range(max_hops):
            next_level = set()

            for current_id in current_level:
                if current_id in visited:
                    continue
                visited.add(current_id)

                # Find edges
                edges = self.knowledge_graph.get("edges", [])
                for edge in edges:
                    if edge.get("source") == current_id:
                        target_id = edge.get("target")
                        if target_id not in visited:
                            next_level.add(target_id)

                            # Get node info
                            nodes = self.knowledge_graph.get("nodes", [])
                            for node in nodes:
                                if node.get("id") == target_id:
                                    related.append(
                                        {
                                            "id": target_id,
                                            "type": node.get("type"),
                                            "content": node.get("content"),
                                            "relationship": edge.get("type"),
                                            "hop": hop + 1,
                                        }
                                    )

                    elif edge.get("target") == current_id:
                        source_id = edge.get("source")
                        if source_id not in visited:
                            next_level.add(source_id)

                            nodes = self.knowledge_graph.get("nodes", [])
                            for node in nodes:
                                if node.get("id") == source_id:
                                    related.append(
                                        {
                                            "id": source_id,
                                            "type": node.get("type"),
                                            "content": node.get("content"),
                                            "relationship": edge.get("type"),
                                            "hop": hop + 1,
                                        }
                                    )

            current_level = next_level

        return related

    # Search Methods
    def search(self, query: str, **options) -> List[RetrievedContext]:
        """
        Simple search (alias for retrieve).
        
        Args:
            query: Search query
            **options: Additional options
        
        Returns:
            List of RetrievedContext objects
        
        Example:
            >>> results = retriever.search("Python", max_results=10)
        """
        return self.retrieve(query, **options)

    def vector_search(self, query: str, **options) -> List[RetrievedContext]:
        """
        Vector-only search.
        
        Args:
            query: Search query
            **options: Additional options
        
        Returns:
            List of RetrievedContext objects from vector store
        
        Example:
            >>> results = retriever.vector_search("Python")
        """
        # Temporarily disable graph and memory
        original_graph = self.knowledge_graph
        original_memory = self.memory_store
        
        self.knowledge_graph = None
        self.memory_store = None
        
        try:
            results = self.retrieve(query, use_graph_expansion=False, **options)
        finally:
            self.knowledge_graph = original_graph
            self.memory_store = original_memory
        
        return results

    def graph_search(self, query: str, **options) -> List[RetrievedContext]:
        """
        Graph-only search.
        
        Args:
            query: Search query
            **options: Additional options
        
        Returns:
            List of RetrievedContext objects from graph
        
        Example:
            >>> results = retriever.graph_search("Python")
        """
        if not self.knowledge_graph:
            return []
        
        # Temporarily disable vector and memory
        original_vector = self.vector_store
        original_memory = self.memory_store
        
        self.vector_store = None
        self.memory_store = None
        
        try:
            results = self.retrieve(query, use_graph_expansion=True, **options)
        finally:
            self.vector_store = original_vector
            self.memory_store = original_memory
        
        return results

    def memory_search(self, query: str, **options) -> List[RetrievedContext]:
        """
        Memory-only search.
        
        Args:
            query: Search query
            **options: Additional options
        
        Returns:
            List of RetrievedContext objects from memory
        
        Example:
            >>> results = retriever.memory_search("Python")
        """
        if not self.memory_store:
            return []
        
        # Use memory store's retrieve method
        memory_results = self.memory_store.retrieve(query, **options)
        
        # Convert to RetrievedContext
        results = []
        for mem in memory_results:
            results.append(
                RetrievedContext(
                    content=mem.get("content", ""),
                    score=mem.get("score", 0.0),
                    source="memory",
                    metadata=mem.get("metadata", {}),
                )
            )
        
        return results

    def hybrid_search(self, query: str, **options) -> List[RetrievedContext]:
        """
        Hybrid search (all sources).
        
        Args:
            query: Search query
            **options: Additional options
        
        Returns:
            List of RetrievedContext objects from all sources
        
        Example:
            >>> results = retriever.hybrid_search("Python")
        """
        return self.retrieve(query, **options)

    # Advanced Retrieval
    def find_similar(self, content: str, limit: int = 5, **options) -> List[RetrievedContext]:
        """
        Find similar content.
        
        Args:
            content: Content to find similar items for
            limit: Maximum results (default: 5)
            **options: Additional options
        
        Returns:
            List of similar RetrievedContext objects
        
        Example:
            >>> similar = retriever.find_similar("Python programming", limit=5)
        """
        return self.retrieve(content, max_results=limit, **options)

    def get_context(self, query: str, max_results: int = 5, **options) -> List[RetrievedContext]:
        """
        Get context for query.
        
        Args:
            query: Query string
            max_results: Maximum results (default: 5)
            **options: Additional options
        
        Returns:
            List of RetrievedContext objects
        
        Example:
            >>> context_data = retriever.get_context("Python", max_results=10)
        """
        return self.retrieve(query, max_results=max_results, **options)

    def expand_query(self, query: str, max_hops: int = 2, **options) -> List[RetrievedContext]:
        """
        Expand query with graph.
        
        Args:
            query: Query string
            max_hops: Maximum expansion hops (default: 2)
            **options: Additional options
        
        Returns:
            List of expanded RetrievedContext objects
        
        Example:
            >>> expanded = retriever.expand_query("Python", max_hops=3)
        """
        return self.retrieve(
            query,
            use_graph_expansion=True,
            max_hops=max_hops,
            **options
        )

    def get_related(self, entity_id: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Get related entities.
        
        Args:
            entity_id: Entity ID
            max_hops: Maximum hops (default: 2)
        
        Returns:
            List of related entity dicts
        
        Example:
            >>> related = retriever.get_related("entity_123", max_hops=2)
        """
        if not self.knowledge_graph:
            return []
        
        return self._get_related_entities(entity_id, max_hops=max_hops)

    def get_path(self, source_id: str, target_id: str, max_hops: int = 5) -> List[Dict[str, Any]]:
        """
        Get path between entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_hops: Maximum hops (default: 5)
        
        Returns:
            List of path nodes/edges
        
        Example:
            >>> path = retriever.get_path("entity_1", "entity_2", max_hops=5)
        """
        if not self.knowledge_graph:
            return []
        
        # Simple BFS path finding
        from collections import deque
        
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            if len(path) > max_hops:
                continue
            
            if current_id == target_id:
                # Return path with node info
                nodes = self.knowledge_graph.get("nodes", [])
                path_info = []
                for node_id in path:
                    for node in nodes:
                        if node.get("id") == node_id:
                            path_info.append({
                                "id": node_id,
                                "content": node.get("content", ""),
                                "type": node.get("type", ""),
                            })
                            break
                return path_info
            
            # Get neighbors
            edges = self.knowledge_graph.get("edges", [])
            for edge in edges:
                neighbor_id = None
                if edge.get("source") == current_id:
                    neighbor_id = edge.get("target")
                elif edge.get("target") == current_id:
                    neighbor_id = edge.get("source")
                
                if neighbor_id and neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        return []

    # Filter Methods
    def filter_by_entity(self, entity_id: str, query: str, **options) -> List[RetrievedContext]:
        """
        Filter by entity.
        
        Args:
            entity_id: Entity ID to filter by
            query: Search query
            **options: Additional options
        
        Returns:
            Filtered RetrievedContext objects
        
        Example:
            >>> results = retriever.filter_by_entity("entity_123", "Python")
        """
        results = self.retrieve(query, **options)
        filtered = []
        for result in results:
            # Check if entity is in related entities
            for entity in result.related_entities:
                if entity.get("id") == entity_id:
                    filtered.append(result)
                    break
        return filtered

    def filter_by_type(self, type: str, query: str, **options) -> List[RetrievedContext]:
        """
        Filter by type.
        
        Args:
            type: Node/entity type to filter by
            query: Search query
            **options: Additional options
        
        Returns:
            Filtered RetrievedContext objects
        
        Example:
            >>> results = retriever.filter_by_type("PROGRAMMING_LANGUAGE", "Python")
        """
        results = self.retrieve(query, **options)
        filtered = []
        for result in results:
            if result.metadata.get("node_type") == type:
                filtered.append(result)
        return filtered

    def filter_by_date(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        query: str,
        **options
    ) -> List[RetrievedContext]:
        """
        Filter by date.
        
        Args:
            start_date: Start date
            end_date: End date
            query: Search query
            **options: Additional options
        
        Returns:
            Filtered RetrievedContext objects
        
        Example:
            >>> results = retriever.filter_by_date("2024-01-01", "2024-12-31", "Python")
        """
        if isinstance(start_date, str):
            from dateutil.parser import parse
            start_date = parse(start_date)
        if isinstance(end_date, str):
            from dateutil.parser import parse
            end_date = parse(end_date)
        
        results = self.retrieve(query, **options)
        filtered = []
        for result in results:
            result_date = result.metadata.get("timestamp")
            if result_date:
                if isinstance(result_date, str):
                    from dateutil.parser import parse
                    result_date = parse(result_date)
                if start_date <= result_date <= end_date:
                    filtered.append(result)
        return filtered

    def filter_by_score(
        self,
        min_score: float,
        query: str,
        **options
    ) -> List[RetrievedContext]:
        """
        Filter by score.
        
        Args:
            min_score: Minimum score threshold
            query: Search query
            **options: Additional options
        
        Returns:
            Filtered RetrievedContext objects
        
        Example:
            >>> results = retriever.filter_by_score(0.7, "Python")
        """
        results = self.retrieve(query, min_relevance_score=min_score, **options)
        return [r for r in results if r.score >= min_score]

    # Batch Operations
    def batch_search(self, queries: List[str], **options) -> Dict[str, List[RetrievedContext]]:
        """
        Search multiple queries.
        
        Args:
            queries: List of queries
            **options: Additional options
        
        Returns:
            Dict mapping query to results
        
        Example:
            >>> results = retriever.batch_search(["Python", "Java", "C++"])
        """
        results = {}
        for query in queries:
            results[query] = self.retrieve(query, **options)
        return results

    def batch_get_context(
        self,
        queries: List[str],
        max_results: int = 5,
        **options
    ) -> Dict[str, List[RetrievedContext]]:
        """
        Get context for multiple queries.
        
        Args:
            queries: List of queries
            max_results: Maximum results per query (default: 5)
            **options: Additional options
        
        Returns:
            Dict mapping query to context results
        
        Example:
            >>> contexts = retriever.batch_get_context(["Python", "Java"], max_results=5)
        """
        results = {}
        for query in queries:
            results[query] = self.get_context(query, max_results=max_results, **options)
        return results