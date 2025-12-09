# Context Module Reference

> **The central nervous system for intelligent agents, managing memory, knowledge graphs, and context retrieval.**

---

## 🎯 System Overview

The **Context Module** provides agents with a persistent, searchable, and structured memory system. It is built on a **Synchronous Architecture 2.0**, ensuring predictable state management and compatibility with modern vector stores and graph databases.

### Key Capabilities

*   **Hierarchical Memory**: A two-tier memory system that mimics human memory (Short-term "Working" Memory + Long-term "Episodic" Memory).
*   **GraphRAG**: Combines unstructured vector search with structured knowledge graph traversal for richer context.
*   **Hybrid Retrieval**: Intelligently blends results from keyword matching, vector similarity, and graph connections.
*   **Token Management**: Automatically prunes short-term memory to fit within LLM context windows using token-based constraints.
*   **Entity Linking**: Resolves ambiguities by linking text mentions to unique entities in the knowledge graph.

---

## 🏗️ Architecture Components

### 1. AgentContext (The Orchestrator)
The high-level facade that unifies all context operations. It routes data to the appropriate subsystems (Memory, Graph, Vector Store) and manages the lifecycle of context.

#### **Constructor Parameters**
*   `vector_store` (Required): The backing vector database instance (e.g., FAISS, Pinecone).
*   `knowledge_graph` (Optional): The graph store instance for structured knowledge.
*   `token_limit` (Default: `2000`): The maximum number of tokens allowed in short-term memory before pruning occurs.
*   `short_term_limit` (Default: `10`): The maximum number of distinct memory items in short-term memory.
*   `hybrid_alpha` (Default: `0.5`): The weighting factor for retrieval (0.0 = Pure Vector, 1.0 = Pure Graph).
*   `use_graph_expansion` (Default: `True`): Whether to fetch neighbors of retrieved nodes from the graph.

#### **Core Methods**

*   **`store(content, ...)`**: Writes information to memory.
    *   *Auto-Detection*: Automatically determines if input is a simple string (memory) or a list of documents.
    *   *Write-Through*: Saves to both Short-term (RAM) and Long-term (Vector Store) memory simultaneously.
    *   *Entity Extraction*: If enabled, extracts entities and relationships to update the Knowledge Graph.
*   **`retrieve(query, ...)`**: Fetches relevant context.
    *   *Hybrid Search*: Queries Vector Store, Short-term Memory, and Knowledge Graph in parallel.
    *   *Reranking*: Merges and ranks results based on relevance scores.
    *   *Context Window Optimization*: Returns results that fit within the agent's context window.

---

### 2. AgentMemory (The Storage Engine)
Manages the storage and lifecycle of memory items. It implements the **Hierarchical Memory** pattern.

#### **Features & Functions**
*   **Short-Term Memory (Working Memory)**
    *   *Structure*: An in-memory list of recent `MemoryItem` objects.
    *   *Purpose*: Provides immediate context for the ongoing conversation.
    *   *Pruning Logic*:
        *   **FIFO**: Removes the oldest items first when limits are reached.
        *   **Token-Aware**: Calculates token counts to ensure the total buffer size stays under `token_limit`.
*   **Long-Term Memory (Episodic Memory)**
    *   *Structure*: Vector embeddings stored in the `vector_store`.
    *   *Purpose*: Persists history indefinitely for semantic retrieval.
    *   *Synchronization*: Automatically syncs with Short-term memory during `store()` operations.
*   **Retention Policy**
    *   *Time-Based*: Can automatically delete memories older than `retention_days`.
    *   *Count-Based*: Can limit the total number of memories to `max_memories`.

#### **Key Methods**
*   `store_vectors()`: Handles the low-level interaction with concrete Vector Store implementations.
*   `_prune_short_term_memory()`: Internal algorithm that enforces token and count limits.
*   `get_conversation_history()`: Retrieves a chronological list of interactions for a specific session.

---

### 3. ContextGraph (The Knowledge Structure)
Manages the structured relationships between entities. It provides the "World Model" for the agent.

#### **Features & Functions**
*   **Dictionary-Based Interface**
    *   *Design*: Uses standard Python dictionaries for nodes and edges, removing dependencies on complex interface classes.
    *   *Benefit*: simpler serialization and easier integration with external APIs.
*   **Graph Traversal**
    *   *Adjacency List*: optimized internal structure for fast neighbor lookups.
    *   *Multi-Hop Search*: Can traverse `k` hops from a starting node to find indirect connections.
*   **Node & Edge Types**
    *   *Typed Schema*: Supports distinct types for nodes (e.g., "Person", "Concept") and edges (e.g., "KNOWS", "RELATED_TO").

#### **Key Methods**
*   `add_nodes(nodes)`: Bulk adds nodes using a list of dictionaries.
*   `add_edges(edges)`: Bulk adds edges using a list of dictionaries.
*   `get_neighbors(node_id, hops)`: Returns connected nodes within a specified distance.
*   `query(query_str)`: Performs keyword-based search specifically on graph nodes.

---

### 4. ContextRetriever (The Search Engine)
The retrieval logic that powers the `retrieve()` command. It implements the **Hybrid Retrieval** algorithm.

#### **Retrieval Strategy**
1.  **Short-Term Check**: Scans the in-memory buffer for immediate, exact-match relevance.
2.  **Vector Search**: Queries the `vector_store` for semantically similar long-term memories.
3.  **Graph Expansion**:
    *   Identifies entities in the query.
    *   Finds those entities in the `ContextGraph`.
    *   Traverses edges to find related concepts that might not match keywords (e.g., finding "Python" when searching for "Coding").
4.  **Hybrid Scoring**:
    *   Formula: `Final_Score = (Vector_Score * (1 - α)) + (Graph_Score * α)`
    *   Allows tuning the balance between semantic similarity and structural relevance.

---

## ⚙️ Configuration & Tuning

### Token Management
*   **`CONTEXT_TOKEN_LIMIT`**: Environment variable to set the global default for short-term memory size.
*   **`short_term_limit`**: Configurable per-agent to limit the *count* of recent items.

### Tuning Retrieval
*   **`hybrid_alpha`**:
    *   `0.0`: Relies entirely on Vector Search (Standard RAG).
    *   `0.5`: Balanced approach (Recommended).
    *   `1.0`: Relies entirely on Graph traversal (Graph RAG).
*   **`max_expansion_hops`**:
    *   `1`: Only direct neighbors.
    *   `2`: Friends of friends (Recommended for discovery).
    *   `3+`: Can introduce noise but finds deep connections.

---

## 📝 Data Structures

### MemoryItem
The fundamental unit of storage.
```python
@dataclass
class MemoryItem:
    content: str              # The actual text content
    timestamp: datetime       # When it was created
    metadata: Dict            # Arbitrary tags (user_id, source, etc.)
    embedding: List[float]    # The vector representation
    entities: List[Dict]      # Entities found in this content
```

### Graph Node (Dict Format)
```python
{
    "id": "node_unique_id",
    "type": "concept",
    "properties": {
        "content": "Description of the node",
        "weight": 1.0
    }
}
```

### Graph Edge (Dict Format)
```python
{
    "source_id": "origin_node",
    "target_id": "destination_node",
    "type": "related_to",
    "weight": 0.8
}
```
