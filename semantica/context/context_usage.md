# Context Module Usage Guide

This guide demonstrates how to use the Semantica context module for building context graphs, managing agent memory, retrieving context, and linking entities.

## Table of Contents

1. [High-Level Interface (Quick Start)](#high-level-interface-quick-start)
2. [Basic Usage](#basic-usage)
3. [Context Graph Construction](#context-graph-construction)
4. [Agent Memory Management](#agent-memory-management)
5. [Context Retrieval](#context-retrieval)
6. [Entity Linking](#entity-linking)

## High-Level Interface (Quick Start)

The `AgentContext` class provides a simplified, generic interface for common use cases. It integrates vector storage, knowledge graphs, and memory management into a unified system.

### Simple RAG (Vector Only)

```python
from semantica.context import AgentContext
from semantica.vector_store import VectorStore

# Initialize vector store
vs = VectorStore(backend="faiss", dimension=768)

# Initialize context
context = AgentContext(vector_store=vs)

# Store a memory
memory_id = context.store("User likes Python programming", conversation_id="conv1")

# Retrieve context
results = context.retrieve("Python programming", max_results=5)

for result in results:
    print(f"Content: {result['content']}")
    print(f"Score: {result['score']:.2f}")
```

### GraphRAG (Vector + Graph)

```python
from semantica.context import AgentContext, ContextGraph
from semantica.graph_store import GraphStore

# Initialize persistent knowledge graph (Recommended for production)
try:
    kg = GraphStore(backend="neo4j", uri="bolt://localhost:7687", user="neo4j", password="password")
    kg.connect()
except:
    print("Neo4j not available, falling back to in-memory graph")
    kg = ContextGraph()

# Initialize context with vector store and knowledge graph
context = AgentContext(vector_store=vs, knowledge_graph=kg)

# Store documents (auto-builds graph)
documents = [
    "Python is a programming language used for machine learning",
    "TensorFlow and PyTorch are popular ML frameworks",
    "Machine learning involves training models on data"
]

stats = context.store(
    documents,
    extract_entities=True,      # Extract entities from documents
    extract_relationships=True,  # Extract relationships
    link_entities=True          # Link entities across documents
)

print(f"Stored {stats['stored_count']} documents")
# Graph stats are available via the graph object directly or context stats
print(f"Graph nodes: {kg.stats()['node_count']}")

# Retrieve with graph context (auto-detects GraphRAG)
results = context.retrieve(
    "Python machine learning",
    use_graph=True,              # Explicitly use graph
    include_entities=True,       # Include related entities
    expand_graph=True            # Use graph expansion
)

for result in results:
    print(f"Content: {result['content']}")
    print(f"Score: {result['score']:.2f}")
    print(f"Related entities: {len(result.get('related_entities', []))}")
```

### Agent Memory Management (Hierarchical)

The system uses a hierarchical memory structure with:
1.  **Short-Term Memory**: Fast, in-memory buffer with token and item count limits.
2.  **Long-Term Memory**: Persistent vector store.

```python
context = AgentContext(
    vector_store=vs,
    retention_days=30,
    short_term_limit=10,  # Max items in short-term buffer
    token_limit=2000      # Max tokens in short-term buffer
)

# Store multiple memories in a conversation
context.store("Hello, I'm interested in Python", conversation_id="conv1", user_id="user123")
context.store("What can you tell me about machine learning?", conversation_id="conv1", user_id="user123")

# Get conversation history
history = context.conversation(
    "conv1",
    reverse=True,           # Most recent first
    include_metadata=True   # Include full metadata
)

for item in history:
    print(f"{item['timestamp']}: {item['content']}")

# Delete old memories
deleted_count = context.forget(days_old=90)
print(f"Deleted {deleted_count} old memories")
```

### Persistence (Save/Load)

You can save the entire state of the agent (Memory, Graph, and Vector Index) to disk and reload it later.

```python
# Save state
context.save("./my_agent_state")

# Load state
new_context = AgentContext(vector_store=VectorStore(), knowledge_graph=ContextGraph())
new_context.load("./my_agent_state")
```

## Basic Usage

### Initialization with Backends

You can configure the `VectorStore` with different backends (`inmemory`, `faiss`, `chroma`, `qdrant`, `weaviate`, `milvus`) and embedding models (including FastEmbed).

```python
from semantica.context import AgentContext, ContextGraph
from semantica.vector_store import VectorStore

# Initialize Vector Store with FastEmbed
vs = VectorStore(backend="inmemory", dimension=384)
if hasattr(vs, "embedder") and vs.embedder:
    vs.embedder.set_text_model(method="fastembed", model_name="BAAI/bge-small-en-v1.5")

# Initialize Context Graph
kg = ContextGraph()

# Initialize Agent Context
context = AgentContext(vector_store=vs, knowledge_graph=kg)
```

## Context Graph Construction

The `ContextGraph` class is an in-memory graph store.

### Building from Entities and Relationships

```python
from semantica.context import ContextGraph

graph = ContextGraph()

entities = [
    {"id": "e1", "text": "Python", "type": "PROGRAMMING_LANGUAGE"},
    {"id": "e2", "text": "Machine Learning", "type": "CONCEPT"},
    {"id": "e3", "text": "TensorFlow", "type": "FRAMEWORK"},
]

relationships = [
    {"source_id": "e1", "target_id": "e2", "type": "used_for", "confidence": 0.9},
    {"source_id": "e3", "target_id": "e2", "type": "implements", "confidence": 0.95},
]

graph_data = graph.build_from_entities_and_relationships(entities, relationships)

print(f"Nodes: {graph.stats()['node_count']}")
print(f"Edges: {graph.stats()['edge_count']}")
```

### Building from Conversations

```python
from semantica.context import ContextGraph

graph = ContextGraph()

conversations = [
    {
        "id": "conv1",
        "content": "User asked about Python programming",
        "entities": [
            {"id": "e1", "text": "Python", "type": "PROGRAMMING_LANGUAGE"}
        ],
        "relationships": []
    }
]

graph_data = graph.build_from_conversations(
    conversations,
    link_entities=True,
    extract_intents=True
)
```

### Adding Nodes and Edges Manually

```python
from semantica.context import ContextGraph

graph = ContextGraph()

# Add nodes
graph.add_node("node1", "entity", "Python programming", confidence=0.9)
graph.add_node("node2", "concept", "Machine Learning", confidence=0.95)

# Add edges
graph.add_edge("node1", "node2", "related_to", weight=0.9)

# Get neighbors
neighbors = graph.get_neighbors("node1", hops=2)
print(f"Neighbors: {neighbors}")

# Query graph
results = graph.query("Python") # Keyword search on nodes
```

### Graph Statistics and Analysis

```python
stats = graph.stats()
print(f"Node types: {stats['node_types']}")
print(f"Density: {stats['density']:.4f}")

# Find specific nodes/edges
entities = graph.find_nodes(node_type="entity")
relations = graph.find_edges(edge_type="related_to")
node = graph.find_node("node1")
```

## Agent Memory Management

The `AgentMemory` class handles short-term and long-term memory with hierarchical storage and token management.

### Storing and Retrieving

```python
from semantica.context import AgentMemory

memory = AgentMemory(
    vector_store=vs,
    knowledge_graph=kg,
    retention_policy="30_days",
    max_memory_size=10000,
    short_term_limit=20,    # 20 items max in short-term
    token_limit=4000        # 4000 tokens max in short-term
)

# Store (automatically updates short-term and long-term)
memory_id = memory.store(
    "User asked about Python programming",
    metadata={"conversation_id": "conv_123"}
)

# Store short-term only (fleeting thoughts)
temp_id = memory.store(
    "Just checking status...",
    skip_vector=True
)

# Retrieve
results = memory.retrieve(
    "Python programming",
    max_results=5,
    type="conversation"
)

# Conversation History
history = memory.get_conversation_history("conv_123")
```

## Context Retrieval

The `ContextRetriever` implements hybrid retrieval strategies.

### Hybrid Retrieval

```python
from semantica.context import ContextRetriever

retriever = ContextRetriever(
    memory_store=memory,
    knowledge_graph=kg,
    vector_store=vs,
    use_graph_expansion=True,
    max_expansion_hops=2,
    hybrid_alpha=0.5  # Balance between vector (0.0) and graph (1.0)
)

results = retriever.retrieve(
    "Python programming",
    max_results=5
)

for result in results:
    print(f"Content: {result.content}")
    print(f"Source: {result.source}") # 'vector', 'graph', or 'memory'
```

## Entity Linking

The `EntityLinker` helps resolve entities to canonical forms or URIs.

```python
from semantica.context import EntityLinker

linker = EntityLinker()
uri = linker.generate_uri("Python Programming Language")
print(uri) # e.g., "python_programming_language"

# Similarity matching
score = linker._calculate_text_similarity("Python", "Python Language")
```
