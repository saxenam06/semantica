# Context

> **Context engineering and memory management system for intelligent agents using RAG and Knowledge Graphs.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-graph:{ .lg .middle } **Context Graph**

    ---

    Build dynamic context graphs from conversations and entities

-   :material-brain:{ .lg .middle } **Agent Memory**

    ---

    Persistent memory management with vector storage integration

-   :material-link-variant:{ .lg .middle } **Entity Linking**

    ---

    Link entities across documents and conversations

-   :material-magnify:{ .lg .middle } **Hybrid Retrieval**

    ---

    Retrieve context using Vector + Graph + Keyword search

-   :material-history:{ .lg .middle } **Conversation History**

    ---

    Manage and synthesize conversation history

-   :material-bullseye-arrow:{ .lg .middle } **Intent Analysis**

    ---

    Extract and track user intent and sentiment

</div>

!!! tip "When to Use"
    - **Agent Development**: When building agents that need long-term memory
    - **RAG Applications**: For advanced Retrieval-Augmented Generation
    - **Personalization**: To maintain user-specific context and preferences

---

## ⚙️ Algorithms Used

### Context Graph Construction
- **Graph Building**: Node/Edge construction from extracted entities
- **Graph Traversal**: BFS/DFS for multi-hop context discovery
- **Intent Extraction**: NLP-based intent classification
- **Sentiment Analysis**: Sentiment scoring and extraction

### Agent Memory
- **Vector Embedding**: Dense vector generation for memory items
- **Vector Search**: Cosine similarity search (k-NN)
- **Retention Policy**: Time-based decay and cleanup
- **Memory Indexing**: Deque-based sliding window for short-term memory

### Entity Linking
- **URI Generation**: Hash-based deterministic IDs
- **Text Similarity**: Jaccard/Levenshtein for name matching
- **Graph Lookup**: Entity resolution against Knowledge Graph
- **Bidirectional Linking**: Symmetric link creation

### Context Retrieval
- **Hybrid Scoring**: `α * VectorScore + β * GraphScore + γ * KeywordScore`
- **Graph Expansion**: Retrieving neighbors of retrieved entities
- **Deduplication**: Content-based result merging
- **Result Ranking**: Weighted aggregation of scores

---

## Main Classes

### AgentContext

High-level interface for agent context management, RAG, and GraphRAG. Provides generic methods (`store`, `retrieve`, `forget`, `conversation`) that auto-detect content types and retrieval strategies.

**Methods:**

| Method | Description | Parameters |
|--------|-------------|------------|
| `store(content, ...)` | Store content (memory or documents) | `extract_entities: bool = True`, `extract_relationships: bool = True`, `link_entities: bool = True`, `auto_extract: bool = False` |
| `retrieve(query, ...)` | Retrieve relevant context | `use_graph: Optional[bool] = None`, `include_entities: bool = True`, `include_relationships: bool = False`, `expand_graph: bool = True`, `deduplicate: bool = True` |
| `forget(...)` | Delete memories | `memory_id`, `conversation_id`, `user_id`, `days_old` |
| `conversation(conversation_id, ...)` | Get conversation history | `reverse: bool = False`, `include_metadata: bool = True` |
| `get_memory(memory_id)` | Get specific memory by ID | `memory_id: str` |
| `stats()` | Get memory statistics | None |
| `link(text, entities, ...)` | Link entities in text | `similarity_threshold: float = 0.8` |
| `build_graph(...)` | Build context graph manually | `entities`, `relationships`, `conversations`, `link_entities: bool = True` |

**Initialization:**

```python
AgentContext(
    vector_store,                    # Required
    knowledge_graph=None,            # Optional (enables GraphRAG)
    retention_days=30,               # Optional
    max_memories=10000,             # Optional
    use_graph_expansion=True,        # Boolean flag
    max_expansion_hops=2,           # Optional
    hybrid_alpha=0.5                # Optional
)
```

**Example:**

```python
from semantica.context import AgentContext

# Simple RAG
context = AgentContext(vector_store=vs)
memory_id = context.store("User likes Python", conversation_id="conv1")
results = context.retrieve("Python programming")

# GraphRAG
context = AgentContext(vector_store=vs, knowledge_graph=kg)
stats = context.store(["Doc 1", "Doc 2"], extract_entities=True)
results = context.retrieve("Python", use_graph=None)  # Auto-detects GraphRAG

# Conversation management
history = context.conversation("conv1", reverse=True)
deleted = context.forget(conversation_id="conv1")
```

**Boolean Flags:**

- **Store**: `extract_entities`, `extract_relationships`, `link_entities`
- **Retrieve**: `use_graph` (None=auto-detect), `include_entities`, `include_relationships`, `expand_graph`
- **Conversation**: `reverse`, `include_metadata`

### ContextGraphBuilder

Builds and manages the context graph.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `build_from_entities_and_relationships(entities, relationships)` | Build graph from entities and relationships | Node/Edge creation |
| `build_from_conversations(conversations, link_entities, extract_intents, extract_sentiments)` | Build graph from conversations | Intent/Entity extraction |
| `add_node(node_id, node_type, content, **metadata)` | Add node to graph | Node creation |
| `add_edge(source_id, target_id, edge_type, weight, **metadata)` | Add edge to graph | Edge creation |
| `get_neighbors(node_id, max_hops)` | Get neighbor nodes | BFS Traversal |
| `query(node_type, edge_type, **filters)` | Query graph nodes and edges | Type-based filtering |

**Example:**

```python
from semantica.context import ContextGraphBuilder

builder = ContextGraphBuilder()
graph = builder.build_from_entities_and_relationships(
    entities=extracted_entities,
    relationships=extracted_rels
)

# Add nodes and edges manually
builder.add_node("node1", "entity", "Python programming")
builder.add_edge("node1", "node2", "related_to", weight=0.9)

# Query and traverse
neighbors = builder.get_neighbors("node1", max_hops=2)
results = builder.query(node_type="entity", confidence=0.8)
```

### AgentMemory

Manages persistent agent memory.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `store(content, metadata, entities, relationships, **options)` | Store memory item | Embedding + Vector Store |
| `retrieve(query, max_results, min_score, **filters)` | Retrieve relevant memories | Vector Similarity |
| `get_memory(memory_id)` | Get specific memory item | Dictionary lookup |
| `delete_memory(memory_id)` | Delete memory item | Cascading deletion |
| `clear_memory(**filters)` | Clear memories by filters | Filter-based deletion |
| `get_conversation_history(conversation_id, max_items)` | Get conversation history | Temporal filtering |
| `get_statistics()` | Get memory statistics | Counter aggregation |

**Example:**

```python
from semantica.context import AgentMemory

memory = AgentMemory(vector_store=vs, knowledge_graph=kg)
memory_id = memory.store("User prefers Python over Java", metadata={"type": "preference"})
relevant = memory.retrieve("What language does the user like?", max_results=5)

# Get specific memory
memory_item = memory.get_memory(memory_id)

# Get statistics
stats = memory.get_statistics()
```

### EntityLinker

Links entities across different contexts.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `assign_uri(entity_id, entity_text, entity_type)` | Assign unique URI to entity | Hash-based/Text-based URI |
| `link(text, entities, context)` | Link entities in text to knowledge graph | Similarity matching |
| `link_entities(entity1_id, entity2_id, link_type, confidence, source, **metadata)` | Create explicit link between entities | Bidirectional linking |
| `get_entity_links(entity_id)` | Get all links for an entity | Dictionary lookup |
| `get_entity_uri(entity_id)` | Get URI for an entity | Registry lookup |
| `find_similar_entities(entity_text, entity_type, threshold)` | Find similar entities in knowledge graph | Text similarity |
| `build_entity_web()` | Build entity connection web | Graph construction |

**Example:**

```python
from semantica.context import EntityLinker

linker = EntityLinker(knowledge_graph=kg, similarity_threshold=0.8)
uri = linker.assign_uri("entity_1", "Python", "PROGRAMMING_LANGUAGE")
linked_entities = linker.link("Python is used for ML", entities=entities)
linker.link_entities("e1", "e2", "related_to", confidence=0.9)
similar = linker.find_similar_entities("Python", entity_type="PROGRAMMING_LANGUAGE")
web = linker.build_entity_web()
```

### ContextRetriever

Orchestrates hybrid retrieval.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `retrieve(query, max_results, use_graph_expansion, min_relevance_score, **options)` | Retrieve relevant context for query | Hybrid (Vector+Graph+Memory) |

**Example:**

```python
from semantica.context import ContextRetriever

retriever = ContextRetriever(
    memory_store=memory,
    knowledge_graph=kg,
    vector_store=vs,
    use_graph_expansion=True,
    max_expansion_hops=2
)

results = retriever.retrieve("Python programming", max_results=5, min_relevance_score=0.5)
for result in results:
    print(f"{result.content}: {result.score:.2f}")
    print(f"Related entities: {len(result.related_entities)}")
```

---

## Methods Module

The methods module provides simple, reusable functions for context operations.

**Functions:**

| Function | Description |
|----------|-------------|
| `build_context_graph(entities, relationships, conversations, method, **kwargs)` | Build context graph using specified method |
| `store_memory(content, vector_store, knowledge_graph, method, **kwargs)` | Store memory using specified method |
| `retrieve_context(query, memory_store, knowledge_graph, vector_store, method, max_results, **kwargs)` | Retrieve context using specified method |
| `link_entities(entities, knowledge_graph, method, **kwargs)` | Link entities using specified method |
| `get_context_method(task, name)` | Get registered context method |
| `list_available_methods(task)` | List all available context methods |

**Example:**

```python
from semantica.context.methods import (
    build_context_graph,
    store_memory,
    retrieve_context,
    link_entities,
    get_context_method,
    list_available_methods
)

# Build graph
graph = build_context_graph(entities, relationships, method="entities_relationships")

# Store memory
memory_id = store_memory("User asked about Python", vector_store=vs, method="store")

# Retrieve context
results = retrieve_context("Python programming", vector_store=vs, method="hybrid")

# Link entities
linked = link_entities(entities, knowledge_graph=kg, method="similarity")

# List available methods
all_methods = list_available_methods()
graph_methods = list_available_methods("graph")
```

## Registry Module

The registry module allows registering custom context methods.

**MethodRegistry Methods:**

| Method | Description |
|--------|-------------|
| `register(task, name, method_func)` | Register a method for a specific task |
| `get(task, name)` | Get a registered method |
| `list_all(task)` | List all registered methods |
| `unregister(task, name)` | Unregister a method |

**Example:**

```python
from semantica.context import registry

def custom_graph_builder(entities, relationships, **kwargs):
    """Custom graph building method."""
    return {"nodes": [], "edges": [], "statistics": {}}

# Register custom method
registry.method_registry.register("graph", "custom_builder", custom_graph_builder)

# List registered methods
methods = registry.method_registry.list_all("graph")

# Unregister method
registry.method_registry.unregister("graph", "custom_builder")
```

## Configuration Module

The configuration module provides centralized configuration management.

**ContextConfig Methods:**

| Method | Description |
|--------|-------------|
| `set(key, value)` | Set a configuration value |
| `get(key, default)` | Get a configuration value |
| `set_method_config(method_name, config)` | Set method-specific configuration |
| `get_method_config(method_name)` | Get method-specific configuration |
| `get_all()` | Get all configurations |

**Example:**

```python
from semantica.context import config

# Get configuration
retention = config.context_config.get("retention_policy", default="unlimited")
max_size = config.context_config.get("max_memory_size", default=10000)

# Set configuration
config.context_config.set("retention_policy", "30_days")
config.context_config.set("max_memory_size", 5000)

# Method-specific configuration
config.context_config.set_method_config("graph", {
    "extract_entities": True,
    "extract_relationships": True
})

method_config = config.context_config.get_method_config("graph")

# Load from config file
context_config = config.ContextConfig(config_file="context_config.yaml")
all_configs = context_config.get_all()
```

---

## Configuration

### Environment Variables

```bash
export CONTEXT_RETENTION_POLICY=30_days
export CONTEXT_MAX_MEMORY_SIZE=5000
export CONTEXT_SIMILARITY_THRESHOLD=0.8
```

### YAML Configuration

```yaml
context:
  retention_policy:
    max_days: 30
    max_items: 1000
    
  retrieval:
    hybrid_weights:
      vector: 0.6
      graph: 0.3
      keyword: 0.1
      
  graph:
    max_depth: 2
    include_attributes: true
```

---

## Integration Examples

### Chatbot with Memory

```python
from semantica.context import AgentMemory, ContextRetriever
from semantica.llm import LLMClient

# 1. Initialize
memory = AgentMemory(vector_store=vs)
retriever = ContextRetriever(memory=memory, graph=kg)
llm = LLMClient()

def chat(user_input):
    # 2. Retrieve Context
    context = retriever.retrieve(user_input)
    
    # 3. Generate Response
    response = llm.generate(user_input, context=context)
    
    # 4. Update Memory
    memory.store(f"User: {user_input}")
    memory.store(f"Agent: {response}")
    
    return response
```

---

## Best Practices

1.  **Prune Regularly**: Use retention policies to keep memory relevant and performant.
2.  **Use Hybrid Retrieval**: Relying solely on vector search misses structural relationships; use graph context too.
3.  **Enrich Metadata**: Store rich metadata (timestamp, source, type) with memories for better filtering.
4.  **Link Entities**: Ensure `EntityLinker` is used to connect mentions of the same entity across conversations.

---

## Troubleshooting

**Issue**: Retrieval returns irrelevant old memories.
**Solution**: Adjust retention policy or increase vector similarity threshold.

```python
memory = AgentMemory(
    retention_policy="7_days",
    max_memory_size=1000
)
```

**Issue**: Context graph growing too large.
**Solution**: Use `prune_graph` or limit hop depth during retrieval.

---

## See Also

- [Vector Store Module](vector_store.md) - Underlying storage for memory
- [Knowledge Graph Module](kg.md) - Underlying graph structure
- [Embeddings Module](embeddings.md) - Vector generation

## Cookbook

- [Context Module](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/19_Context_Module.ipynb)
