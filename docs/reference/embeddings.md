# Embeddings

> **Text embedding generation with multiple model support.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-text-box:{ .lg .middle } **Text Embeddings**

    ---

    Sentence-transformers, OpenAI, BGE, and FastEmbed model support

-   :material-vector-square:{ .lg .middle } **Vector Databases**

    ---

    Specialized managers for vector database integration (FAISS, Pinecone, Qdrant, etc.)

-   :material-graph:{ .lg .middle } **Graph Databases**

    ---

    Specialized managers for graph database integration (Neo4j, NetworkX, etc.)

-   :material-compress:{ .lg .middle } **Pooling Strategies**

    ---

    Mean, Max, CLS, Attention, and Hierarchical pooling for aggregation

</div>

!!! tip "When to Use"
    - **Semantic Search**: Converting text to vectors for similarity search
    - **Clustering**: Grouping similar documents
    - **Classification**: Using embeddings as features for ML models
    - **RAG**: Embedding chunks for retrieval
    - **Vector Databases**: Preparing embeddings for FAISS, Pinecone, Qdrant, etc.
    - **Graph Databases**: Creating node and edge embeddings for Neo4j, NetworkX, etc.

---

## ⚙️ Algorithms Used

### Generation
- **Transformer Encoding**: BERT/RoBERTa based models for text (sentence-transformers).
- **FastEmbed Encoding**: Fast and efficient embedding generation using FastEmbed library.
- **Provider Adapters**: OpenAI, BGE, Llama, and FastEmbed adapters for different providers.

### Pooling
- **Mean Pooling**: Arithmetic mean across embedding dimension.
- **Max Pooling**: Element-wise maximum across embedding dimension.
- **CLS Token Pooling**: First token/embedding extraction (for transformer models).
- **Attention-based Pooling**: Softmax-weighted sum using dot product attention scores.
- **Hierarchical Pooling**: Two-level pooling (chunk-level then global-level mean pooling).

---

## Main Classes

### EmbeddingGenerator

Unified interface for text embedding generation.

**Methods:**

| Method | Description |
|--------|-------------|
| `generate_embeddings(data, data_type="text")` | Generate embedding |
| `process_batch(items)` | Batch generation |
| `compare_embeddings(emb1, emb2, method="cosine")` | Calculate similarity |
| `get_text_method()` | Get active text embedding method |
| `get_methods_info()` | Get detailed method information |

**Example:**

```python
from semantica.embeddings import EmbeddingGenerator

gen = EmbeddingGenerator()
vec = gen.generate_embeddings("Hello world", data_type="text")
```

### TextEmbedder

Specialized text embedding generation.

**Methods:**

| Method | Description |
|--------|-------------|
| `embed_text(text)` | Generate vector for single text |
| `embed_batch(texts)` | Batch processing |
| `get_method()` | Get active embedding method |
| `get_model_info()` | Get detailed model information |
| `get_embedding_dimension()` | Get embedding dimension |

**Example:**

```python
from semantica.embeddings import TextEmbedder

embedder = TextEmbedder(method="fastembed")
vec = embedder.embed_text("Hello world")
```

### VectorEmbeddingManager

Manages embeddings for vector databases.

**Methods:**

| Method | Description |
|--------|-------------|
| `prepare_for_vector_db(embeddings, backend, ...)` | Prepare embeddings for vector DB |
| `batch_prepare(embeddings_list, ...)` | Batch preparation |
| `validate_dimensions(embeddings, expected_dim)` | Validate embedding dimensions |

### GraphEmbeddingManager

Manages embeddings for graph databases.

**Methods:**

| Method | Description |
|--------|-------------|
| `prepare_for_graph_db(entities, relationships, ...)` | Prepare embeddings for graph DB |
| `embed_entities(entities, ...)` | Generate entity embeddings |
| `embed_relationships(relationships, ...)` | Generate relationship embeddings |

---

## Convenience Functions

```python
from semantica.embeddings import embed_text, calculate_similarity, check_available_providers

# Generate embeddings
emb1 = embed_text("text1", method="sentence_transformers")
emb2 = embed_text("text2", method="fastembed")

# Similarity
score = calculate_similarity(emb1, emb2)

# Check available providers
providers = check_available_providers()
if providers["fastembed"]:
    print("FastEmbed is available")
```

---

## Configuration

### Environment Variables

```bash
export EMBEDDING_MODEL=all-MiniLM-L6-v2
export EMBEDDING_DEVICE=cuda
export OPENAI_API_KEY=sk-...
```

### YAML Configuration

```yaml
embeddings:
  text:
    model: all-MiniLM-L6-v2
    method: sentence_transformers
    batch_size: 32
    normalize: true
```

---

## Integration Examples

### Text Embedding with Multiple Methods

```python
from semantica.embeddings import TextEmbedder, check_available_providers
from semantica.vector_store import VectorStore

# Check available providers
providers = check_available_providers()

# Use FastEmbed if available, otherwise sentence-transformers
if providers["fastembed"]:
    embedder = TextEmbedder(method="fastembed", model_name="BAAI/bge-small-en-v1.5")
else:
    embedder = TextEmbedder(method="sentence_transformers")

# Generate embeddings
texts = ["Machine learning", "Artificial intelligence", "Deep learning"]
embeddings = embedder.embed_batch(texts)

# Store in vector database
store = VectorStore()
store.store_vectors(embeddings, metadata=[{"text": t} for t in texts])

# Search
query_emb = embedder.embed_text("neural networks")
results = store.search(query_emb, k=2)
print(f"Found {len(results)} similar texts")
```

### Using Vector Embedding Manager

```python
from semantica.embeddings import VectorEmbeddingManager, TextEmbedder

# Generate embeddings
embedder = TextEmbedder()
texts = ["Document 1", "Document 2", "Document 3"]
embeddings = embedder.embed_batch(texts)

# Prepare for vector database
manager = VectorEmbeddingManager()
formatted = manager.prepare_for_vector_db(
    embeddings,
    metadata=[{"id": i, "text": t} for i, t in enumerate(texts)],
    backend="faiss"
)

# Use formatted data with your vector database
print(f"Prepared {len(formatted['ids'])} vectors for {formatted['backend']}")
```

---

## Best Practices

1.  **Batch Processing**: Always use batch methods (`embed_batch`, `process_batch`) for >1 item. It's much faster on GPU.
2.  **Use Caching**: Embeddings are expensive to compute. Cache them if possible.
3.  **Match Dimensions**: Ensure your vector store is configured with the correct dimension for your chosen model.
4.  **Normalize**: L2 normalization is usually required for Cosine Similarity to work correctly.

---

## See Also

- [Vector Store Module](vector_store.md) - Storing the generated vectors
- [Ingest Module](ingest.md) - Loading data to embed
- [Pipeline Module](pipeline.md) - Orchestrating the embedding process
